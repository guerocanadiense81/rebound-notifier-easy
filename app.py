# app.py
# Web service that runs the "Ultimate Lowest Rebound" ENTRY logic (Easy preset) for IOTA/USDT, PEPE/USDT, DOGE/USDT
# and sends Telegram DMs on signals. Designed for Render.com (or any hosts). Python 3.10+.

import os, asyncio, math, time
from datetime import datetime, timezone
from typing import Dict, List, Optional

import ccxt
import numpy as np
import pandas as pd
import requests
from fastapi import FastAPI
from pydantic import BaseModel

# ===================== Config (Easy preset defaults) =====================
# You can override any of these via environment variables.

PAIRS_ENV = os.getenv("PAIRS", "IOTA/USDT,PEPE/USDT,DOGE/USDT")
PAIRS: List[str] = [s.strip().replace(":", "/") for s in PAIRS_ENV.split(",") if s.strip()]

TF      = os.getenv("TF", "4h")     # working timeframe
HTF     = os.getenv("HTF", "1d")    # higher timeframe for trend filter

# ----- Easy profile knobs -----
LOOKBACK_1M_D = int(os.getenv("LOOKBACK_1M_D", "30"))
LOOKBACK_3M_D = int(os.getenv("LOOKBACK_3M_D", "90"))
NEAR_PCT      = float(os.getenv("NEAR_PCT", "2.0"))     # Easy: 2.0
RSI_LEN       = int(os.getenv("RSI_LEN", "14"))
RSI_LEVEL     = float(os.getenv("RSI_LEVEL", "30"))
EMA_LEN       = int(os.getenv("EMA_LEN", "20"))
CHECK_VWAP    = os.getenv("CHECK_VWAP", "false").lower() not in ("0", "false", "no")  # Easy: false
CORR_LEN      = int(os.getenv("CORR_LEN", "50"))
MIN_CORR      = float(os.getenv("MIN_CORR", "0.15"))     # Easy: 0.15
VOL_LEN       = int(os.getenv("VOL_LEN", "20"))
VOL_MULT      = float(os.getenv("VOL_MULT", "1.2"))      # Easy: 1.2
ATR_LEN       = int(os.getenv("ATR_LEN", "14"))
ATR_MULT      = float(os.getenv("ATR_MULT", "1.2"))      # Easy: 1.2
SCORE_NEEDED  = int(os.getenv("SCORE_NEEDED", "5"))      # Easy: 5
TARGET_TPM    = int(os.getenv("TARGET_TPM", "3"))        # Easy: 3 (set 2 if you want fewer)
MAX_PER_MONTH = int(os.getenv("MAX_PER_MONTH", "3"))     # Easy: 3 (set 2 if you want fewer)

# Telegram
TG_BOT_TOKEN = os.getenv("TG_BOT_TOKEN", "")  # set in Render env
TG_CHAT_ID   = os.getenv("TG_CHAT_ID", "")    # set in Render env

# Polling cadence
POLL_SECS    = int(os.getenv("POLL_SECS", "60"))

# ===================== Exchange =====================
binance = ccxt.binance({"enableRateLimit": True})

def fetch_ohlcv(symbol: str, tf: str, limit: int = 1500) -> pd.DataFrame:
    o = binance.fetch_ohlcv(symbol, timeframe=tf, limit=limit)
    df = pd.DataFrame(o, columns=["ts", "open", "high", "low", "close", "volume"])
    df["ts"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
    df.set_index("ts", inplace=True)
    return df

# ===================== Indicators =====================
def rsi(series: pd.Series, length: int = 14) -> pd.Series:
    delta = series.diff()
    up = delta.clip(lower=0.0)
    down = (-delta).clip(lower=0.0)
    roll_up = up.ewm(alpha=1/length, adjust=False).mean()
    roll_down = down.ewm(alpha=1/length, adjust=False).mean().replace(0, np.nan)
    rs = roll_up / roll_down
    return 100 - (100 / (1 + rs))

def ema(series: pd.Series, length: int) -> pd.Series:
    return series.ewm(span=length, adjust=False).mean()

def atr(df: pd.DataFrame, length: int = 14) -> pd.Series:
    hl = df["high"] - df["low"]
    hc = (df["high"] - df["close"].shift()).abs()
    lc = (df["low"] - df["close"].shift()).abs()
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    return tr.ewm(alpha=1/length, adjust=False).mean()

def to_seconds(tf: str) -> int:
    unit = tf[-1]
    val = int(tf[:-1])
    return val * {"s":1, "m":60, "h":3600, "d":86400}[unit]

def bars_from_days(days: int, tf_seconds: int) -> int:
    return max(2, round(days * 86400 / max(1, tf_seconds)))

def monthly_prev_low(df: pd.DataFrame) -> pd.Series:
    # Calendar-month low of previous month, forward-filled to bar index.
    m = df["low"].to_period("M")
    low_by_month = df["low"].groupby(m).min()
    low_prev = low_by_month.shift(1).to_timestamp(how="end")  # align to month end
    # Reindex to original bar index, forward-fill
    low_prev = low_prev.reindex(pd.to_datetime(low_prev.index, utc=True), copy=False)
    # Build a series over df.index: take latest known prev-month-low
    out = pd.Series(index=df.index, dtype=float)
    last_val = np.nan
    for i, ts in enumerate(df.index):
        key = ts.to_period("M").to_timestamp("M").tz_localize("UTC")
        if key in low_prev.index and not np.isnan(low_prev.loc[key]):
            last_val = low_prev.loc[key]
        out.iat[i] = last_val
    return out

def rolling_vwap(df: pd.DataFrame, length: Optional[int] = None) -> pd.Series:
    tp = (df["high"] + df["low"] + df["close"]) / 3.0
    pv = tp * df["volume"]
    if length is None:
        return pv.cumsum() / (df["volume"].replace(0, np.nan).cumsum())
    return pv.rolling(length).sum() / df["volume"].rolling(length).sum()

def is_pivot_low(low: pd.Series, left: int = 2, right: int = 2) -> pd.Series:
    arr = low.values
    out = np.zeros_like(arr, dtype=bool)
    for i in range(left, len(arr) - right):
        window = arr[i-left:i+right+1]
        out[i] = arr[i] == window.min()
    return pd.Series(out, index=low.index)

# ===================== Telegram =====================
def send_telegram(text: str):
    if not TG_BOT_TOKEN or not TG_CHAT_ID:
        print("[Telegram not configured] ", text)
        return
    try:
        url = f"https://api.telegram.org/bot{TG_BOT_TOKEN}/sendMessage"
        requests.post(url, json={"chat_id": TG_CHAT_ID, "text": text, "disable_web_page_preview": True}, timeout=10)
    except Exception as e:
        print("Telegram error:", e)

# ===================== Signal Logic =====================
class State(BaseModel):
    last_entry_bar: Optional[int] = None
    month_key: Optional[int] = None
    trades_this_month: int = 0
    last_signal_iso: Optional[str] = None
    last_score: Optional[int] = None

pair_state: Dict[str, State] = {p: State() for p in PAIRS}

def evaluate_pair(symbol: str):
    sec_per_bar = to_seconds(TF)

    df = fetch_ohlcv(symbol, TF, limit=1500)
    btc = fetch_ohlcv("BTC/USDT", TF, limit=1500)
    htf = fetch_ohlcv(symbol, HTF, limit=400)

    n1 = bars_from_days(LOOKBACK_1M_D, sec_per_bar)
    n3 = bars_from_days(LOOKBACK_3M_D, sec_per_bar)

    low1m = df["low"].rolling(n1).min().shift(1)
    low3m = df["low"].rolling(n3).min().shift(1)
    prev_m_low = monthly_prev_low(df)
    near_band = pd.concat([low1m, low3m, prev_m_low], axis=1).min(axis=1)

    near_ok = df["low"] <= near_band * (1 + NEAR_PCT/100.0)
    rsi_val = rsi(df["close"], RSI_LEN)
    rsi_rebound = (rsi_val.shift(1) < RSI_LEVEL) & (rsi_val >= RSI_LEVEL)

    ema_s = ema(df["close"], EMA_LEN)
    ema_reclaim = (df["close"].shift(1) < ema_s.shift(1)) & (df["close"] >= ema_s)

    if CHECK_VWAP:
        vwap_s = rolling_vwap(df, length=VOL_LEN)
        vwap_reclaim = (df["close"].shift(1) < vwap_s.shift(1)) & (df["close"] >= vwap_s)
    else:
        vwap_reclaim = pd.Series(True, index=df.index)

    # HTF trend (aligned onto TF index)
    htf_ema = ema(htf["close"], EMA_LEN)
    htf_up = (htf["close"] > htf_ema).reindex(df.index, method="ffill")

    # BTC regime & correlation
    btc_ema200 = ema(btc["close"], 200)
    btc_up = (btc["close"] > btc_ema200) & (btc_ema200 > btc_ema200.shift(1))
    btc_up = btc_up.reindex(df.index, method="ffill")

    corr = df["close"].rolling(CORR_LEN).corr(btc["close"].reindex(df.index, method="ffill"))
    corr_ok = corr > MIN_CORR
    btc_regime = btc_up & corr_ok

    # Volume & ATR
    vol_ma = df["volume"].rolling(VOL_LEN).mean()
    vol_climax = df["volume"] > vol_ma * VOL_MULT

    atr_s = atr(df, ATR_LEN)
    atr_ma = atr_s.rolling(ATR_LEN).mean()
    atr_spike = atr_s > atr_ma * ATR_MULT
    atr_cooling = atr_spike.shift(1).fillna(False) & (atr_s < atr_s.shift(1))

    # Candle patterns
    body = (df["close"] - df["open"]).abs()
    upper = df["high"] - np.maximum(df["close"], df["open"])
    lower = np.minimum(df["close"], df["open"]) - df["low"]
    rng = df["high"] - df["low"]
    hammer = (rng > 0) & (lower >= body*2) & (upper <= body)
    engulf = (df["close"] > df["open"]) & (df["close"].shift(1) < df["open"].shift(1)) & \
             (df["close"] >= df["open"].shift(1)) & (df["open"] <= df["close"].shift(1))
    pattern_ok = hammer | engulf

    piv = is_pivot_low(df["low"])
    score = (near_ok.astype(int) + piv.astype(int) + rsi_rebound.astype(int) + ema_reclaim.astype(int) +
             vwap_reclaim.astype(int) + htf_up.astype(int) + btc_regime.astype(int) +
             vol_climax.astype(int) + (atr_spike | atr_cooling).astype(int) + pattern_ok.astype(int))

    base_signal = score >= SCORE_NEEDED

    # Auto cooldown
    bars_per_month = max(10, round(30*24*3600 / max(1, sec_per_bar)))
    cooldown_bars = max(5, round(bars_per_month / max(1, TARGET_TPM)))

    # Monthly key
    st = pair_state[symbol]
    now_idx = df.index[-1]
    now_bar = len(df) - 1
    current_month_key = now_idx.year * 100 + now_idx.month
    if st.month_key is None or st.month_key != current_month_key:
        st.month_key = current_month_key
        st.trades_this_month = 0

    cooldown_ok = (st.last_entry_bar is None) or ((now_bar - st.last_entry_bar) >= cooldown_bars)
    month_ok = st.trades_this_month < MAX_PER_MONTH

    signal_now = bool(base_signal.iloc[-1] and cooldown_ok and month_ok)

    if signal_now:
        st.last_entry_bar = now_bar
        st.trades_this_month += 1
        st.last_signal_iso = now_idx.isoformat()
        st.last_score = int(score.iloc[-1])
        msg = (f"ENTRY (Easy) — {symbol} {TF}\n"
               f"score={st.last_score}/10  corr={corr.iloc[-1]:.2f}\n"
               f"close={df['close'].iloc[-1]:.6g}  nearBand={near_band.iloc[-1]:.6g}")
        print(msg, flush=True)
        send_telegram(msg)

# ===================== Web app / background loop =====================
app = FastAPI(title="Rebound Notifier (Easy)")

@app.get("/")
def root():
    return {
        "status": "ok",
        "preset": "Easy",
        "pairs": PAIRS,
        "tf": TF,
        "htf": HTF,
        "last": {p: pair_state[p].dict() for p in PAIRS},
    }

@app.get("/trigger")
def trigger_once():
    for p in PAIRS:
        try:
            evaluate_pair(p)
        except Exception as e:
            print(f"Trigger error on {p}: {e}", flush=True)
    return {"ok": True}

async def loop():
    print("Notifier loop started (Easy preset).", flush=True)
    while True:
        try:
            for p in PAIRS:
                evaluate_pair(p)
        except Exception as e:
            print("Loop error:", e, flush=True)
        await asyncio.sleep(POLL_SECS)

@app.on_event("startup")
async def startup_event():
    asyncio.create_task(loop())
