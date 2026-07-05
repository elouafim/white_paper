"""
paper_trading_live.py -- Swing RSI BTC/USDT
=============================================
Correction : utilise l API publique Binance via requests
(pas de bibliotheque python-binance qui declenche le blocage geo)
ou fallback sur KuCoin si Binance bloque.
"""

import pandas as pd
import numpy as np
import requests
from datetime import datetime, timezone
import os
import json
import time
import warnings
warnings.filterwarnings("ignore")

TRAILING_PCT  = 0.035
COOLDOWN_H    = 32
MIN_ATR_PCT   = 0.0015
RSI_OS        = 35
RSI_OB        = 65
CAPITAL_PAPER = 500
STATE_FILE    = "paper_state.json"
JOURNAL_FILE  = "paper_journal.csv"

# ============================================================
# DONNEES -- API publique (pas de cle requise)
# ============================================================

BINANCE_BASE  = "https://api.binance.com"
KUCOIN_BASE   = "https://api.kucoin.com"

def get_data_binance(symbol, interval, limit=500):
    """
    API publique Binance -- pas de cle, pas de restriction geo
    sur les endpoints publics de certains serveurs.
    interval : 15m, 1h, 4h
    """
    url    = "{}/api/v3/klines".format(BINANCE_BASE)
    params = {"symbol": symbol, "interval": interval, "limit": limit}
    resp   = requests.get(url, params=params, timeout=15)
    resp.raise_for_status()
    data   = resp.json()

    df = pd.DataFrame(data, columns=[
        "timestamp","open","high","low","close","volume",
        "close_time","quote_vol","trades","taker_buy_base","taker_buy_quote","ignore"
    ])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df = df.set_index("timestamp")
    df = df[["open","high","low","close","volume"]].astype(float)
    return df.dropna()

def get_data_kucoin(symbol, interval, limit=500):
    """
    Fallback KuCoin si Binance inaccessible.
    symbol   : BTC-USDT
    interval : 15min, 1hour, 4hour
    """
    url    = "{}/api/v1/market/candles".format(KUCOIN_BASE)
    params = {"symbol": symbol, "type": interval}
    resp   = requests.get(url, params=params, timeout=15)
    resp.raise_for_status()
    data   = resp.json().get("data", [])

    df = pd.DataFrame(data, columns=[
        "timestamp","open","close","high","low","volume","amount"
    ])
    df["timestamp"] = pd.to_datetime(df["timestamp"].astype(int), unit="s")
    df = df.set_index("timestamp")
    df = df[["open","high","low","close","volume"]].astype(float)
    df = df.sort_index()
    return df.tail(limit).dropna()

def get_data(interval_15m="15m", interval_1h="1h", interval_4h="4h"):
    """
    Essaie Binance public d abord, fallback KuCoin.
    Retourne (data_15m, data_1h, data_4h)
    """
    # Map intervalles Binance -> KuCoin
    kucoin_map = {"15m": "15min", "1h": "1hour", "4h": "4hour"}

    try:
        print("  Source : Binance API publique...")
        d15 = get_data_binance("BTCUSDT", interval_15m, limit=300)
        d1h = get_data_binance("BTCUSDT", interval_1h,  limit=200)
        d4h = get_data_binance("BTCUSDT", interval_4h,  limit=200)
        print("  Binance OK -- BTC : {:,.0f} USDT".format(d15["close"].iloc[-1]))
        return d15, d1h, d4h

    except Exception as e:
        print("  Binance indisponible : {}".format(str(e)))
        print("  Fallback : KuCoin API publique...")
        try:
            d15 = get_data_kucoin("BTC-USDT", kucoin_map[interval_15m], limit=300)
            d1h = get_data_kucoin("BTC-USDT", kucoin_map[interval_1h],  limit=200)
            d4h = get_data_kucoin("BTC-USDT", kucoin_map[interval_4h],  limit=200)
            print("  KuCoin OK -- BTC : {:,.0f} USDT".format(d15["close"].iloc[-1]))
            return d15, d1h, d4h
        except Exception as e2:
            raise RuntimeError("Binance et KuCoin inaccessibles : {}".format(str(e2)))

# ============================================================
# INDICATEURS
# ============================================================

def compute_rsi(close, window=14):
    delta = close.diff()
    gain  = delta.clip(lower=0).rolling(window).mean()
    loss  = (-delta.clip(upper=0)).rolling(window).mean()
    return 100 - (100 / (1 + gain / loss))

def compute_ema(close, span):
    return close.ewm(span=span, adjust=False).mean()

def compute_atr(df, window=14):
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - df["close"].shift()).abs(),
        (df["low"]  - df["close"].shift()).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(window).mean()

def resample(series, idx):
    return series.reindex(idx, method="ffill")

def get_h4_trend(data_4h, data_15m):
    e20 = compute_ema(data_4h["close"], 20)
    e50 = compute_ema(data_4h["close"], 50)
    t   = pd.Series(0.0, index=data_4h.index)
    t[e20 > e50] =  1.0
    t[e20 < e50] = -1.0
    return resample(t, data_15m.index)

def get_h1_momentum(data_1h, data_15m):
    rsi = compute_rsi(data_1h["close"], 14)
    m   = pd.Series(0.0, index=data_1h.index)
    m[rsi > 52] =  1.0
    m[rsi < 48] = -1.0
    return resample(m, data_15m.index)

# ============================================================
# ETAT
# ============================================================

def load_state():
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE) as f:
            return json.load(f)
    return {
        "position"        : 0,
        "entry_price"     : None,
        "entry_time"      : None,
        "highest_price"   : None,
        "lowest_price"    : None,
        "last_entry_time" : None,
        "capital"         : CAPITAL_PAPER,
        "n_trades"        : 0,
        "n_wins"          : 0,
        "paper_start"     : datetime.now(timezone.utc).isoformat(),
    }

def save_state(state):
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2, default=str)

def append_journal(row_dict):
    import csv
    file_exists = os.path.exists(JOURNAL_FILE)
    with open(JOURNAL_FILE, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "date","type","price_signal","direction","pnl_pct","capital_after","notes"
        ])
        if not file_exists:
            writer.writeheader()
        writer.writerow(row_dict)

# ============================================================
# SIGNAL
# ============================================================

def get_current_signal(data_15m, data_1h, data_4h, state):
    close    = data_15m["close"]
    rsi_15m  = compute_rsi(close, 14)
    h4_trend = get_h4_trend(data_4h, data_15m)
    h1_mom   = get_h1_momentum(data_1h, data_15m)
    atr      = compute_atr(data_15m)

    i  = -2
    t  = data_15m.index[i]
    cl = close.iloc[i]
    r  = rsi_15m.iloc[i]
    h4 = h4_trend.iloc[i]
    h1 = h1_mom.iloc[i]
    a  = atr.iloc[i]

    greens  = (close.iloc[i] > close.iloc[i-1]) and (close.iloc[i-1] > close.iloc[i-2])
    reds    = (close.iloc[i] < close.iloc[i-1]) and (close.iloc[i-1] < close.iloc[i-2])
    atr_ok  = (a / cl) >= MIN_ATR_PCT
    atr_pct = round(a / cl * 100, 3)

    cooldown_ok = True
    elapsed_h   = 999.0
    if state["last_entry_time"]:
        last_ts     = pd.Timestamp(state["last_entry_time"])
        elapsed_h   = (t - last_ts).total_seconds() / 3600
        cooldown_ok = elapsed_h >= COOLDOWN_H

    trail_hit        = False
    trail_stop_price = None
    if state["position"] == 1 and state["highest_price"]:
        trail_stop_price = state["highest_price"] * (1 - TRAILING_PCT)
        trail_hit        = cl < trail_stop_price
    elif state["position"] == -1 and state["lowest_price"]:
        trail_stop_price = state["lowest_price"] * (1 + TRAILING_PCT)
        trail_hit        = cl > trail_stop_price

    inverse_exit = False
    if state["position"] == 1  and h4 == -1 and h1 == -1:
        inverse_exit = True
    if state["position"] == -1 and h4 ==  1 and h1 ==  1:
        inverse_exit = True

    long_signal  = (h4 == 1 and h1 >= 0 and r > RSI_OS and greens
                    and atr_ok and cooldown_ok and state["position"] == 0)
    short_signal = (h4 == -1 and h1 <= 0 and r < RSI_OB and reds
                    and atr_ok and cooldown_ok and state["position"] == 0)
    exit_signal  = (trail_hit or inverse_exit) and state["position"] != 0

    return {
        "timestamp"       : t,
        "close"           : cl,
        "rsi"             : round(r, 1),
        "h4_trend"        : h4,
        "h1_momentum"     : h1,
        "atr_pct"         : atr_pct,
        "atr_ok"          : atr_ok,
        "greens"          : greens,
        "reds"            : reds,
        "cooldown_ok"     : cooldown_ok,
        "elapsed_h"       : round(elapsed_h, 1),
        "trail_stop_price": trail_stop_price,
        "trail_hit"       : trail_hit,
        "inverse_exit"    : inverse_exit,
        "long_signal"     : long_signal,
        "short_signal"    : short_signal,
        "exit_signal"     : exit_signal,
    }

def update_trailing(state, current_price):
    if state["position"] == 1 and state["highest_price"]:
        if current_price > state["highest_price"]:
            state["highest_price"] = current_price
    elif state["position"] == -1 and state["lowest_price"]:
        if current_price < state["lowest_price"]:
            state["lowest_price"] = current_price

# ============================================================
# TRAITEMENT DU SIGNAL
# ============================================================

def process_signal(state, sig):
    from telegram_notify import notify_signal

    if sig["exit_signal"] and state["position"] != 0:
        exec_price = sig["close"]
        if state["position"] == 1:
            pnl = (exec_price - state["entry_price"]) / state["entry_price"] * 100
        else:
            pnl = (state["entry_price"] - exec_price) / state["entry_price"] * 100
        pnl_net = pnl - 0.15

        state["capital"]  *= (1 + pnl_net / 100)
        state["n_trades"] += 1
        if pnl_net > 0:
            state["n_wins"] += 1

        note = "trailing_stop" if sig["trail_hit"] else "inverse_exit"
        append_journal({
            "date"         : str(sig["timestamp"]),
            "type"         : "EXIT",
            "price_signal" : sig["close"],
            "direction"    : state["position"],
            "pnl_pct"      : round(pnl_net, 3),
            "capital_after": round(state["capital"], 2),
            "notes"        : note,
        })
        trail_stop = sig["trail_stop_price"] or 0
        notify_signal("EXIT", sig["close"], trail_stop,
                      state["capital"], state["n_trades"])
        print("  EXIT -- PnL net : {:+.2f}%  Capital : {:,.0f}$".format(
            pnl_net, state["capital"]))

        state["position"]      = 0
        state["entry_price"]   = None
        state["entry_time"]    = None
        state["highest_price"] = None
        state["lowest_price"]  = None

    elif sig["long_signal"] and state["position"] == 0:
        exec_price               = sig["close"]
        state["position"]        = 1
        state["entry_price"]     = exec_price
        state["entry_time"]      = str(sig["timestamp"])
        state["last_entry_time"] = str(sig["timestamp"])
        state["highest_price"]   = exec_price
        state["lowest_price"]    = exec_price
        trail_stop = exec_price * (1 - TRAILING_PCT)
        append_journal({
            "date"         : str(sig["timestamp"]),
            "type"         : "ENTRY",
            "price_signal" : exec_price,
            "direction"    : 1,
            "pnl_pct"      : None,
            "capital_after": round(state["capital"], 2),
            "notes"        : "LONG",
        })
        notify_signal("LONG", exec_price, trail_stop,
                      state["capital"], state["n_trades"])
        print("  LONG enregistre a {:,.0f} USDT".format(exec_price))

    elif sig["short_signal"] and state["position"] == 0:
        exec_price               = sig["close"]
        state["position"]        = -1
        state["entry_price"]     = exec_price
        state["entry_time"]      = str(sig["timestamp"])
        state["last_entry_time"] = str(sig["timestamp"])
        state["highest_price"]   = exec_price
        state["lowest_price"]    = exec_price
        trail_stop = exec_price * (1 + TRAILING_PCT)
        append_journal({
            "date"         : str(sig["timestamp"]),
            "type"         : "ENTRY",
            "price_signal" : exec_price,
            "direction"    : -1,
            "pnl_pct"      : None,
            "capital_after": round(state["capital"], 2),
            "notes"        : "SHORT",
        })
        notify_signal("SHORT", exec_price, trail_stop,
                      state["capital"], state["n_trades"])
        print("  SHORT enregistre a {:,.0f} USDT".format(exec_price))

    else:
        print("  Pas de signal -- pos={} long={} short={} exit={}".format(
            state["position"], sig["long_signal"],
            sig["short_signal"], sig["exit_signal"]))

    return state
