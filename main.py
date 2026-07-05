"""
main.py -- Point d entree GitHub Actions
==========================================
GitHub Actions lance ce script toutes les 15 minutes.
Une seule execution, pas de boucle infinie.

Workflow :
  1. Charge les donnees Binance
  2. Calcule le signal
  3. Si signal : met a jour l etat + envoie notification Telegram
  4. Sauvegarde l etat dans paper_state.json
  5. GitHub Actions commite le fichier mis a jour
"""

import sys
import os
from datetime import datetime, timezone

from paper_trading_live import (
    get_data,
    load_state,
    save_state,
    update_trailing,
    get_current_signal,
    process_signal,
)
from telegram_notify import notify_daily_report, notify_error

def main():
    now = datetime.now(timezone.utc)
    print("=" * 56)
    print("  Swing RSI Paper Trading -- {}".format(
        now.strftime("%Y-%m-%d %H:%M UTC")))
    print("=" * 56)

    try:
        # 1. Donnees
        print("  Telechargement donnees Binance...")
        data_15m = get_data("BTCUSDT", "15m", "60 days ago UTC")
        data_1h  = get_data("BTCUSDT", "1h",  "90 days ago UTC")
        data_4h  = get_data("BTCUSDT", "4h",  "180 days ago UTC")
        btc_price = data_15m["close"].iloc[-1]
        print("  BTC : {:,.0f} USDT".format(btc_price))

        # 2. Etat actuel
        state = load_state()
        update_trailing(state, btc_price)

        # 3. Signal
        sig = get_current_signal(data_15m, data_1h, data_4h, state)
        print("  Signal : long={} short={} exit={}".format(
            sig["long_signal"], sig["short_signal"], sig["exit_signal"]))
        print("  RSI={} H4={} H1={} ATR={}%".format(
            sig["rsi"], sig["h4_trend"], sig["h1_momentum"], sig["atr_pct"]))

        # 4. Traitement
        state = process_signal(state, sig)

        # 5. Sauvegarde
        save_state(state)
        print("  Etat sauvegarde.")

        # 6. Rapport quotidien (a 08:00 UTC chaque jour)
        if now.hour == 8 and now.minute < 15:
            winrate = 0.0
            if state["n_trades"] > 0:
                winrate = state["n_wins"] / state["n_trades"] * 100
            total_ret = (state["capital"] / 500 - 1) * 100
            notify_daily_report(
                state["capital"], state["n_trades"],
                winrate, total_ret, btc_price
            )
            print("  Rapport quotidien envoye.")

        print("  Termine.")
        sys.exit(0)

    except Exception as e:
        error_msg = str(e)
        print("  ERREUR : {}".format(error_msg))
        notify_error(error_msg)
        sys.exit(1)

if __name__ == "__main__":
    main()
