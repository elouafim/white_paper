"""
main.py -- Point d entree GitHub Actions
==========================================
Utilise l API publique Binance/KuCoin (pas de cle requise).
"""

import sys
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
        # 1. Donnees (API publique, pas de cle Binance requise)
        print("  Telechargement donnees...")
        data_15m, data_1h, data_4h = get_data()
        btc_price = data_15m["close"].iloc[-1]

        # 2. Etat
        state = load_state()
        update_trailing(state, btc_price)

        # 3. Signal
        sig = get_current_signal(data_15m, data_1h, data_4h, state)
        print("  RSI={} H4={} H1={} ATR={}% long={} short={} exit={}".format(
            sig["rsi"], sig["h4_trend"], sig["h1_momentum"],
            sig["atr_pct"], sig["long_signal"],
            sig["short_signal"], sig["exit_signal"]))

        # 4. Traitement
        state = process_signal(state, sig)

        # 5. Sauvegarde
        save_state(state)
        print("  Etat sauvegarde.")

        # 6. Rapport quotidien a 08:00 UTC
        if now.hour == 8 and now.minute < 15:
            winrate   = 0.0
            if state["n_trades"] > 0:
                winrate = state["n_wins"] / state["n_trades"] * 100
            total_ret = (state["capital"] / 500 - 1) * 100
            notify_daily_report(state["capital"], state["n_trades"],
                                winrate, total_ret, btc_price)
            print("  Rapport quotidien envoye.")

        print("  Termine.")
        sys.exit(0)

    except Exception as e:
        print("  ERREUR : {}".format(str(e)))
        notify_error(str(e))
        sys.exit(1)

if __name__ == "__main__":
    main()
