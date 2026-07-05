"""
telegram_notify.py -- Notifications Telegram
=============================================
Envoie un message Telegram quand un signal est detecte.

Setup :
  1. Parle a @BotFather sur Telegram -> /newbot -> copie le token
  2. Envoie un message a ton bot
  3. Va sur https://api.telegram.org/bot<TOKEN>/getUpdates
     -> copie le chat_id
  4. Ajoute dans GitHub Secrets :
       TELEGRAM_TOKEN = ton_token
       TELEGRAM_CHAT_ID = ton_chat_id
"""

import os
import requests

TELEGRAM_TOKEN   = os.getenv("TELEGRAM_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")

def send_message(text):
    """Envoie un message Telegram. Retourne True si succes."""
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        print("  [Telegram] Variables non configurees, notification ignoree.")
        return False
    try:
        url  = "https://api.telegram.org/bot{}/sendMessage".format(TELEGRAM_TOKEN)
        data = {"chat_id": TELEGRAM_CHAT_ID, "text": text, "parse_mode": "HTML"}
        resp = requests.post(url, data=data, timeout=10)
        if resp.status_code == 200:
            print("  [Telegram] Message envoye.")
            return True
        else:
            print("  [Telegram] Erreur HTTP {}".format(resp.status_code))
            return False
    except Exception as e:
        print("  [Telegram] Erreur : {}".format(str(e)))
        return False

def notify_signal(sig_type, price, trailing_stop, capital, n_trades):
    """Signal d entree ou de sortie."""
    icons = {"LONG": "LONG", "SHORT": "SHORT", "EXIT": "SORTIE"}
    icon  = icons.get(sig_type, sig_type)

    if sig_type == "LONG":
        msg = (
            "<b>SIGNAL {} BTC/USDT</b>\n"
            "Prix signal   : {:,.0f} USDT\n"
            "Trailing stop : {:,.0f} USDT (-3.5%)\n"
            "Capital paper : {:,.0f}$\n"
            "Trades total  : {}"
        ).format(icon, price, trailing_stop, capital, n_trades)

    elif sig_type == "SHORT":
        msg = (
            "<b>SIGNAL {} BTC/USDT</b>\n"
            "Prix signal   : {:,.0f} USDT\n"
            "Trailing stop : {:,.0f} USDT (+3.5%)\n"
            "Capital paper : {:,.0f}$\n"
            "Trades total  : {}"
        ).format(icon, price, trailing_stop, capital, n_trades)

    else:  # EXIT
        msg = (
            "<b>SIGNAL {} BTC/USDT</b>\n"
            "Prix signal   : {:,.0f} USDT\n"
            "Capital paper : {:,.0f}$\n"
            "Trades total  : {}"
        ).format(icon, price, capital, n_trades)

    return send_message(msg)

def notify_daily_report(capital, n_trades, winrate, total_return, btc_price):
    """Rapport quotidien automatique."""
    msg = (
        "<b>Rapport quotidien -- Swing RSI BTC</b>\n"
        "Capital simule : {:,.0f}$  ({:+.1f}%)\n"
        "Trades         : {}  |  WR : {:.1f}%\n"
        "BTC actuel     : {:,.0f} USDT"
    ).format(capital, total_return, n_trades, winrate, btc_price)
    return send_message(msg)

def notify_error(error_msg):
    """Alerte en cas d erreur."""
    msg = "<b>ERREUR Paper Trading</b>\n{}".format(error_msg)
    return send_message(msg)
