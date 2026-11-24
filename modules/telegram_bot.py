import os
import requests


def send_telegram_alert(bot_token, chat_id, message, image_path=None):
    """
    Sends a formatted security alert (with optional snapshot photo) to Telegram.
    Returns True if delivered, False otherwise.
    """
    if not bot_token or not chat_id:
        return False

    try:
        if image_path and os.path.exists(image_path):
            url = f"https://api.telegram.org/bot{bot_token}/sendPhoto"
            with open(image_path, "rb") as photo_file:
                resp = requests.post(
                    url,
                    data={"chat_id": chat_id, "caption": message, "parse_mode": "HTML"},
                    files={"photo": photo_file},
                    timeout=10
                )
        else:
            url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
            resp = requests.post(
                url,
                data={"chat_id": chat_id, "text": message, "parse_mode": "HTML"},
                timeout=10
            )
        return resp.status_code == 200
    except Exception:
        return False