import os
import requests
from dotenv import load_dotenv

load_dotenv()

WEBHOOK_URL = os.getenv("DISCORD_SLACK_WEBHOOK_URL", "")

def send_drift_alert(drift_score: float, threshold: float, model_version: str):
    """Sends a alert notification to Discord or Slack webhook."""
    if not WEBHOOK_URL:
        print("[Notifier Warning] DISCORD_SLACK_WEBHOOK_URL not configured. Skipping alert.")
        return

    payload = {
        "content": f" **MLOps Data Drift Detected!** \n"
                   f"• **Model Version:** `{model_version}`\n"
                   f"• **Centroid Cosine Distance:** `{drift_score:.4f}` (Threshold: `{threshold}`)\n"
                   f"• **Status:** Automated Prefect re-training pipeline triggered."
    }

    try:
        response = requests.post(WEBHOOK_URL, json=payload, timeout=5)
        if response.status_code in [200, 204]:
            print("[Notifier] Alert successfully sent to Webhook.")
        else:
            print(f"[Notifier Error] Webhook returned status {response.status_code}")
    except Exception as e:
        print(f"[Notifier Error] Failed to send notification: {e}")