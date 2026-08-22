import os
import requests

WEBHOOK = os.environ["WEBHOOK"]

requests.post(WEBHOOK, json={
    "content": "✅ Alpha Alerts is ONLINE!"
})

print("Sent successfully")