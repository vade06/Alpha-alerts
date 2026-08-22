import os
import time
import requests
from bs4 import BeautifulSoup

WEBHOOK = os.environ["WEBHOOK"]
WEBHOOK2 = os.environ["WEBHOOK2"]

URL = "https://www.coinbase.com/blog"
seen = set()

while True:
    try:
        html = requests.get(URL, timeout=10).text
        soup = BeautifulSoup(html, "html.parser")

        for a in soup.find_all("a", href=True):
            title = a.get_text(" ", strip=True)

            if "listing" in title.lower() or "roadmap" in title.lower():
                link = "https://www.coinbase.com" + a["href"]

                if link not in seen:
                    seen.add(link)

                    message = {
                        "content": f"🚨 **{title}**\n{link}"
                    }

                    requests.post(WEBHOOK, json=message, timeout=10)
                    requests.post(WEBHOOK2, json=message, timeout=10)

        time.sleep(30)

    except Exception as e:
        print(f"Error: {e}")
        time.sleep(30)