import os, time, requests
from bs4 import BeautifulSoup

WEBHOOK = os.environ["WEBHOOK"]
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
                    requests.post(WEBHOOK, json={
                        "content": f"🚨 **{title}**\n{link}"
                    })
        time.sleep(30)
    except:
        time.sleep(30)