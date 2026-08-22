import os
import json
import time
from pathlib import Path

import requests
from dotenv import load_dotenv

load_dotenv()

WEBHOOKS = [
    url for url in (
        os.getenv("WEBHOOK"),
        os.getenv("WEBHOOK2"),
    )
    if url
]

CHECK_INTERVAL = int(os.getenv("CHECK_INTERVAL", "60"))
STATE_FILE = Path("data/coinbase_state.json")

COINBASE_PRODUCTS_URL = "https://api.exchange.coinbase.com/products"

session = requests.Session()
session.headers.update({
    "User-Agent": "Ade-Market-Monitor/1.0"
})


def check_config():
    if not WEBHOOKS:
        raise RuntimeError(
            "Add WEBHOOK and/or WEBHOOK2 to your environment variables."
        )


def send_discord(title, description, fields=None):
    embed = {
        "title": title,
        "description": description,
        "color": 3447003,
    }

    if fields:
        embed["fields"] = fields

    payload = {"embeds": [embed]}

    for webhook in WEBHOOKS:
        try:
            response = session.post(
                webhook,
                json=payload,
                timeout=15,
            )
            response.raise_for_status()
        except Exception as exc:
            print(f"Webhook error: {exc}")


def get_products():
    response = session.get(
        COINBASE_PRODUCTS_URL,
        timeout=15,
    )
    response.raise_for_status()

    products = response.json()

    online = {}

    for product in products:
        if product.get("status") != "online":
            continue

        product_id = product.get("id")
        base = product.get("base_currency")
        quote = product.get("quote_currency")

        if not product_id or not base or not quote:
            continue

        online[product_id] = {
            "base": base,
            "quote": quote,
        }

    return online


def load_state():
    if not STATE_FILE.exists():
        return None

    try:
        with STATE_FILE.open("r", encoding="utf-8") as file:
            data = json.load(file)

        return {
            "assets": set(data.get("assets", [])),
            "products": set(data.get("products", [])),
        }

    except Exception:
        return None


def save_state(assets, products):
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)

    with STATE_FILE.open("w", encoding="utf-8") as file:
        json.dump(
            {
                "assets": sorted(assets),
                "products": sorted(products),
            },
            file,
            indent=2,
        )


def main():
    check_config()
    state = load_state()

    print("Coinbase webhook monitor starting...")

    while True:
        try:
            online = get_products()

            current_products = set(online)
            current_assets = {
                info["base"]
                for info in online.values()
            }

            if state is None:
                state = {
                    "assets": current_assets,
                    "products": current_products,
                }

                save_state(
                    current_assets,
                    current_products,
                )

                print(
                    f"Baseline saved: "
                    f"{len(current_assets)} assets, "
                    f"{len(current_products)} markets."
                )

                send_discord(
                    "✅ Coinbase monitor online",
                    (
                        f"Baseline created. Monitoring "
                        f"{len(current_assets)} existing assets "
                        f"for future additions."
                    ),
                )

            else:
                new_assets = current_assets - state["assets"]
                new_products = current_products - state["products"]

                for asset in sorted(new_assets):
                    markets = sorted(
                        product_id
                        for product_id, info in online.items()
                        if info["base"] == asset
                    )

                    send_discord(
                        "🚨 New Coinbase asset detected",
                        (
                            f"**{asset}** has appeared as an "
                            f"online Coinbase Exchange asset."
                        ),
                        [
                            {
                                "name": "Markets",
                                "value": (
                                    ", ".join(
                                        f"`{market}`"
                                        for market in markets[:20]
                                    )
                                    or "Unknown"
                                ),
                                "inline": False,
                            }
                        ],
                    )

                    print(
                        f"NEW ASSET: {asset} | "
                        f"{', '.join(markets)}"
                    )

                for product_id in sorted(new_products):
                    if online[product_id]["base"] not in new_assets:
                        print(f"New Coinbase market: {product_id}")

                if new_assets or new_products:
                    state["assets"].update(current_assets)
                    state["products"].update(current_products)

                    save_state(
                        state["assets"],
                        state["products"],
                    )

        except Exception as exc:
            print(f"Monitor error: {exc}")

        time.sleep(CHECK_INTERVAL)


if __name__ == "__main__":
    main()
