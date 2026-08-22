import os
import time
import requests
import psycopg2
from dotenv import load_dotenv

load_dotenv()

WEBHOOKS = [
    url for url in (
        os.getenv("WEBHOOK"),
        os.getenv("WEBHOOK2"),
    )
    if url
]

DATABASE_URL = os.getenv("DATABASE_URL")

CHECK_INTERVAL = int(
    os.getenv("CHECK_INTERVAL", "60")
)

COINBASE_PRODUCTS_URL = (
    "https://api.exchange.coinbase.com/products"
)

session = requests.Session()

session.headers.update({
    "User-Agent": "Ade-Market-Monitor/2.0"
})


def check_config():
    if not WEBHOOKS:
        raise RuntimeError(
            "WEBHOOK or WEBHOOK2 is missing."
        )

    if not DATABASE_URL:
        raise RuntimeError(
            "DATABASE_URL is missing."
        )


def get_database():
    return psycopg2.connect(DATABASE_URL)


def create_table():
    with get_database() as conn:
        with conn.cursor() as cursor:
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS seen_assets (
                    asset_name TEXT PRIMARY KEY
                );
            """)


def get_seen_assets():
    with get_database() as conn:
        with conn.cursor() as cursor:
            cursor.execute(
                "SELECT asset_name FROM seen_assets;"
            )

            rows = cursor.fetchall()

            return {
                row[0]
                for row in rows
            }


def save_asset(asset):
    with get_database() as conn:
        with conn.cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO seen_assets (asset_name)
                VALUES (%s)
                ON CONFLICT (asset_name)
                DO NOTHING;
                """,
                (asset,)
            )


def save_assets(assets):
    with get_database() as conn:
        with conn.cursor() as cursor:
            for asset in assets:
                cursor.execute(
                    """
                    INSERT INTO seen_assets (asset_name)
                    VALUES (%s)
                    ON CONFLICT (asset_name)
                    DO NOTHING;
                    """,
                    (asset,)
                )


def send_discord(
    title,
    description,
    fields=None
):
    embed = {
        "title": title,
        "description": description,
        "color": 3447003
    }

    if fields:
        embed["fields"] = fields

    payload = {
        "embeds": [embed]
    }

    for webhook in WEBHOOKS:
        try:
            response = session.post(
                webhook,
                json=payload,
                timeout=15
            )

            response.raise_for_status()

        except Exception as exc:
            print(
                f"Webhook error: {exc}"
            )


def get_coinbase_products():
    response = session.get(
        COINBASE_PRODUCTS_URL,
        timeout=15
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

        if not product_id:
            continue

        if not base:
            continue

        if not quote:
            continue

        online[product_id] = {
            "base": base,
            "quote": quote
        }

    return online


def main():
    check_config()

    print(
        "Starting Coinbase monitor..."
    )

    create_table()

    seen_assets = get_seen_assets()

    while True:

        try:
            online = get_coinbase_products()

            current_assets = {
                info["base"]
                for info in online.values()
            }

            # FIRST RUN
            #
            # If the database is empty,
            # save everything currently on Coinbase.
            # This prevents hundreds of old alerts.
            if not seen_assets:

                save_assets(
                    current_assets
                )

                seen_assets.update(
                    current_assets
                )

                print(
                    f"Baseline saved to PostgreSQL: "
                    f"{len(current_assets)} assets."
                )

                send_discord(
                    "✅ Coinbase monitor online",
                    (
                        f"Database baseline created.\n\n"
                        f"Monitoring "
                        f"**{len(current_assets)}** "
                        f"existing Coinbase assets "
                        f"for new listings."
                    )
                )

            else:

                new_assets = (
                    current_assets
                    - seen_assets
                )

                for asset in sorted(
                    new_assets
                ):

                    markets = sorted(
                        product_id
                        for product_id, info
                        in online.items()
                        if info["base"] == asset
                    )

                    # Save it permanently
                    # BEFORE the next polling cycle.
                    save_asset(asset)

                    seen_assets.add(asset)

                    market_text = ", ".join(
                        f"`{market}`"
                        for market in markets[:20]
                    )

                    send_discord(
                        "🚨 NEW COINBASE ASSET",
                        (
                            f"**{asset}** has appeared "
                            f"as an online Coinbase "
                            f"Exchange asset."
                        ),
                        [
                            {
                                "name": "Markets",
                                "value": (
                                    market_text
                                    or "Unknown"
                                ),
                                "inline": False
                            }
                        ]
                    )

                    print(
                        f"NEW COINBASE ASSET: "
                        f"{asset}"
                    )

            time.sleep(
                CHECK_INTERVAL
            )

        except Exception as exc:

            print(
                f"Monitor error: {exc}"
            )

            time.sleep(30)


if __name__ == "__main__":
    main()