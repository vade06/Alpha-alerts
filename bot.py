import os
import re
import time
import requests
import psycopg2

from bs4 import BeautifulSoup
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

COINBASE_ROADMAP_URL = (
    "https://www.coinbase.com/blog/"
    "increasing-transparency-for-new-asset-listings-on-coinbase"
)

session = requests.Session()

session.headers.update({
    "User-Agent": "Ade-Alpha-Alerts/3.0"
})


# --------------------------------------------------
# CONFIG
# --------------------------------------------------

def check_config():
    if not WEBHOOKS:
        raise RuntimeError(
            "WEBHOOK or WEBHOOK2 is missing."
        )

    if not DATABASE_URL:
        raise RuntimeError(
            "DATABASE_URL is missing."
        )


# --------------------------------------------------
# DATABASE
# --------------------------------------------------

def get_database():
    return psycopg2.connect(DATABASE_URL)


def create_tables():
    with get_database() as conn:
        with conn.cursor() as cursor:

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS seen_assets (
                    asset_name TEXT PRIMARY KEY
                );
            """)

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS seen_roadmap_assets (
                    asset_key TEXT PRIMARY KEY,
                    asset_name TEXT,
                    symbol TEXT,
                    first_seen TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)


def get_seen_assets():
    with get_database() as conn:
        with conn.cursor() as cursor:

            cursor.execute(
                "SELECT asset_name FROM seen_assets;"
            )

            return {
                row[0]
                for row in cursor.fetchall()
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


def get_seen_roadmap_assets():
    with get_database() as conn:
        with conn.cursor() as cursor:

            cursor.execute("""
                SELECT asset_key
                FROM seen_roadmap_assets;
            """)

            return {
                row[0]
                for row in cursor.fetchall()
            }


def save_roadmap_asset(
    asset_key,
    asset_name,
    symbol
):
    with get_database() as conn:
        with conn.cursor() as cursor:

            cursor.execute(
                """
                INSERT INTO seen_roadmap_assets (
                    asset_key,
                    asset_name,
                    symbol
                )
                VALUES (%s, %s, %s)
                ON CONFLICT (asset_key)
                DO NOTHING;
                """,
                (
                    asset_key,
                    asset_name,
                    symbol
                )
            )


# --------------------------------------------------
# DISCORD
# --------------------------------------------------

def send_discord(
    title,
    description,
    fields=None,
    color=3447003
):

    embed = {
        "title": title,
        "description": description,
        "color": color
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


# --------------------------------------------------
# COINBASE LIVE PRODUCTS
# --------------------------------------------------

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


# --------------------------------------------------
# COINBASE ROADMAP
# --------------------------------------------------

def get_coinbase_roadmap():

    response = session.get(
        COINBASE_ROADMAP_URL,
        timeout=20
    )

    response.raise_for_status()

    soup = BeautifulSoup(
        response.text,
        "html.parser"
    )

    text = soup.get_text(
        "\n",
        strip=True
    )

    # Limit parsing to the roadmap section.
    if "Roadmap" in text:
        text = text.split(
            "Roadmap",
            1
        )[1]

    if "Why have you decided" in text:
        text = text.split(
            "Why have you decided",
            1
        )[0]

    assets = []

    # Looks for entries such as:
    # Arcium (ARX)
    #
    # This intentionally keeps parsing simple.
    pattern = re.compile(
        r"([A-Za-z0-9 .+'\-]{2,80})"
        r"\s*\(([A-Z0-9]{2,15})\)"
    )

    matches = pattern.findall(text)

    for name, symbol in matches:

        name = name.strip()
        symbol = symbol.strip().upper()

        # Filter common false matches.
        if len(name) > 80:
            continue

        asset_key = (
            f"{symbol}:{name}".lower()
        )

        assets.append({
            "key": asset_key,
            "name": name,
            "symbol": symbol
        })

    # Deduplicate
    unique = {}

    for asset in assets:
        unique[asset["key"]] = asset

    return list(
        unique.values()
    )


# --------------------------------------------------
# MAIN
# --------------------------------------------------

def main():

    check_config()

    print(
        "Starting Alpha Alerts..."
    )

    create_tables()

    seen_assets = get_seen_assets()

    seen_roadmap = (
        get_seen_roadmap_assets()
    )

    roadmap_initialized = bool(
        seen_roadmap
    )

    while True:

        # ==========================================
        # ROADMAP CHECK
        # ==========================================

        try:

            roadmap_assets = (
                get_coinbase_roadmap()
            )

            current_roadmap_keys = {
                asset["key"]
                for asset in roadmap_assets
            }

            # First roadmap run:
            # learn existing entries without spam.
            if not roadmap_initialized:

                for asset in roadmap_assets:

                    save_roadmap_asset(
                        asset["key"],
                        asset["name"],
                        asset["symbol"]
                    )

                    seen_roadmap.add(
                        asset["key"]
                    )

                roadmap_initialized = True

                print(
                    f"Roadmap baseline created: "
                    f"{len(roadmap_assets)} assets."
                )

            else:

                new_roadmap = [
                    asset
                    for asset in roadmap_assets
                    if asset["key"]
                    not in seen_roadmap
                ]

                for asset in new_roadmap:

                    save_roadmap_asset(
                        asset["key"],
                        asset["name"],
                        asset["symbol"]
                    )

                    seen_roadmap.add(
                        asset["key"]
                    )

                    send_discord(
                        "📢 COINBASE ROADMAP ADDITION",
                        (
                            f"**{asset['name']} "
                            f"({asset['symbol']})** "
                            f"has appeared on Coinbase's "
                            f"asset listing roadmap."
                        ),
                        [
                            {
                                "name":
                                    "Status",
                                "value":
                                    "Decision to list / roadmap stage",
                                "inline":
                                    False
                            },
                            {
                                "name":
                                    "Important",
                                "value":
                                    (
                                        "Trading is not live yet. "
                                        "The monitor will alert again "
                                        "if an online Coinbase market "
                                        "appears."
                                    ),
                                "inline":
                                    False
                            }
                        ],
                        color=16753920
                    )

                    print(
                        "NEW ROADMAP ASSET: "
                        f"{asset['name']} "
                        f"({asset['symbol']})"
                    )

        except Exception as exc:

            print(
                f"Roadmap monitor error: {exc}"
            )

        # ==========================================
        # LIVE TRADING CHECK
        # ==========================================

        try:

            online = get_coinbase_products()

            current_assets = {
                info["base"]
                for info in online.values()
            }

            # Only relevant if this is a brand-new DB.
            if not seen_assets:

                save_assets(
                    current_assets
                )

                seen_assets.update(
                    current_assets
                )

                print(
                    f"Trading baseline created: "
                    f"{len(current_assets)} assets."
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

                    save_asset(asset)

                    seen_assets.add(asset)

                    market_text = ", ".join(
                        f"`{market}`"
                        for market in markets[:20]
                    )

                    send_discord(
                        "🚨 COINBASE TRADING LIVE",
                        (
                            f"**{asset}** has appeared "
                            f"as an online Coinbase "
                            f"Exchange asset."
                        ),
                        [
                            {
                                "name":
                                    "Markets",
                                "value":
                                    (
                                        market_text
                                        or "Unknown"
                                    ),
                                "inline":
                                    False
                            },
                            {
                                "name":
                                    "Stage",
                                "value":
                                    "Online Coinbase market detected",
                                "inline":
                                    False
                            }
                        ],
                        color=5763719
                    )

                    print(
                        "COINBASE TRADING LIVE: "
                        f"{asset}"
                    )

        except Exception as exc:

            print(
                f"Trading monitor error: {exc}"
            )

        time.sleep(
            CHECK_INTERVAL
        )


if __name__ == "__main__":
    main()