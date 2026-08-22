import os
import time
import requests
import psycopg2

from dotenv import load_dotenv

load_dotenv()


# =========================================================
# CONFIG
# =========================================================

# Existing Coinbase/private webhook + friends webhook
WEBHOOKS = [
    url for url in (
        os.getenv("WEBHOOK"),
        os.getenv("WEBHOOK2"),
    )
    if url
]

# New private crypto-only webhook
PRIVATE_CRYPTO_WEBHOOK = os.getenv(
    "PRIVATE_CRYPTO_WEBHOOK"
)

DATABASE_URL = os.getenv("DATABASE_URL")

CHECK_INTERVAL = int(
    os.getenv("CHECK_INTERVAL", "60")
)

CRYPTO_SYMBOLS = [
    symbol.strip().upper()
    for symbol in os.getenv(
        "CRYPTO_SYMBOLS",
        "BTC,ETH,SOL"
    ).split(",")
    if symbol.strip()
]

CRYPTO_SPIKE_PERCENT = float(
    os.getenv(
        "CRYPTO_SPIKE_PERCENT",
        "5"
    )
)

CRYPTO_WINDOW_MINUTES = int(
    os.getenv(
        "CRYPTO_WINDOW_MINUTES",
        "15"
    )
)

CRYPTO_ALERT_COOLDOWN_MINUTES = int(
    os.getenv(
        "CRYPTO_ALERT_COOLDOWN_MINUTES",
        "30"
    )
)

COINBASE_PRODUCTS_URL = (
    "https://api.exchange.coinbase.com/products"
)

COINBASE_TICKER_URL = (
    "https://api.exchange.coinbase.com/"
    "products/{product_id}/ticker"
)

session = requests.Session()

session.headers.update({
    "User-Agent": "Alpha-Alerts/5.0"
})


# =========================================================
# CONFIG CHECK
# =========================================================

def check_config():

    if not WEBHOOKS:
        raise RuntimeError(
            "WEBHOOK or WEBHOOK2 is missing."
        )

    if not DATABASE_URL:
        raise RuntimeError(
            "DATABASE_URL is missing."
        )


# =========================================================
# DATABASE
# =========================================================

def get_database():

    return psycopg2.connect(
        DATABASE_URL
    )


def create_tables():

    with get_database() as conn:

        with conn.cursor() as cursor:

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS seen_assets (
                    asset_name TEXT PRIMARY KEY
                );
            """)

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS crypto_price_samples (
                    symbol TEXT NOT NULL,
                    price NUMERIC NOT NULL,
                    sampled_at TIMESTAMP
                    DEFAULT CURRENT_TIMESTAMP
                );
            """)

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS crypto_alerts (
                    symbol TEXT PRIMARY KEY,
                    last_alerted_at TIMESTAMP,
                    last_alert_percent NUMERIC
                );
            """)


# =========================================================
# COINBASE LISTING DATABASE
# =========================================================

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
                INSERT INTO seen_assets (
                    asset_name
                )
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
                    INSERT INTO seen_assets (
                        asset_name
                    )
                    VALUES (%s)
                    ON CONFLICT (asset_name)
                    DO NOTHING;
                    """,
                    (asset,)
                )


# =========================================================
# CRYPTO DATABASE
# =========================================================

def save_crypto_sample(
    symbol,
    price
):

    with get_database() as conn:

        with conn.cursor() as cursor:

            cursor.execute(
                """
                INSERT INTO crypto_price_samples (
                    symbol,
                    price
                )
                VALUES (%s, %s);
                """,
                (
                    symbol,
                    price
                )
            )


def get_old_crypto_price(
    symbol
):

    with get_database() as conn:

        with conn.cursor() as cursor:

            cursor.execute(
                """
                SELECT
                    price,
                    sampled_at
                FROM crypto_price_samples
                WHERE
                    symbol = %s
                    AND sampled_at <=
                        NOW() - (%s * INTERVAL '1 minute')
                ORDER BY sampled_at DESC
                LIMIT 1;
                """,
                (
                    symbol,
                    CRYPTO_WINDOW_MINUTES
                )
            )

            result = cursor.fetchone()

            if not result:
                return None

            return {
                "price": float(result[0]),
                "sampled_at": result[1]
            }


def can_send_crypto_alert(
    symbol
):

    with get_database() as conn:

        with conn.cursor() as cursor:

            cursor.execute(
                """
                SELECT last_alerted_at
                FROM crypto_alerts
                WHERE symbol = %s;
                """,
                (symbol,)
            )

            result = cursor.fetchone()

            if not result:
                return True

            last_alert = result[0]

            cursor.execute(
                """
                SELECT
                    NOW() - %s >=
                    (%s * INTERVAL '1 minute');
                """,
                (
                    last_alert,
                    CRYPTO_ALERT_COOLDOWN_MINUTES
                )
            )

            return cursor.fetchone()[0]


def record_crypto_alert(
    symbol,
    percent
):

    with get_database() as conn:

        with conn.cursor() as cursor:

            cursor.execute(
                """
                INSERT INTO crypto_alerts (
                    symbol,
                    last_alerted_at,
                    last_alert_percent
                )
                VALUES (
                    %s,
                    NOW(),
                    %s
                )
                ON CONFLICT (symbol)
                DO UPDATE SET
                    last_alerted_at =
                        EXCLUDED.last_alerted_at,
                    last_alert_percent =
                        EXCLUDED.last_alert_percent;
                """,
                (
                    symbol,
                    percent
                )
            )


def clean_old_crypto_samples():

    with get_database() as conn:

        with conn.cursor() as cursor:

            cursor.execute(
                """
                DELETE FROM crypto_price_samples
                WHERE sampled_at <
                    NOW() - INTERVAL '24 hours';
                """
            )


# =========================================================
# DISCORD
# =========================================================

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


def send_crypto_discord(
    title,
    description,
    fields=None,
    color=3447003
):

    crypto_webhooks = [
        url for url in (
            PRIVATE_CRYPTO_WEBHOOK,
            os.getenv("WEBHOOK2"),
        )
        if url
    ]

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

    for webhook in crypto_webhooks:

        try:

            response = session.post(
                webhook,
                json=payload,
                timeout=15
            )

            response.raise_for_status()

        except Exception as exc:

            print(
                f"Crypto webhook error: {exc}"
            )


# =========================================================
# COINBASE LISTING MONITOR
# =========================================================

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


def check_new_coinbase_assets(
    seen_assets
):

    online = get_coinbase_products()

    current_assets = {
        info["base"]
        for info in online.values()
    }

    if not seen_assets:

        save_assets(
            current_assets
        )

        seen_assets.update(
            current_assets
        )

        print(
            f"Coinbase baseline created: "
            f"{len(current_assets)} assets."
        )

        return

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
                f"as a new online Coinbase asset."
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
            ],
            color=5763719
        )

        print(
            f"NEW COINBASE ASSET: "
            f"{asset}"
        )


# =========================================================
# CRYPTO PRICE MONITOR
# =========================================================

def get_crypto_price(
    symbol
):

    product_id = (
        f"{symbol}-USD"
    )

    url = COINBASE_TICKER_URL.format(
        product_id=product_id
    )

    response = session.get(
        url,
        timeout=15
    )

    response.raise_for_status()

    data = response.json()

    return float(
        data["price"]
    )


def check_crypto_symbol(
    symbol
):

    current_price = get_crypto_price(
        symbol
    )

    old = get_old_crypto_price(
        symbol
    )

    save_crypto_sample(
        symbol,
        current_price
    )

    print(
        f"{symbol}: "
        f"${current_price:,.4f}"
    )

    if not old:
        return

    old_price = old["price"]

    if old_price == 0:
        return

    percent_change = (
        (
            current_price
            - old_price
        )
        / old_price
    ) * 100

    print(
        f"{symbol} "
        f"{CRYPTO_WINDOW_MINUTES}m move: "
        f"{percent_change:+.2f}%"
    )

    if abs(
        percent_change
    ) < CRYPTO_SPIKE_PERCENT:

        return

    if not can_send_crypto_alert(
        symbol
    ):

        print(
            f"{symbol} spike alert "
            f"suppressed by cooldown."
        )

        return

    if percent_change > 0:

        emoji = "🚀"
        direction = "UP"
        color = 5763719

    else:

        emoji = "🔻"
        direction = "DOWN"
        color = 15548997

    send_crypto_discord(
        f"{emoji} CRYPTO PRICE SPIKE",
        (
            f"**{symbol}** is "
            f"**{direction} "
            f"{abs(percent_change):.2f}%** "
            f"in approximately "
            f"{CRYPTO_WINDOW_MINUTES} minutes."
        ),
        [
            {
                "name": "Current price",
                "value": (
                    f"${current_price:,.6f}"
                ),
                "inline": True
            },
            {
                "name": "Earlier price",
                "value": (
                    f"${old_price:,.6f}"
                ),
                "inline": True
            },
            {
                "name": "Move",
                "value": (
                    f"{percent_change:+.2f}%"
                ),
                "inline": True
            }
        ],
        color=color
    )

    record_crypto_alert(
        symbol,
        percent_change
    )

    print(
        f"CRYPTO ALERT SENT: "
        f"{symbol} "
        f"{percent_change:+.2f}%"
    )


# =========================================================
# MAIN
# =========================================================

def main():

    check_config()

    create_tables()

    seen_assets = (
        get_seen_assets()
    )

    print(
        "================================"
    )

    print(
        "Alpha Alerts is ONLINE"
    )

    print(
        f"Coinbase assets remembered: "
        f"{len(seen_assets)}"
    )

    print(
        "Watching crypto: "
        + ", ".join(
            CRYPTO_SYMBOLS
        )
    )

    print(
        f"Spike threshold: "
        f"{CRYPTO_SPIKE_PERCENT}% "
        f"in {CRYPTO_WINDOW_MINUTES} minutes"
    )

    print(
        "================================"
    )

    loop_count = 0

    while True:

        try:

            check_new_coinbase_assets(
                seen_assets
            )

        except Exception as exc:

            print(
                f"Coinbase listing error: "
                f"{exc}"
            )

        for symbol in CRYPTO_SYMBOLS:

            try:

                check_crypto_symbol(
                    symbol
                )

            except Exception as exc:

                print(
                    f"{symbol} price monitor "
                    f"error: {exc}"
                )

        loop_count += 1

        if loop_count >= 60:

            try:

                clean_old_crypto_samples()

                print(
                    "Old crypto samples cleaned."
                )

            except Exception as exc:

                print(
                    f"Sample cleanup error: "
                    f"{exc}"
                )

            loop_count = 0

        time.sleep(
            CHECK_INTERVAL
        )


if __name__ == "__main__":
    main()