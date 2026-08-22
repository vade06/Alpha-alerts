import os
import json
import time
import threading
import requests
import psycopg2
import websocket

from psycopg2.extras import execute_values
from dotenv import load_dotenv

load_dotenv()


# =========================================================
# CONFIG
# =========================================================

WEBHOOK = os.getenv("WEBHOOK")
WEBHOOK2 = os.getenv("WEBHOOK2")

PRIVATE_CRYPTO_WEBHOOK = os.getenv(
    "PRIVATE_CRYPTO_WEBHOOK"
)

PRIVATE_STOCK_WEBHOOK = os.getenv(
    "PRIVATE_STOCK_WEBHOOK"
)

DATABASE_URL = os.getenv(
    "DATABASE_URL"
)

FINNHUB_API_KEY = os.getenv(
    "FINNHUB_API_KEY"
)

CHECK_INTERVAL = int(
    os.getenv(
        "CHECK_INTERVAL",
        "60"
    )
)


# =========================================================
# CRYPTO SETTINGS
# =========================================================

CRYPTO_WINDOW_MINUTES = int(
    os.getenv(
        "CRYPTO_WINDOW_MINUTES",
        "15"
    )
)

# Major coins can trigger at a lower threshold.
CRYPTO_MAJOR_SPIKE_PERCENT = float(
    os.getenv(
        "CRYPTO_MAJOR_SPIKE_PERCENT",
        "5"
    )
)

# Smaller coins are naturally more volatile,
# so use a larger trigger.
CRYPTO_SMALL_SPIKE_PERCENT = float(
    os.getenv(
        "CRYPTO_SMALL_SPIKE_PERCENT",
        "10"
    )
)

CRYPTO_ALERT_COOLDOWN_MINUTES = int(
    os.getenv(
        "CRYPTO_ALERT_COOLDOWN_MINUTES",
        "30"
    )
)

MAJOR_CRYPTO = {
    "BTC",
    "ETH",
    "SOL",
    "XRP",
    "ADA",
    "DOGE",
    "AVAX",
    "LINK",
    "LTC",
    "BCH",
}


# =========================================================
# STOCK SETTINGS
# =========================================================

STOCK_SYMBOLS = [
    symbol.strip().upper()
    for symbol in os.getenv(
        "STOCK_SYMBOLS",
        (
            "NVDA,TSLA,AAPL,MSFT,AMZN,"
            "META,GOOGL,AMD,PLTR,RKLB,"
            "ASTS,LUNR,AVGO,ARM,COIN,MSTR"
        )
    ).split(",")
    if symbol.strip()
]

STOCK_MOVE_PERCENT = float(
    os.getenv(
        "STOCK_MOVE_PERCENT",
        "5"
    )
)

STOCK_WINDOW_MINUTES = int(
    os.getenv(
        "STOCK_WINDOW_MINUTES",
        "30"
    )
)

STOCK_ALERT_COOLDOWN_MINUTES = int(
    os.getenv(
        "STOCK_ALERT_COOLDOWN_MINUTES",
        "60"
    )
)

STOCK_CHECK_INTERVAL = int(
    os.getenv(
        "STOCK_CHECK_INTERVAL",
        "300"
    )
)


# =========================================================
# API URLS
# =========================================================

COINBASE_PRODUCTS_URL = (
    "https://api.exchange.coinbase.com/products"
)

COINBASE_WEBSOCKET_URL = (
    "wss://ws-feed.exchange.coinbase.com"
)

FINNHUB_QUOTE_URL = (
    "https://finnhub.io/api/v1/quote"
)


# =========================================================
# GLOBAL PRICE CACHE
# =========================================================

crypto_prices = {}

crypto_prices_lock = threading.Lock()

coinbase_product_ids = []

session = requests.Session()

session.headers.update({
    "User-Agent": "Alpha-Alerts/8.0"
})


# =========================================================
# CONFIG
# =========================================================

def check_config():

    if not WEBHOOK and not WEBHOOK2:
        raise RuntimeError(
            "Discord webhook missing."
        )

    if not DATABASE_URL:
        raise RuntimeError(
            "DATABASE_URL missing."
        )

    if not FINNHUB_API_KEY:
        print(
            "WARNING: Finnhub API key missing. "
            "Stock monitoring disabled."
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
                CREATE INDEX IF NOT EXISTS
                idx_crypto_samples_symbol_time
                ON crypto_price_samples (
                    symbol,
                    sampled_at DESC
                );
            """)

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS crypto_alerts (
                    symbol TEXT PRIMARY KEY,
                    last_alerted_at TIMESTAMP,
                    last_alert_percent NUMERIC
                );
            """)

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS stock_price_samples (
                    symbol TEXT NOT NULL,
                    price NUMERIC NOT NULL,
                    sampled_at TIMESTAMP
                    DEFAULT CURRENT_TIMESTAMP
                );
            """)

            cursor.execute("""
                CREATE INDEX IF NOT EXISTS
                idx_stock_samples_symbol_time
                ON stock_price_samples (
                    symbol,
                    sampled_at DESC
                );
            """)

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS stock_alerts (
                    symbol TEXT PRIMARY KEY,
                    last_alerted_at TIMESTAMP,
                    last_alert_percent NUMERIC
                );
            """)


# =========================================================
# DISCORD
# =========================================================

def post_to_webhooks(
    webhooks,
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

    sent = set()

    for webhook in webhooks:

        if not webhook:
            continue

        if webhook in sent:
            continue

        sent.add(webhook)

        try:

            response = session.post(
                webhook,
                json=payload,
                timeout=15
            )

            response.raise_for_status()

        except Exception as exc:

            print(
                f"Discord webhook error: {exc}"
            )


def send_coinbase_discord(
    title,
    description,
    fields=None,
    color=3447003
):

    post_to_webhooks(
        [
            WEBHOOK,
            WEBHOOK2
        ],
        title,
        description,
        fields,
        color
    )


def send_crypto_discord(
    title,
    description,
    fields=None,
    color=3447003
):

    post_to_webhooks(
        [
            PRIVATE_CRYPTO_WEBHOOK,
            WEBHOOK2
        ],
        title,
        description,
        fields,
        color
    )


def send_stock_discord(
    title,
    description,
    fields=None,
    color=3447003
):

    post_to_webhooks(
        [
            PRIVATE_STOCK_WEBHOOK,
            WEBHOOK2
        ],
        title,
        description,
        fields,
        color
    )


# =========================================================
# COINBASE PRODUCTS
# =========================================================

def get_coinbase_products():

    response = session.get(
        COINBASE_PRODUCTS_URL,
        timeout=20
    )

    response.raise_for_status()

    return response.json()


def get_online_coinbase_assets():

    products = get_coinbase_products()

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


def save_assets(assets):

    if not assets:
        return

    rows = [
        (asset,)
        for asset in assets
    ]

    with get_database() as conn:

        with conn.cursor() as cursor:

            execute_values(
                cursor,
                """
                INSERT INTO seen_assets (
                    asset_name
                )
                VALUES %s
                ON CONFLICT (asset_name)
                DO NOTHING
                """,
                rows
            )


def check_new_coinbase_assets(
    seen_assets
):

    online = get_online_coinbase_assets()

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

        save_assets(
            {asset}
        )

        seen_assets.add(
            asset
        )

        market_text = ", ".join(
            f"`{market}`"
            for market in markets[:20]
        )

        send_coinbase_discord(
            "🚨 COINBASE TRADING LIVE",
            (
                f"**{asset}** has appeared "
                f"as a new online Coinbase asset."
            ),
            [
                {
                    "name": "Markets",
                    "value":
                        market_text or "Unknown",
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
# ALL COINBASE USD CRYPTO MARKETS
# =========================================================

def get_crypto_products():

    products = get_coinbase_products()

    product_ids = []

    for product in products:

        if product.get("status") != "online":
            continue

        if product.get("quote_currency") != "USD":
            continue

        product_id = product.get("id")

        base = product.get(
            "base_currency"
        )

        if not product_id or not base:
            continue

        # Avoid obvious fiat / stable quote-like products.
        if base in {
            "USD",
            "USDC",
            "USDT",
            "EUR",
            "GBP"
        }:
            continue

        product_ids.append(
            product_id
        )

    return sorted(
        set(product_ids)
    )


# =========================================================
# COINBASE WEBSOCKET
# =========================================================

def websocket_on_open(ws):

    print(
        f"Coinbase WebSocket connected. "
        f"Subscribing to "
        f"{len(coinbase_product_ids)} markets."
    )

    # Subscribe in chunks to avoid one
    # enormous WebSocket message.
    chunk_size = 100

    for i in range(
        0,
        len(coinbase_product_ids),
        chunk_size
    ):

        chunk = (
            coinbase_product_ids[
                i:i + chunk_size
            ]
        )

        message = {
            "type": "subscribe",
            "product_ids": chunk,
            "channels": [
                "ticker"
            ]
        }

        ws.send(
            json.dumps(
                message
            )
        )

        time.sleep(
            0.25
        )


def websocket_on_message(
    ws,
    message
):

    try:

        data = json.loads(
            message
        )

        if data.get("type") != "ticker":
            return

        product_id = data.get(
            "product_id"
        )

        price = data.get(
            "price"
        )

        if not product_id or not price:
            return

        if not product_id.endswith(
            "-USD"
        ):
            return

        symbol = product_id[:-4]

        price = float(
            price
        )

        with crypto_prices_lock:

            crypto_prices[
                symbol
            ] = price

    except Exception as exc:

        print(
            f"WebSocket message error: "
            f"{exc}"
        )


def websocket_on_error(
    ws,
    error
):

    print(
        f"Coinbase WebSocket error: "
        f"{error}"
    )


def websocket_on_close(
    ws,
    close_status_code,
    close_msg
):

    print(
        "Coinbase WebSocket disconnected."
    )


def run_coinbase_websocket():

    while True:

        try:

            ws = websocket.WebSocketApp(
                COINBASE_WEBSOCKET_URL,
                on_open=websocket_on_open,
                on_message=websocket_on_message,
                on_error=websocket_on_error,
                on_close=websocket_on_close
            )

            ws.run_forever(
                ping_interval=30,
                ping_timeout=10
            )

        except Exception as exc:

            print(
                f"WebSocket failure: "
                f"{exc}"
            )

        print(
            "Reconnecting Coinbase "
            "WebSocket in 5 seconds..."
        )

        time.sleep(
            5
        )


# =========================================================
# CRYPTO DATABASE
# =========================================================

def save_crypto_samples(
    prices
):

    if not prices:
        return

    rows = [
        (
            symbol,
            price
        )
        for symbol, price
        in prices.items()
    ]

    with get_database() as conn:

        with conn.cursor() as cursor:

            execute_values(
                cursor,
                """
                INSERT INTO crypto_price_samples (
                    symbol,
                    price
                )
                VALUES %s
                """,
                rows
            )


def get_old_crypto_prices():

    with get_database() as conn:

        with conn.cursor() as cursor:

            cursor.execute(
                """
                SELECT DISTINCT ON (symbol)
                    symbol,
                    price
                FROM crypto_price_samples
                WHERE sampled_at <=
                    NOW() -
                    (%s * INTERVAL '1 minute')
                ORDER BY
                    symbol,
                    sampled_at DESC;
                """,
                (
                    CRYPTO_WINDOW_MINUTES,
                )
            )

            rows = cursor.fetchall()

            return {
                symbol: float(price)
                for symbol, price
                in rows
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
                (
                    symbol,
                )
            )

            result = cursor.fetchone()

            if not result:
                return True

            cursor.execute(
                """
                SELECT
                    NOW() - %s >=
                    (%s * INTERVAL '1 minute');
                """,
                (
                    result[0],
                    CRYPTO_ALERT_COOLDOWN_MINUTES
                )
            )

            return bool(
                cursor.fetchone()[0]
            )


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


# =========================================================
# ALL-CRYPTO SPIKE MONITOR
# =========================================================

def get_crypto_threshold(
    symbol
):

    if symbol in MAJOR_CRYPTO:

        return (
            CRYPTO_MAJOR_SPIKE_PERCENT
        )

    return (
        CRYPTO_SMALL_SPIKE_PERCENT
    )


def check_all_crypto():

    with crypto_prices_lock:

        current_prices = dict(
            crypto_prices
        )

    if not current_prices:

        print(
            "Waiting for Coinbase "
            "WebSocket prices..."
        )

        return

    old_prices = (
        get_old_crypto_prices()
    )

    # Save current snapshot after loading
    # the old snapshot.
    save_crypto_samples(
        current_prices
    )

    print(
        f"Tracking "
        f"{len(current_prices)} "
        f"live Coinbase USD markets."
    )

    alert_candidates = []

    for symbol, current_price in (
        current_prices.items()
    ):

        old_price = old_prices.get(
            symbol
        )

        if old_price is None:
            continue

        if old_price <= 0:
            continue

        percent_change = (
            (
                current_price
                - old_price
            )
            / old_price
        ) * 100

        threshold = (
            get_crypto_threshold(
                symbol
            )
        )

        if abs(
            percent_change
        ) < threshold:

            continue

        alert_candidates.append(
            (
                symbol,
                current_price,
                old_price,
                percent_change,
                threshold
            )
        )

    # Biggest movers first.
    alert_candidates.sort(
        key=lambda item:
            abs(item[3]),
        reverse=True
    )

    # Safety cap:
    # don't send 50 Discord alerts
    # during a market-wide crash.
    for (
        symbol,
        current_price,
        old_price,
        percent_change,
        threshold
    ) in alert_candidates[:10]:

        if not can_send_crypto_alert(
            symbol
        ):
            continue

        if percent_change >= 0:

            emoji = "🚀"
            direction = "UP"
            color = 5763719

        else:

            emoji = "🔻"
            direction = "DOWN"
            color = 15548997

        send_crypto_discord(
            (
                f"{emoji} CRYPTO "
                f"PRICE SPIKE"
            ),
            (
                f"**{symbol}** is "
                f"**{direction} "
                f"{abs(percent_change):.2f}%** "
                f"in approximately "
                f"{CRYPTO_WINDOW_MINUTES} "
                f"minutes."
            ),
            [
                {
                    "name":
                        "Current Price",
                    "value":
                        f"${current_price:,.8f}",
                    "inline":
                        True
                },
                {
                    "name":
                        "Earlier Price",
                    "value":
                        f"${old_price:,.8f}",
                    "inline":
                        True
                },
                {
                    "name":
                        "Move",
                    "value":
                        f"{percent_change:+.2f}%",
                    "inline":
                        True
                },
                {
                    "name":
                        "Alert Threshold",
                    "value":
                        f"{threshold:.1f}%",
                    "inline":
                        True
                }
            ],
            color=color
        )

        record_crypto_alert(
            symbol,
            percent_change
        )

        print(
            f"CRYPTO ALERT: "
            f"{symbol} "
            f"{percent_change:+.2f}%"
        )


# =========================================================
# STOCK DATABASE
# =========================================================

def save_stock_sample(
    symbol,
    price
):

    with get_database() as conn:

        with conn.cursor() as cursor:

            cursor.execute(
                """
                INSERT INTO stock_price_samples (
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


def get_old_stock_price(
    symbol
):

    with get_database() as conn:

        with conn.cursor() as cursor:

            cursor.execute(
                """
                SELECT price
                FROM stock_price_samples
                WHERE
                    symbol = %s
                    AND sampled_at <=
                        NOW() -
                        (%s * INTERVAL '1 minute')
                ORDER BY sampled_at DESC
                LIMIT 1;
                """,
                (
                    symbol,
                    STOCK_WINDOW_MINUTES
                )
            )

            result = cursor.fetchone()

            if not result:
                return None

            return float(
                result[0]
            )


def can_send_stock_alert(
    symbol
):

    with get_database() as conn:

        with conn.cursor() as cursor:

            cursor.execute(
                """
                SELECT last_alerted_at
                FROM stock_alerts
                WHERE symbol = %s;
                """,
                (
                    symbol,
                )
            )

            result = cursor.fetchone()

            if not result:
                return True

            cursor.execute(
                """
                SELECT
                    NOW() - %s >=
                    (%s * INTERVAL '1 minute');
                """,
                (
                    result[0],
                    STOCK_ALERT_COOLDOWN_MINUTES
                )
            )

            return bool(
                cursor.fetchone()[0]
            )


def record_stock_alert(
    symbol,
    percent
):

    with get_database() as conn:

        with conn.cursor() as cursor:

            cursor.execute(
                """
                INSERT INTO stock_alerts (
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


# =========================================================
# STOCK MONITOR
# =========================================================

def get_stock_quote(
    symbol
):

    response = session.get(
        FINNHUB_QUOTE_URL,
        params={
            "symbol": symbol,
            "token": FINNHUB_API_KEY
        },
        timeout=15
    )

    response.raise_for_status()

    data = response.json()

    price = float(
        data.get(
            "c",
            0
        )
    )

    if price <= 0:

        raise RuntimeError(
            f"No valid price for "
            f"{symbol}"
        )

    return price


def check_stock_symbol(
    symbol
):

    current_price = (
        get_stock_quote(
            symbol
        )
    )

    old_price = (
        get_old_stock_price(
            symbol
        )
    )

    save_stock_sample(
        symbol,
        current_price
    )

    print(
        f"STOCK {symbol}: "
        f"${current_price:,.2f}"
    )

    if old_price is None:
        return

    if old_price <= 0:
        return

    percent_change = (
        (
            current_price
            - old_price
        )
        / old_price
    ) * 100

    if abs(
        percent_change
    ) < STOCK_MOVE_PERCENT:

        return

    if not can_send_stock_alert(
        symbol
    ):
        return

    if percent_change >= 0:

        emoji = "📈"
        direction = "UP"
        color = 5763719

    else:

        emoji = "📉"
        direction = "DOWN"
        color = 15548997

    send_stock_discord(
        f"{emoji} STOCK MOVE ALERT",
        (
            f"**{symbol}** is "
            f"**{direction} "
            f"{abs(percent_change):.2f}%** "
            f"in approximately "
            f"{STOCK_WINDOW_MINUTES} "
            f"minutes."
        ),
        [
            {
                "name":
                    "Current Price",
                "value":
                    f"${current_price:,.2f}",
                "inline":
                    True
            },
            {
                "name":
                    "Earlier Price",
                "value":
                    f"${old_price:,.2f}",
                "inline":
                    True
            },
            {
                "name":
                    "Move",
                "value":
                    f"{percent_change:+.2f}%",
                "inline":
                    True
            }
        ],
        color=color
    )

    record_stock_alert(
        symbol,
        percent_change
    )

    print(
        f"STOCK ALERT: "
        f"{symbol} "
        f"{percent_change:+.2f}%"
    )


# =========================================================
# CLEANUP
# =========================================================

def clean_old_samples():

    with get_database() as conn:

        with conn.cursor() as cursor:

            cursor.execute("""
                DELETE FROM crypto_price_samples
                WHERE sampled_at <
                    NOW() -
                    INTERVAL '24 hours';
            """)

            cursor.execute("""
                DELETE FROM stock_price_samples
                WHERE sampled_at <
                    NOW() -
                    INTERVAL '24 hours';
            """)


# =========================================================
# MAIN
# =========================================================

def main():

    global coinbase_product_ids

    check_config()

    create_tables()

    seen_assets = (
        get_seen_assets()
    )

    coinbase_product_ids = (
        get_crypto_products()
    )

    print(
        "================================"
    )

    print(
        "ALPHA ALERTS ONLINE"
    )

    print(
        f"Coinbase assets remembered: "
        f"{len(seen_assets)}"
    )

    print(
        f"Coinbase USD crypto markets: "
        f"{len(coinbase_product_ids)}"
    )

    print(
        f"Major crypto trigger: "
        f"{CRYPTO_MAJOR_SPIKE_PERCENT}%"
    )

    print(
        f"Other crypto trigger: "
        f"{CRYPTO_SMALL_SPIKE_PERCENT}%"
    )

    print(
        f"Crypto window: "
        f"{CRYPTO_WINDOW_MINUTES}m"
    )

    print(
        f"Stocks: "
        f"{len(STOCK_SYMBOLS)}"
    )

    print(
        "================================"
    )

    websocket_thread = threading.Thread(
        target=run_coinbase_websocket,
        daemon=True
    )

    websocket_thread.start()

    last_stock_check = 0

    last_listing_refresh = 0

    loop_count = 0

    while True:

        now = time.time()

        # -----------------------------------------
        # COINBASE LISTING DETECTION
        # -----------------------------------------

        if (
            now - last_listing_refresh
            >= 60
        ):

            try:

                check_new_coinbase_assets(
                    seen_assets
                )

            except Exception as exc:

                print(
                    f"Coinbase listing error: "
                    f"{exc}"
                )

            last_listing_refresh = now


        # -----------------------------------------
        # ALL CRYPTO
        # -----------------------------------------

        try:

            check_all_crypto()

        except Exception as exc:

            print(
                f"Crypto monitor error: "
                f"{exc}"
            )


        # -----------------------------------------
        # STOCKS
        # -----------------------------------------

        if (
            FINNHUB_API_KEY
            and (
                now - last_stock_check
                >= STOCK_CHECK_INTERVAL
            )
        ):

            print(
                "Checking stock watchlist..."
            )

            for symbol in STOCK_SYMBOLS:

                try:

                    check_stock_symbol(
                        symbol
                    )

                except Exception as exc:

                    print(
                        f"Stock {symbol} error: "
                        f"{exc}"
                    )

                time.sleep(
                    1
                )

            last_stock_check = (
                time.time()
            )


        # -----------------------------------------
        # CLEANUP
        # -----------------------------------------

        loop_count += 1

        if loop_count >= 60:

            try:

                clean_old_samples()

                print(
                    "Old samples cleaned."
                )

            except Exception as exc:

                print(
                    f"Cleanup error: "
                    f"{exc}"
                )

            loop_count = 0


        time.sleep(
            CHECK_INTERVAL
        )


if __name__ == "__main__":

    main()