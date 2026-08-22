import os
import json
import time
import asyncio
import threading
import requests
import psycopg2
import websocket
import discord

from discord import app_commands
from discord.ext import commands
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

DISCORD_BOT_TOKEN = os.getenv(
    "DISCORD_BOT_TOKEN"
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

CRYPTO_MAJOR_SPIKE_PERCENT = float(
    os.getenv(
        "CRYPTO_MAJOR_SPIKE_PERCENT",
        "5"
    )
)

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
# GLOBAL STATE
# =========================================================

crypto_prices = {}

crypto_prices_lock = threading.Lock()

coinbase_product_ids = []

monitor_started = False

session = requests.Session()

session.headers.update({
    "User-Agent": "Alpha-Alerts/9.0"
})


# =========================================================
# CONFIG CHECK
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

    if not DISCORD_BOT_TOKEN:
        raise RuntimeError(
            "DISCORD_BOT_TOKEN missing."
        )

    if not FINNHUB_API_KEY:
        print(
            "WARNING: FINNHUB_API_KEY missing."
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
# DISCORD WEBHOOKS
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

        if not product_id or not base or not quote:
            continue

        online[product_id] = {
            "base": base,
            "quote": quote
        }

    return online


def get_crypto_products():

    products = get_coinbase_products()

    result = []

    for product in products:

        if product.get("status") != "online":
            continue

        if product.get("quote_currency") != "USD":
            continue

        product_id = product.get("id")
        base = product.get("base_currency")

        if not product_id or not base:
            continue

        if base in {
            "USD",
            "USDC",
            "USDT",
            "EUR",
            "GBP"
        }:
            continue

        result.append(
            product_id
        )

    return sorted(
        set(result)
    )


# =========================================================
# COINBASE LISTINGS
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
            f"NEW COINBASE ASSET: {asset}"
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

        ws.send(
            json.dumps({
                "type": "subscribe",
                "product_ids": chunk,
                "channels": [
                    "ticker"
                ]
            })
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

        with crypto_prices_lock:

            crypto_prices[
                symbol
            ] = float(price)

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
                f"WebSocket failure: {exc}"
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

            return {
                symbol: float(price)
                for symbol, price
                in cursor.fetchall()
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


# =========================================================
# CRYPTO MONITOR
# =========================================================

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

    save_crypto_samples(
        current_prices
    )

    print(
        f"Tracking "
        f"{len(current_prices)} "
        f"live Coinbase USD markets."
    )

    candidates = []

    for symbol, current_price in (
        current_prices.items()
    ):

        old_price = old_prices.get(
            symbol
        )

        if not old_price:
            continue

        percent = (
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

        if abs(percent) >= threshold:

            candidates.append(
                (
                    symbol,
                    current_price,
                    old_price,
                    percent,
                    threshold
                )
            )

    candidates.sort(
        key=lambda item:
            abs(item[3]),
        reverse=True
    )

    for (
        symbol,
        current_price,
        old_price,
        percent,
        threshold
    ) in candidates[:10]:

        if not can_send_crypto_alert(
            symbol
        ):
            continue

        positive = percent >= 0

        send_crypto_discord(
            (
                "🚀 CRYPTO PRICE SPIKE"
                if positive
                else
                "🔻 CRYPTO PRICE SPIKE"
            ),
            (
                f"**{symbol}** is "
                f"**{'UP' if positive else 'DOWN'} "
                f"{abs(percent):.2f}%** "
                f"in approximately "
                f"{CRYPTO_WINDOW_MINUTES} minutes."
            ),
            [
                {
                    "name": "Price",
                    "value":
                        f"${current_price:,.8f}",
                    "inline": True
                },
                {
                    "name": "Move",
                    "value":
                        f"{percent:+.2f}%",
                    "inline": True
                },
                {
                    "name": "Threshold",
                    "value":
                        f"{threshold:.1f}%",
                    "inline": True
                }
            ],
            color=(
                5763719
                if positive
                else 15548997
            )
        )

        record_crypto_alert(
            symbol,
            percent
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
        data.get("c", 0)
    )

    if price <= 0:

        raise RuntimeError(
            f"No stock price for "
            f"{symbol}"
        )

    return data


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
                (symbol,)
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


def check_stock_symbol(
    symbol
):

    data = get_stock_quote(
        symbol
    )

    current_price = float(
        data["c"]
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

    if not old_price:
        return

    percent = (
        (
            current_price
            - old_price
        )
        / old_price
    ) * 100

    if abs(percent) < STOCK_MOVE_PERCENT:
        return

    if not can_send_stock_alert(
        symbol
    ):
        return

    positive = percent >= 0

    send_stock_discord(
        (
            "📈 STOCK MOVE ALERT"
            if positive
            else
            "📉 STOCK MOVE ALERT"
        ),
        (
            f"**{symbol}** is "
            f"**{'UP' if positive else 'DOWN'} "
            f"{abs(percent):.2f}%** "
            f"in approximately "
            f"{STOCK_WINDOW_MINUTES} minutes."
        ),
        [
            {
                "name": "Price",
                "value":
                    f"${current_price:,.2f}",
                "inline": True
            },
            {
                "name": "Move",
                "value":
                    f"{percent:+.2f}%",
                "inline": True
            }
        ],
        color=(
            5763719
            if positive
            else 15548997
        )
    )

    record_stock_alert(
        symbol,
        percent
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
# BACKGROUND MONITOR
# =========================================================

def monitor_main():

    global coinbase_product_ids
    global monitor_started

    if monitor_started:
        return

    monitor_started = True

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
    last_listing_check = 0
    loop_count = 0

    while True:

        now = time.time()

        if (
            now - last_listing_check
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

            last_listing_check = now

        try:

            check_all_crypto()

        except Exception as exc:

            print(
                f"Crypto monitor error: "
                f"{exc}"
            )

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

        loop_count += 1

        if loop_count >= 60:

            try:

                clean_old_samples()

            except Exception as exc:

                print(
                    f"Cleanup error: {exc}"
                )

            loop_count = 0

        time.sleep(
            CHECK_INTERVAL
        )


# =========================================================
# DISCORD BOT
# =========================================================

intents = discord.Intents.default()

bot = commands.Bot(
    command_prefix="!",
    intents=intents
)


@bot.event
async def on_ready():

    print(
        f"Discord bot online as "
        f"{bot.user}"
    )

    try:

        synced = await bot.tree.sync()

        print(
            f"Synced "
            f"{len(synced)} slash commands."
        )

    except Exception as exc:

        print(
            f"Command sync error: "
            f"{exc}"
        )


# =========================================================
# /status
# =========================================================

@bot.tree.command(
    name="status",
    description="Show Alpha Alerts system status"
)
async def status(
    interaction: discord.Interaction
):

    with crypto_prices_lock:

        market_count = len(
            crypto_prices
        )

    embed = discord.Embed(
        title="🟢 Alpha Alerts Status",
        color=5763719
    )

    embed.add_field(
        name="Crypto Markets",
        value=str(
            market_count
        ),
        inline=True
    )

    embed.add_field(
        name="Stock Watchlist",
        value=str(
            len(STOCK_SYMBOLS)
        ),
        inline=True
    )

    embed.add_field(
        name="Crypto Window",
        value=(
            f"{CRYPTO_WINDOW_MINUTES} min"
        ),
        inline=True
    )

    embed.add_field(
        name="Major Crypto Trigger",
        value=(
            f"{CRYPTO_MAJOR_SPIKE_PERCENT}%"
        ),
        inline=True
    )

    embed.add_field(
        name="Altcoin Trigger",
        value=(
            f"{CRYPTO_SMALL_SPIKE_PERCENT}%"
        ),
        inline=True
    )

    embed.add_field(
        name="Stock Trigger",
        value=(
            f"{STOCK_MOVE_PERCENT}%"
        ),
        inline=True
    )

    await interaction.response.send_message(
        embed=embed
    )


# =========================================================
# /crypto
# =========================================================

@bot.tree.command(
    name="crypto",
    description="Get the latest crypto price"
)
@app_commands.describe(
    symbol="Example: BTC, ETH, SOL"
)
async def crypto(
    interaction: discord.Interaction,
    symbol: str
):

    symbol = (
        symbol.strip().upper()
    )

    with crypto_prices_lock:

        price = crypto_prices.get(
            symbol
        )

    if price is None:

        await interaction.response.send_message(
            (
                f"I don't currently have "
                f"a live Coinbase USD price "
                f"for `{symbol}`."
            ),
            ephemeral=True
        )

        return

    old_prices = await asyncio.to_thread(
        get_old_crypto_prices
    )

    old_price = old_prices.get(
        symbol
    )

    move_text = "Collecting history"

    if old_price:

        percent = (
            (
                price
                - old_price
            )
            / old_price
        ) * 100

        move_text = (
            f"{percent:+.2f}% "
            f"over ~"
            f"{CRYPTO_WINDOW_MINUTES}m"
        )

    embed = discord.Embed(
        title=f"🪙 {symbol}",
        description=(
            f"**${price:,.8f}**"
        ),
        color=3447003
    )

    embed.add_field(
        name="Recent Move",
        value=move_text,
        inline=False
    )

    embed.set_footer(
        text="Coinbase live market data"
    )

    await interaction.response.send_message(
        embed=embed
    )


# =========================================================
# /stock
# =========================================================

@bot.tree.command(
    name="stock",
    description="Get the latest stock quote"
)
@app_commands.describe(
    ticker="Example: NVDA, TSLA, AAPL"
)
async def stock(
    interaction: discord.Interaction,
    ticker: str
):

    await interaction.response.defer()

    ticker = (
        ticker.strip().upper()
    )

    try:

        data = await asyncio.to_thread(
            get_stock_quote,
            ticker
        )

        current = float(
            data.get("c", 0)
        )

        change = float(
            data.get("d", 0) or 0
        )

        percent = float(
            data.get("dp", 0) or 0
        )

        high = float(
            data.get("h", 0) or 0
        )

        low = float(
            data.get("l", 0) or 0
        )

        previous_close = float(
            data.get("pc", 0) or 0
        )

        if current <= 0:

            raise RuntimeError(
                "No valid quote"
            )

        positive = change >= 0

        embed = discord.Embed(
            title=(
                f"{'📈' if positive else '📉'} "
                f"{ticker}"
            ),
            description=(
                f"**${current:,.2f}**\n"
                f"{percent:+.2f}% today"
            ),
            color=(
                5763719
                if positive
                else 15548997
            )
        )

        embed.add_field(
            name="Change",
            value=(
                f"${change:+,.2f}"
            ),
            inline=True
        )

        embed.add_field(
            name="Day High",
            value=(
                f"${high:,.2f}"
            ),
            inline=True
        )

        embed.add_field(
            name="Day Low",
            value=(
                f"${low:,.2f}"
            ),
            inline=True
        )

        embed.add_field(
            name="Previous Close",
            value=(
                f"${previous_close:,.2f}"
            ),
            inline=True
        )

        embed.set_footer(
            text="Stock data: Finnhub"
        )

        await interaction.followup.send(
            embed=embed
        )

    except Exception as exc:

        print(
            f"/stock error: {exc}"
        )

        await interaction.followup.send(
            (
                f"I couldn't find a valid "
                f"quote for `{ticker}`."
            )
        )


# =========================================================
# /watchlist
# =========================================================

@bot.tree.command(
    name="watchlist",
    description="Show the stock watchlist"
)
async def watchlist(
    interaction: discord.Interaction
):

    stock_text = ", ".join(
        f"`{symbol}`"
        for symbol in STOCK_SYMBOLS
    )

    embed = discord.Embed(
        title="📊 Alpha Alerts Watchlist",
        color=3447003
    )

    embed.add_field(
        name="Stocks",
        value=stock_text,
        inline=False
    )

    embed.add_field(
        name="Crypto",
        value=(
            "Automatically monitoring "
            "all online Coinbase USD markets."
        ),
        inline=False
    )

    await interaction.response.send_message(
        embed=embed
    )


# =========================================================
# /movers
# =========================================================

@bot.tree.command(
    name="movers",
    description="Show the biggest current crypto movers"
)
async def movers(
    interaction: discord.Interaction
):

    await interaction.response.defer()

    with crypto_prices_lock:

        current = dict(
            crypto_prices
        )

    old = await asyncio.to_thread(
        get_old_crypto_prices
    )

    results = []

    for symbol, price in current.items():

        old_price = old.get(
            symbol
        )

        if not old_price:
            continue

        percent = (
            (
                price
                - old_price
            )
            / old_price
        ) * 100

        results.append(
            (
                symbol,
                price,
                percent
            )
        )

    results.sort(
        key=lambda item:
            abs(item[2]),
        reverse=True
    )

    top = results[:10]

    if not top:

        await interaction.followup.send(
            "Still collecting enough price history."
        )

        return

    lines = []

    for symbol, price, percent in top:

        emoji = (
            "🟢"
            if percent >= 0
            else "🔴"
        )

        lines.append(
            (
                f"{emoji} **{symbol}** "
                f"{percent:+.2f}% "
                f"— ${price:,.6f}"
            )
        )

    embed = discord.Embed(
        title=(
            f"🔥 Biggest Crypto Movers "
            f"(~{CRYPTO_WINDOW_MINUTES}m)"
        ),
        description="\n".join(
            lines
        ),
        color=3447003
    )

    await interaction.followup.send(
        embed=embed
    )


# =========================================================
# START
# =========================================================

if __name__ == "__main__":

    check_config()

    monitor_thread = threading.Thread(
        target=monitor_main,
        daemon=True
    )

    monitor_thread.start()

    bot.run(
        DISCORD_BOT_TOKEN
    )