import os
import json
import time
import asyncio
import threading
import xml.etree.ElementTree as ET

import requests
import psycopg2
import websocket
import discord

from discord import app_commands
from discord.ext import commands
from dotenv import load_dotenv
from psycopg2.extras import execute_values

from ai_trader import run_ai_cycle


# =========================================================
# ENVIRONMENT
# =========================================================

load_dotenv()

WEBHOOK = os.getenv("WEBHOOK")
WEBHOOK2 = os.getenv("WEBHOOK2")

PRIVATE_CRYPTO_WEBHOOK = os.getenv("PRIVATE_CRYPTO_WEBHOOK")
PRIVATE_STOCK_WEBHOOK = os.getenv("PRIVATE_STOCK_WEBHOOK")
PRIVATE_INSIDER_WEBHOOK = os.getenv("PRIVATE_INSIDER_WEBHOOK")

DATABASE_URL = os.getenv("DATABASE_URL")
FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY")
DISCORD_BOT_TOKEN = os.getenv("DISCORD_BOT_TOKEN")
SEC_USER_AGENT = os.getenv("SEC_USER_AGENT")

# Your Discord USER ID.
# Keep this in Railway Variables, not in GitHub.
OWNER_DISCORD_USER_ID = os.getenv("OWNER_DISCORD_USER_ID")


# =========================================================
# GENERAL SETTINGS
# =========================================================

CHECK_INTERVAL = int(
    os.getenv("CHECK_INTERVAL", "60")
)

AI_CHECK_INTERVAL = int(
    os.getenv("AI_CHECK_INTERVAL", "300")
)


# =========================================================
# CRYPTO SETTINGS
# =========================================================

CRYPTO_WINDOW_MINUTES = int(
    os.getenv("CRYPTO_WINDOW_MINUTES", "15")
)

CRYPTO_MAJOR_SPIKE_PERCENT = float(
    os.getenv("CRYPTO_MAJOR_SPIKE_PERCENT", "7")
)

CRYPTO_SMALL_SPIKE_PERCENT = float(
    os.getenv("CRYPTO_SMALL_SPIKE_PERCENT", "15")
)

CRYPTO_ALERT_COOLDOWN_MINUTES = int(
    os.getenv("CRYPTO_ALERT_COOLDOWN_MINUTES", "60")
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
    os.getenv("STOCK_MOVE_PERCENT", "5")
)

STOCK_WINDOW_MINUTES = int(
    os.getenv("STOCK_WINDOW_MINUTES", "30")
)

STOCK_ALERT_COOLDOWN_MINUTES = int(
    os.getenv("STOCK_ALERT_COOLDOWN_MINUTES", "60")
)

STOCK_CHECK_INTERVAL = int(
    os.getenv("STOCK_CHECK_INTERVAL", "300")
)


# =========================================================
# SEC INSIDER SETTINGS
# =========================================================

SEC_CHECK_INTERVAL = int(
    os.getenv("SEC_CHECK_INTERVAL", "300")
)

SEC_TICKERS_URL = (
    "https://www.sec.gov/files/company_tickers.json"
)

SEC_SUBMISSIONS_URL = (
    "https://data.sec.gov/submissions/CIK{cik}.json"
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

ai_last_result = None
ai_result_lock = threading.Lock()

http = requests.Session()

http.headers.update({
    "User-Agent": "Alpha-Alerts/13.0"
})


# =========================================================
# OWNER SECURITY
# =========================================================

def is_bot_owner(interaction: discord.Interaction) -> bool:
    """
    Fail closed:
    if OWNER_DISCORD_USER_ID is missing, nobody gets owner access.
    """
    if not OWNER_DISCORD_USER_ID:
        return False

    return str(interaction.user.id) == str(OWNER_DISCORD_USER_ID)


async def reject_non_owner(interaction: discord.Interaction):
    await interaction.response.send_message(
        "â You are not authorised to use the AI trader.",
        ephemeral=True
    )


# =========================================================
# CONFIG CHECK
# =========================================================

def check_config():

    if not DATABASE_URL:
        raise RuntimeError("DATABASE_URL is missing.")

    if not DISCORD_BOT_TOKEN:
        raise RuntimeError("DISCORD_BOT_TOKEN is missing.")

    if not OWNER_DISCORD_USER_ID:
        print(
            "WARNING: OWNER_DISCORD_USER_ID missing. "
            "Owner-only commands will be locked for everyone."
        )

    if not WEBHOOK and not WEBHOOK2:
        print("WARNING: General Discord webhook missing.")

    if not FINNHUB_API_KEY:
        print(
            "WARNING: FINNHUB_API_KEY missing. "
            "Stock monitoring disabled."
        )

    if not PRIVATE_INSIDER_WEBHOOK:
        print(
            "WARNING: PRIVATE_INSIDER_WEBHOOK missing. "
            "SEC insider alerts will not be sent."
        )

    if not SEC_USER_AGENT:
        print(
            "WARNING: SEC_USER_AGENT missing. "
            "SEC insider monitoring disabled."
        )


# =========================================================
# DATABASE
# =========================================================

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

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS seen_sec_filings (
                    accession_number TEXT PRIMARY KEY,
                    ticker TEXT,
                    company_name TEXT,
                    filing_date TEXT,
                    form_type TEXT,
                    created_at TIMESTAMP
                    DEFAULT CURRENT_TIMESTAMP
                );
            """)


# =========================================================
# DISCORD WEBHOOK HELPERS
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

    already_sent = set()

    for webhook in webhooks:

        if not webhook:
            continue

        if webhook in already_sent:
            continue

        already_sent.add(webhook)

        try:

            response = http.post(
                webhook,
                json=payload,
                timeout=15
            )

            response.raise_for_status()

        except Exception as exc:

            print(f"Discord webhook error: {exc}")


def send_coinbase_alert(
    title,
    description,
    fields=None,
    color=3447003
):

    post_to_webhooks(
        [WEBHOOK, WEBHOOK2],
        title,
        description,
        fields,
        color
    )


def send_crypto_alert(
    title,
    description,
    fields=None,
    color=3447003
):

    post_to_webhooks(
        [PRIVATE_CRYPTO_WEBHOOK, WEBHOOK2],
        title,
        description,
        fields,
        color
    )


def send_stock_alert(
    title,
    description,
    fields=None,
    color=3447003
):

    post_to_webhooks(
        [PRIVATE_STOCK_WEBHOOK, WEBHOOK2],
        title,
        description,
        fields,
        color
    )


def send_insider_alert(
    title,
    description,
    fields=None,
    color=10181046
):

    post_to_webhooks(
        [PRIVATE_INSIDER_WEBHOOK],
        title,
        description,
        fields,
        color
    )


# =========================================================
# COINBASE PRODUCTS
# =========================================================

def get_coinbase_products():

    response = http.get(
        COINBASE_PRODUCTS_URL,
        timeout=20
    )

    response.raise_for_status()

    return response.json()


def get_crypto_products():

    products = get_coinbase_products()

    result = []

    ignored = {
        "USD",
        "USDC",
        "USDT",
        "EUR",
        "GBP"
    }

    for product in products:

        if product.get("status") != "online":
            continue

        if product.get("quote_currency") != "USD":
            continue

        product_id = product.get("id")
        base = product.get("base_currency")

        if not product_id or not base:
            continue

        if base in ignored:
            continue

        result.append(product_id)

    return sorted(set(result))


# =========================================================
# COINBASE NEW ASSET TRACKING
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


def check_new_coinbase_assets(seen_assets):

    products = get_coinbase_products()

    current_assets = set()
    markets_by_asset = {}

    for product in products:

        if product.get("status") != "online":
            continue

        base = product.get("base_currency")
        product_id = product.get("id")

        if not base or not product_id:
            continue

        current_assets.add(base)

        markets_by_asset.setdefault(
            base,
            []
        ).append(product_id)

    if not seen_assets:

        save_assets(current_assets)
        seen_assets.update(current_assets)

        print(
            f"Coinbase baseline created: "
            f"{len(current_assets)} assets."
        )

        return

    new_assets = current_assets - seen_assets

    for asset in sorted(new_assets):

        save_assets({asset})
        seen_assets.add(asset)

        markets = sorted(
            markets_by_asset.get(asset, [])
        )

        market_text = ", ".join(
            f"`{market}`"
            for market in markets[:20]
        )

        send_coinbase_alert(
            "ð¨ COINBASE TRADING LIVE",
            (
                f"**{asset}** has appeared "
                f"as a new online Coinbase asset."
            ),
            [
                {
                    "name": "Markets",
                    "value": market_text or "Unknown",
                    "inline": False
                }
            ],
            color=5763719
        )

        print(f"NEW COINBASE ASSET: {asset}")


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

    for start in range(
        0,
        len(coinbase_product_ids),
        chunk_size
    ):

        chunk = coinbase_product_ids[
            start:start + chunk_size
        ]

        ws.send(
            json.dumps({
                "type": "subscribe",
                "product_ids": chunk,
                "channels": ["ticker"]
            })
        )

        time.sleep(0.25)


def websocket_on_message(ws, message):

    try:

        data = json.loads(message)

        if data.get("type") != "ticker":
            return

        product_id = data.get("product_id")
        price = data.get("price")

        if not product_id or not price:
            return

        if not product_id.endswith("-USD"):
            return

        symbol = product_id[:-4]

        with crypto_prices_lock:
            crypto_prices[symbol] = float(price)

    except Exception as exc:

        print(f"WebSocket message error: {exc}")


def websocket_on_error(ws, error):
    print(f"Coinbase WebSocket error: {error}")


def websocket_on_close(
    ws,
    close_status_code,
    close_msg
):
    print("Coinbase WebSocket disconnected.")


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

            print(f"WebSocket failure: {exc}")

        print(
            "Reconnecting WebSocket "
            "in 5 seconds..."
        )

        time.sleep(5)


# =========================================================
# CRYPTO DATABASE
# =========================================================

def save_crypto_samples(prices):

    if not prices:
        return

    rows = [
        (symbol, price)
        for symbol, price in prices.items()
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
                (CRYPTO_WINDOW_MINUTES,)
            )

            return {
                symbol: float(price)
                for symbol, price
                in cursor.fetchall()
            }


def can_send_crypto_alert(symbol):

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

            return bool(cursor.fetchone()[0])


def record_crypto_alert(symbol, percent):

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
                (symbol, percent)
            )


def crypto_threshold(symbol):

    if symbol in MAJOR_CRYPTO:
        return CRYPTO_MAJOR_SPIKE_PERCENT

    return CRYPTO_SMALL_SPIKE_PERCENT


# =========================================================
# CRYPTO MONITOR
# =========================================================

def check_all_crypto():

    with crypto_prices_lock:
        current_prices = dict(crypto_prices)

    if not current_prices:

        print(
            "Waiting for Coinbase "
            "WebSocket prices..."
        )

        return

    old_prices = get_old_crypto_prices()

    save_crypto_samples(current_prices)

    print(
        f"Tracking "
        f"{len(current_prices)} "
        f"live Coinbase USD markets."
    )

    candidates = []

    for symbol, price in current_prices.items():

        old_price = old_prices.get(symbol)

        if not old_price:
            continue

        percent = (
            (price - old_price)
            / old_price
        ) * 100

        threshold = crypto_threshold(symbol)

        if abs(percent) >= threshold:

            candidates.append(
                (
                    symbol,
                    price,
                    old_price,
                    percent,
                    threshold
                )
            )

    candidates.sort(
        key=lambda item: abs(item[3]),
        reverse=True
    )

    for (
        symbol,
        price,
        old_price,
        percent,
        threshold
    ) in candidates[:3]:

        if not can_send_crypto_alert(symbol):
            continue

        positive = percent >= 0

        send_crypto_alert(
            (
                "ð CRYPTO PRICE SPIKE"
                if positive
                else
                "ð» CRYPTO PRICE SPIKE"
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
                    "name": "Current Price",
                    "value": f"${price:,.8f}",
                    "inline": True
                },
                {
                    "name": "Earlier Price",
                    "value": f"${old_price:,.8f}",
                    "inline": True
                },
                {
                    "name": "Move",
                    "value": f"{percent:+.2f}%",
                    "inline": True
                }
            ],
            color=(
                5763719
                if positive
                else 15548997
            )
        )

        record_crypto_alert(symbol, percent)


# =========================================================
# STOCK DATABASE
# =========================================================

def save_stock_sample(symbol, price):

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
                (symbol, price)
            )


def get_old_stock_price(symbol):

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

            return float(result[0])


def can_send_stock_alert(symbol):

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

            return bool(cursor.fetchone()[0])


def record_stock_alert(symbol, percent):

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
                (symbol, percent)
            )


# =========================================================
# FINNHUB STOCK DATA
# =========================================================

def get_stock_quote(symbol):

    if not FINNHUB_API_KEY:
        raise RuntimeError(
            "FINNHUB_API_KEY missing."
        )

    response = http.get(
        FINNHUB_QUOTE_URL,
        params={
            "symbol": symbol,
            "token": FINNHUB_API_KEY
        },
        timeout=15
    )

    response.raise_for_status()

    data = response.json()

    current = float(
        data.get("c", 0) or 0
    )

    if current <= 0:
        raise RuntimeError(
            f"No valid quote for {symbol}"
        )

    return data


def check_stock_symbol(symbol):

    data = get_stock_quote(symbol)

    current_price = float(data["c"])
    old_price = get_old_stock_price(symbol)

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
        (current_price - old_price)
        / old_price
    ) * 100

    if abs(percent) < STOCK_MOVE_PERCENT:
        return

    if not can_send_stock_alert(symbol):
        return

    positive = percent >= 0

    send_stock_alert(
        (
            "ð STOCK MOVE ALERT"
            if positive
            else
            "ð STOCK MOVE ALERT"
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
                "name": "Current Price",
                "value": f"${current_price:,.2f}",
                "inline": True
            },
            {
                "name": "Move",
                "value": f"{percent:+.2f}%",
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
# SEC HELPERS
# =========================================================

def sec_headers():

    return {
        "User-Agent": SEC_USER_AGENT,
        "Accept-Encoding": "gzip, deflate",
        "Accept": "*/*"
    }


def get_sec_ticker_map():

    response = http.get(
        SEC_TICKERS_URL,
        headers=sec_headers(),
        timeout=20
    )

    response.raise_for_status()

    data = response.json()

    ticker_map = {}

    for item in data.values():

        ticker = (
            item["ticker"]
            .strip()
            .upper()
        )

        ticker_map[ticker] = {
            "cik": str(
                item["cik_str"]
            ).zfill(10),
            "company": item["title"]
        }

    return ticker_map


def sec_database_empty():

    with get_database() as conn:
        with conn.cursor() as cursor:

            cursor.execute(
                """
                SELECT COUNT(*)
                FROM seen_sec_filings;
                """
            )

            return cursor.fetchone()[0] == 0


def sec_filing_seen(accession):

    with get_database() as conn:
        with conn.cursor() as cursor:

            cursor.execute(
                """
                SELECT 1
                FROM seen_sec_filings
                WHERE accession_number = %s;
                """,
                (accession,)
            )

            return cursor.fetchone() is not None


def save_sec_filing(
    accession,
    ticker,
    company,
    filing_date,
    form_type
):

    with get_database() as conn:
        with conn.cursor() as cursor:

            cursor.execute(
                """
                INSERT INTO seen_sec_filings (
                    accession_number,
                    ticker,
                    company_name,
                    filing_date,
                    form_type
                )
                VALUES (
                    %s,
                    %s,
                    %s,
                    %s,
                    %s
                )
                ON CONFLICT (
                    accession_number
                )
                DO NOTHING;
                """,
                (
                    accession,
                    ticker,
                    company,
                    filing_date,
                    form_type
                )
            )


def get_recent_form4_filings(
    ticker,
    cik,
    company
):

    url = SEC_SUBMISSIONS_URL.format(
        cik=cik
    )

    response = http.get(
        url,
        headers=sec_headers(),
        timeout=20
    )

    response.raise_for_status()

    data = response.json()

    recent = (
        data.get("filings", {})
        .get("recent", {})
    )

    forms = recent.get("form", [])
    accession_numbers = recent.get(
        "accessionNumber",
        []
    )
    filing_dates = recent.get(
        "filingDate",
        []
    )
    primary_documents = recent.get(
        "primaryDocument",
        []
    )

    filings = []

    for index, form_type in enumerate(forms):

        if form_type != "4":
            continue

        if index >= len(accession_numbers):
            continue

        accession = accession_numbers[index]

        filing_date = (
            filing_dates[index]
            if index < len(filing_dates)
            else "Unknown"
        )

        primary_document = (
            primary_documents[index]
            if index < len(primary_documents)
            else ""
        )

        accession_clean = accession.replace(
            "-",
            ""
        )

        cik_clean = str(int(cik))

        filing_url = (
            "https://www.sec.gov/Archives/"
            f"edgar/data/{cik_clean}/"
            f"{accession_clean}/"
            f"{primary_document}"
        )

        filings.append({
            "ticker": ticker,
            "company": company,
            "accession": accession,
            "filing_date": filing_date,
            "url": filing_url
        })

    return filings


# =========================================================
# FORM 4 XML PARSER
# =========================================================

def safe_xml_text(
    element,
    path,
    default="Unknown"
):

    if element is None:
        return default

    found = element.find(path)

    if found is None:
        return default

    if found.text is None:
        return default

    return found.text.strip()


def format_money(value):

    try:

        number = float(value)

        if number >= 1_000_000_000:
            return (
                f"${number / 1_000_000_000:.2f}B"
            )

        if number >= 1_000_000:
            return (
                f"${number / 1_000_000:.2f}M"
            )

        if number >= 1_000:
            return (
                f"${number / 1_000:.2f}K"
            )

        return f"${number:,.2f}"

    except Exception:
        return "Unknown"


def transaction_name(code):

    names = {
        "P": "ð¢ Open-market BUY",
        "S": "ð´ Open-market SELL",
        "A": "ðµ Grant / Award",
        "M": "ð¡ Option Exercise",
        "F": "ð  Tax / Withholding",
        "G": "ð£ Gift",
        "C": "Conversion",
        "D": "Disposition",
        "J": "Other",
    }

    return names.get(
        code,
        f"Other ({code})"
    )


def get_form4_xml(filing_url):

    response = http.get(
        filing_url,
        headers=sec_headers(),
        timeout=20
    )

    response.raise_for_status()

    return response.text


def parse_form4(xml_text):

    root = ET.fromstring(xml_text)

    owner = root.find(
        ".//reportingOwner"
    )

    insider_name = safe_xml_text(
        owner,
        ".//rptOwnerName"
    )

    relationship = None

    if owner is not None:
        relationship = owner.find(
            ".//reportingOwnerRelationship"
        )

    roles = []

    if relationship is not None:

        is_director = safe_xml_text(
            relationship,
            "isDirector",
            "0"
        )

        is_officer = safe_xml_text(
            relationship,
            "isOfficer",
            "0"
        )

        is_ten_percent = safe_xml_text(
            relationship,
            "isTenPercentOwner",
            "0"
        )

        officer_title = safe_xml_text(
            relationship,
            "officerTitle",
            ""
        )

        if is_director == "1":
            roles.append("Director")

        if is_officer == "1":

            if officer_title:
                roles.append(officer_title)

            else:
                roles.append("Officer")

        if is_ten_percent == "1":
            roles.append("10% Owner")

    role = (
        ", ".join(roles)
        if roles
        else "Insider"
    )

    transactions = []

    for transaction in root.findall(
        ".//nonDerivativeTransaction"
    ):

        code = safe_xml_text(
            transaction,
            ".//transactionCoding/transactionCode",
            "?"
        )

        shares_text = safe_xml_text(
            transaction,
            ".//transactionAmounts/transactionShares/value",
            "0"
        )

        price_text = safe_xml_text(
            transaction,
            ".//transactionAmounts/transactionPricePerShare/value",
            ""
        )

        acquired_disposed = safe_xml_text(
            transaction,
            ".//transactionAmounts/transactionAcquiredDisposedCode/value",
            ""
        )

        try:
            shares = float(shares_text)
        except Exception:
            shares = 0

        try:
            price = float(price_text)
        except Exception:
            price = None

        value = None

        if shares > 0 and price is not None:
            value = shares * price

        transactions.append({
            "code": code,
            "action": transaction_name(code),
            "shares": shares,
            "price": price,
            "value": value,
            "direction": acquired_disposed
        })

    return {
        "insider": insider_name,
        "role": role,
        "transactions": transactions
    }


# =========================================================
# SEC INSIDER MONITOR
# =========================================================

def send_parsed_form4_alert(
    ticker,
    company,
    filing
):

    try:

        xml_text = get_form4_xml(
            filing["url"]
        )

        parsed = parse_form4(xml_text)

        insider = parsed["insider"]
        role = parsed["role"]
        transactions = parsed["transactions"]

        important = [
            transaction
            for transaction in transactions
            if transaction["code"] in {"P", "S"}
        ]

        display_transactions = (
            important
            if important
            else transactions
        )

        fields = [
            {
                "name": "Insider",
                "value": insider,
                "inline": True
            },
            {
                "name": "Role",
                "value": role,
                "inline": True
            },
            {
                "name": "Filed",
                "value": filing["filing_date"],
                "inline": True
            }
        ]

        for transaction in display_transactions[:4]:

            shares = transaction["shares"]
            price = transaction["price"]
            value = transaction["value"]

            transaction_text = (
                f"**{transaction['action']}**\n"
                f"Shares: {shares:,.0f}"
            )

            if price is not None:

                transaction_text += (
                    f"\nPrice: "
                    f"${price:,.4f}"
                )

            if value is not None:

                transaction_text += (
                    f"\nApprox value: "
                    f"{format_money(value)}"
                )

            fields.append({
                "name": "Transaction",
                "value": transaction_text,
                "inline": False
            })

        fields.append({
            "name": "Official SEC Filing",
            "value": filing["url"],
            "inline": False
        })

        has_purchase = any(
            transaction["code"] == "P"
            for transaction in transactions
        )

        has_sale = any(
            transaction["code"] == "S"
            for transaction in transactions
        )

        if has_purchase:

            title = "ð¢ INSIDER BUY"
            color = 5763719

        elif has_sale:

            title = "ð´ INSIDER SALE"
            color = 15548997

        else:

            title = "ð§  INSIDER FORM 4"
            color = 10181046

        send_insider_alert(
            title,
            (
                f"**{company} "
                f"({ticker})**\n"
                f"New SEC Form 4 transaction."
            ),
            fields,
            color=color
        )

    except Exception as parse_error:

        print(
            f"Form 4 parsing error "
            f"{ticker}: "
            f"{parse_error}"
        )

        send_insider_alert(
            "ð§  NEW INSIDER FILING",
            (
                f"**{company} "
                f"({ticker})** filed "
                f"a new SEC Form 4."
            ),
            [
                {
                    "name": "Filed",
                    "value": filing["filing_date"],
                    "inline": True
                },
                {
                    "name": "Official SEC Filing",
                    "value": filing["url"],
                    "inline": False
                }
            ]
        )


def check_sec_insider_filings(
    ticker_map,
    baseline=False
):

    for ticker in STOCK_SYMBOLS:

        info = ticker_map.get(ticker)

        if not info:

            print(
                f"SEC ticker not found: "
                f"{ticker}"
            )

            continue

        try:

            filings = get_recent_form4_filings(
                ticker,
                info["cik"],
                info["company"]
            )

            for filing in filings[:10]:

                accession = filing["accession"]

                if sec_filing_seen(accession):
                    continue

                save_sec_filing(
                    accession,
                    ticker,
                    info["company"],
                    filing["filing_date"],
                    "4"
                )

                if baseline:
                    continue

                send_parsed_form4_alert(
                    ticker,
                    info["company"],
                    filing
                )

                print(
                    f"FORM 4 ALERT: "
                    f"{ticker} "
                    f"{accession}"
                )

            time.sleep(0.25)

        except Exception as exc:

            print(
                f"SEC {ticker} error: "
                f"{exc}"
            )


# =========================================================
# DATABASE CLEANUP
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
# AI PAPER TRADER
# =========================================================

def run_ai_trader_cycle():

    global ai_last_result

    try:

        print("Running AI trader cycle...")

        result = run_ai_cycle()

        with ai_result_lock:
            ai_last_result = result

        print(
            f"AI TRADER: "
            f"{result['product']} | "
            f"{result['decision']} | "
            f"Confidence "
            f"{result['confidence'] * 100:.1f}% | "
            f"Portfolio "
            f"Â£{result['portfolio_value']:.2f}"
        )

        if result.get("opened_position"):

            position = result["opened_position"]

            print(
                f"AI PAPER BUY: "
                f"{result['product']} "
                f"@ ${position['entry_price']:,.2f}"
            )

        if result.get("closed_trade"):

            trade = result["closed_trade"]

            print(
                f"AI PAPER CLOSE: "
                f"PnL Â£{trade['pnl']:+.2f} "
                f"({trade['reason']})"
            )

    except Exception as exc:

        print(
            f"AI trader error: {exc}"
        )


# =========================================================
# BACKGROUND MONITOR
# =========================================================

def monitor_main():

    global monitor_started
    global coinbase_product_ids

    if monitor_started:
        return

    monitor_started = True

    create_tables()

    seen_assets = get_seen_assets()

    coinbase_product_ids = (
        get_crypto_products()
    )

    sec_ticker_map = {}

    if SEC_USER_AGENT:

        try:

            sec_ticker_map = get_sec_ticker_map()

            first_sec_run = sec_database_empty()

            check_sec_insider_filings(
                sec_ticker_map,
                baseline=first_sec_run
            )

            if first_sec_run:
                print(
                    "SEC Form 4 baseline created."
                )

            print(
                f"SEC insider monitor loaded "
                f"for {len(STOCK_SYMBOLS)} "
                f"watchlist stocks."
            )

        except Exception as exc:

            print(
                f"SEC setup error: {exc}"
            )

    print("================================")
    print("ALPHA ALERTS ONLINE")

    print(
        f"Coinbase assets remembered: "
        f"{len(seen_assets)}"
    )

    print(
        f"Coinbase USD crypto markets: "
        f"{len(coinbase_product_ids)}"
    )

    print(
        f"Stocks watched: "
        f"{len(STOCK_SYMBOLS)}"
    )

    print(
        f"SEC insider monitoring: "
        f"{'ON' if sec_ticker_map else 'OFF'}"
    )

    print(
        f"AI paper trader interval: "
        f"{AI_CHECK_INTERVAL}s"
    )

    print("================================")

    websocket_thread = threading.Thread(
        target=run_coinbase_websocket,
        daemon=True
    )

    websocket_thread.start()

    last_stock_check = 0
    last_listing_check = 0
    last_sec_check = time.time()
    last_ai_check = 0

    cleanup_counter = 0

    while True:

        now = time.time()

        # =================================================
        # COINBASE LISTINGS
        # =================================================

        if now - last_listing_check >= 60:

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

        # =================================================
        # CRYPTO
        # =================================================

        try:
            check_all_crypto()

        except Exception as exc:

            print(
                f"Crypto monitor error: "
                f"{exc}"
            )

        # =================================================
        # AI PAPER TRADER
        # =================================================

        if (
            now - last_ai_check
            >= AI_CHECK_INTERVAL
        ):

            run_ai_trader_cycle()

            last_ai_check = time.time()

        # =================================================
        # STOCKS
        # =================================================

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
                    check_stock_symbol(symbol)

                except Exception as exc:

                    print(
                        f"Stock {symbol} error: "
                        f"{exc}"
                    )

                time.sleep(1)

            last_stock_check = time.time()

        # =================================================
        # SEC INSIDER
        # =================================================

        if (
            SEC_USER_AGENT
            and sec_ticker_map
            and (
                now - last_sec_check
                >= SEC_CHECK_INTERVAL
            )
        ):

            try:

                print(
                    "Checking SEC Form 4 filings..."
                )

                check_sec_insider_filings(
                    sec_ticker_map,
                    baseline=False
                )

            except Exception as exc:

                print(
                    f"SEC insider monitor error: "
                    f"{exc}"
                )

            last_sec_check = time.time()

        # =================================================
        # CLEANUP
        # =================================================

        cleanup_counter += 1

        if cleanup_counter >= 60:

            try:

                clean_old_samples()

                print(
                    "Old samples cleaned."
                )

            except Exception as exc:

                print(
                    f"Cleanup error: {exc}"
                )

            cleanup_counter = 0

        time.sleep(CHECK_INTERVAL)


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
            f"Slash command sync error: "
            f"{exc}"
        )


# =========================================================
# /PING
# =========================================================

@bot.tree.command(
    name="ping",
    description="Check whether Alpha Alerts is online"
)
async def ping(
    interaction: discord.Interaction
):

    await interaction.response.send_message(
        (
            f"ð Alpha Alerts online â "
            f"{round(bot.latency * 1000)} ms"
        )
    )


# =========================================================
# /STATUS
# =========================================================

@bot.tree.command(
    name="status",
    description="Show Alpha Alerts status"
)
async def status(
    interaction: discord.Interaction
):

    with crypto_prices_lock:
        crypto_count = len(crypto_prices)

    with ai_result_lock:
        ai_ready = ai_last_result is not None

    embed = discord.Embed(
        title="ð¢ Alpha Alerts Status",
        color=5763719
    )

    embed.add_field(
        name="Live Crypto Markets",
        value=str(crypto_count),
        inline=True
    )

    embed.add_field(
        name="Stock Watchlist",
        value=str(len(STOCK_SYMBOLS)),
        inline=True
    )

    embed.add_field(
        name="SEC Insider",
        value=(
            "ð¢ Active"
            if SEC_USER_AGENT
            and PRIVATE_INSIDER_WEBHOOK
            else
            "ð´ Disabled"
        ),
        inline=True
    )

    embed.add_field(
        name="AI Paper Trader",
        value=(
            "ð¢ Active"
            if ai_ready
            else
            "ð¡ Starting"
        ),
        inline=True
    )

    embed.add_field(
        name="Crypto Window",
        value=f"{CRYPTO_WINDOW_MINUTES}m",
        inline=True
    )

    embed.add_field(
        name="Major Crypto Trigger",
        value=f"{CRYPTO_MAJOR_SPIKE_PERCENT}%",
        inline=True
    )

    embed.add_field(
        name="Altcoin Trigger",
        value=f"{CRYPTO_SMALL_SPIKE_PERCENT}%",
        inline=True
    )

    embed.add_field(
        name="Stock Trigger",
        value=f"{STOCK_MOVE_PERCENT}%",
        inline=True
    )

    await interaction.response.send_message(
        embed=embed
    )


# =========================================================
# /CRYPTO
# =========================================================

@bot.tree.command(
    name="crypto",
    description="Get a live crypto price"
)
@app_commands.describe(
    symbol="Example: BTC, ETH, SOL"
)
async def crypto(
    interaction: discord.Interaction,
    symbol: str
):

    symbol = symbol.strip().upper()

    with crypto_prices_lock:
        price = crypto_prices.get(symbol)

    if price is None:

        await interaction.response.send_message(
            (
                f"No live Coinbase USD price "
                f"found for `{symbol}`."
            ),
            ephemeral=True
        )

        return

    old_prices = await asyncio.to_thread(
        get_old_crypto_prices
    )

    old_price = old_prices.get(symbol)

    move_text = (
        "Still collecting history."
    )

    if old_price:

        percent = (
            (price - old_price)
            / old_price
        ) * 100

        move_text = (
            f"{percent:+.2f}% "
            f"over ~"
            f"{CRYPTO_WINDOW_MINUTES}m"
        )

    embed = discord.Embed(
        title=f"ðª {symbol}",
        description=f"**${price:,.8f}**",
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
# /STOCK
# =========================================================

@bot.tree.command(
    name="stock",
    description="Get a stock quote"
)
@app_commands.describe(
    ticker="Example: NVDA, TSLA, AAPL"
)
async def stock(
    interaction: discord.Interaction,
    ticker: str
):

    await interaction.response.defer()

    ticker = ticker.strip().upper()

    try:

        data = await asyncio.to_thread(
            get_stock_quote,
            ticker
        )

        current = float(
            data.get("c", 0) or 0
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

        positive = change >= 0

        embed = discord.Embed(
            title=(
                f"{'ð' if positive else 'ð'} "
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
            value=f"${change:+,.2f}",
            inline=True
        )

        embed.add_field(
            name="High",
            value=f"${high:,.2f}",
            inline=True
        )

        embed.add_field(
            name="Low",
            value=f"${low:,.2f}",
            inline=True
        )

        embed.add_field(
            name="Previous Close",
            value=f"${previous_close:,.2f}",
            inline=True
        )

        embed.set_footer(
            text="Market data: Finnhub"
        )

        await interaction.followup.send(
            embed=embed
        )

    except Exception as exc:

        print(f"/stock error: {exc}")

        await interaction.followup.send(
            (
                f"Couldn't get a valid quote "
                f"for `{ticker}`."
            )
        )


# =========================================================
# /WATCHLIST
# =========================================================

@bot.tree.command(
    name="watchlist",
    description="Show the Alpha Alerts watchlist"
)
async def watchlist(
    interaction: discord.Interaction
):

    stock_text = ", ".join(
        f"`{symbol}`"
        for symbol in STOCK_SYMBOLS
    )

    embed = discord.Embed(
        title="ð Alpha Alerts Watchlist",
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
            "All online Coinbase USD "
            "crypto markets."
        ),
        inline=False
    )

    embed.add_field(
        name="Insider Monitoring",
        value=(
            "SEC Form 4 filings for "
            "the stock watchlist."
        ),
        inline=False
    )

    await interaction.response.send_message(
        embed=embed
    )


# =========================================================
# CRYPTO MOVERS HELPER
# =========================================================

async def calculate_crypto_moves():

    with crypto_prices_lock:
        current = dict(crypto_prices)

    old = await asyncio.to_thread(
        get_old_crypto_prices
    )

    results = []

    for symbol, price in current.items():

        old_price = old.get(symbol)

        if not old_price:
            continue

        percent = (
            (price - old_price)
            / old_price
        ) * 100

        results.append(
            (
                symbol,
                price,
                percent
            )
        )

    return results


# =========================================================
# /MOVERS
# =========================================================

@bot.tree.command(
    name="movers",
    description="Show the biggest crypto movers"
)
async def movers(
    interaction: discord.Interaction
):

    await interaction.response.defer()

    results = await calculate_crypto_moves()

    results.sort(
        key=lambda item: abs(item[2]),
        reverse=True
    )

    top = results[:10]

    if not top:

        await interaction.followup.send(
            "Still collecting price history."
        )

        return

    lines = []

    for symbol, price, percent in top:

        emoji = (
            "ð¢"
            if percent >= 0
            else "ð´"
        )

        lines.append(
            (
                f"{emoji} **{symbol}** "
                f"{percent:+.2f}% "
                f"â ${price:,.6f}"
            )
        )

    embed = discord.Embed(
        title=(
            f"ð¥ Biggest Crypto Movers "
            f"(~{CRYPTO_WINDOW_MINUTES}m)"
        ),
        description="\n".join(lines),
        color=3447003
    )

    await interaction.followup.send(
        embed=embed
    )


# =========================================================
# /TOPGAINERS
# =========================================================

@bot.tree.command(
    name="topgainers",
    description="Show the biggest crypto gainers"
)
async def topgainers(
    interaction: discord.Interaction
):

    await interaction.response.defer()

    results = await calculate_crypto_moves()

    results.sort(
        key=lambda item: item[2],
        reverse=True
    )

    top = results[:10]

    if not top:

        await interaction.followup.send(
            "Still collecting price history."
        )

        return

    lines = [
        (
            f"ð¢ **{symbol}** "
            f"{percent:+.2f}% "
            f"â ${price:,.6f}"
        )
        for symbol, price, percent
        in top
    ]

    embed = discord.Embed(
        title="ð Top Crypto Gainers",
        description="\n".join(lines),
        color=5763719
    )

    await interaction.followup.send(
        embed=embed
    )


# =========================================================
# /TOPLOSERS
# =========================================================

@bot.tree.command(
    name="toplosers",
    description="Show the biggest crypto losers"
)
async def toplosers(
    interaction: discord.Interaction
):

    await interaction.response.defer()

    results = await calculate_crypto_moves()

    results.sort(
        key=lambda item: item[2]
    )

    bottom = results[:10]

    if not bottom:

        await interaction.followup.send(
            "Still collecting price history."
        )

        return

    lines = [
        (
            f"ð´ **{symbol}** "
            f"{percent:+.2f}% "
            f"â ${price:,.6f}"
        )
        for symbol, price, percent
        in bottom
    ]

    embed = discord.Embed(
        title="ð Top Crypto Losers",
        description="\n".join(lines),
        color=15548997
    )

    await interaction.followup.send(
        embed=embed
    )


# =========================================================
# /ALERTS
# =========================================================

@bot.tree.command(
    name="alerts",
    description="Show current alert settings"
)
async def alerts(
    interaction: discord.Interaction
):

    embed = discord.Embed(
        title="âï¸ Alert Settings",
        color=3447003
    )

    embed.add_field(
        name="Major Crypto",
        value=(
            f"{CRYPTO_MAJOR_SPIKE_PERCENT}% "
            f"in {CRYPTO_WINDOW_MINUTES}m"
        ),
        inline=False
    )

    embed.add_field(
        name="Other Crypto",
        value=(
            f"{CRYPTO_SMALL_SPIKE_PERCENT}% "
            f"in {CRYPTO_WINDOW_MINUTES}m"
        ),
        inline=False
    )

    embed.add_field(
        name="Stocks",
        value=(
            f"{STOCK_MOVE_PERCENT}% "
            f"in {STOCK_WINDOW_MINUTES}m"
        ),
        inline=False
    )

    embed.add_field(
        name="Crypto Cooldown",
        value=(
            f"{CRYPTO_ALERT_COOLDOWN_MINUTES}m"
        ),
        inline=True
    )

    embed.add_field(
        name="Stock Cooldown",
        value=(
            f"{STOCK_ALERT_COOLDOWN_MINUTES}m"
        ),
        inline=True
    )

    embed.add_field(
        name="SEC Check",
        value=(
            f"Every "
            f"{SEC_CHECK_INTERVAL // 60}m"
        ),
        inline=True
    )

    await interaction.response.send_message(
        embed=embed
    )


# =========================================================
# /AI â OWNER ONLY
# =========================================================

@bot.tree.command(
    name="ai",
    description="Show your private AI paper trader"
)
async def ai(
    interaction: discord.Interaction
):

    if not is_bot_owner(interaction):

        await reject_non_owner(
            interaction
        )

        return

    with ai_result_lock:

        result = (
            dict(ai_last_result)
            if ai_last_result
            else None
        )

    if not result:

        await interaction.response.send_message(
            (
                "ð§  AI trader is still "
                "waiting for its first cycle."
            ),
            ephemeral=True
        )

        return

    decision = result["decision"]

    confidence = (
        result["confidence"]
        * 100
    )

    if decision == "BUY":
        decision_emoji = "ð¢"

    elif decision == "BEARISH":
        decision_emoji = "ð´"

    else:
        decision_emoji = "ð¡"

    embed = discord.Embed(
        title="ð§  Alpha AI Trader",
        description=(
            f"**{result['product']}**\n"
            f"${result['price']:,.2f}"
        ),
        color=3447003
    )

    embed.add_field(
        name="Decision",
        value=(
            f"{decision_emoji} "
            f"**{decision}**"
        ),
        inline=True
    )

    embed.add_field(
        name="Confidence",
        value=f"{confidence:.1f}%",
        inline=True
    )

    embed.add_field(
        name="Paper Portfolio",
        value=(
            f"Â£{result['portfolio_value']:,.2f}"
        ),
        inline=True
    )

    embed.add_field(
        name="Cash",
        value=f"Â£{result['cash']:,.2f}",
        inline=True
    )

    embed.add_field(
        name="Realised P&L",
        value=(
            f"Â£{result['realized_pnl']:+,.2f}"
        ),
        inline=True
    )

    embed.add_field(
        name="Record",
        value=(
            f"{result['wins']}W / "
            f"{result['losses']}L"
        ),
        inline=True
    )

    position = result.get("position")

    if position:

        entry = position["entry_price"]
        current = result["price"]

        unrealized = (
            (current / entry) - 1
        ) * 100

        embed.add_field(
            name="ð Open Paper Trade",
            value=(
                f"Entry: **${entry:,.2f}**\n"
                f"Current: **${current:,.2f}**\n"
                f"Move: **{unrealized:+.2f}%**\n"
                f"Size: **Â£{position['value']:.2f}**\n"
                f"Stop: **${position['stop_loss']:,.2f}**\n"
                f"Target: **${position['take_profit']:,.2f}**"
            ),
            inline=False
        )

    else:

        embed.add_field(
            name="Position",
            value="No open paper trade.",
            inline=False
        )

    embed.set_footer(
        text=(
            "OWNER ONLY â¢ PAPER TRADING â¢ "
            "No real funds"
        )
    )

    await interaction.response.send_message(
        embed=embed,
        ephemeral=True
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