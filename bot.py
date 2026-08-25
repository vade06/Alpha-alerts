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

from ai_trader import run_ai_cycle, get_learning_stats
from ai_backtest import run_backtest
from stock_backtest import run_stock_backtest


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

OWNER_DISCORD_USER_ID = os.getenv("OWNER_DISCORD_USER_ID")


# =========================================================
# GENERAL SETTINGS
# =========================================================

CHECK_INTERVAL = int(os.getenv("CHECK_INTERVAL", "60"))
AI_CHECK_INTERVAL = int(os.getenv("AI_CHECK_INTERVAL", "300"))


# =========================================================
# CRYPTO SETTINGS
# =========================================================

CRYPTO_WINDOW_MINUTES = int(os.getenv("CRYPTO_WINDOW_MINUTES", "15"))
CRYPTO_MAJOR_SPIKE_PERCENT = float(os.getenv("CRYPTO_MAJOR_SPIKE_PERCENT", "7"))
CRYPTO_SMALL_SPIKE_PERCENT = float(os.getenv("CRYPTO_SMALL_SPIKE_PERCENT", "15"))
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
        ),
    ).split(",")
    if symbol.strip()
]

STOCK_MOVE_PERCENT = float(os.getenv("STOCK_MOVE_PERCENT", "5"))
STOCK_WINDOW_MINUTES = int(os.getenv("STOCK_WINDOW_MINUTES", "30"))
STOCK_ALERT_COOLDOWN_MINUTES = int(
    os.getenv("STOCK_ALERT_COOLDOWN_MINUTES", "60")
)
STOCK_CHECK_INTERVAL = int(os.getenv("STOCK_CHECK_INTERVAL", "300"))


# =========================================================
# SEC / INSIDER SETTINGS
# =========================================================

SEC_CHECK_INTERVAL = int(os.getenv("SEC_CHECK_INTERVAL", "300"))
SEC_REQUEST_DELAY = float(os.getenv("SEC_REQUEST_DELAY", "0.25"))

INSIDER_CLUSTER_DAYS = int(os.getenv("INSIDER_CLUSTER_DAYS", "14"))
INSIDER_FEATURE_DAYS = int(os.getenv("INSIDER_FEATURE_DAYS", "30"))
INSIDER_MIN_ALERT_SCORE = int(os.getenv("INSIDER_MIN_ALERT_SCORE", "10"))
INSIDER_MAJOR_BUY_VALUE = float(os.getenv("INSIDER_MAJOR_BUY_VALUE", "1000000"))
INSIDER_HUGE_BUY_VALUE = float(os.getenv("INSIDER_HUGE_BUY_VALUE", "5000000"))
INSIDER_MAJOR_SELL_VALUE = float(os.getenv("INSIDER_MAJOR_SELL_VALUE", "5000000"))

SEC_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
SEC_SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik}.json"

# EDGAR has used both SC 13D/G and SCHEDULE 13D/G labels.
SEC_FORMS_TO_MONITOR = {
    "4",
    "4/A",
    "SC 13D",
    "SC 13D/A",
    "SC 13G",
    "SC 13G/A",
    "SCHEDULE 13D",
    "SCHEDULE 13D/A",
    "SCHEDULE 13G",
    "SCHEDULE 13G/A",
}

OWNERSHIP_FORMS = {
    "SC 13D",
    "SC 13D/A",
    "SC 13G",
    "SC 13G/A",
    "SCHEDULE 13D",
    "SCHEDULE 13D/A",
    "SCHEDULE 13G",
    "SCHEDULE 13G/A",
}


# =========================================================
# API URLS
# =========================================================

COINBASE_PRODUCTS_URL = "https://api.exchange.coinbase.com/products"
COINBASE_WEBSOCKET_URL = "wss://ws-feed.exchange.coinbase.com"
FINNHUB_QUOTE_URL = "https://finnhub.io/api/v1/quote"


# =========================================================
# GLOBAL STATE
# =========================================================

crypto_prices = {}
crypto_prices_lock = threading.Lock()

coinbase_product_ids = []
monitor_started = False

ai_last_result = None
ai_result_lock = threading.Lock()

discord_event_loop = None

backtest_running = False
backtest_lock = threading.Lock()

stock_backtest_running = False
stock_backtest_lock = threading.Lock()

http = requests.Session()
http.headers.update({"User-Agent": "Alpha-Alerts/15.0"})


# =========================================================
# OWNER SECURITY
# =========================================================

def is_bot_owner(interaction: discord.Interaction) -> bool:
    if not OWNER_DISCORD_USER_ID:
        return False
    return str(interaction.user.id) == str(OWNER_DISCORD_USER_ID)


async def reject_non_owner(interaction: discord.Interaction):
    await interaction.response.send_message(
        "You are not authorised to use the AI trader.",
        ephemeral=True,
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
        print("WARNING: FINNHUB_API_KEY missing. Stock monitoring disabled.")

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
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS seen_assets (
                    asset_name TEXT PRIMARY KEY
                );
                """
            )

            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS crypto_price_samples (
                    symbol TEXT NOT NULL,
                    price NUMERIC NOT NULL,
                    sampled_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
                """
            )

            cursor.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_crypto_samples_symbol_time
                ON crypto_price_samples (symbol, sampled_at DESC);
                """
            )

            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS crypto_alerts (
                    symbol TEXT PRIMARY KEY,
                    last_alerted_at TIMESTAMP,
                    last_alert_percent NUMERIC
                );
                """
            )

            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS stock_price_samples (
                    symbol TEXT NOT NULL,
                    price NUMERIC NOT NULL,
                    sampled_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
                """
            )

            cursor.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_stock_samples_symbol_time
                ON stock_price_samples (symbol, sampled_at DESC);
                """
            )

            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS stock_alerts (
                    symbol TEXT PRIMARY KEY,
                    last_alerted_at TIMESTAMP,
                    last_alert_percent NUMERIC
                );
                """
            )

            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS seen_sec_filings (
                    accession_number TEXT PRIMARY KEY,
                    ticker TEXT,
                    company_name TEXT,
                    filing_date TEXT,
                    form_type TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
                """
            )

            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS insider_transactions (
                    id BIGSERIAL PRIMARY KEY,
                    accession_number TEXT NOT NULL,
                    transaction_index INTEGER NOT NULL,
                    ticker TEXT NOT NULL,
                    company_name TEXT,
                    insider_name TEXT,
                    insider_role TEXT,
                    filing_date DATE,
                    transaction_date DATE,
                    transaction_code TEXT,
                    transaction_type TEXT,
                    security_title TEXT,
                    shares NUMERIC,
                    price NUMERIC,
                    transaction_value NUMERIC,
                    acquired_disposed TEXT,
                    shares_owned_after NUMERIC,
                    ownership_type TEXT,
                    insider_score INTEGER DEFAULT 0,
                    sec_url TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE (
                        accession_number,
                        insider_name,
                        transaction_index
                    )
                );
                """
            )

            cursor.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_insider_ticker_date
                ON insider_transactions (
                    ticker,
                    transaction_date DESC,
                    filing_date DESC
                );
                """
            )

            cursor.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_insider_code
                ON insider_transactions (transaction_code);
                """
            )

            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS ownership_filings (
                    accession_number TEXT PRIMARY KEY,
                    ticker TEXT NOT NULL,
                    company_name TEXT,
                    form_type TEXT,
                    filing_date DATE,
                    sec_url TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
                """
            )

            cursor.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_ownership_ticker_date
                ON ownership_filings (ticker, filing_date DESC);
                """
            )


# =========================================================
# DISCORD WEBHOOK HELPERS
# =========================================================

def post_to_webhooks(webhooks, title, description, fields=None, color=3447003):
    embed = {
        "title": title,
        "description": description,
        "color": color,
    }

    if fields:
        embed["fields"] = fields

    payload = {"embeds": [embed]}
    already_sent = set()

    for webhook in webhooks:
        if not webhook or webhook in already_sent:
            continue

        already_sent.add(webhook)

        try:
            response = http.post(webhook, json=payload, timeout=15)
            response.raise_for_status()
        except Exception as exc:
            print(f"Discord webhook error: {exc}")


def send_coinbase_alert(title, description, fields=None, color=3447003):
    post_to_webhooks([WEBHOOK, WEBHOOK2], title, description, fields, color)


def send_crypto_alert(title, description, fields=None, color=3447003):
    post_to_webhooks(
        [PRIVATE_CRYPTO_WEBHOOK, WEBHOOK2],
        title,
        description,
        fields,
        color,
    )


def send_stock_alert(title, description, fields=None, color=3447003):
    post_to_webhooks(
        [PRIVATE_STOCK_WEBHOOK, WEBHOOK2],
        title,
        description,
        fields,
        color,
    )


def send_insider_alert(title, description, fields=None, color=10181046):
    post_to_webhooks(
        [PRIVATE_INSIDER_WEBHOOK],
        title,
        description,
        fields,
        color,
    )


# =========================================================
# COINBASE PRODUCTS
# =========================================================

def get_coinbase_products():
    response = http.get(COINBASE_PRODUCTS_URL, timeout=20)
    response.raise_for_status()
    return response.json()


def get_crypto_products():
    products = get_coinbase_products()
    result = []
    ignored = {"USD", "USDC", "USDT", "EUR", "GBP"}

    for product in products:
        if product.get("status") != "online":
            continue
        if product.get("quote_currency") != "USD":
            continue

        product_id = product.get("id")
        base = product.get("base_currency")

        if not product_id or not base or base in ignored:
            continue

        result.append(product_id)

    return sorted(set(result))


# =========================================================
# COINBASE NEW ASSET TRACKING
# =========================================================

def get_seen_assets():
    with get_database() as conn:
        with conn.cursor() as cursor:
            cursor.execute("SELECT asset_name FROM seen_assets;")
            return {row[0] for row in cursor.fetchall()}


def save_assets(assets):
    if not assets:
        return

    rows = [(asset,) for asset in assets]

    with get_database() as conn:
        with conn.cursor() as cursor:
            execute_values(
                cursor,
                """
                INSERT INTO seen_assets (asset_name)
                VALUES %s
                ON CONFLICT (asset_name) DO NOTHING
                """,
                rows,
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
        markets_by_asset.setdefault(base, []).append(product_id)

    if not seen_assets:
        save_assets(current_assets)
        seen_assets.update(current_assets)
        print(f"Coinbase baseline created: {len(current_assets)} assets.")
        return

    new_assets = current_assets - seen_assets

    for asset in sorted(new_assets):
        save_assets({asset})
        seen_assets.add(asset)

        markets = sorted(markets_by_asset.get(asset, []))
        market_text = ", ".join(f"`{market}`" for market in markets[:20])

        send_coinbase_alert(
            "COINBASE TRADING LIVE",
            f"**{asset}** has appeared as a new online Coinbase asset.",
            [
                {
                    "name": "Markets",
                    "value": market_text or "Unknown",
                    "inline": False,
                }
            ],
            color=5763719,
        )

        print(f"NEW COINBASE ASSET: {asset}")


# =========================================================
# COINBASE WEBSOCKET
# =========================================================

def websocket_on_open(ws):
    print(
        f"Coinbase WebSocket connected. Subscribing to "
        f"{len(coinbase_product_ids)} markets."
    )

    chunk_size = 100

    for start in range(0, len(coinbase_product_ids), chunk_size):
        chunk = coinbase_product_ids[start : start + chunk_size]
        ws.send(
            json.dumps(
                {
                    "type": "subscribe",
                    "product_ids": chunk,
                    "channels": ["ticker"],
                }
            )
        )
        time.sleep(0.25)


def websocket_on_message(ws, message):
    try:
        data = json.loads(message)

        if data.get("type") != "ticker":
            return

        product_id = data.get("product_id")
        price = data.get("price")

        if not product_id or not price or not product_id.endswith("-USD"):
            return

        symbol = product_id[:-4]

        with crypto_prices_lock:
            crypto_prices[symbol] = float(price)

    except Exception as exc:
        print(f"WebSocket message error: {exc}")


def websocket_on_error(ws, error):
    print(f"Coinbase WebSocket error: {error}")


def websocket_on_close(ws, close_status_code, close_msg):
    print("Coinbase WebSocket disconnected.")


def run_coinbase_websocket():
    while True:
        try:
            ws = websocket.WebSocketApp(
                COINBASE_WEBSOCKET_URL,
                on_open=websocket_on_open,
                on_message=websocket_on_message,
                on_error=websocket_on_error,
                on_close=websocket_on_close,
            )

            ws.run_forever(ping_interval=30, ping_timeout=10)

        except Exception as exc:
            print(f"WebSocket failure: {exc}")

        print("Reconnecting WebSocket in 5 seconds...")
        time.sleep(5)


# =========================================================
# CRYPTO DATABASE
# =========================================================

def save_crypto_samples(prices):
    if not prices:
        return

    rows = [(symbol, price) for symbol, price in prices.items()]

    with get_database() as conn:
        with conn.cursor() as cursor:
            execute_values(
                cursor,
                """
                INSERT INTO crypto_price_samples (symbol, price)
                VALUES %s
                """,
                rows,
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
                WHERE sampled_at <= NOW() - (%s * INTERVAL '1 minute')
                ORDER BY symbol, sampled_at DESC;
                """,
                (CRYPTO_WINDOW_MINUTES,),
            )

            return {symbol: float(price) for symbol, price in cursor.fetchall()}


def can_send_crypto_alert(symbol):
    with get_database() as conn:
        with conn.cursor() as cursor:
            cursor.execute(
                """
                SELECT last_alerted_at
                FROM crypto_alerts
                WHERE symbol = %s;
                """,
                (symbol,),
            )

            result = cursor.fetchone()

            if not result:
                return True

            cursor.execute(
                """
                SELECT NOW() - %s >= (%s * INTERVAL '1 minute');
                """,
                (result[0], CRYPTO_ALERT_COOLDOWN_MINUTES),
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
                VALUES (%s, NOW(), %s)
                ON CONFLICT (symbol)
                DO UPDATE SET
                    last_alerted_at = EXCLUDED.last_alerted_at,
                    last_alert_percent = EXCLUDED.last_alert_percent;
                """,
                (symbol, percent),
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
        print("Waiting for Coinbase WebSocket prices...")
        return

    old_prices = get_old_crypto_prices()
    save_crypto_samples(current_prices)

    print(f"Tracking {len(current_prices)} live Coinbase USD markets.")

    candidates = []

    for symbol, price in current_prices.items():
        old_price = old_prices.get(symbol)

        if not old_price:
            continue

        percent = ((price - old_price) / old_price) * 100
        threshold = crypto_threshold(symbol)

        if abs(percent) >= threshold:
            candidates.append((symbol, price, old_price, percent, threshold))

    candidates.sort(key=lambda item: abs(item[3]), reverse=True)

    for symbol, price, old_price, percent, threshold in candidates[:3]:
        if not can_send_crypto_alert(symbol):
            continue

        positive = percent >= 0

        send_crypto_alert(
            "CRYPTO PRICE SPIKE",
            (
                f"**{symbol}** is **{'UP' if positive else 'DOWN'} "
                f"{abs(percent):.2f}%** in approximately "
                f"{CRYPTO_WINDOW_MINUTES} minutes."
            ),
            [
                {
                    "name": "Current Price",
                    "value": f"${price:,.8f}",
                    "inline": True,
                },
                {
                    "name": "Earlier Price",
                    "value": f"${old_price:,.8f}",
                    "inline": True,
                },
                {
                    "name": "Move",
                    "value": f"{percent:+.2f}%",
                    "inline": True,
                },
            ],
            color=5763719 if positive else 15548997,
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
                INSERT INTO stock_price_samples (symbol, price)
                VALUES (%s, %s);
                """,
                (symbol, price),
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
                    AND sampled_at <= NOW() - (%s * INTERVAL '1 minute')
                ORDER BY sampled_at DESC
                LIMIT 1;
                """,
                (symbol, STOCK_WINDOW_MINUTES),
            )

            result = cursor.fetchone()
            return float(result[0]) if result else None


def can_send_stock_alert(symbol):
    with get_database() as conn:
        with conn.cursor() as cursor:
            cursor.execute(
                """
                SELECT last_alerted_at
                FROM stock_alerts
                WHERE symbol = %s;
                """,
                (symbol,),
            )

            result = cursor.fetchone()

            if not result:
                return True

            cursor.execute(
                """
                SELECT NOW() - %s >= (%s * INTERVAL '1 minute');
                """,
                (result[0], STOCK_ALERT_COOLDOWN_MINUTES),
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
                VALUES (%s, NOW(), %s)
                ON CONFLICT (symbol)
                DO UPDATE SET
                    last_alerted_at = EXCLUDED.last_alerted_at,
                    last_alert_percent = EXCLUDED.last_alert_percent;
                """,
                (symbol, percent),
            )


# =========================================================
# FINNHUB STOCK DATA
# =========================================================

def get_stock_quote(symbol):
    if not FINNHUB_API_KEY:
        raise RuntimeError("FINNHUB_API_KEY missing.")

    response = http.get(
        FINNHUB_QUOTE_URL,
        params={"symbol": symbol, "token": FINNHUB_API_KEY},
        timeout=15,
    )
    response.raise_for_status()

    data = response.json()
    current = float(data.get("c", 0) or 0)

    if current <= 0:
        raise RuntimeError(f"No valid quote for {symbol}")

    return data


def check_stock_symbol(symbol):
    data = get_stock_quote(symbol)

    current_price = float(data["c"])
    old_price = get_old_stock_price(symbol)

    save_stock_sample(symbol, current_price)
    print(f"STOCK {symbol}: ${current_price:,.2f}")

    if not old_price:
        return

    percent = ((current_price - old_price) / old_price) * 100

    if abs(percent) < STOCK_MOVE_PERCENT:
        return

    if not can_send_stock_alert(symbol):
        return

    positive = percent >= 0

    send_stock_alert(
        "STOCK MOVE ALERT",
        (
            f"**{symbol}** is **{'UP' if positive else 'DOWN'} "
            f"{abs(percent):.2f}%** in approximately "
            f"{STOCK_WINDOW_MINUTES} minutes."
        ),
        [
            {
                "name": "Current Price",
                "value": f"${current_price:,.2f}",
                "inline": True,
            },
            {
                "name": "Move",
                "value": f"{percent:+.2f}%",
                "inline": True,
            },
        ],
        color=5763719 if positive else 15548997,
    )

    record_stock_alert(symbol, percent)


# =========================================================
# SEC HELPERS
# =========================================================

def sec_headers():
    return {
        "User-Agent": SEC_USER_AGENT or "Alpha-Alerts contact@example.com",
        "Accept-Encoding": "gzip, deflate",
        "Accept": "*/*",
    }


def get_sec_ticker_map():
    response = http.get(
        SEC_TICKERS_URL,
        headers=sec_headers(),
        timeout=20,
    )
    response.raise_for_status()
    data = response.json()

    ticker_map = {}

    for item in data.values():
        ticker = item["ticker"].strip().upper()
        ticker_map[ticker] = {
            "cik": str(item["cik_str"]).zfill(10),
            "company": item["title"],
        }

    return ticker_map


def sec_database_empty():
    with get_database() as conn:
        with conn.cursor() as cursor:
            cursor.execute("SELECT COUNT(*) FROM seen_sec_filings;")
            return cursor.fetchone()[0] == 0


def insider_database_empty():
    with get_database() as conn:
        with conn.cursor() as cursor:
            cursor.execute("SELECT COUNT(*) FROM insider_transactions;")
            return cursor.fetchone()[0] == 0


def ownership_database_empty():
    with get_database() as conn:
        with conn.cursor() as cursor:
            cursor.execute("SELECT COUNT(*) FROM ownership_filings;")
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
                (accession,),
            )
            return cursor.fetchone() is not None


def save_sec_filing(accession, ticker, company, filing_date, form_type):
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
                VALUES (%s, %s, %s, %s, %s)
                ON CONFLICT (accession_number) DO NOTHING;
                """,
                (accession, ticker, company, filing_date, form_type),
            )


def get_recent_sec_filings(ticker, cik, company):
    url = SEC_SUBMISSIONS_URL.format(cik=cik)

    response = http.get(
        url,
        headers=sec_headers(),
        timeout=20,
    )
    response.raise_for_status()

    data = response.json()
    recent = data.get("filings", {}).get("recent", {})

    forms = recent.get("form", [])
    accession_numbers = recent.get("accessionNumber", [])
    filing_dates = recent.get("filingDate", [])
    primary_documents = recent.get("primaryDocument", [])

    filings = []

    for index, form_type in enumerate(forms):
        if form_type not in SEC_FORMS_TO_MONITOR:
            continue

        if index >= len(accession_numbers):
            continue

        accession = accession_numbers[index]
        filing_date = filing_dates[index] if index < len(filing_dates) else ""
        primary_document = (
            primary_documents[index] if index < len(primary_documents) else ""
        )

        if not primary_document:
            continue

        accession_clean = accession.replace("-", "")
        cik_clean = str(int(cik))

        filing_url = (
            "https://www.sec.gov/Archives/edgar/data/"
            f"{cik_clean}/{accession_clean}/{primary_document}"
        )

        filings.append(
            {
                "ticker": ticker,
                "company": company,
                "accession": accession,
                "filing_date": filing_date,
                "form_type": form_type,
                "url": filing_url,
            }
        )

    return filings


# =========================================================
# FORM 4 XML PARSER / INSIDER SCORING
# =========================================================

def safe_xml_text(element, path, default="Unknown"):
    if element is None:
        return default

    found = element.find(path)

    if found is None or found.text is None:
        return default

    return found.text.strip()


def format_money(value, signed=False):
    try:
        number = float(value)
        prefix = "+" if signed and number > 0 else ""

        if abs(number) >= 1_000_000_000:
            return f"{prefix}${number / 1_000_000_000:.2f}B"
        if abs(number) >= 1_000_000:
            return f"{prefix}${number / 1_000_000:.2f}M"
        if abs(number) >= 1_000:
            return f"{prefix}${number / 1_000:.2f}K"
        return f"{prefix}${number:,.2f}"

    except Exception:
        return "Unknown"


def transaction_name(code):
    names = {
        "P": "Open-market / private BUY",
        "S": "Open-market / private SELL",
        "A": "Grant / Award",
        "D": "Sale back to issuer",
        "F": "Tax / Exercise Payment",
        "M": "Option Exercise",
        "C": "Conversion",
        "G": "Gift",
        "J": "Other",
        "L": "Small Acquisition",
        "W": "Will / Inheritance",
        "X": "Option Exercise",
        "Z": "Voting Trust",
    }
    return names.get(code, f"Other ({code})")


def normalise_role(role):
    return (role or "").lower().replace(".", "").strip()


def score_insider_transaction(code, value, role, shares, shares_owned_after):
    score = 0
    value = float(value or 0)
    shares = float(shares or 0)
    shares_owned_after = float(shares_owned_after or 0)
    role_lower = normalise_role(role)

    if code == "P":
        score += 25

        if "chief executive" in role_lower or "ceo" in role_lower:
            score += 20
        elif "chief financial" in role_lower or "cfo" in role_lower:
            score += 15
        elif "president" in role_lower:
            score += 12
        elif "director" in role_lower:
            score += 8
        elif "10% owner" in role_lower:
            score += 8

        if value >= 10_000_000:
            score += 35
        elif value >= INSIDER_HUGE_BUY_VALUE:
            score += 30
        elif value >= INSIDER_MAJOR_BUY_VALUE:
            score += 20
        elif value >= 500_000:
            score += 15
        elif value >= 100_000:
            score += 10
        elif value >= 25_000:
            score += 5

        if shares > 0 and shares_owned_after > shares:
            previous_shares = shares_owned_after - shares

            if previous_shares > 0:
                ownership_increase = shares / previous_shares

                if ownership_increase >= 1.0:
                    score += 20
                elif ownership_increase >= 0.50:
                    score += 15
                elif ownership_increase >= 0.25:
                    score += 10
                elif ownership_increase >= 0.10:
                    score += 5

    elif code == "S":
        score -= 10

        if value >= 20_000_000:
            score -= 20
        elif value >= 10_000_000:
            score -= 15
        elif value >= INSIDER_MAJOR_SELL_VALUE:
            score -= 10

        if "chief executive" in role_lower or "ceo" in role_lower:
            score -= 5
        if "chief financial" in role_lower or "cfo" in role_lower:
            score -= 4

    # Awards, exercises, gifts, withholding etc. are intentionally neutral.
    return max(-100, min(100, int(score)))


def get_form4_xml(filing_url):
    response = http.get(
        filing_url,
        headers=sec_headers(),
        timeout=20,
    )
    response.raise_for_status()
    return response.text


def parse_form4(xml_text):
    root = ET.fromstring(xml_text)

    owner = root.find(".//reportingOwner")
    insider_name = safe_xml_text(owner, ".//rptOwnerName")

    relationship = None
    if owner is not None:
        relationship = owner.find(".//reportingOwnerRelationship")

    roles = []

    if relationship is not None:
        is_director = safe_xml_text(relationship, "isDirector", "0")
        is_officer = safe_xml_text(relationship, "isOfficer", "0")
        is_ten_percent = safe_xml_text(relationship, "isTenPercentOwner", "0")
        is_other = safe_xml_text(relationship, "isOther", "0")
        officer_title = safe_xml_text(relationship, "officerTitle", "")
        other_text = safe_xml_text(relationship, "otherText", "")

        if is_director == "1":
            roles.append("Director")

        if is_officer == "1":
            roles.append(officer_title if officer_title else "Officer")

        if is_ten_percent == "1":
            roles.append("10% Owner")

        if is_other == "1" and other_text:
            roles.append(other_text)

    role = ", ".join(roles) if roles else "Insider"
    transactions = []

    for index, transaction in enumerate(
        root.findall(".//nonDerivativeTransaction"),
        start=1,
    ):
        code = safe_xml_text(
            transaction,
            ".//transactionCoding/transactionCode",
            "?",
        )
        transaction_date = safe_xml_text(
            transaction,
            ".//transactionDate/value",
            "",
        )
        shares_text = safe_xml_text(
            transaction,
            ".//transactionAmounts/transactionShares/value",
            "0",
        )
        price_text = safe_xml_text(
            transaction,
            ".//transactionAmounts/transactionPricePerShare/value",
            "",
        )
        acquired_disposed = safe_xml_text(
            transaction,
            ".//transactionAmounts/transactionAcquiredDisposedCode/value",
            "",
        )
        owned_after_text = safe_xml_text(
            transaction,
            ".//postTransactionAmounts/sharesOwnedFollowingTransaction/value",
            "0",
        )
        ownership_type = safe_xml_text(
            transaction,
            ".//ownershipNature/directOrIndirectOwnership/value",
            "",
        )
        security_title = safe_xml_text(
            transaction,
            ".//securityTitle/value",
            "Security",
        )

        try:
            shares = float(shares_text)
        except Exception:
            shares = 0.0

        try:
            price = float(price_text)
        except Exception:
            price = None

        try:
            shares_owned_after = float(owned_after_text)
        except Exception:
            shares_owned_after = 0.0

        value = shares * price if shares > 0 and price is not None else None

        score = score_insider_transaction(
            code=code,
            value=value,
            role=role,
            shares=shares,
            shares_owned_after=shares_owned_after,
        )

        transactions.append(
            {
                "index": index,
                "code": code,
                "action": transaction_name(code),
                "transaction_date": transaction_date,
                "security": security_title,
                "shares": shares,
                "price": price,
                "value": value,
                "direction": acquired_disposed,
                "shares_owned_after": shares_owned_after,
                "ownership_type": ownership_type,
                "score": score,
            }
        )

    return {
        "insider": insider_name,
        "role": role,
        "transactions": transactions,
    }


# =========================================================
# INSIDER DATABASE / FEATURES
# =========================================================

def save_insider_transaction(
    ticker,
    company,
    accession,
    filing_date,
    filing_url,
    insider,
    role,
    transaction,
):
    with get_database() as conn:
        with conn.cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO insider_transactions (
                    accession_number,
                    transaction_index,
                    ticker,
                    company_name,
                    insider_name,
                    insider_role,
                    filing_date,
                    transaction_date,
                    transaction_code,
                    transaction_type,
                    security_title,
                    shares,
                    price,
                    transaction_value,
                    acquired_disposed,
                    shares_owned_after,
                    ownership_type,
                    insider_score,
                    sec_url
                )
                VALUES (
                    %s, %s, %s, %s, %s, %s,
                    NULLIF(%s, '')::date,
                    NULLIF(%s, '')::date,
                    %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
                )
                ON CONFLICT (
                    accession_number,
                    insider_name,
                    transaction_index
                )
                DO NOTHING;
                """,
                (
                    accession,
                    int(transaction.get("index", 0)),
                    ticker,
                    company,
                    insider,
                    role,
                    filing_date,
                    transaction.get("transaction_date", ""),
                    transaction.get("code"),
                    transaction.get("action"),
                    transaction.get("security"),
                    transaction.get("shares"),
                    transaction.get("price"),
                    transaction.get("value"),
                    transaction.get("direction"),
                    transaction.get("shares_owned_after"),
                    transaction.get("ownership_type"),
                    transaction.get("score", 0),
                    filing_url,
                ),
            )


def get_cluster_buy_data(ticker, days=None):
    days = int(days or INSIDER_CLUSTER_DAYS)

    with get_database() as conn:
        with conn.cursor() as cursor:
            cursor.execute(
                """
                SELECT
                    COUNT(DISTINCT insider_name),
                    COUNT(*),
                    COALESCE(SUM(transaction_value), 0)
                FROM insider_transactions
                WHERE
                    ticker = %s
                    AND transaction_code = 'P'
                    AND COALESCE(transaction_date, filing_date)
                        >= CURRENT_DATE - %s;
                """,
                (ticker, days),
            )
            result = cursor.fetchone()

    return {
        "insiders": int(result[0] or 0),
        "transactions": int(result[1] or 0),
        "value": float(result[2] or 0),
    }


def calculate_cluster_bonus(ticker):
    cluster = get_cluster_buy_data(ticker)
    insiders = cluster["insiders"]

    if insiders >= 4:
        return 30
    if insiders >= 3:
        return 25
    if insiders >= 2:
        return 15
    return 0


def get_stock_insider_features(ticker, days=None):
    ticker = ticker.strip().upper()
    days = int(days or INSIDER_FEATURE_DAYS)

    with get_database() as conn:
        with conn.cursor() as cursor:
            cursor.execute(
                """
                SELECT
                    COALESCE(SUM(
                        CASE WHEN transaction_code = 'P'
                        THEN transaction_value ELSE 0 END
                    ), 0),
                    COALESCE(SUM(
                        CASE WHEN transaction_code = 'S'
                        THEN transaction_value ELSE 0 END
                    ), 0),
                    COUNT(*) FILTER (WHERE transaction_code = 'P'),
                    COUNT(*) FILTER (WHERE transaction_code = 'S'),
                    COUNT(DISTINCT insider_name)
                        FILTER (WHERE transaction_code = 'P'),
                    COUNT(DISTINCT insider_name)
                        FILTER (WHERE transaction_code = 'S'),
                    COALESCE(SUM(insider_score), 0)
                FROM insider_transactions
                WHERE
                    ticker = %s
                    AND COALESCE(transaction_date, filing_date)
                        >= CURRENT_DATE - %s;
                """,
                (ticker, days),
            )
            row = cursor.fetchone()

            cursor.execute(
                """
                SELECT insider_role
                FROM insider_transactions
                WHERE
                    ticker = %s
                    AND transaction_code = 'P'
                    AND COALESCE(transaction_date, filing_date)
                        >= CURRENT_DATE - %s
                ORDER BY insider_score DESC, transaction_value DESC NULLS LAST
                LIMIT 10;
                """,
                (ticker, days),
            )
            buy_roles = [str(item[0] or "").lower() for item in cursor.fetchall()]

            cursor.execute(
                """
                SELECT COUNT(*)
                FROM ownership_filings
                WHERE
                    ticker = %s
                    AND filing_date >= CURRENT_DATE - %s;
                """,
                (ticker, days),
            )
            ownership_count = int(cursor.fetchone()[0] or 0)

    buy_value = float(row[0] or 0)
    sell_value = float(row[1] or 0)
    buy_transactions = int(row[2] or 0)
    sell_transactions = int(row[3] or 0)
    unique_buyers = int(row[4] or 0)
    unique_sellers = int(row[5] or 0)
    raw_score = float(row[6] or 0)

    cluster_bonus = calculate_cluster_bonus(ticker)
    insider_score = max(-100.0, min(100.0, raw_score + cluster_bonus))

    ceo_buy = int(
        any("ceo" in role or "chief executive" in role for role in buy_roles)
    )
    cfo_buy = int(
        any("cfo" in role or "chief financial" in role for role in buy_roles)
    )

    return {
        "ticker": ticker,
        "insider_score": insider_score,
        "insider_buy_value": buy_value,
        "insider_sell_value": sell_value,
        "insider_net_value": buy_value - sell_value,
        "insider_buy_transactions": buy_transactions,
        "insider_sell_transactions": sell_transactions,
        "insider_buyers": unique_buyers,
        "insider_sellers": unique_sellers,
        "cluster_buy": int(unique_buyers >= 2),
        "cluster_buyers": unique_buyers,
        "ceo_buy": ceo_buy,
        "cfo_buy": cfo_buy,
        "ownership_filings": ownership_count,
    }


def get_recent_insider_transactions(ticker, limit=8):
    with get_database() as conn:
        with conn.cursor() as cursor:
            cursor.execute(
                """
                SELECT
                    insider_name,
                    insider_role,
                    transaction_code,
                    transaction_type,
                    transaction_date,
                    shares,
                    price,
                    transaction_value,
                    insider_score,
                    sec_url
                FROM insider_transactions
                WHERE ticker = %s
                ORDER BY
                    COALESCE(transaction_date, filing_date) DESC,
                    created_at DESC
                LIMIT %s;
                """,
                (ticker, int(limit)),
            )
            return cursor.fetchall()


# =========================================================
# FORM 4 PROCESSING / ALERTS
# =========================================================

def process_form4_filing(ticker, company, filing, send_alert=True):
    xml_text = get_form4_xml(filing["url"])
    parsed = parse_form4(xml_text)

    insider = parsed["insider"]
    role = parsed["role"]
    transactions = parsed["transactions"]

    if not transactions:
        return

    for transaction in transactions:
        save_insider_transaction(
            ticker=ticker,
            company=company,
            accession=filing["accession"],
            filing_date=filing["filing_date"],
            filing_url=filing["url"],
            insider=insider,
            role=role,
            transaction=transaction,
        )

    if not send_alert:
        return

    important = [tx for tx in transactions if tx["code"] in {"P", "S"}]
    display_transactions = important if important else transactions

    has_purchase = any(tx["code"] == "P" for tx in transactions)
    has_sale = any(tx["code"] == "S" for tx in transactions)

    transaction_score = sum(int(tx.get("score", 0)) for tx in transactions)
    cluster = get_cluster_buy_data(ticker)
    cluster_bonus = calculate_cluster_bonus(ticker) if has_purchase else 0

    final_score = max(-100, min(100, transaction_score + cluster_bonus))

    if has_purchase and final_score >= 75:
        title = "STRONG INSIDER BUY"
        color = 5763719
        interpretation = "Strong bullish insider conviction signal."
    elif has_purchase:
        title = "INSIDER BUY"
        color = 5763719
        interpretation = "Bullish insider activity detected."
    elif has_sale:
        title = "INSIDER SALE"
        color = 15548997
        interpretation = (
            "Insider selling detected. A sale is not automatically bearish "
            "and may be routine."
        )
    else:
        title = "INSIDER FORM 4"
        color = 10181046
        interpretation = (
            "Equity transaction detected, but it is not classified as "
            "a conviction purchase or sale."
        )

    fields = [
        {"name": "Insider", "value": insider, "inline": True},
        {"name": "Role", "value": role, "inline": True},
        {
            "name": "Filed",
            "value": filing["filing_date"] or "Unknown",
            "inline": True,
        },
    ]

    for transaction in display_transactions[:4]:
        shares = transaction["shares"]
        price = transaction["price"]
        value = transaction["value"]
        owned_after = transaction.get("shares_owned_after", 0)
        score = int(transaction.get("score", 0))

        text = f"**{transaction['action']}**\nShares: **{shares:,.0f}**"

        if transaction.get("transaction_date"):
            text += f"\nTransaction date: **{transaction['transaction_date']}**"

        if price is not None:
            text += f"\nPrice: **${price:,.4f}**"

        if value is not None:
            text += f"\nApprox value: **{format_money(value)}**"

        if owned_after:
            text += f"\nShares owned after: **{owned_after:,.0f}**"

        text += f"\nSignal score: **{score:+d}**"

        fields.append(
            {
                "name": transaction.get("security", "Transaction"),
                "value": text,
                "inline": False,
            }
        )

    if has_purchase and cluster["insiders"] >= 2:
        fields.append(
            {
                "name": "Cluster Buying Detected",
                "value": (
                    f"**{cluster['insiders']}** different insiders bought "
                    f"within the last {INSIDER_CLUSTER_DAYS} days.\n"
                    f"Recorded buying: **{format_money(cluster['value'])}**"
                ),
                "inline": False,
            }
        )

    fields.append(
        {
            "name": "Insider Intelligence Score",
            "value": f"**{final_score:+d} / 100**\n{interpretation}",
            "inline": False,
        }
    )

    fields.append(
        {
            "name": "Official SEC Filing",
            "value": filing["url"],
            "inline": False,
        }
    )

    if not important and abs(final_score) < INSIDER_MIN_ALERT_SCORE:
        print(
            f"LOW SIGNAL FORM 4: {ticker} | {insider} | {final_score:+d}"
        )
        return

    send_insider_alert(
        title,
        f"**{company} ({ticker})**\n\n{interpretation}",
        fields,
        color=color,
    )


# =========================================================
# 13D / 13G OWNERSHIP MONITOR
# =========================================================

def save_ownership_filing(filing):
    with get_database() as conn:
        with conn.cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO ownership_filings (
                    accession_number,
                    ticker,
                    company_name,
                    form_type,
                    filing_date,
                    sec_url
                )
                VALUES (
                    %s,
                    %s,
                    %s,
                    %s,
                    NULLIF(%s, '')::date,
                    %s
                )
                ON CONFLICT (accession_number) DO NOTHING;
                """,
                (
                    filing["accession"],
                    filing["ticker"],
                    filing["company"],
                    filing["form_type"],
                    filing["filing_date"],
                    filing["url"],
                ),
            )


def is_schedule_13d(form_type):
    return "13D" in str(form_type).upper()


def is_schedule_13g(form_type):
    return "13G" in str(form_type).upper()


def send_ownership_alert(filing):
    form_type = filing["form_type"]
    ticker = filing["ticker"]
    company = filing["company"]

    if is_schedule_13d(form_type):
        title = "MAJOR OWNERSHIP FILING"
        explanation = (
            "A Schedule 13D filing or amendment was detected. "
            "This can indicate a significant shareholder position and "
            "deserves further analysis before treating it as bullish or bearish."
        )
        color = 16776960
    else:
        title = "LARGE SHAREHOLDER FILING"
        explanation = (
            "A Schedule 13G filing or amendment was detected. "
            "This reports significant beneficial ownership but is not, by itself, "
            "a buy signal."
        )
        color = 3447003

    send_insider_alert(
        title,
        f"**{company} ({ticker})**\n\n{explanation}",
        [
            {"name": "Form", "value": f"**{form_type}**", "inline": True},
            {
                "name": "Filed",
                "value": filing["filing_date"] or "Unknown",
                "inline": True,
            },
            {
                "name": "Official SEC Filing",
                "value": filing["url"],
                "inline": False,
            },
        ],
        color=color,
    )


# =========================================================
# SEC MONITOR
# =========================================================

def backfill_recent_form4_transactions(ticker_map):
    print("Backfilling recent Form 4 transactions into insider database...")

    for ticker in STOCK_SYMBOLS:
        info = ticker_map.get(ticker)

        if not info:
            continue

        try:
            filings = get_recent_sec_filings(
                ticker,
                info["cik"],
                info["company"],
            )

            form4_filings = [
                filing for filing in filings if filing["form_type"] in {"4", "4/A"}
            ][:15]

            for filing in form4_filings:
                try:
                    process_form4_filing(
                        ticker,
                        info["company"],
                        filing,
                        send_alert=False,
                    )
                except Exception as exc:
                    print(
                        f"Form 4 backfill parse error {ticker} "
                        f"{filing['accession']}: {exc}"
                    )

                time.sleep(SEC_REQUEST_DELAY)

        except Exception as exc:
            print(f"SEC backfill {ticker} error: {exc}")

        time.sleep(SEC_REQUEST_DELAY)

    print("Recent Form 4 backfill finished.")


def check_sec_insider_filings(
    ticker_map,
    baseline_all=False,
    baseline_ownership=False,
):
    for ticker in STOCK_SYMBOLS:
        info = ticker_map.get(ticker)

        if not info:
            print(f"SEC ticker not found: {ticker}")
            continue

        try:
            filings = get_recent_sec_filings(
                ticker,
                info["cik"],
                info["company"],
            )

            for filing in filings[:25]:
                accession = filing["accession"]
                form_type = filing["form_type"]

                if sec_filing_seen(accession):
                    continue

                save_sec_filing(
                    accession,
                    ticker,
                    info["company"],
                    filing["filing_date"],
                    form_type,
                )

                if form_type in {"4", "4/A"}:
                    try:
                        process_form4_filing(
                            ticker,
                            info["company"],
                            filing,
                            send_alert=not baseline_all,
                        )
                    except Exception as parse_error:
                        print(
                            f"Form 4 parsing error {ticker} "
                            f"{accession}: {parse_error}"
                        )

                        if not baseline_all:
                            send_insider_alert(
                                "NEW INSIDER FILING",
                                (
                                    f"**{info['company']} ({ticker})** filed "
                                    f"a new SEC {form_type}."
                                ),
                                [
                                    {
                                        "name": "Filed",
                                        "value": filing["filing_date"] or "Unknown",
                                        "inline": True,
                                    },
                                    {
                                        "name": "Official SEC Filing",
                                        "value": filing["url"],
                                        "inline": False,
                                    },
                                ],
                            )

                    print(f"FORM {form_type}: {ticker} | {accession}")

                elif form_type in OWNERSHIP_FORMS:
                    save_ownership_filing(filing)

                    if not baseline_all and not baseline_ownership:
                        send_ownership_alert(filing)

                    print(f"{form_type}: {ticker} | {accession}")

            time.sleep(SEC_REQUEST_DELAY)

        except Exception as exc:
            print(f"SEC {ticker} error: {exc}")


# =========================================================
# DATABASE CLEANUP
# =========================================================

def clean_old_samples():
    with get_database() as conn:
        with conn.cursor() as cursor:
            cursor.execute(
                """
                DELETE FROM crypto_price_samples
                WHERE sampled_at < NOW() - INTERVAL '24 hours';
                """
            )

            cursor.execute(
                """
                DELETE FROM stock_price_samples
                WHERE sampled_at < NOW() - INTERVAL '24 hours';
                """
            )


# =========================================================
# DISCORD BOT
# =========================================================

intents = discord.Intents.default()

bot = commands.Bot(
    command_prefix="!",
    intents=intents,
)


# =========================================================
# PRIVATE OWNER DMS
# =========================================================

async def send_owner_dm(title, message):
    try:
        if not OWNER_DISCORD_USER_ID:
            print("Owner DM skipped: OWNER_DISCORD_USER_ID missing.")
            return

        user = await bot.fetch_user(int(OWNER_DISCORD_USER_ID))

        embed = discord.Embed(
            title=title,
            description=message,
            color=3447003,
        )
        embed.set_footer(text="Alpha AI Trader | Private owner notification")

        await user.send(embed=embed)

    except Exception as exc:
        print(f"Owner DM error: {exc}")


def queue_owner_dm(title, message):
    if discord_event_loop is None:
        print("Owner DM skipped: Discord event loop not ready.")
        return

    try:
        asyncio.run_coroutine_threadsafe(
            send_owner_dm(title, message),
            discord_event_loop,
        )
    except Exception as exc:
        print(f"Owner DM queue error: {exc}")


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
            f"AI TRADER: {result['product']} | {result['decision']} | "
            f"Confidence {result['confidence'] * 100:.1f}% | "
            f"Portfolio GBP {result['portfolio_value']:.2f}"
        )

        opened_position = result.get("opened_position")

        if opened_position:
            print(
                f"AI PAPER BUY: {opened_position['product']} @ "
                f"${opened_position['entry_price']:,.6f}"
            )

            queue_owner_dm(
                "AI PAPER TRADE OPENED",
                (
                    f"Market: **{opened_position['product']}**\n"
                    f"Entry: **${opened_position['entry_price']:,.6f}**\n"
                    f"Size: **GBP {opened_position['value']:.2f}**\n"
                    f"AI upside probability: "
                    f"**{opened_position.get('probability_up', 0) * 100:.1f}%**\n"
                    f"Stop: **${opened_position['stop_loss']:,.6f}**\n"
                    f"Target: **${opened_position['take_profit']:,.6f}**\n\n"
                    f"This is a **paper trade**. No real funds were used."
                ),
            )

        closed_trade = result.get("closed_trade")

        if closed_trade:
            print(
                f"AI PAPER CLOSE: {closed_trade['product']} | "
                f"PnL GBP {closed_trade['pnl']:+.2f} | "
                f"{closed_trade['reason']}"
            )

            queue_owner_dm(
                "AI PAPER TRADE CLOSED",
                (
                    f"Market: **{closed_trade['product']}**\n"
                    f"Entry: **${closed_trade['entry_price']:,.6f}**\n"
                    f"Exit: **${closed_trade['exit_price']:,.6f}**\n"
                    f"P&L: **GBP {closed_trade['pnl']:+.2f}**\n"
                    f"Reason: **{closed_trade['reason']}**\n\n"
                    f"This was a **paper trade**."
                ),
            )

    except Exception as exc:
        print(f"AI trader error: {exc}")


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
    coinbase_product_ids = get_crypto_products()

    sec_ticker_map = {}

    if SEC_USER_AGENT:
        try:
            sec_ticker_map = get_sec_ticker_map()

            first_sec_run = sec_database_empty()
            first_ownership_run = ownership_database_empty()
            first_insider_data_run = insider_database_empty()

            # If this upgraded version is being installed over the old bot,
            # backfill recent Form 4 transaction details without sending old alerts.
            if first_insider_data_run:
                backfill_recent_form4_transactions(sec_ticker_map)

            check_sec_insider_filings(
                sec_ticker_map,
                baseline_all=first_sec_run,
                baseline_ownership=first_ownership_run,
            )

            if first_sec_run:
                print("SEC filing baseline created.")

            if first_ownership_run:
                print("SEC 13D/13G ownership baseline created.")

            print(
                f"SEC insider monitor loaded for {len(STOCK_SYMBOLS)} "
                f"watchlist stocks."
            )

        except Exception as exc:
            print(f"SEC setup error: {exc}")

    print("================================")
    print("ALPHA ALERTS ONLINE")
    print(f"Coinbase assets remembered: {len(seen_assets)}")
    print(f"Coinbase USD crypto markets: {len(coinbase_product_ids)}")
    print(f"Stocks watched: {len(STOCK_SYMBOLS)}")
    print(f"SEC insider monitoring: {'ON' if sec_ticker_map else 'OFF'}")
    print(f"AI paper trader interval: {AI_CHECK_INTERVAL}s")
    print("================================")

    websocket_thread = threading.Thread(
        target=run_coinbase_websocket,
        daemon=True,
    )
    websocket_thread.start()

    last_stock_check = 0
    last_listing_check = 0
    last_sec_check = time.time()
    last_ai_check = 0
    cleanup_counter = 0

    while True:
        now = time.time()

        if now - last_listing_check >= 60:
            try:
                check_new_coinbase_assets(seen_assets)
            except Exception as exc:
                print(f"Coinbase listing error: {exc}")

            last_listing_check = now

        try:
            check_all_crypto()
        except Exception as exc:
            print(f"Crypto monitor error: {exc}")

        if now - last_ai_check >= AI_CHECK_INTERVAL:
            run_ai_trader_cycle()
            last_ai_check = time.time()

        if FINNHUB_API_KEY and now - last_stock_check >= STOCK_CHECK_INTERVAL:
            print("Checking stock watchlist...")

            for symbol in STOCK_SYMBOLS:
                try:
                    check_stock_symbol(symbol)
                except Exception as exc:
                    print(f"Stock {symbol} error: {exc}")

                time.sleep(1)

            last_stock_check = time.time()

        if (
            SEC_USER_AGENT
            and sec_ticker_map
            and now - last_sec_check >= SEC_CHECK_INTERVAL
        ):
            try:
                print("Checking SEC insider / ownership filings...")
                check_sec_insider_filings(
                    sec_ticker_map,
                    baseline_all=False,
                    baseline_ownership=False,
                )
            except Exception as exc:
                print(f"SEC insider monitor error: {exc}")

            last_sec_check = time.time()

        cleanup_counter += 1

        if cleanup_counter >= 60:
            try:
                clean_old_samples()
                print("Old samples cleaned.")
            except Exception as exc:
                print(f"Cleanup error: {exc}")

            cleanup_counter = 0

        time.sleep(CHECK_INTERVAL)


# =========================================================
# DISCORD READY
# =========================================================

@bot.event
async def on_ready():
    global discord_event_loop

    discord_event_loop = asyncio.get_running_loop()
    print(f"Discord bot online as {bot.user}")

    try:
        synced = await bot.tree.sync()
        print(f"Synced {len(synced)} slash commands.")
    except Exception as exc:
        print(f"Slash command sync error: {exc}")


# =========================================================
# /PING
# =========================================================

@bot.tree.command(
    name="ping",
    description="Check whether Alpha Alerts is online",
)
async def ping(interaction: discord.Interaction):
    await interaction.response.send_message(
        f"Alpha Alerts online - {round(bot.latency * 1000)} ms"
    )


# =========================================================
# /STATUS
# =========================================================

@bot.tree.command(
    name="status",
    description="Show Alpha Alerts status",
)
async def status(interaction: discord.Interaction):
    with crypto_prices_lock:
        crypto_count = len(crypto_prices)

    with ai_result_lock:
        ai_ready = ai_last_result is not None

    embed = discord.Embed(title="Alpha Alerts Status", color=5763719)

    embed.add_field(name="Live Crypto Markets", value=str(crypto_count), inline=True)
    embed.add_field(name="Stock Watchlist", value=str(len(STOCK_SYMBOLS)), inline=True)
    embed.add_field(
        name="SEC Insider",
        value=(
            "Active"
            if SEC_USER_AGENT and PRIVATE_INSIDER_WEBHOOK
            else "Disabled"
        ),
        inline=True,
    )
    embed.add_field(
        name="AI Paper Trader",
        value="Active" if ai_ready else "Starting",
        inline=True,
    )
    embed.add_field(
        name="Crypto Window",
        value=f"{CRYPTO_WINDOW_MINUTES}m",
        inline=True,
    )
    embed.add_field(
        name="Major Crypto Trigger",
        value=f"{CRYPTO_MAJOR_SPIKE_PERCENT}%",
        inline=True,
    )
    embed.add_field(
        name="Altcoin Trigger",
        value=f"{CRYPTO_SMALL_SPIKE_PERCENT}%",
        inline=True,
    )
    embed.add_field(
        name="Stock Trigger",
        value=f"{STOCK_MOVE_PERCENT}%",
        inline=True,
    )

    await interaction.response.send_message(embed=embed)


# =========================================================
# /CRYPTO
# =========================================================

@bot.tree.command(
    name="crypto",
    description="Get a live crypto price",
)
@app_commands.describe(symbol="Example: BTC, ETH, SOL")
async def crypto(interaction: discord.Interaction, symbol: str):
    symbol = symbol.strip().upper()

    with crypto_prices_lock:
        price = crypto_prices.get(symbol)

    if price is None:
        await interaction.response.send_message(
            f"No live Coinbase USD price found for `{symbol}`.",
            ephemeral=True,
        )
        return

    old_prices = await asyncio.to_thread(get_old_crypto_prices)
    old_price = old_prices.get(symbol)
    move_text = "Still collecting history."

    if old_price:
        percent = ((price - old_price) / old_price) * 100
        move_text = f"{percent:+.2f}% over ~{CRYPTO_WINDOW_MINUTES}m"

    embed = discord.Embed(
        title=symbol,
        description=f"**${price:,.8f}**",
        color=3447003,
    )
    embed.add_field(name="Recent Move", value=move_text, inline=False)
    embed.set_footer(text="Coinbase live market data")

    await interaction.response.send_message(embed=embed)


# =========================================================
# /STOCK
# =========================================================

@bot.tree.command(
    name="stock",
    description="Get a stock quote",
)
@app_commands.describe(ticker="Example: NVDA, TSLA, AAPL")
async def stock(interaction: discord.Interaction, ticker: str):
    await interaction.response.defer()
    ticker = ticker.strip().upper()

    try:
        data = await asyncio.to_thread(get_stock_quote, ticker)

        current = float(data.get("c", 0) or 0)
        change = float(data.get("d", 0) or 0)
        percent = float(data.get("dp", 0) or 0)
        high = float(data.get("h", 0) or 0)
        low = float(data.get("l", 0) or 0)
        previous_close = float(data.get("pc", 0) or 0)
        positive = change >= 0

        embed = discord.Embed(
            title=ticker,
            description=f"**${current:,.2f}**\n{percent:+.2f}% today",
            color=5763719 if positive else 15548997,
        )

        embed.add_field(name="Change", value=f"${change:+,.2f}", inline=True)
        embed.add_field(name="High", value=f"${high:,.2f}", inline=True)
        embed.add_field(name="Low", value=f"${low:,.2f}", inline=True)
        embed.add_field(
            name="Previous Close",
            value=f"${previous_close:,.2f}",
            inline=True,
        )
        embed.set_footer(text="Market data: Finnhub")

        await interaction.followup.send(embed=embed)

    except Exception as exc:
        print(f"/stock error: {exc}")
        await interaction.followup.send(
            f"Couldn't get a valid quote for `{ticker}`."
        )


# =========================================================
# /INSIDER
# =========================================================

@bot.tree.command(
    name="insider",
    description="Show recent SEC insider intelligence for a stock",
)
@app_commands.describe(ticker="Example: NVDA, TSLA, PLTR")
async def insider(interaction: discord.Interaction, ticker: str):
    await interaction.response.defer()
    ticker = ticker.strip().upper()

    try:
        features = await asyncio.to_thread(get_stock_insider_features, ticker)
        recent = await asyncio.to_thread(get_recent_insider_transactions, ticker, 5)

        score = float(features["insider_score"])

        if score >= 60:
            interpretation = "Strong bullish insider activity"
            color = 5763719
        elif score >= 20:
            interpretation = "Moderately bullish insider activity"
            color = 5763719
        elif score <= -40:
            interpretation = "Heavy insider selling activity"
            color = 15548997
        elif score < 0:
            interpretation = "Some insider selling activity"
            color = 15548997
        else:
            interpretation = "No strong insider signal"
            color = 3447003

        embed = discord.Embed(
            title=f"{ticker} Insider Intelligence",
            description=f"**Score: {score:+.0f}/100**\n{interpretation}",
            color=color,
        )

        embed.add_field(
            name="Insider Buys",
            value=(
                f"{features['insider_buy_transactions']} transactions\n"
                f"{format_money(features['insider_buy_value'])}"
            ),
            inline=True,
        )
        embed.add_field(
            name="Insider Sales",
            value=(
                f"{features['insider_sell_transactions']} transactions\n"
                f"{format_money(features['insider_sell_value'])}"
            ),
            inline=True,
        )
        embed.add_field(
            name="Net Insider Flow",
            value=format_money(features["insider_net_value"], signed=True),
            inline=True,
        )
        embed.add_field(
            name="Unique Buyers",
            value=str(features["insider_buyers"]),
            inline=True,
        )
        embed.add_field(
            name="Unique Sellers",
            value=str(features["insider_sellers"]),
            inline=True,
        )
        embed.add_field(
            name="Cluster Buy",
            value="YES" if features["cluster_buy"] else "NO",
            inline=True,
        )
        embed.add_field(
            name="CEO Buy",
            value="YES" if features["ceo_buy"] else "NO",
            inline=True,
        )
        embed.add_field(
            name="CFO Buy",
            value="YES" if features["cfo_buy"] else "NO",
            inline=True,
        )
        embed.add_field(
            name="13D / 13G Filings",
            value=str(features["ownership_filings"]),
            inline=True,
        )

        if recent:
            lines = []

            for row in recent:
                (
                    insider_name,
                    insider_role,
                    transaction_code,
                    transaction_type,
                    transaction_date,
                    shares,
                    price,
                    transaction_value,
                    transaction_score,
                    sec_url,
                ) = row

                value_text = (
                    format_money(transaction_value)
                    if transaction_value is not None
                    else "Value unknown"
                )

                lines.append(
                    f"**{transaction_code}** | {insider_name} | "
                    f"{value_text} | score {int(transaction_score or 0):+d}"
                )

            embed.add_field(
                name="Recent Transactions",
                value="\n".join(lines)[:1024],
                inline=False,
            )

        embed.set_footer(
            text=f"SEC activity | Rolling {INSIDER_FEATURE_DAYS}-day window"
        )

        await interaction.followup.send(embed=embed)

    except Exception as exc:
        print(f"/insider error: {exc}")
        await interaction.followup.send(
            f"Couldn't load insider information for `{ticker}`."
        )


# =========================================================
# /WATCHLIST
# =========================================================

@bot.tree.command(
    name="watchlist",
    description="Show the Alpha Alerts watchlist",
)
async def watchlist(interaction: discord.Interaction):
    stock_text = ", ".join(f"`{symbol}`" for symbol in STOCK_SYMBOLS)

    embed = discord.Embed(title="Alpha Alerts Watchlist", color=3447003)
    embed.add_field(name="Stocks", value=stock_text, inline=False)
    embed.add_field(
        name="Crypto",
        value="All online Coinbase USD crypto markets.",
        inline=False,
    )
    embed.add_field(
        name="Insider Monitoring",
        value=(
            "SEC Form 4/4-A insider transactions plus Schedule 13D/13G "
            "ownership filings for the stock watchlist."
        ),
        inline=False,
    )

    await interaction.response.send_message(embed=embed)


# =========================================================
# CRYPTO MOVERS HELPER
# =========================================================

async def calculate_crypto_moves():
    with crypto_prices_lock:
        current = dict(crypto_prices)

    old = await asyncio.to_thread(get_old_crypto_prices)
    results = []

    for symbol, price in current.items():
        old_price = old.get(symbol)

        if not old_price:
            continue

        percent = ((price - old_price) / old_price) * 100
        results.append((symbol, price, percent))

    return results


# =========================================================
# /MOVERS
# =========================================================

@bot.tree.command(
    name="movers",
    description="Show the biggest crypto movers",
)
async def movers(interaction: discord.Interaction):
    await interaction.response.defer()
    results = await calculate_crypto_moves()
    results.sort(key=lambda item: abs(item[2]), reverse=True)
    top = results[:10]

    if not top:
        await interaction.followup.send("Still collecting price history.")
        return

    lines = [
        f"**{symbol}** {percent:+.2f}% - ${price:,.6f}"
        for symbol, price, percent in top
    ]

    embed = discord.Embed(
        title=f"Biggest Crypto Movers (~{CRYPTO_WINDOW_MINUTES}m)",
        description="\n".join(lines),
        color=3447003,
    )

    await interaction.followup.send(embed=embed)


# =========================================================
# /TOPGAINERS
# =========================================================

@bot.tree.command(
    name="topgainers",
    description="Show the biggest crypto gainers",
)
async def topgainers(interaction: discord.Interaction):
    await interaction.response.defer()
    results = await calculate_crypto_moves()
    results.sort(key=lambda item: item[2], reverse=True)
    top = results[:10]

    if not top:
        await interaction.followup.send("Still collecting price history.")
        return

    lines = [
        f"**{symbol}** {percent:+.2f}% - ${price:,.6f}"
        for symbol, price, percent in top
    ]

    embed = discord.Embed(
        title="Top Crypto Gainers",
        description="\n".join(lines),
        color=5763719,
    )
    await interaction.followup.send(embed=embed)


# =========================================================
# /TOPLOSERS
# =========================================================

@bot.tree.command(
    name="toplosers",
    description="Show the biggest crypto losers",
)
async def toplosers(interaction: discord.Interaction):
    await interaction.response.defer()
    results = await calculate_crypto_moves()
    results.sort(key=lambda item: item[2])
    bottom = results[:10]

    if not bottom:
        await interaction.followup.send("Still collecting price history.")
        return

    lines = [
        f"**{symbol}** {percent:+.2f}% - ${price:,.6f}"
        for symbol, price, percent in bottom
    ]

    embed = discord.Embed(
        title="Top Crypto Losers",
        description="\n".join(lines),
        color=15548997,
    )
    await interaction.followup.send(embed=embed)


# =========================================================
# /ALERTS
# =========================================================

@bot.tree.command(
    name="alerts",
    description="Show current alert settings",
)
async def alerts(interaction: discord.Interaction):
    embed = discord.Embed(title="Alert Settings", color=3447003)

    embed.add_field(
        name="Major Crypto",
        value=f"{CRYPTO_MAJOR_SPIKE_PERCENT}% in {CRYPTO_WINDOW_MINUTES}m",
        inline=False,
    )
    embed.add_field(
        name="Other Crypto",
        value=f"{CRYPTO_SMALL_SPIKE_PERCENT}% in {CRYPTO_WINDOW_MINUTES}m",
        inline=False,
    )
    embed.add_field(
        name="Stocks",
        value=f"{STOCK_MOVE_PERCENT}% in {STOCK_WINDOW_MINUTES}m",
        inline=False,
    )
    embed.add_field(
        name="Crypto Cooldown",
        value=f"{CRYPTO_ALERT_COOLDOWN_MINUTES}m",
        inline=True,
    )
    embed.add_field(
        name="Stock Cooldown",
        value=f"{STOCK_ALERT_COOLDOWN_MINUTES}m",
        inline=True,
    )
    embed.add_field(
        name="SEC Check",
        value=f"Every {SEC_CHECK_INTERVAL // 60}m",
        inline=True,
    )
    embed.add_field(
        name="Insider Window",
        value=f"{INSIDER_FEATURE_DAYS} days",
        inline=True,
    )
    embed.add_field(
        name="Cluster Window",
        value=f"{INSIDER_CLUSTER_DAYS} days",
        inline=True,
    )

    await interaction.response.send_message(embed=embed)


# =========================================================
# /AI - OWNER ONLY
# =========================================================

@bot.tree.command(
    name="ai",
    description="Show your private AI paper trader",
)
async def ai(interaction: discord.Interaction):
    if not is_bot_owner(interaction):
        await reject_non_owner(interaction)
        return

    with ai_result_lock:
        result = dict(ai_last_result) if ai_last_result else None

    if not result:
        await interaction.response.send_message(
            "AI trader is still waiting for its first cycle.",
            ephemeral=True,
        )
        return

    decision = result["decision"]
    confidence = result["confidence"] * 100

    embed = discord.Embed(
        title="Alpha AI Trader",
        description=f"**{result['product']}**\n${result['price']:,.2f}",
        color=3447003,
    )

    embed.add_field(name="Decision", value=f"**{decision}**", inline=True)
    embed.add_field(
        name="Confidence",
        value=f"{confidence:.1f}%",
        inline=True,
    )
    embed.add_field(
        name="Paper Portfolio",
        value=f"GBP {result['portfolio_value']:,.2f}",
        inline=True,
    )
    embed.add_field(
        name="Cash",
        value=f"GBP {result['cash']:,.2f}",
        inline=True,
    )
    embed.add_field(
        name="Realised P&L",
        value=f"GBP {result['realized_pnl']:+,.2f}",
        inline=True,
    )
    embed.add_field(
        name="Record",
        value=f"{result['wins']}W / {result['losses']}L",
        inline=True,
    )

    position = result.get("position")

    if position:
        entry = position["entry_price"]
        current = result["price"]
        unrealized = ((current / entry) - 1) * 100

        embed.add_field(
            name="Open Paper Trade",
            value=(
                f"Entry: **${entry:,.2f}**\n"
                f"Current: **${current:,.2f}**\n"
                f"Move: **{unrealized:+.2f}%**\n"
                f"Size: **GBP {position['value']:.2f}**\n"
                f"Stop: **${position['stop_loss']:,.2f}**\n"
                f"Target: **${position['take_profit']:,.2f}**"
            ),
            inline=False,
        )
    else:
        embed.add_field(
            name="Position",
            value="No open paper trade.",
            inline=False,
        )

    embed.set_footer(text="OWNER ONLY | PAPER TRADING | No real funds")

    await interaction.response.send_message(embed=embed, ephemeral=True)


# =========================================================
# CRYPTO BACKTEST FORMATTER
# =========================================================

def format_backtest_dm(result, days):
    trades = int(result.get("trades", 0))
    wins = int(result.get("wins", 0))
    losses = int(result.get("losses", 0))
    win_rate = float(result.get("win_rate", 0)) * 100
    pnl = float(result.get("pnl", 0))
    gross_pnl = float(result.get("gross_pnl", 0))
    fees = float(result.get("fees", 0))
    max_drawdown = float(result.get("max_drawdown", 0))
    profit_factor = float(result.get("profit_factor", 0))
    expectancy = float(result.get("expectancy", 0))
    fee_per_side = float(result.get("fee_per_side", 0)) * 100

    lines = [
        f"Strategy: **{result.get('strategy', 'AI')}**",
        f"Period: **{result.get('days', days)} days**",
        f"Markets completed: **{result.get('markets', 0)}**",
    ]

    if "markets_discovered" in result:
        lines.append(
            f"Markets discovered: **{result.get('markets_discovered', 0)}**"
        )
        lines.append(f"Markets skipped: **{result.get('markets_skipped', 0)}**")

    lines.extend(
        [
            f"Trades: **{trades}**",
            f"Trades/day: **{float(result.get('trades_per_day', 0)):.2f}**",
            f"Record: **{wins}W / {losses}L**",
            f"Win rate: **{win_rate:.1f}%**",
            f"Gross P&L: **GBP {gross_pnl:+.2f}**",
            f"Fees: **GBP {fees:.2f}**",
            f"Net P&L: **GBP {pnl:+.2f}**",
            f"Profit factor: **{profit_factor:.2f}**",
            f"Expectancy/trade: **GBP {expectancy:+.3f}**",
            f"Max drawdown: **GBP {max_drawdown:.2f}**",
            (
                f"Best market: **{result.get('best_market', 'N/A')}** "
                f"(GBP {float(result.get('best_market_pnl', 0)):+.2f})"
            ),
            (
                f"Worst market: **{result.get('worst_market', 'N/A')}** "
                f"(GBP {float(result.get('worst_market_pnl', 0)):+.2f})"
            ),
            f"Estimated fee / side: **{fee_per_side:.3f}%**",
        ]
    )

    confidence_report = result.get("confidence_report", [])

    if confidence_report:
        lines.append("")
        lines.append("**Confidence performance:**")

        for item in confidence_report:
            lines.append(
                f"{item.get('bucket', '?')}% | "
                f"{item.get('trades', 0)} trades | "
                f"{float(item.get('win_rate', 0)) * 100:.1f}% WR | "
                f"GBP {float(item.get('pnl', 0)):+.2f}"
            )

    exit_reasons = result.get("exit_reasons", {})

    if exit_reasons:
        lines.append("")
        lines.append("**Exit reasons:**")

        for reason, count in exit_reasons.items():
            lines.append(f"{reason}: **{count}**")

    top_markets = result.get("top_markets", result.get("by_market", []))

    if top_markets:
        lines.append("")
        lines.append("**Top markets:**")

        for market in top_markets[:10]:
            lines.append(
                f"{market['product']} | "
                f"{market['trades']} trades | "
                f"{market['win_rate'] * 100:.1f}% WR | "
                f"GBP {market['pnl']:+.2f}"
            )

    return "\n".join(lines)


# =========================================================
# STOCK BACKTEST FORMATTER
# =========================================================

def format_stock_backtest_dm(result, days):
    trades = int(result.get("trades", 0))
    wins = int(result.get("wins", 0))
    losses = int(result.get("losses", 0))
    win_rate = float(result.get("win_rate", 0)) * 100

    pnl = float(result.get("pnl", 0))
    gross_pnl = float(result.get("gross_pnl", 0))
    fees = float(result.get("fees", 0))

    fixed_size_pnl = float(result.get("fixed_size_pnl", 0))
    sizing_improvement = float(
        result.get("dynamic_sizing_improvement", pnl - fixed_size_pnl)
    )
    average_trade_size = float(result.get("average_trade_size", 0))
    profit_factor = float(result.get("profit_factor", 0))
    expectancy = float(result.get("expectancy", 0))
    max_drawdown = float(result.get("max_drawdown", 0))
    worst_symbol_drawdown = float(result.get("worst_symbol_drawdown", 0))

    lines = [
        f"Strategy: **{result.get('strategy', 'STOCK_AI')}**",
        f"Unseen test: **{result.get('days', days)} days**",
        f"Training period: **~{result.get('training_days', 0)} days**",
        f"Symbols configured: **{result.get('symbols_configured', 0)}**",
        f"Symbols completed: **{result.get('symbols_completed', 0)}**",
        f"Symbols skipped: **{result.get('symbols_skipped', 0)}**",
        "",
        f"Trades: **{trades}**",
        f"Trades/day: **{float(result.get('trades_per_day', 0)):.2f}**",
        f"Record: **{wins}W / {losses}L**",
        f"Win rate: **{win_rate:.1f}%**",
        "",
        f"Gross P&L: **GBP {gross_pnl:+.2f}**",
        f"Trading costs: **GBP {fees:.2f}**",
        f"Dynamic-size net P&L: **GBP {pnl:+.2f}**",
        f"Fixed GBP 100 P&L: **GBP {fixed_size_pnl:+.2f}**",
        f"Sizing improvement: **GBP {sizing_improvement:+.2f}**",
        f"Average trade size: **GBP {average_trade_size:.2f}**",
        "",
        f"Profit factor: **{profit_factor:.2f}**",
        f"Expectancy/trade: **GBP {expectancy:+.3f}**",
        f"Portfolio max drawdown: **GBP {max_drawdown:.2f}**",
        f"Worst symbol drawdown: **GBP {worst_symbol_drawdown:.2f}**",
        "",
        (
            f"Best symbol: **{result.get('best_symbol', 'N/A')}** "
            f"(GBP {float(result.get('best_symbol_pnl', 0)):+.2f})"
        ),
        (
            f"Worst symbol: **{result.get('worst_symbol', 'N/A')}** "
            f"(GBP {float(result.get('worst_symbol_pnl', 0)):+.2f})"
        ),
    ]

    confidence_report = result.get("confidence_report", [])

    if confidence_report:
        lines.append("")
        lines.append("**Calibrated confidence:**")

        for item in confidence_report:
            lines.append(
                f"{item.get('bucket', '?')}% | "
                f"{item.get('trades', 0)} trades | "
                f"{float(item.get('win_rate', 0)) * 100:.1f}% WR | "
                f"GBP {float(item.get('pnl', 0)):+.2f}"
            )

    exit_reasons = result.get("exit_reasons", {})

    if exit_reasons:
        lines.append("")
        lines.append("**Exit reasons:**")

        for reason, count in exit_reasons.items():
            lines.append(f"{reason}: **{count}**")

    top_symbols = result.get("top_symbols", [])

    if top_symbols:
        lines.append("")
        lines.append("**Top symbols:**")

        for item in top_symbols[:10]:
            lines.append(
                f"{item['symbol']} | "
                f"{item['trades']} trades | "
                f"{item['win_rate'] * 100:.1f}% WR | "
                f"GBP {item['pnl']:+.2f}"
            )

    return "\n".join(lines)


# =========================================================
# CRYPTO BACKTEST BACKGROUND
# =========================================================

def run_backtest_background(days):
    global backtest_running

    try:
        print(f"BACKGROUND BACKTEST STARTED: {days} days")
        result = run_backtest(days)

        print(
            f"BACKGROUND BACKTEST FINISHED: {days} days | "
            f"{result.get('markets', 0)} markets | "
            f"GBP {float(result.get('pnl', 0)):+.2f}"
        )

        queue_owner_dm(
            "AI BACKTEST FINISHED",
            format_backtest_dm(result, days),
        )

    except Exception as exc:
        print(f"Background backtest error: {exc}")
        queue_owner_dm(
            "AI BACKTEST FAILED",
            (
                f"The {days}-day backtest failed.\n\n"
                f"Error: {exc}\n\n"
                f"Check Railway logs for more detail."
            ),
        )

    finally:
        with backtest_lock:
            backtest_running = False


# =========================================================
# STOCK BACKTEST BACKGROUND
# =========================================================

def run_stock_backtest_background(days):
    global stock_backtest_running

    try:
        print(f"STOCK BACKTEST STARTED: {days} days")
        result = run_stock_backtest(days)

        print(
            f"STOCK BACKTEST FINISHED: {days} days | "
            f"{result.get('symbols_completed', 0)} symbols | "
            f"GBP {float(result.get('pnl', 0)):+.2f}"
        )

        queue_owner_dm(
            "STOCK AI BACKTEST FINISHED",
            format_stock_backtest_dm(result, days),
        )

    except Exception as exc:
        print(f"Stock backtest error: {exc}")
        queue_owner_dm(
            "STOCK AI BACKTEST FAILED",
            (
                f"The {days}-day stock backtest failed.\n\n"
                f"Error: {exc}\n\n"
                f"Check Railway logs for more detail."
            ),
        )

    finally:
        with stock_backtest_lock:
            stock_backtest_running = False


# =========================================================
# /AIBACKTEST - OWNER ONLY
# =========================================================

@bot.tree.command(
    name="aibacktest",
    description="Start a private crypto AI backtest and DM the result",
)
@app_commands.describe(days="Number of days to test, from 3 to 30")
async def aibacktest(interaction: discord.Interaction, days: int = 7):
    global backtest_running

    if not is_bot_owner(interaction):
        await reject_non_owner(interaction)
        return

    days = max(3, min(int(days), 30))

    with backtest_lock:
        if backtest_running:
            await interaction.response.send_message(
                "A crypto backtest is already running. "
                "Wait for the DM when it finishes.",
                ephemeral=True,
            )
            return

        backtest_running = True

    thread = threading.Thread(
        target=run_backtest_background,
        args=(days,),
        daemon=True,
    )
    thread.start()

    await interaction.response.send_message(
        f"Crypto backtest started for **{days} days**.\n"
        f"I will DM you when it finishes.",
        ephemeral=True,
    )


# =========================================================
# /STOCKBACKTEST - OWNER ONLY
# =========================================================

@bot.tree.command(
    name="stockbacktest",
    description="Start a private AI stock backtest and DM the result",
)
@app_commands.describe(days="Number of unseen calendar days to test, from 5 to 180")
async def stockbacktest(interaction: discord.Interaction, days: int = 30):
    global stock_backtest_running

    if not is_bot_owner(interaction):
        await reject_non_owner(interaction)
        return

    days = max(5, min(int(days), 180))

    with stock_backtest_lock:
        if stock_backtest_running:
            await interaction.response.send_message(
                "A stock backtest is already running. "
                "Wait for the DM when it finishes.",
                ephemeral=True,
            )
            return

        stock_backtest_running = True

    thread = threading.Thread(
        target=run_stock_backtest_background,
        args=(days,),
        daemon=True,
    )
    thread.start()

    await interaction.response.send_message(
        (
            f"Stock backtest started for **{days} days**.\n"
            f"I will DM you when it finishes.\n\n"
            f"This is a paper backtest only. No real stock trades will be placed."
        ),
        ephemeral=True,
    )


# =========================================================
# /AIRANKINGS - OWNER ONLY
# =========================================================

@bot.tree.command(
    name="airankings",
    description="Rank markets from your private AI trader",
)
async def airankings(interaction: discord.Interaction):
    if not is_bot_owner(interaction):
        await reject_non_owner(interaction)
        return

    with ai_result_lock:
        result = dict(ai_last_result) if ai_last_result else None

    if not result:
        await interaction.response.send_message(
            "AI trader is still waiting for its first cycle.",
            ephemeral=True,
        )
        return

    rankings = result.get("market_rankings", [])

    if not rankings:
        await interaction.response.send_message(
            "No market rankings are available yet.",
            ephemeral=True,
        )
        return

    lines = []

    for index, market in enumerate(rankings, start=1):
        product = market.get("product", "Unknown")
        decision = market.get("decision", "HOLD")
        probability_up = float(market.get("probability_up", 0)) * 100
        price = float(market.get("price", 0))

        lines.append(
            f"**{index}. {product}** [{decision}]\n"
            f"Price: `${price:,.6f}` | "
            f"Upside probability: **{probability_up:.1f}%**"
        )

    embed = discord.Embed(
        title="Alpha AI Market Rankings",
        description="\n\n".join(lines),
        color=3447003,
    )

    best = result.get("best_opportunity")

    if best:
        embed.add_field(
            name="Highest-Ranked Setup",
            value=(
                f"**{best.get('product', 'Unknown')}** | "
                f"{float(best.get('probability_up', 0)) * 100:.1f}% "
                f"estimated upside probability"
            ),
            inline=False,
        )

    embed.add_field(
        name="Markets Scanned",
        value=str(result.get("markets_scanned", len(rankings))),
        inline=True,
    )

    position = result.get("position")
    embed.add_field(
        name="Paper Position",
        value=position.get("product", "Unknown") if position else "None",
        inline=True,
    )

    embed.set_footer(
        text="OWNER ONLY | PAPER TRADING | Model estimates are not guarantees"
    )

    await interaction.response.send_message(embed=embed, ephemeral=True)


# =========================================================
# /AILEARNING - OWNER ONLY
# =========================================================

@bot.tree.command(
    name="ailearning",
    description="Show how your AI is learning from old predictions",
)
async def ailearning(interaction: discord.Interaction):
    if not is_bot_owner(interaction):
        await reject_non_owner(interaction)
        return

    await interaction.response.defer(ephemeral=True)

    try:
        stats = await asyncio.to_thread(get_learning_stats)
    except Exception as exc:
        print(f"/ailearning error: {exc}")
        await interaction.followup.send(
            "Could not load AI learning statistics.",
            ephemeral=True,
        )
        return

    buy_accuracy = stats.get("buy_accuracy")
    bearish_accuracy = stats.get("bearish_accuracy")

    embed = discord.Embed(
        title="Alpha AI Learning",
        description="Shows the trader's stored learning/performance statistics.",
        color=3447003,
    )

    embed.add_field(
        name="Resolved Predictions",
        value=str(stats.get("resolved_predictions", 0)),
        inline=True,
    )
    embed.add_field(
        name="BUY Signals",
        value=f"{stats.get('buy_correct', 0)}/{stats.get('buy_signals', 0)} correct",
        inline=True,
    )
    embed.add_field(
        name="BUY Accuracy",
        value=(
            f"{buy_accuracy * 100:.1f}%"
            if buy_accuracy is not None
            else "Not enough data"
        ),
        inline=True,
    )
    embed.add_field(
        name="BEARISH Signals",
        value=(
            f"{stats.get('bearish_correct', 0)}/"
            f"{stats.get('bearish_signals', 0)} correct"
        ),
        inline=True,
    )
    embed.add_field(
        name="BEARISH Accuracy",
        value=(
            f"{bearish_accuracy * 100:.1f}%"
            if bearish_accuracy is not None
            else "Not enough data"
        ),
        inline=True,
    )

    product_lines = []

    for product in stats.get("products", []):
        accuracy = product.get("directional_accuracy")
        accuracy_text = f"{accuracy * 100:.1f}%" if accuracy is not None else "N/A"

        product_lines.append(
            f"**{product['product']}** | "
            f"{product['resolved']} resolved | "
            f"{accuracy_text}"
        )

    if product_lines:
        embed.add_field(
            name="By Market",
            value="\n".join(product_lines),
            inline=False,
        )

    embed.set_footer(text="OWNER ONLY | AI performance tracking")

    await interaction.followup.send(embed=embed, ephemeral=True)


# =========================================================
# START
# =========================================================

if __name__ == "__main__":
    check_config()

    monitor_thread = threading.Thread(
        target=monitor_main,
        daemon=True,
    )
    monitor_thread.start()

    bot.run(DISCORD_BOT_TOKEN)