import os
import json
import time
import asyncio
import threading
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone

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

CHECK_INTERVAL = int(os.getenv("CHECK_INTERVAL", "60"))
AI_CHECK_INTERVAL = int(os.getenv("AI_CHECK_INTERVAL", "300"))

CRYPTO_WINDOW_MINUTES = int(os.getenv("CRYPTO_WINDOW_MINUTES", "15"))
CRYPTO_MAJOR_SPIKE_PERCENT = float(os.getenv("CRYPTO_MAJOR_SPIKE_PERCENT", "7"))
CRYPTO_SMALL_SPIKE_PERCENT = float(os.getenv("CRYPTO_SMALL_SPIKE_PERCENT", "15"))
CRYPTO_ALERT_COOLDOWN_MINUTES = int(os.getenv("CRYPTO_ALERT_COOLDOWN_MINUTES", "60"))

MAJOR_CRYPTO = {"BTC","ETH","SOL","XRP","ADA","DOGE","AVAX","LINK","LTC","BCH"}

STOCK_SYMBOLS = [
    symbol.strip().upper()
    for symbol in os.getenv(
        "STOCK_SYMBOLS",
        "NVDA,TSLA,AAPL,MSFT,AMZN,META,GOOGL,AMD,PLTR,RKLB,ASTS,LUNR,AVGO,ARM,COIN,MSTR",
    ).split(",")
    if symbol.strip()
]

STOCK_MOVE_PERCENT = float(os.getenv("STOCK_MOVE_PERCENT", "5"))
STOCK_WINDOW_MINUTES = int(os.getenv("STOCK_WINDOW_MINUTES", "30"))
STOCK_ALERT_COOLDOWN_MINUTES = int(os.getenv("STOCK_ALERT_COOLDOWN_MINUTES", "60"))
STOCK_CHECK_INTERVAL = int(os.getenv("STOCK_CHECK_INTERVAL", "300"))

SEC_CHECK_INTERVAL = int(os.getenv("SEC_CHECK_INTERVAL", "300"))
SEC_REQUEST_DELAY = float(os.getenv("SEC_REQUEST_DELAY", "0.25"))
SEC_ALERT_MAX_AGE_DAYS = int(os.getenv("SEC_ALERT_MAX_AGE_DAYS", "3"))

INSIDER_CLUSTER_DAYS = int(os.getenv("INSIDER_CLUSTER_DAYS", "14"))
INSIDER_FEATURE_DAYS = int(os.getenv("INSIDER_FEATURE_DAYS", "30"))
INSIDER_MIN_ALERT_SCORE = int(os.getenv("INSIDER_MIN_ALERT_SCORE", "10"))
INSIDER_MAJOR_BUY_VALUE = float(os.getenv("INSIDER_MAJOR_BUY_VALUE", "1000000"))
INSIDER_HUGE_BUY_VALUE = float(os.getenv("INSIDER_HUGE_BUY_VALUE", "5000000"))
INSIDER_MAJOR_SELL_VALUE = float(os.getenv("INSIDER_MAJOR_SELL_VALUE", "5000000"))

SEC_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
SEC_SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik}.json"

SEC_FORMS_TO_MONITOR = {
    "4","4/A",
    "SC 13D","SC 13D/A","SC 13G","SC 13G/A",
    "SCHEDULE 13D","SCHEDULE 13D/A","SCHEDULE 13G","SCHEDULE 13G/A",
}
OWNERSHIP_FORMS = SEC_FORMS_TO_MONITOR - {"4","4/A"}

COINBASE_PRODUCTS_URL = "https://api.exchange.coinbase.com/products"
COINBASE_WEBSOCKET_URL = "wss://ws-feed.exchange.coinbase.com"
FINNHUB_QUOTE_URL = "https://finnhub.io/api/v1/quote"

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
http.headers.update({"User-Agent": "Alpha-Alerts/16.0"})


def is_bot_owner(interaction: discord.Interaction) -> bool:
    if not OWNER_DISCORD_USER_ID:
        return False
    return str(interaction.user.id) == str(OWNER_DISCORD_USER_ID)


async def reject_non_owner(interaction: discord.Interaction):
    await interaction.response.send_message(
        "You are not authorised to use the AI trader.",
        ephemeral=True,
    )


def check_config():
    if not DATABASE_URL:
        raise RuntimeError("DATABASE_URL is missing.")
    if not DISCORD_BOT_TOKEN:
        raise RuntimeError("DISCORD_BOT_TOKEN is missing.")
    if not OWNER_DISCORD_USER_ID:
        print("WARNING: OWNER_DISCORD_USER_ID missing. Owner-only commands locked.")
    if not FINNHUB_API_KEY:
        print("WARNING: FINNHUB_API_KEY missing. Stock monitoring disabled.")
    if not SEC_USER_AGENT:
        print("WARNING: SEC_USER_AGENT missing. SEC monitoring disabled.")


def get_database():
    return psycopg2.connect(DATABASE_URL)


def create_tables():
    with get_database() as conn:
        with conn.cursor() as cursor:
            cursor.execute("CREATE TABLE IF NOT EXISTS seen_assets (asset_name TEXT PRIMARY KEY);")
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS crypto_price_samples (
                    symbol TEXT NOT NULL,
                    price NUMERIC NOT NULL,
                    sampled_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_crypto_samples_symbol_time
                ON crypto_price_samples (symbol, sampled_at DESC);
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
                    sampled_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_stock_samples_symbol_time
                ON stock_price_samples (symbol, sampled_at DESC);
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
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)
            cursor.execute("""
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
                    UNIQUE (accession_number, insider_name, transaction_index)
                );
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_insider_ticker_date
                ON insider_transactions (ticker, transaction_date DESC, filing_date DESC);
            """)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS ownership_filings (
                    accession_number TEXT PRIMARY KEY,
                    ticker TEXT NOT NULL,
                    company_name TEXT,
                    form_type TEXT,
                    filing_date DATE,
                    sec_url TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)


def post_to_webhooks(webhooks, title, description, fields=None, color=3447003):
    embed = {"title": title, "description": description, "color": color}
    if fields:
        embed["fields"] = fields
    payload = {"embeds": [embed]}
    sent = set()
    for webhook in webhooks:
        if not webhook or webhook in sent:
            continue
        sent.add(webhook)
        try:
            response = http.post(webhook, json=payload, timeout=15)
            response.raise_for_status()
        except Exception as exc:
            print(f"Discord webhook error: {exc}")


def send_coinbase_alert(title, description, fields=None, color=3447003):
    post_to_webhooks([WEBHOOK, WEBHOOK2], title, description, fields, color)


def send_crypto_alert(title, description, fields=None, color=3447003):
    post_to_webhooks([PRIVATE_CRYPTO_WEBHOOK, WEBHOOK2], title, description, fields, color)


def send_stock_alert(title, description, fields=None, color=3447003):
    post_to_webhooks([PRIVATE_STOCK_WEBHOOK, WEBHOOK2], title, description, fields, color)


def send_insider_alert(title, description, fields=None, color=10181046):
    post_to_webhooks([PRIVATE_INSIDER_WEBHOOK], title, description, fields, color)


def get_coinbase_products():
    response = http.get(COINBASE_PRODUCTS_URL, timeout=20)
    response.raise_for_status()
    return response.json()


def get_crypto_products():
    ignored = {"USD","USDC","USDT","EUR","GBP"}
    result = []
    for product in get_coinbase_products():
        if product.get("status") != "online":
            continue
        if product.get("quote_currency") != "USD":
            continue
        product_id = product.get("id")
        base = product.get("base_currency")
        if product_id and base and base not in ignored:
            result.append(product_id)
    return sorted(set(result))


def get_seen_assets():
    with get_database() as conn:
        with conn.cursor() as cursor:
            cursor.execute("SELECT asset_name FROM seen_assets;")
            return {row[0] for row in cursor.fetchall()}


def save_assets(assets):
    if not assets:
        return
    with get_database() as conn:
        with conn.cursor() as cursor:
            execute_values(
                cursor,
                "INSERT INTO seen_assets (asset_name) VALUES %s ON CONFLICT DO NOTHING",
                [(asset,) for asset in assets],
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

    for asset in sorted(current_assets - seen_assets):
        save_assets({asset})
        seen_assets.add(asset)
        markets = sorted(markets_by_asset.get(asset, []))
        market_text = ", ".join(f"`{m}`" for m in markets[:20])
        send_coinbase_alert(
            "COINBASE TRADING LIVE",
            f"**{asset}** has appeared as a new online Coinbase asset.",
            [{"name":"Markets","value":market_text or "Unknown","inline":False}],
            color=5763719,
        )


def websocket_on_open(ws):
    print(f"Coinbase WebSocket connected. Subscribing to {len(coinbase_product_ids)} markets.")
    for start in range(0, len(coinbase_product_ids), 100):
        ws.send(json.dumps({
            "type":"subscribe",
            "product_ids":coinbase_product_ids[start:start+100],
            "channels":["ticker"],
        }))
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
        with crypto_prices_lock:
            crypto_prices[product_id[:-4]] = float(price)
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
        time.sleep(5)


def save_crypto_samples(prices):
    if not prices:
        return
    with get_database() as conn:
        with conn.cursor() as cursor:
            execute_values(
                cursor,
                "INSERT INTO crypto_price_samples (symbol, price) VALUES %s",
                list(prices.items()),
            )


def get_old_crypto_prices():
    with get_database() as conn:
        with conn.cursor() as cursor:
            cursor.execute("""
                SELECT DISTINCT ON (symbol) symbol, price
                FROM crypto_price_samples
                WHERE sampled_at <= NOW() - (%s * INTERVAL '1 minute')
                ORDER BY symbol, sampled_at DESC;
            """, (CRYPTO_WINDOW_MINUTES,))
            return {symbol: float(price) for symbol, price in cursor.fetchall()}


def can_send_crypto_alert(symbol):
    with get_database() as conn:
        with conn.cursor() as cursor:
            cursor.execute("SELECT last_alerted_at FROM crypto_alerts WHERE symbol=%s", (symbol,))
            result = cursor.fetchone()
            if not result:
                return True
            cursor.execute(
                "SELECT NOW() - %s >= (%s * INTERVAL '1 minute')",
                (result[0], CRYPTO_ALERT_COOLDOWN_MINUTES),
            )
            return bool(cursor.fetchone()[0])


def record_crypto_alert(symbol, percent):
    with get_database() as conn:
        with conn.cursor() as cursor:
            cursor.execute("""
                INSERT INTO crypto_alerts(symbol,last_alerted_at,last_alert_percent)
                VALUES(%s,NOW(),%s)
                ON CONFLICT(symbol) DO UPDATE SET
                    last_alerted_at=EXCLUDED.last_alerted_at,
                    last_alert_percent=EXCLUDED.last_alert_percent;
            """, (symbol, percent))


def crypto_threshold(symbol):
    return CRYPTO_MAJOR_SPIKE_PERCENT if symbol in MAJOR_CRYPTO else CRYPTO_SMALL_SPIKE_PERCENT


def check_all_crypto():
    with crypto_prices_lock:
        current_prices = dict(crypto_prices)
    if not current_prices:
        print("Waiting for Coinbase WebSocket prices...")
        return
    old_prices = get_old_crypto_prices()
    save_crypto_samples(current_prices)

    candidates = []
    for symbol, price in current_prices.items():
        old_price = old_prices.get(symbol)
        if not old_price:
            continue
        percent = ((price - old_price) / old_price) * 100
        if abs(percent) >= crypto_threshold(symbol):
            candidates.append((symbol, price, old_price, percent))

    candidates.sort(key=lambda item: abs(item[3]), reverse=True)

    for symbol, price, old_price, percent in candidates[:3]:
        if not can_send_crypto_alert(symbol):
            continue
        positive = percent >= 0
        send_crypto_alert(
            "CRYPTO PRICE SPIKE",
            f"**{symbol}** is **{'UP' if positive else 'DOWN'} {abs(percent):.2f}%** in approximately {CRYPTO_WINDOW_MINUTES} minutes.",
            [
                {"name":"Current Price","value":f"${price:,.8f}","inline":True},
                {"name":"Earlier Price","value":f"${old_price:,.8f}","inline":True},
                {"name":"Move","value":f"{percent:+.2f}%","inline":True},
            ],
            color=5763719 if positive else 15548997,
        )
        record_crypto_alert(symbol, percent)


def save_stock_sample(symbol, price):
    with get_database() as conn:
        with conn.cursor() as cursor:
            cursor.execute("INSERT INTO stock_price_samples(symbol,price) VALUES(%s,%s)", (symbol, price))


def get_old_stock_price(symbol):
    with get_database() as conn:
        with conn.cursor() as cursor:
            cursor.execute("""
                SELECT price FROM stock_price_samples
                WHERE symbol=%s
                AND sampled_at <= NOW() - (%s * INTERVAL '1 minute')
                ORDER BY sampled_at DESC LIMIT 1;
            """, (symbol, STOCK_WINDOW_MINUTES))
            result = cursor.fetchone()
            return float(result[0]) if result else None


def can_send_stock_alert(symbol):
    with get_database() as conn:
        with conn.cursor() as cursor:
            cursor.execute("SELECT last_alerted_at FROM stock_alerts WHERE symbol=%s", (symbol,))
            result = cursor.fetchone()
            if not result:
                return True
            cursor.execute(
                "SELECT NOW() - %s >= (%s * INTERVAL '1 minute')",
                (result[0], STOCK_ALERT_COOLDOWN_MINUTES),
            )
            return bool(cursor.fetchone()[0])


def record_stock_alert(symbol, percent):
    with get_database() as conn:
        with conn.cursor() as cursor:
            cursor.execute("""
                INSERT INTO stock_alerts(symbol,last_alerted_at,last_alert_percent)
                VALUES(%s,NOW(),%s)
                ON CONFLICT(symbol) DO UPDATE SET
                    last_alerted_at=EXCLUDED.last_alerted_at,
                    last_alert_percent=EXCLUDED.last_alert_percent;
            """, (symbol, percent))


def get_stock_quote(symbol):
    if not FINNHUB_API_KEY:
        raise RuntimeError("FINNHUB_API_KEY missing.")
    response = http.get(
        FINNHUB_QUOTE_URL,
        params={"symbol":symbol,"token":FINNHUB_API_KEY},
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
    if abs(percent) < STOCK_MOVE_PERCENT or not can_send_stock_alert(symbol):
        return
    positive = percent >= 0
    send_stock_alert(
        "STOCK MOVE ALERT",
        f"**{symbol}** is **{'UP' if positive else 'DOWN'} {abs(percent):.2f}%** in approximately {STOCK_WINDOW_MINUTES} minutes.",
        [
            {"name":"Current Price","value":f"${current_price:,.2f}","inline":True},
            {"name":"Move","value":f"{percent:+.2f}%","inline":True},
        ],
        color=5763719 if positive else 15548997,
    )
    record_stock_alert(symbol, percent)


def sec_headers():
    return {
        "User-Agent": SEC_USER_AGENT or "Alpha-Alerts contact@example.com",
        "Accept-Encoding":"gzip, deflate",
        "Accept":"*/*",
    }


def get_sec_ticker_map():
    response = http.get(SEC_TICKERS_URL, headers=sec_headers(), timeout=20)
    response.raise_for_status()
    data = response.json()
    result = {}
    for item in data.values():
        ticker = item["ticker"].strip().upper()
        result[ticker] = {
            "cik":str(item["cik_str"]).zfill(10),
            "company":item["title"],
        }
    return result


def sec_database_empty():
    with get_database() as conn:
        with conn.cursor() as cursor:
            cursor.execute("SELECT COUNT(*) FROM seen_sec_filings;")
            return cursor.fetchone()[0] == 0


def ownership_database_empty():
    with get_database() as conn:
        with conn.cursor() as cursor:
            cursor.execute("SELECT COUNT(*) FROM ownership_filings;")
            return cursor.fetchone()[0] == 0


def sec_filing_seen(accession):
    with get_database() as conn:
        with conn.cursor() as cursor:
            cursor.execute("SELECT 1 FROM seen_sec_filings WHERE accession_number=%s", (accession,))
            return cursor.fetchone() is not None


def save_sec_filing(accession, ticker, company, filing_date, form_type):
    with get_database() as conn:
        with conn.cursor() as cursor:
            cursor.execute("""
                INSERT INTO seen_sec_filings(accession_number,ticker,company_name,filing_date,form_type)
                VALUES(%s,%s,%s,%s,%s)
                ON CONFLICT(accession_number) DO NOTHING;
            """, (accession, ticker, company, filing_date, form_type))


def sec_filing_is_fresh(filing_date):
    try:
        filed = datetime.strptime(filing_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        cutoff = datetime.now(timezone.utc) - timedelta(days=SEC_ALERT_MAX_AGE_DAYS)
        return filed >= cutoff
    except Exception:
        return False


def get_recent_sec_filings(ticker, cik, company):
    response = http.get(
        SEC_SUBMISSIONS_URL.format(cik=cik),
        headers=sec_headers(),
        timeout=20,
    )
    response.raise_for_status()
    recent = response.json().get("filings", {}).get("recent", {})

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
        primary_document = primary_documents[index] if index < len(primary_documents) else ""
        if not primary_document:
            continue

        accession_clean = accession.replace("-", "")
        cik_clean = str(int(cik))

        filings.append({
            "ticker":ticker,
            "company":company,
            "accession":accession,
            "filing_date":filing_date,
            "form_type":form_type,
            "url":f"https://www.sec.gov/Archives/edgar/data/{cik_clean}/{accession_clean}/{primary_document}",
        })
    return filings


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
    return {
        "P":"Open-market / private BUY",
        "S":"Open-market / private SELL",
        "A":"Grant / Award",
        "D":"Sale back to issuer",
        "F":"Tax / Exercise Payment",
        "M":"Option Exercise",
        "C":"Conversion",
        "G":"Gift",
        "J":"Other",
        "L":"Small Acquisition",
        "W":"Will / Inheritance",
        "X":"Option Exercise",
        "Z":"Voting Trust",
    }.get(code, f"Other ({code})")


def score_insider_transaction(code, value, role, shares, shares_owned_after):
    score = 0
    value = float(value or 0)
    shares = float(shares or 0)
    shares_owned_after = float(shares_owned_after or 0)
    role_lower = (role or "").lower().replace(".", "").strip()

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
                ratio = shares / previous_shares
                if ratio >= 1:
                    score += 20
                elif ratio >= 0.5:
                    score += 15
                elif ratio >= 0.25:
                    score += 10
                elif ratio >= 0.10:
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

    return max(-100, min(100, int(score)))


def get_form4_xml(filing_url):
    response = http.get(filing_url, headers=sec_headers(), timeout=20)
    response.raise_for_status()
    return response.text


def parse_form4(xml_text):
    root = ET.fromstring(xml_text)
    owner = root.find(".//reportingOwner")
    insider_name = safe_xml_text(owner, ".//rptOwnerName")
    relationship = owner.find(".//reportingOwnerRelationship") if owner is not None else None

    roles = []
    if relationship is not None:
        if safe_xml_text(relationship, "isDirector", "0") == "1":
            roles.append("Director")
        if safe_xml_text(relationship, "isOfficer", "0") == "1":
            title = safe_xml_text(relationship, "officerTitle", "")
            roles.append(title if title else "Officer")
        if safe_xml_text(relationship, "isTenPercentOwner", "0") == "1":
            roles.append("10% Owner")
        if safe_xml_text(relationship, "isOther", "0") == "1":
            other = safe_xml_text(relationship, "otherText", "")
            if other:
                roles.append(other)

    role = ", ".join(roles) if roles else "Insider"
    transactions = []

    for index, transaction in enumerate(root.findall(".//nonDerivativeTransaction"), start=1):
        code = safe_xml_text(transaction, ".//transactionCoding/transactionCode", "?")
        transaction_date = safe_xml_text(transaction, ".//transactionDate/value", "")
        shares_text = safe_xml_text(transaction, ".//transactionAmounts/transactionShares/value", "0")
        price_text = safe_xml_text(transaction, ".//transactionAmounts/transactionPricePerShare/value", "")
        direction = safe_xml_text(transaction, ".//transactionAmounts/transactionAcquiredDisposedCode/value", "")
        owned_after_text = safe_xml_text(transaction, ".//postTransactionAmounts/sharesOwnedFollowingTransaction/value", "0")
        ownership_type = safe_xml_text(transaction, ".//ownershipNature/directOrIndirectOwnership/value", "")
        security = safe_xml_text(transaction, ".//securityTitle/value", "Security")

        try:
            shares = float(shares_text)
        except Exception:
            shares = 0.0
        try:
            price = float(price_text)
        except Exception:
            price = None
        try:
            owned_after = float(owned_after_text)
        except Exception:
            owned_after = 0.0

        value = shares * price if shares > 0 and price is not None else None
        score = score_insider_transaction(code, value, role, shares, owned_after)

        transactions.append({
            "index":index,
            "code":code,
            "action":transaction_name(code),
            "transaction_date":transaction_date,
            "security":security,
            "shares":shares,
            "price":price,
            "value":value,
            "direction":direction,
            "shares_owned_after":owned_after,
            "ownership_type":ownership_type,
            "score":score,
        })

    return {"insider":insider_name,"role":role,"transactions":transactions}


def save_insider_transaction(ticker, company, accession, filing_date, filing_url, insider, role, transaction):
    with get_database() as conn:
        with conn.cursor() as cursor:
            cursor.execute("""
                INSERT INTO insider_transactions(
                    accession_number,transaction_index,ticker,company_name,
                    insider_name,insider_role,filing_date,transaction_date,
                    transaction_code,transaction_type,security_title,shares,
                    price,transaction_value,acquired_disposed,shares_owned_after,
                    ownership_type,insider_score,sec_url
                )
                VALUES(
                    %s,%s,%s,%s,%s,%s,NULLIF(%s,'')::date,NULLIF(%s,'')::date,
                    %s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s
                )
                ON CONFLICT(accession_number,insider_name,transaction_index) DO NOTHING;
            """, (
                accession, int(transaction.get("index",0)), ticker, company,
                insider, role, filing_date, transaction.get("transaction_date",""),
                transaction.get("code"), transaction.get("action"), transaction.get("security"),
                transaction.get("shares"), transaction.get("price"), transaction.get("value"),
                transaction.get("direction"), transaction.get("shares_owned_after"),
                transaction.get("ownership_type"), transaction.get("score",0), filing_url,
            ))


def get_cluster_buy_data(ticker, days=None):
    days = int(days or INSIDER_CLUSTER_DAYS)
    with get_database() as conn:
        with conn.cursor() as cursor:
            cursor.execute("""
                SELECT COUNT(DISTINCT insider_name), COUNT(*), COALESCE(SUM(transaction_value),0)
                FROM insider_transactions
                WHERE ticker=%s AND transaction_code='P'
                AND COALESCE(transaction_date, filing_date) >= CURRENT_DATE - %s;
            """, (ticker, days))
            result = cursor.fetchone()
    return {"insiders":int(result[0] or 0),"transactions":int(result[1] or 0),"value":float(result[2] or 0)}


def calculate_cluster_bonus(ticker):
    insiders = get_cluster_buy_data(ticker)["insiders"]
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
            cursor.execute("""
                SELECT
                    COALESCE(SUM(CASE WHEN transaction_code='P' THEN transaction_value ELSE 0 END),0),
                    COALESCE(SUM(CASE WHEN transaction_code='S' THEN transaction_value ELSE 0 END),0),
                    COUNT(*) FILTER (WHERE transaction_code='P'),
                    COUNT(*) FILTER (WHERE transaction_code='S'),
                    COUNT(DISTINCT insider_name) FILTER (WHERE transaction_code='P'),
                    COUNT(DISTINCT insider_name) FILTER (WHERE transaction_code='S'),
                    COALESCE(SUM(insider_score),0)
                FROM insider_transactions
                WHERE ticker=%s
                AND COALESCE(transaction_date, filing_date) >= CURRENT_DATE - %s;
            """, (ticker, days))
            row = cursor.fetchone()

            cursor.execute("""
                SELECT insider_role FROM insider_transactions
                WHERE ticker=%s AND transaction_code='P'
                AND COALESCE(transaction_date, filing_date) >= CURRENT_DATE - %s
                ORDER BY insider_score DESC, transaction_value DESC NULLS LAST
                LIMIT 10;
            """, (ticker, days))
            buy_roles = [str(item[0] or "").lower() for item in cursor.fetchall()]

            cursor.execute("""
                SELECT COUNT(*) FROM ownership_filings
                WHERE ticker=%s AND filing_date >= CURRENT_DATE - %s;
            """, (ticker, days))
            ownership_count = int(cursor.fetchone()[0] or 0)

    buy_value = float(row[0] or 0)
    sell_value = float(row[1] or 0)
    unique_buyers = int(row[4] or 0)
    score = max(-100.0, min(100.0, float(row[6] or 0) + calculate_cluster_bonus(ticker)))

    return {
        "ticker":ticker,
        "insider_score":score,
        "insider_buy_value":buy_value,
        "insider_sell_value":sell_value,
        "insider_net_value":buy_value-sell_value,
        "insider_buy_transactions":int(row[2] or 0),
        "insider_sell_transactions":int(row[3] or 0),
        "insider_buyers":unique_buyers,
        "insider_sellers":int(row[5] or 0),
        "cluster_buy":int(unique_buyers >= 2),
        "cluster_buyers":unique_buyers,
        "ceo_buy":int(any("ceo" in r or "chief executive" in r for r in buy_roles)),
        "cfo_buy":int(any("cfo" in r or "chief financial" in r for r in buy_roles)),
        "ownership_filings":ownership_count,
    }


def get_recent_insider_transactions(ticker, limit=8):
    with get_database() as conn:
        with conn.cursor() as cursor:
            cursor.execute("""
                SELECT insider_name,insider_role,transaction_code,transaction_type,
                       transaction_date,shares,price,transaction_value,insider_score,sec_url
                FROM insider_transactions
                WHERE ticker=%s
                ORDER BY COALESCE(transaction_date, filing_date) DESC, created_at DESC
                LIMIT %s;
            """, (ticker, int(limit)))
            return cursor.fetchall()


def process_form4_filing(ticker, company, filing, send_alert=True):
    parsed = parse_form4(get_form4_xml(filing["url"]))
    insider = parsed["insider"]
    role = parsed["role"]
    transactions = parsed["transactions"]

    if not transactions:
        return

    for transaction in transactions:
        save_insider_transaction(
            ticker, company, filing["accession"], filing["filing_date"],
            filing["url"], insider, role, transaction,
        )

    if not send_alert:
        return

    important = [tx for tx in transactions if tx["code"] in {"P","S"}]
    display_transactions = important if important else transactions
    has_purchase = any(tx["code"] == "P" for tx in transactions)
    has_sale = any(tx["code"] == "S" for tx in transactions)
    transaction_score = sum(int(tx.get("score",0)) for tx in transactions)
    cluster = get_cluster_buy_data(ticker)
    final_score = max(-100, min(100, transaction_score + (calculate_cluster_bonus(ticker) if has_purchase else 0)))

    if has_purchase and final_score >= 75:
        title, color = "STRONG INSIDER BUY", 5763719
        interpretation = "Strong bullish insider conviction signal."
    elif has_purchase:
        title, color = "INSIDER BUY", 5763719
        interpretation = "Bullish insider activity detected."
    elif has_sale:
        title, color = "INSIDER SALE", 15548997
        interpretation = "Insider selling detected. A sale is not automatically bearish and may be routine."
    else:
        title, color = "INSIDER FORM 4", 10181046
        interpretation = "Equity transaction detected, but it is not classified as a conviction purchase or sale."

    fields = [
        {"name":"Insider","value":insider,"inline":True},
        {"name":"Role","value":role,"inline":True},
        {"name":"Filed","value":filing["filing_date"] or "Unknown","inline":True},
    ]

    for tx in display_transactions[:4]:
        text = f"**{tx['action']}**\nShares: **{tx['shares']:,.0f}**"
        if tx.get("transaction_date"):
            text += f"\nTransaction date: **{tx['transaction_date']}**"
        if tx["price"] is not None:
            text += f"\nPrice: **${tx['price']:,.4f}**"
        if tx["value"] is not None:
            text += f"\nApprox value: **{format_money(tx['value'])}**"
        if tx.get("shares_owned_after"):
            text += f"\nShares owned after: **{tx['shares_owned_after']:,.0f}**"
        text += f"\nSignal score: **{int(tx.get('score',0)):+d}**"
        fields.append({"name":tx.get("security","Transaction"),"value":text,"inline":False})

    if has_purchase and cluster["insiders"] >= 2:
        fields.append({
            "name":"Cluster Buying Detected",
            "value":f"**{cluster['insiders']}** different insiders bought within the last {INSIDER_CLUSTER_DAYS} days.\nRecorded buying: **{format_money(cluster['value'])}**",
            "inline":False,
        })

    fields.append({
        "name":"Insider Intelligence Score",
        "value":f"**{final_score:+d} / 100**\n{interpretation}",
        "inline":False,
    })
    fields.append({"name":"Official SEC Filing","value":filing["url"],"inline":False})

    if not important and abs(final_score) < INSIDER_MIN_ALERT_SCORE:
        print(f"LOW SIGNAL FORM 4: {ticker} | {insider} | {final_score:+d}")
        return

    send_insider_alert(
        title,
        f"**{company} ({ticker})**\n\n{interpretation}",
        fields,
        color=color,
    )


def save_ownership_filing(filing):
    with get_database() as conn:
        with conn.cursor() as cursor:
            cursor.execute("""
                INSERT INTO ownership_filings(accession_number,ticker,company_name,form_type,filing_date,sec_url)
                VALUES(%s,%s,%s,%s,NULLIF(%s,'')::date,%s)
                ON CONFLICT(accession_number) DO NOTHING;
            """, (
                filing["accession"], filing["ticker"], filing["company"],
                filing["form_type"], filing["filing_date"], filing["url"],
            ))


def send_ownership_alert(filing):
    is_13d = "13D" in str(filing["form_type"]).upper()
    title = "MAJOR OWNERSHIP FILING" if is_13d else "LARGE SHAREHOLDER FILING"
    explanation = (
        "A Schedule 13D filing or amendment was detected. This can indicate a significant shareholder position."
        if is_13d else
        "A Schedule 13G filing or amendment was detected. This reports significant beneficial ownership but is not, by itself, a buy signal."
    )
    send_insider_alert(
        title,
        f"**{filing['company']} ({filing['ticker']})**\n\n{explanation}",
        [
            {"name":"Form","value":f"**{filing['form_type']}**","inline":True},
            {"name":"Filed","value":filing["filing_date"] or "Unknown","inline":True},
            {"name":"Official SEC Filing","value":filing["url"],"inline":False},
        ],
        color=16776960 if is_13d else 3447003,
    )


def check_sec_insider_filings(ticker_map, baseline_all=False, baseline_ownership=False):
    for ticker in STOCK_SYMBOLS:
        info = ticker_map.get(ticker)
        if not info:
            print(f"SEC ticker not found: {ticker}")
            continue

        try:
            filings = get_recent_sec_filings(ticker, info["cik"], info["company"])

            for filing in filings[:25]:
                accession = filing["accession"]
                form_type = filing["form_type"]
                filing_date = filing["filing_date"]

                if sec_filing_seen(accession):
                    continue

                save_sec_filing(accession, ticker, info["company"], filing_date, form_type)

                if form_type in {"4","4/A"}:
                    if baseline_all:
                        print(f"SEC BASELINE REMEMBERED: {ticker} | {form_type} | {filing_date} | {accession}")
                        continue

                    if not sec_filing_is_fresh(filing_date):
                        print(f"OLD SEC FILING REMEMBERED - NO ALERT: {ticker} | {form_type} | {filing_date} | {accession}")
                        continue

                    try:
                        process_form4_filing(ticker, info["company"], filing, send_alert=True)
                        print(f"NEW FORM {form_type}: {ticker} | {filing_date} | {accession}")
                    except Exception as parse_error:
                        print(f"NEW FORM 4 PARSE ERROR: {ticker} | {accession} | {parse_error}")
                    continue

                if form_type in OWNERSHIP_FORMS:
                    try:
                        save_ownership_filing(filing)
                    except Exception as exc:
                        print(f"Ownership DB error: {ticker} | {accession} | {exc}")
                        continue

                    if baseline_all or baseline_ownership:
                        print(f"OWNERSHIP BASELINE REMEMBERED: {ticker} | {form_type} | {filing_date}")
                        continue

                    if not sec_filing_is_fresh(filing_date):
                        print(f"OLD OWNERSHIP FILING REMEMBERED - NO ALERT: {ticker} | {form_type} | {filing_date}")
                        continue

                    send_ownership_alert(filing)
                    print(f"NEW {form_type}: {ticker} | {filing_date} | {accession}")

            time.sleep(SEC_REQUEST_DELAY)

        except Exception as exc:
            print(f"SEC {ticker} error: {exc}")


def clean_old_samples():
    with get_database() as conn:
        with conn.cursor() as cursor:
            cursor.execute("DELETE FROM crypto_price_samples WHERE sampled_at < NOW() - INTERVAL '24 hours';")
            cursor.execute("DELETE FROM stock_price_samples WHERE sampled_at < NOW() - INTERVAL '24 hours';")


intents = discord.Intents.default()
bot = commands.Bot(command_prefix="!", intents=intents)


async def send_owner_dm(title, message):
    try:
        if not OWNER_DISCORD_USER_ID:
            return
        user = await bot.fetch_user(int(OWNER_DISCORD_USER_ID))
        embed = discord.Embed(title=title, description=message, color=3447003)
        embed.set_footer(text="Alpha AI Trader | Private owner notification")
        await user.send(embed=embed)
    except Exception as exc:
        print(f"Owner DM error: {exc}")


def queue_owner_dm(title, message):
    if discord_event_loop is None:
        print("Owner DM skipped: Discord event loop not ready.")
        return
    asyncio.run_coroutine_threadsafe(send_owner_dm(title, message), discord_event_loop)


def run_ai_trader_cycle():
    global ai_last_result
    try:
        result = run_ai_cycle()
        with ai_result_lock:
            ai_last_result = result

        opened_position = result.get("opened_position")
        if opened_position:
            queue_owner_dm(
                "AI PAPER TRADE OPENED",
                f"Market: **{opened_position['product']}**\n"
                f"Entry: **${opened_position['entry_price']:,.6f}**\n"
                f"Size: **GBP {opened_position['value']:.2f}**\n"
                f"AI upside probability: **{opened_position.get('probability_up',0)*100:.1f}%**\n"
                f"Stop: **${opened_position['stop_loss']:,.6f}**\n"
                f"Target: **${opened_position['take_profit']:,.6f}**",
            )

        closed_trade = result.get("closed_trade")
        if closed_trade:
            queue_owner_dm(
                "AI PAPER TRADE CLOSED",
                f"Market: **{closed_trade['product']}**\n"
                f"Entry: **${closed_trade['entry_price']:,.6f}**\n"
                f"Exit: **${closed_trade['exit_price']:,.6f}**\n"
                f"P&L: **GBP {closed_trade['pnl']:+.2f}**\n"
                f"Reason: **{closed_trade['reason']}**",
            )
    except Exception as exc:
        print(f"AI trader error: {exc}")


def monitor_main():
    global monitor_started, coinbase_product_ids

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

            check_sec_insider_filings(
                sec_ticker_map,
                baseline_all=first_sec_run,
                baseline_ownership=first_ownership_run,
            )

            if first_sec_run:
                print("SEC filing baseline created. Existing filings were remembered only.")
            if first_ownership_run:
                print("SEC 13D/13G ownership baseline created.")

            print("Historical Form 4 backfill: OFF")
            print(f"SEC insider monitor loaded for {len(STOCK_SYMBOLS)} watchlist stocks.")
            print(f"SEC alerts limited to filings no more than {SEC_ALERT_MAX_AGE_DAYS} days old.")
        except Exception as exc:
            print(f"SEC setup error: {exc}")

    print("================================")
    print("ALPHA ALERTS ONLINE")
    print(f"Coinbase assets remembered: {len(seen_assets)}")
    print(f"Coinbase USD crypto markets: {len(coinbase_product_ids)}")
    print(f"Stocks watched: {len(STOCK_SYMBOLS)}")
    print(f"SEC insider monitoring: {'ON' if sec_ticker_map else 'OFF'}")
    print("================================")

    threading.Thread(target=run_coinbase_websocket, daemon=True).start()

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
            for symbol in STOCK_SYMBOLS:
                try:
                    check_stock_symbol(symbol)
                except Exception as exc:
                    print(f"Stock {symbol} error: {exc}")
                time.sleep(1)
            last_stock_check = time.time()

        if SEC_USER_AGENT and sec_ticker_map and now - last_sec_check >= SEC_CHECK_INTERVAL:
            try:
                print("Checking SEC insider / ownership filings...")
                check_sec_insider_filings(sec_ticker_map, False, False)
            except Exception as exc:
                print(f"SEC insider monitor error: {exc}")
            last_sec_check = time.time()

        cleanup_counter += 1
        if cleanup_counter >= 60:
            try:
                clean_old_samples()
            except Exception as exc:
                print(f"Cleanup error: {exc}")
            cleanup_counter = 0

        time.sleep(CHECK_INTERVAL)


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


@bot.tree.command(name="ping", description="Check whether Alpha Alerts is online")
async def ping(interaction: discord.Interaction):
    await interaction.response.send_message(f"Alpha Alerts online - {round(bot.latency * 1000)} ms")


@bot.tree.command(name="status", description="Show Alpha Alerts status")
async def status(interaction: discord.Interaction):
    with crypto_prices_lock:
        crypto_count = len(crypto_prices)

    embed = discord.Embed(title="Alpha Alerts Status", color=5763719)
    embed.add_field(name="Live Crypto Markets", value=str(crypto_count), inline=True)
    embed.add_field(name="Stock Watchlist", value=str(len(STOCK_SYMBOLS)), inline=True)
    embed.add_field(name="SEC Insider", value="Active" if SEC_USER_AGENT and PRIVATE_INSIDER_WEBHOOK else "Disabled", inline=True)
    embed.add_field(name="SEC Backfill", value="OFF", inline=True)
    embed.add_field(name="SEC Alert Max Age", value=f"{SEC_ALERT_MAX_AGE_DAYS} days", inline=True)
    await interaction.response.send_message(embed=embed)


@bot.tree.command(name="crypto", description="Get a live crypto price")
@app_commands.describe(symbol="Example: BTC, ETH, SOL")
async def crypto(interaction: discord.Interaction, symbol: str):
    symbol = symbol.strip().upper()
    with crypto_prices_lock:
        price = crypto_prices.get(symbol)
    if price is None:
        await interaction.response.send_message(f"No live Coinbase USD price found for `{symbol}`.", ephemeral=True)
        return
    await interaction.response.send_message(f"**{symbol}**: ${price:,.8f}")


@bot.tree.command(name="stock", description="Get a stock quote")
@app_commands.describe(ticker="Example: NVDA, TSLA, AAPL")
async def stock(interaction: discord.Interaction, ticker: str):
    await interaction.response.defer()
    ticker = ticker.strip().upper()
    try:
        data = await asyncio.to_thread(get_stock_quote, ticker)
        current = float(data.get("c",0) or 0)
        percent = float(data.get("dp",0) or 0)
        await interaction.followup.send(f"**{ticker}**: ${current:,.2f} ({percent:+.2f}% today)")
    except Exception:
        await interaction.followup.send(f"Couldn't get a valid quote for `{ticker}`.")


@bot.tree.command(name="insider", description="Show recent SEC insider intelligence for a stock")
@app_commands.describe(ticker="Example: NVDA, TSLA, PLTR")
async def insider(interaction: discord.Interaction, ticker: str):
    await interaction.response.defer()
    ticker = ticker.strip().upper()
    try:
        features = await asyncio.to_thread(get_stock_insider_features, ticker)
        score = float(features["insider_score"])
        embed = discord.Embed(
            title=f"{ticker} Insider Intelligence",
            description=f"**Score: {score:+.0f}/100**",
            color=3447003,
        )
        embed.add_field(name="Buys", value=f"{features['insider_buy_transactions']} | {format_money(features['insider_buy_value'])}", inline=True)
        embed.add_field(name="Sales", value=f"{features['insider_sell_transactions']} | {format_money(features['insider_sell_value'])}", inline=True)
        embed.add_field(name="Net Flow", value=format_money(features["insider_net_value"], signed=True), inline=True)
        embed.add_field(name="Cluster Buy", value="YES" if features["cluster_buy"] else "NO", inline=True)
        embed.add_field(name="CEO Buy", value="YES" if features["ceo_buy"] else "NO", inline=True)
        embed.add_field(name="CFO Buy", value="YES" if features["cfo_buy"] else "NO", inline=True)
        await interaction.followup.send(embed=embed)
    except Exception as exc:
        print(f"/insider error: {exc}")
        await interaction.followup.send(f"Couldn't load insider information for `{ticker}`.")


@bot.tree.command(name="watchlist", description="Show the Alpha Alerts watchlist")
async def watchlist(interaction: discord.Interaction):
    await interaction.response.send_message(
        "**Stocks:** " + ", ".join(f"`{s}`" for s in STOCK_SYMBOLS) +
        "\n**Crypto:** All online Coinbase USD markets."
    )


@bot.tree.command(name="alerts", description="Show current alert settings")
async def alerts(interaction: discord.Interaction):
    await interaction.response.send_message(
        f"SEC check: every {SEC_CHECK_INTERVAL // 60}m\n"
        f"SEC backfill: OFF\n"
        f"SEC alert max age: {SEC_ALERT_MAX_AGE_DAYS} days"
    )



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
        await interaction.followup.send(
            "Still collecting price history."
        )
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
        await interaction.followup.send(
            "Still collecting price history."
        )
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
        await interaction.followup.send(
            "Still collecting price history."
        )
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

    decision = result.get("decision", "HOLD")
    confidence = float(result.get("confidence", 0)) * 100

    embed = discord.Embed(
        title="Alpha AI Trader",
        description=(
            f"**{result.get('product', 'Unknown')}**\n"
            f"${float(result.get('price', 0)):,.2f}"
        ),
        color=3447003,
    )

    embed.add_field(
        name="Decision",
        value=f"**{decision}**",
        inline=True,
    )

    embed.add_field(
        name="Confidence",
        value=f"{confidence:.1f}%",
        inline=True,
    )

    embed.add_field(
        name="Paper Portfolio",
        value=f"GBP {float(result.get('portfolio_value', 0)):,.2f}",
        inline=True,
    )

    embed.add_field(
        name="Cash",
        value=f"GBP {float(result.get('cash', 0)):,.2f}",
        inline=True,
    )

    embed.add_field(
        name="Realised P&L",
        value=f"GBP {float(result.get('realized_pnl', 0)):+,.2f}",
        inline=True,
    )

    embed.add_field(
        name="Record",
        value=(
            f"{int(result.get('wins', 0))}W / "
            f"{int(result.get('losses', 0))}L"
        ),
        inline=True,
    )

    position = result.get("position")

    if position:
        entry = float(position.get("entry_price", 0))
        current = float(result.get("price", 0))

        unrealized = (
            ((current / entry) - 1) * 100
            if entry > 0
            else 0
        )

        embed.add_field(
            name="Open Paper Trade",
            value=(
                f"Entry: **${entry:,.2f}**\n"
                f"Current: **${current:,.2f}**\n"
                f"Move: **{unrealized:+.2f}%**\n"
                f"Size: **GBP {float(position.get('value', 0)):.2f}**\n"
                f"Stop: **${float(position.get('stop_loss', 0)):,.2f}**\n"
                f"Target: **${float(position.get('take_profit', 0)):,.2f}**"
            ),
            inline=False,
        )
    else:
        embed.add_field(
            name="Position",
            value="No open paper trade.",
            inline=False,
        )

    embed.set_footer(
        text="OWNER ONLY | PAPER TRADING | No real funds"
    )

    await interaction.response.send_message(
        embed=embed,
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
        value=(
            position.get("product", "Unknown")
            if position
            else "None"
        ),
        inline=True,
    )

    embed.set_footer(
        text=(
            "OWNER ONLY | PAPER TRADING | "
            "Model estimates are not guarantees"
        )
    )

    await interaction.response.send_message(
        embed=embed,
        ephemeral=True,
    )


def format_backtest_dm(result, days):
    return (
        f"Strategy: **{result.get('strategy','AI')}**\n"
        f"Period: **{result.get('days',days)} days**\n"
        f"Trades: **{result.get('trades',0)}**\n"
        f"Win rate: **{float(result.get('win_rate',0))*100:.1f}%**\n"
        f"Net P&L: **GBP {float(result.get('pnl',0)):+.2f}**"
    )


def format_stock_backtest_dm(result, days):
    return (
        f"Strategy: **{result.get('strategy','STOCK_AI')}**\n"
        f"Unseen test: **{result.get('days',days)} days**\n"
        f"Trades: **{result.get('trades',0)}**\n"
        f"Win rate: **{float(result.get('win_rate',0))*100:.1f}%**\n"
        f"Net P&L: **GBP {float(result.get('pnl',0)):+.2f}**"
    )


def run_backtest_background(days):
    global backtest_running
    try:
        result = run_backtest(days)
        queue_owner_dm("AI BACKTEST FINISHED", format_backtest_dm(result, days))
    finally:
        with backtest_lock:
            backtest_running = False


def run_stock_backtest_background(days):
    global stock_backtest_running
    try:
        result = run_stock_backtest(days)
        queue_owner_dm("STOCK AI BACKTEST FINISHED", format_stock_backtest_dm(result, days))
    finally:
        with stock_backtest_lock:
            stock_backtest_running = False


@bot.tree.command(name="aibacktest", description="Start a private crypto AI backtest and DM the result")
@app_commands.describe(days="Number of days to test, from 3 to 30")
async def aibacktest(interaction: discord.Interaction, days: int = 7):
    global backtest_running
    if not is_bot_owner(interaction):
        await reject_non_owner(interaction)
        return
    days = max(3, min(int(days), 30))
    with backtest_lock:
        if backtest_running:
            await interaction.response.send_message("A crypto backtest is already running.", ephemeral=True)
            return
        backtest_running = True
    threading.Thread(target=run_backtest_background, args=(days,), daemon=True).start()
    await interaction.response.send_message(f"Crypto backtest started for **{days} days**.", ephemeral=True)


@bot.tree.command(name="stockbacktest", description="Start a private AI stock backtest and DM the result")
@app_commands.describe(days="Number of unseen calendar days to test, from 5 to 180")
async def stockbacktest(interaction: discord.Interaction, days: int = 30):
    global stock_backtest_running
    if not is_bot_owner(interaction):
        await reject_non_owner(interaction)
        return
    days = max(5, min(int(days), 180))
    with stock_backtest_lock:
        if stock_backtest_running:
            await interaction.response.send_message("A stock backtest is already running.", ephemeral=True)
            return
        stock_backtest_running = True
    threading.Thread(target=run_stock_backtest_background, args=(days,), daemon=True).start()
    await interaction.response.send_message(f"Stock backtest started for **{days} days**.", ephemeral=True)


@bot.tree.command(name="ailearning", description="Show how your AI is learning from old predictions")
async def ailearning(interaction: discord.Interaction):
    if not is_bot_owner(interaction):
        await reject_non_owner(interaction)
        return
    await interaction.response.defer(ephemeral=True)
    stats = await asyncio.to_thread(get_learning_stats)
    await interaction.followup.send(
        f"Resolved predictions: **{stats.get('resolved_predictions',0)}**",
        ephemeral=True,
    )


if __name__ == "__main__":
    check_config()
    threading.Thread(target=monitor_main, daemon=True).start()
    bot.run(DISCORD_BOT_TOKEN)
