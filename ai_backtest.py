import json
import os
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd
import requests

from coinbase.rest import RESTClient
from sklearn.ensemble import (
    RandomForestClassifier,
    RandomForestRegressor,
)


# =========================================================
# STRATEGY V6
# =========================================================

STRATEGY_NAME = "V6"

GRANULARITY = "FIVE_MINUTE"
CANDLE_SECONDS = 300
PAGE_SIZE = 300


# =========================================================
# COINBASE MARKET UNIVERSE
# =========================================================

COINBASE_PRODUCTS_URL = (
    "https://api.exchange.coinbase.com/products"
)

IGNORED_BASE_ASSETS = {
    "USD",
    "USDC",
    "USDT",
    "EUR",
    "GBP",
    "DAI",
    "PYUSD",
}


# =========================================================
# CORE TRADING SETTINGS
# =========================================================

# V5 = 0.60
# V6 deliberately demands much stronger AI confidence.
BUY_THRESHOLD = float(
    os.getenv("AI_BUY_THRESHOLD", "0.68")
)

STOP_LOSS_PCT = float(
    os.getenv("AI_STOP_LOSS_PCT", "0.015")
)

TAKE_PROFIT_PCT = float(
    os.getenv("AI_TAKE_PROFIT_PCT", "0.040")
)

TRADE_SIZE = float(
    os.getenv("AI_TRADE_SIZE", "25.0")
)

# Keep this configurable.
#
# 0.006 = 0.60% per side.
ESTIMATED_FEE_PER_SIDE = float(
    os.getenv(
        "AI_BACKTEST_FEE_PER_SIDE",
        "0.006",
    )
)


# =========================================================
# TRADE HORIZON
# =========================================================

# 72 x 5 minutes = 6 hours.
TARGET_HORIZON_CANDLES = int(
    os.getenv(
        "AI_TARGET_HORIZON_CANDLES",
        "72",
    )
)

MAX_HOLD_CANDLES = int(
    os.getenv(
        "AI_MAX_HOLD_CANDLES",
        "72",
    )
)

# V5 = 6 candles / 30 minutes.
# V6 uses 12 candles / 60 minutes.
COOLDOWN_CANDLES = int(
    os.getenv(
        "AI_COOLDOWN_CANDLES",
        "12",
    )
)


# =========================================================
# TRAINING SETTINGS
# =========================================================

# V5 = 10 days.
# V6 gets more historical examples.
TRAINING_LOOKBACK_DAYS = int(
    os.getenv(
        "AI_TRAINING_LOOKBACK_DAYS",
        "14",
    )
)

MIN_TRAINING_ROWS = int(
    os.getenv(
        "AI_MIN_TRAINING_ROWS",
        "500",
    )
)

MIN_POSITIVE_TRAINING_EXAMPLES = int(
    os.getenv(
        "AI_MIN_POSITIVE_TRAINING_EXAMPLES",
        "8",
    )
)


# =========================================================
# V6 LIQUIDITY FILTERS
# =========================================================

# Average five-minute dollar volume.
#
# V5 = 15,000.
# V6 is deliberately much more restrictive.
MIN_DOLLAR_VOLUME_5M = float(
    os.getenv(
        "AI_MIN_DOLLAR_VOLUME_5M",
        "50000",
    )
)

# Longer-term market liquidity check.
MIN_DOLLAR_VOLUME_24H = float(
    os.getenv(
        "AI_MIN_DOLLAR_VOLUME_24H",
        "3000000",
    )
)

# Current volume relative to recent average.
#
# V5 = 0.80.
MIN_VOLUME_RATIO = float(
    os.getenv(
        "AI_MIN_VOLUME_RATIO",
        "1.00",
    )
)


# =========================================================
# VOLATILITY FILTERS
# =========================================================

MIN_ATR_PCT = float(
    os.getenv(
        "AI_MIN_ATR_PCT",
        "0.0020",
    )
)

MAX_ATR_PCT = float(
    os.getenv(
        "AI_MAX_ATR_PCT",
        "0.050",
    )
)


# =========================================================
# MOMENTUM FILTERS
# =========================================================

MIN_RSI = float(
    os.getenv(
        "AI_MIN_RSI",
        "48",
    )
)

MAX_RSI = float(
    os.getenv(
        "AI_MAX_RSI",
        "72",
    )
)

MIN_CLOSE_POSITION = float(
    os.getenv(
        "AI_MIN_CLOSE_POSITION",
        "0.55",
    )
)

MIN_RETURN_12 = float(
    os.getenv(
        "AI_MIN_RETURN_12",
        "-0.005",
    )
)


# =========================================================
# EXPECTED EDGE FILTER
# =========================================================

# The regression model predicts the NET percentage return
# after simulated entry + exit fees.
#
# Example:
# 0.006 = require at least +0.60% expected NET return.
MIN_PREDICTED_NET_RETURN = float(
    os.getenv(
        "AI_MIN_PREDICTED_NET_RETURN",
        "0.006",
    )
)

# Optional safety requirement:
# predicted gross move needs to comfortably exceed
# expected round-trip fees.
FEE_EDGE_MULTIPLIER = float(
    os.getenv(
        "AI_FEE_EDGE_MULTIPLIER",
        "1.50",
    )
)


# =========================================================
# BACKTEST / API SETTINGS
# =========================================================

CACHE_DIR = os.getenv(
    "AI_BACKTEST_CACHE_DIR",
    "backtest_cache_v6",
)

REQUEST_DELAY_SECONDS = float(
    os.getenv(
        "AI_BACKTEST_REQUEST_DELAY",
        "0.35",
    )
)

MAX_RETRIES = int(
    os.getenv(
        "AI_BACKTEST_MAX_RETRIES",
        "6",
    )
)

BACKOFF_START_SECONDS = float(
    os.getenv(
        "AI_BACKTEST_BACKOFF_START",
        "2",
    )
)


# =========================================================
# CLIENTS
# =========================================================

client = RESTClient()

http = requests.Session()

http.headers.update(
    {
        "User-Agent":
            "Alpha-Alerts-Backtest-V6/1.0"
    }
)

os.makedirs(
    CACHE_DIR,
    exist_ok=True,
)


# =========================================================
# AI FEATURES
# =========================================================

FEATURE_COLUMNS = [
    "return_1",
    "return_3",
    "return_6",
    "return_12",
    "return_24",

    "ma_ratio_5_20",
    "ma_ratio_20_50",

    "ema_ratio_9_21",
    "ema_ratio_21_50",

    "price_vs_ema21",
    "price_vs_ema50",

    "volatility_12",
    "volatility_24",

    "volume_ratio",
    "volume_zscore",

    "rsi",
    "atr_pct",

    "range_pct",
    "body_pct",
    "close_position",
]


# =========================================================
# MARKET DISCOVERY
# =========================================================

def get_all_coinbase_usd_markets():

    response = http.get(
        COINBASE_PRODUCTS_URL,
        timeout=30,
    )

    response.raise_for_status()

    markets = []

    for product in response.json():

        if product.get("status") != "online":
            continue

        if product.get("quote_currency") != "USD":
            continue

        product_id = product.get("id")
        base = product.get("base_currency")

        if not product_id or not base:
            continue

        if base in IGNORED_BASE_ASSETS:
            continue

        if not product_id.endswith("-USD"):
            continue

        markets.append(product_id)

    return sorted(set(markets))


# =========================================================
# CACHE
# =========================================================

def cache_path(
    product_id,
    requested_days,
):

    safe = (
        product_id
        .replace("/", "_")
        .replace("-", "_")
    )

    total_days = (
        requested_days
        + TRAINING_LOOKBACK_DAYS
    )

    return os.path.join(
        CACHE_DIR,
        f"{safe}_{total_days}d_v6.json",
    )


def load_cache(
    product_id,
    requested_days,
):

    path = cache_path(
        product_id,
        requested_days,
    )

    if not os.path.exists(path):
        return None

    try:

        age = (
            time.time()
            - os.path.getmtime(path)
        )

        if age > 6 * 60 * 60:
            return None

        with open(
            path,
            "r",
            encoding="utf-8",
        ) as f:

            rows = json.load(f)

        if not rows:
            return None

        return pd.DataFrame(rows)

    except Exception:
        return None


def save_cache(
    product_id,
    requested_days,
    df,
):

    path = cache_path(
        product_id,
        requested_days,
    )

    with open(
        path,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            df.to_dict(
                orient="records"
            ),
            f,
        )


# =========================================================
# COINBASE CANDLE DOWNLOAD
# =========================================================

def get_candle_page(
    product_id,
    start,
    end,
):

    delay = BACKOFF_START_SECONDS
    last_error = None

    for attempt in range(MAX_RETRIES):

        try:

            response = (
                client.get_public_candles(
                    product_id=product_id,
                    start=str(int(start)),
                    end=str(int(end)),
                    granularity=GRANULARITY,
                    limit=PAGE_SIZE,
                )
            )

            time.sleep(
                REQUEST_DELAY_SECONDS
            )

            return response

        except Exception as exc:

            last_error = exc
            text = str(exc)

            if (
                "429" in text
                or
                "Too Many Requests" in text
            ):

                print(
                    f"RATE LIMIT "
                    f"{product_id}: "
                    f"waiting "
                    f"{delay:.0f}s "
                    f"(attempt "
                    f"{attempt + 1}/"
                    f"{MAX_RETRIES})"
                )

                time.sleep(delay)

                delay = min(
                    delay * 2,
                    60,
                )

                continue

            raise

    raise RuntimeError(
        f"Coinbase rate limit persisted "
        f"for {product_id}: "
        f"{last_error}"
    )


def get_historical_candles(
    product_id,
    days,
):

    cached = load_cache(
        product_id,
        days,
    )

    if cached is not None:

        print(
            f"CACHE HIT: "
            f"{product_id}"
        )

        return cached

    end_time = int(time.time())

    total_days = (
        days
        + TRAINING_LOOKBACK_DAYS
    )

    start_time = (
        end_time
        - total_days
        * 24
        * 60
        * 60
    )

    rows = []

    cursor = start_time

    while cursor < end_time:

        page_end = min(
            cursor
            + PAGE_SIZE
            * CANDLE_SECONDS,
            end_time,
        )

        response = get_candle_page(
            product_id,
            cursor,
            page_end,
        )

        for candle in response.candles:

            rows.append(
                {
                    "time":
                        int(candle.start),

                    "open":
                        float(candle.open),

                    "high":
                        float(candle.high),

                    "low":
                        float(candle.low),

                    "close":
                        float(candle.close),

                    "volume":
                        float(candle.volume),
                }
            )

        cursor = (
            page_end
            + CANDLE_SECONDS
        )

    if not rows:

        raise RuntimeError(
            f"No historical candles "
            f"returned for "
            f"{product_id}"
        )

    df = (
        pd.DataFrame(rows)
        .sort_values("time")
        .drop_duplicates(
            subset=["time"]
        )
        .reset_index(drop=True)
    )

    save_cache(
        product_id,
        days,
        df,
    )

    return df


# =========================================================
# TECHNICAL INDICATORS
# =========================================================

def calculate_rsi(
    series,
    period=14,
):

    delta = series.diff()

    gain = delta.clip(
        lower=0
    )

    loss = -delta.clip(
        upper=0
    )

    avg_gain = gain.ewm(
        alpha=1 / period,
        adjust=False,
    ).mean()

    avg_loss = loss.ewm(
        alpha=1 / period,
        adjust=False,
    ).mean()

    rs = (
        avg_gain
        /
        avg_loss.replace(
            0,
            np.nan,
        )
    )

    return (
        100
        - 100
        / (1 + rs)
    ).fillna(50)


def calculate_atr(
    data,
    period=14,
):

    previous_close = (
        data["close"]
        .shift(1)
    )

    true_range = pd.concat(
        [
            (
                data["high"]
                - data["low"]
            ),

            (
                data["high"]
                - previous_close
            ).abs(),

            (
                data["low"]
                - previous_close
            ).abs(),
        ],
        axis=1,
    ).max(axis=1)

    return true_range.ewm(
        alpha=1 / period,
        adjust=False,
    ).mean()


# =========================================================
# FEATURE ENGINEERING
# =========================================================

def build_feature_frame(df):

    data = df.copy()

    for n in (
        1,
        3,
        6,
        12,
        24,
    ):

        data[
            f"return_{n}"
        ] = (
            data["close"]
            .pct_change(n)
        )

    data["ma_5"] = (
        data["close"]
        .rolling(5)
        .mean()
    )

    data["ma_20"] = (
        data["close"]
        .rolling(20)
        .mean()
    )

    data["ma_50"] = (
        data["close"]
        .rolling(50)
        .mean()
    )

    data["ma_ratio_5_20"] = (
        data["ma_5"]
        /
        data["ma_20"]
    )

    data["ma_ratio_20_50"] = (
        data["ma_20"]
        /
        data["ma_50"]
    )

    data["ema_9"] = (
        data["close"]
        .ewm(
            span=9,
            adjust=False,
        )
        .mean()
    )

    data["ema_21"] = (
        data["close"]
        .ewm(
            span=21,
            adjust=False,
        )
        .mean()
    )

    data["ema_50"] = (
        data["close"]
        .ewm(
            span=50,
            adjust=False,
        )
        .mean()
    )

    data["ema_ratio_9_21"] = (
        data["ema_9"]
        /
        data["ema_21"]
    )

    data["ema_ratio_21_50"] = (
        data["ema_21"]
        /
        data["ema_50"]
    )

    data["price_vs_ema21"] = (
        data["close"]
        /
        data["ema_21"]
    )

    data["price_vs_ema50"] = (
        data["close"]
        /
        data["ema_50"]
    )

    data["volatility_12"] = (
        data["return_1"]
        .rolling(12)
        .std()
    )

    data["volatility_24"] = (
        data["return_1"]
        .rolling(24)
        .std()
    )

    data["volume_ma_20"] = (
        data["volume"]
        .rolling(20)
        .mean()
    )

    data["volume_std_20"] = (
        data["volume"]
        .rolling(20)
        .std()
    )

    data["volume_ratio"] = (
        data["volume"]
        /
        data["volume_ma_20"]
        .replace(
            0,
            np.nan,
        )
    )

    data["volume_zscore"] = (
        (
            data["volume"]
            -
            data["volume_ma_20"]
        )
        /
        data["volume_std_20"]
        .replace(
            0,
            np.nan,
        )
    )

    data["rsi"] = (
        calculate_rsi(
            data["close"]
        )
    )

    data["atr"] = (
        calculate_atr(data)
    )

    data["atr_pct"] = (
        data["atr"]
        /
        data["close"]
        .replace(
            0,
            np.nan,
        )
    )

    candle_range = (
        data["high"]
        - data["low"]
    )

    data["range_pct"] = (
        candle_range
        /
        data["close"]
        .replace(
            0,
            np.nan,
        )
    )

    data["body_pct"] = (
        (
            data["close"]
            - data["open"]
        )
        /
        data["open"]
        .replace(
            0,
            np.nan,
        )
    )

    data["close_position"] = (
        (
            data["close"]
            - data["low"]
        )
        /
        candle_range.replace(
            0,
            np.nan,
        )
    )

    data["dollar_volume"] = (
        data["close"]
        *
        data["volume"]
    )

    # Average dollar turnover per 5-minute candle.
    data["dollar_volume_ma20"] = (
        data["dollar_volume"]
        .rolling(20)
        .mean()
    )

    # Approximate trailing 24-hour dollar turnover.
    # 288 five-minute candles = 24 hours.
    data["dollar_volume_24h"] = (
        data["dollar_volume"]
        .rolling(288)
        .sum()
    )

    data.replace(
        [
            np.inf,
            -np.inf,
        ],
        np.nan,
        inplace=True,
    )

    return data


# =========================================================
# FORWARD TRADE SIMULATION
# =========================================================

def simulate_forward_trade(
    data,
    index,
):

    final_index = (
        index
        + TARGET_HORIZON_CANDLES
    )

    if final_index >= len(data):
        return None

    entry = float(
        data.iloc[index]["close"]
    )

    stop = (
        entry
        * (1 - STOP_LOSS_PCT)
    )

    target = (
        entry
        * (1 + TAKE_PROFIT_PCT)
    )

    exit_price = None
    success = 0

    for future_index in range(
        index + 1,
        final_index + 1,
    ):

        future_row = (
            data.iloc[future_index]
        )

        future_low = float(
            future_row["low"]
        )

        future_high = float(
            future_row["high"]
        )

        stop_hit = (
            future_low <= stop
        )

        target_hit = (
            future_high >= target
        )

        # Conservative assumption:
        # if both happen in one candle,
        # the stop is assumed hit first.
        if stop_hit:

            exit_price = stop
            success = 0
            break

        if target_hit:

            exit_price = target
            success = 1
            break

    if exit_price is None:

        exit_price = float(
            data.iloc[
                final_index
            ]["close"]
        )

        success = (
            1
            if exit_price >= target
            else 0
        )

    gross_return = (
        exit_price
        /
        entry
        - 1
    )

    entry_fee_pct = (
        ESTIMATED_FEE_PER_SIDE
    )

    exit_fee_pct = (
        (
            exit_price
            / entry
        )
        *
        ESTIMATED_FEE_PER_SIDE
    )

    net_return = (
        gross_return
        -
        entry_fee_pct
        -
        exit_fee_pct
    )

    return {
        "success":
            success,

        "gross_return":
            gross_return,

        "net_return":
            net_return,
    }


# =========================================================
# AI TARGET CREATION
# =========================================================

def create_trade_targets(data):

    total = len(data)

    success_targets = np.full(
        total,
        np.nan,
    )

    net_return_targets = np.full(
        total,
        np.nan,
    )

    for index in range(total):

        result = (
            simulate_forward_trade(
                data,
                index,
            )
        )

        if result is None:
            continue

        success_targets[index] = (
            result["success"]
        )

        net_return_targets[index] = (
            result["net_return"]
        )

    return (
        success_targets,
        net_return_targets,
    )


# =========================================================
# TRAINING DATA
# =========================================================

def build_training_data(df):

    data = (
        build_feature_frame(df)
    )

    (
        success_target,
        net_return_target,
    ) = create_trade_targets(data)

    data[
        "target_success"
    ] = success_target

    data[
        "target_net_return"
    ] = net_return_target

    required_columns = (
        FEATURE_COLUMNS
        +
        [
            "target_success",
            "target_net_return",
        ]
    )

    data = (
        data
        .dropna(
            subset=required_columns
        )
        .reset_index(drop=True)
    )

    data[
        "target_success"
    ] = (
        data[
            "target_success"
        ]
        .astype(int)
    )

    return data


# =========================================================
# TRAIN MODELS
# =========================================================

def train_models(
    training_data,
):

    if (
        len(training_data)
        < MIN_TRAINING_ROWS
    ):

        raise RuntimeError(
            "Not enough training history: "
            f"{len(training_data)} rows"
        )

    X_train = (
        training_data[
            FEATURE_COLUMNS
        ]
    )

    y_classifier = (
        training_data[
            "target_success"
        ]
    )

    y_regressor = (
        training_data[
            "target_net_return"
        ]
    )

    if (
        y_classifier.nunique()
        < 2
    ):

        raise RuntimeError(
            "Training sample contains "
            "only one target class."
        )

    positive_count = int(
        (
            y_classifier == 1
        ).sum()
    )

    if (
        positive_count
        <
        MIN_POSITIVE_TRAINING_EXAMPLES
    ):

        raise RuntimeError(
            "Not enough successful "
            "historical setups."
        )

    classifier = (
        RandomForestClassifier(
            n_estimators=500,
            max_depth=8,
            min_samples_leaf=10,
            max_features="sqrt",
            random_state=42,
            class_weight=(
                "balanced_subsample"
            ),
            n_jobs=-1,
        )
    )

    classifier.fit(
        X_train,
        y_classifier,
    )

    regressor = (
        RandomForestRegressor(
            n_estimators=500,
            max_depth=8,
            min_samples_leaf=10,
            max_features="sqrt",
            random_state=43,
            n_jobs=-1,
        )
    )

    regressor.fit(
        X_train,
        y_regressor,
    )

    return (
        classifier,
        regressor,
    )


# =========================================================
# MODEL PREDICTIONS
# =========================================================

def probability_success(
    model,
    row,
):

    X = (
        row[
            FEATURE_COLUMNS
        ]
        .to_frame()
        .T
    )

    probabilities = (
        model.predict_proba(X)[0]
    )

    mapping = {
        int(model.classes_[i]):
            float(
                probabilities[i]
            )
        for i
        in range(
            len(model.classes_)
        )
    }

    return mapping.get(
        1,
        0.0,
    )


def predicted_net_return(
    model,
    row,
):

    X = (
        row[
            FEATURE_COLUMNS
        ]
        .to_frame()
        .T
    )

    prediction = (
        model.predict(X)[0]
    )

    return float(prediction)


# =========================================================
# V6 ENTRY FILTER
# =========================================================

def passes_entry_filter(row):

    dollar_volume_5m = float(
        row[
            "dollar_volume_ma20"
        ]
    )

    dollar_volume_24h = float(
        row[
            "dollar_volume_24h"
        ]
    )

    volume_ratio = float(
        row["volume_ratio"]
    )

    atr_pct = float(
        row["atr_pct"]
    )

    rsi = float(
        row["rsi"]
    )

    close_position = float(
        row["close_position"]
    )

    return_6 = float(
        row["return_6"]
    )

    return_12 = float(
        row["return_12"]
    )

    price = float(
        row["close"]
    )

    ema9 = float(
        row["ema_9"]
    )

    ema21 = float(
        row["ema_21"]
    )

    ema50 = float(
        row["ema_50"]
    )

    # -----------------------------------------
    # Liquidity
    # -----------------------------------------

    if (
        dollar_volume_5m
        <
        MIN_DOLLAR_VOLUME_5M
    ):
        return False

    if (
        dollar_volume_24h
        <
        MIN_DOLLAR_VOLUME_24H
    ):
        return False

    # -----------------------------------------
    # Volume
    # -----------------------------------------

    if (
        volume_ratio
        <
        MIN_VOLUME_RATIO
    ):
        return False

    # -----------------------------------------
    # Volatility
    # -----------------------------------------

    if not (
        MIN_ATR_PCT
        <= atr_pct
        <= MAX_ATR_PCT
    ):
        return False

    # -----------------------------------------
    # RSI
    # -----------------------------------------

    if not (
        MIN_RSI
        <= rsi
        <= MAX_RSI
    ):
        return False

    # -----------------------------------------
    # Candle quality
    # -----------------------------------------

    if (
        close_position
        <
        MIN_CLOSE_POSITION
    ):
        return False

    # -----------------------------------------
    # Momentum
    # -----------------------------------------

    if (
        return_12
        <
        MIN_RETURN_12
    ):
        return False

    if (
        return_6
        <= -0.003
    ):
        return False

    # -----------------------------------------
    # Trend filter
    # -----------------------------------------

    # V5 rejected only very obviously
    # bearish structures.
    #
    # V6 requires at least a healthier
    # medium-term price structure.

    if price < ema21:
        return False

    if (
        ema9 < ema21
        and
        ema21 < ema50
    ):
        return False

    return True


# =========================================================
# POSITION
# =========================================================

@dataclass
class Position:

    entry_price: float
    quantity: float

    entry_time: int
    entry_index: int

    entry_probability: float
    predicted_net_return: float

    stop_loss: float
    take_profit: float


# =========================================================
# PNL
# =========================================================

def calculate_trade_pnl(
    entry_price,
    exit_price,
    quantity,
):

    entry_value = (
        entry_price
        * quantity
    )

    exit_value = (
        exit_price
        * quantity
    )

    gross_pnl = (
        exit_value
        -
        entry_value
    )

    entry_fee = (
        entry_value
        *
        ESTIMATED_FEE_PER_SIDE
    )

    exit_fee = (
        exit_value
        *
        ESTIMATED_FEE_PER_SIDE
    )

    fees = (
        entry_fee
        +
        exit_fee
    )

    net_pnl = (
        gross_pnl
        -
        fees
    )

    net_return_pct = (
        net_pnl
        /
        entry_value
        if entry_value > 0
        else 0.0
    )

    return {
        "gross_pnl":
            gross_pnl,

        "fees":
            fees,

        "net_pnl":
            net_pnl,

        "net_return_pct":
            net_return_pct,
    }


# =========================================================
# SINGLE-MARKET BACKTEST
# =========================================================

def run_product_backtest(
    product_id,
    days=7,
):

    raw = (
        get_historical_candles(
            product_id,
            days,
        )
    )

    feature_data = (
        build_feature_frame(raw)
    )

    now = int(time.time())

    test_start_time = (
        now
        -
        days
        * 24
        * 60
        * 60
    )

    # -----------------------------------------
    # Training data
    # -----------------------------------------

    raw_training = (
        raw[
            raw["time"]
            <
            test_start_time
        ]
        .copy()
        .reset_index(drop=True)
    )

    training_data = (
        build_training_data(
            raw_training
        )
    )

    (
        classifier,
        regressor,
    ) = train_models(
        training_data
    )

    # -----------------------------------------
    # Unseen test data
    # -----------------------------------------

    required_test_columns = (
        FEATURE_COLUMNS
        +
        [
            "atr_pct",
            "dollar_volume_ma20",
            "dollar_volume_24h",
            "ema_9",
            "ema_21",
            "ema_50",
        ]
    )

    test = (
        feature_data[
            feature_data["time"]
            >=
            test_start_time
        ]
        .dropna(
            subset=required_test_columns
        )
        .reset_index(drop=True)
    )

    if len(test) < 60:

        raise RuntimeError(
            "Not enough unseen "
            "test candles"
        )

    position = None

    trades = []

    net_equity = 0.0
    gross_equity = 0.0
    total_fees = 0.0

    equity_curve = [0.0]

    cooldown_until = -1

    signals_checked = 0
    classifier_passes = 0
    edge_passes = 0
    buy_signals = 0

    # -----------------------------------------
    # Test loop
    # -----------------------------------------

    for index, row in test.iterrows():

        price = float(
            row["close"]
        )

        high = float(
            row["high"]
        )

        low = float(
            row["low"]
        )

        timestamp = int(
            row["time"]
        )

        # =====================================
        # MANAGE OPEN POSITION
        # =====================================

        if position is not None:

            exit_price = None
            reason = None

            stop_hit = (
                low
                <=
                position.stop_loss
            )

            target_hit = (
                high
                >=
                position.take_profit
            )

            # Conservative ordering.
            if stop_hit:

                exit_price = (
                    position.stop_loss
                )

                reason = (
                    "STOP LOSS"
                )

            elif target_hit:

                exit_price = (
                    position.take_profit
                )

                reason = (
                    "TAKE PROFIT"
                )

            elif (
                index
                -
                position.entry_index
                >=
                MAX_HOLD_CANDLES
            ):

                exit_price = price

                reason = (
                    "MAX HOLD TIME"
                )

            if exit_price is not None:

                pnl = (
                    calculate_trade_pnl(
                        position.entry_price,
                        exit_price,
                        position.quantity,
                    )
                )

                gross_equity += (
                    pnl["gross_pnl"]
                )

                total_fees += (
                    pnl["fees"]
                )

                net_equity += (
                    pnl["net_pnl"]
                )

                trades.append(
                    {
                        "product":
                            product_id,

                        "entry_price":
                            position.entry_price,

                        "exit_price":
                            exit_price,

                        "gross_pnl":
                            pnl[
                                "gross_pnl"
                            ],

                        "fees":
                            pnl["fees"],

                        "pnl":
                            pnl["net_pnl"],

                        "net_return_pct":
                            pnl[
                                "net_return_pct"
                            ],

                        "reason":
                            reason,

                        "entry_probability":
                            position
                            .entry_probability,

                        "predicted_net_return":
                            position
                            .predicted_net_return,

                        "entry_time":
                            position
                            .entry_time,

                        "exit_time":
                            timestamp,
                    }
                )

                position = None

                cooldown_until = (
                    index
                    +
                    COOLDOWN_CANDLES
                )

        # =====================================
        # SEARCH FOR NEW POSITION
        # =====================================

        if position is None:

            if (
                index
                <
                cooldown_until
            ):

                equity_curve.append(
                    net_equity
                )

                continue

            # ---------------------------------
            # Technical / liquidity filter
            # ---------------------------------

            if not passes_entry_filter(
                row
            ):

                equity_curve.append(
                    net_equity
                )

                continue

            signals_checked += 1

            # ---------------------------------
            # Model 1:
            # probability of TP success
            # ---------------------------------

            probability = (
                probability_success(
                    classifier,
                    row,
                )
            )

            if (
                probability
                <
                BUY_THRESHOLD
            ):

                equity_curve.append(
                    net_equity
                )

                continue

            classifier_passes += 1

            # ---------------------------------
            # Model 2:
            # expected AFTER-FEE return
            # ---------------------------------

            expected_net = (
                predicted_net_return(
                    regressor,
                    row,
                )
            )

            if (
                expected_net
                <
                MIN_PREDICTED_NET_RETURN
            ):

                equity_curve.append(
                    net_equity
                )

                continue

            # ---------------------------------
            # Fee edge check
            # ---------------------------------

            round_trip_fee_estimate = (
                ESTIMATED_FEE_PER_SIDE
                * 2
            )

            required_edge = (
                round_trip_fee_estimate
                *
                FEE_EDGE_MULTIPLIER
            )

            expected_gross_equivalent = (
                expected_net
                +
                round_trip_fee_estimate
            )

            if (
                expected_gross_equivalent
                <
                required_edge
            ):

                equity_curve.append(
                    net_equity
                )

                continue

            edge_passes += 1

            # ---------------------------------
            # BUY
            # ---------------------------------

            buy_signals += 1

            quantity = (
                TRADE_SIZE
                /
                price
            )

            position = Position(
                entry_price=price,
                quantity=quantity,

                entry_time=timestamp,
                entry_index=index,

                entry_probability=(
                    probability
                ),

                predicted_net_return=(
                    expected_net
                ),

                stop_loss=(
                    price
                    *
                    (
                        1
                        -
                        STOP_LOSS_PCT
                    )
                ),

                take_profit=(
                    price
                    *
                    (
                        1
                        +
                        TAKE_PROFIT_PCT
                    )
                ),
            )

        equity_curve.append(
            net_equity
        )

    # =========================================
    # CLOSE REMAINING POSITION
    # =========================================

    if (
        position is not None
        and
        len(test) > 0
    ):

        final_row = (
            test.iloc[-1]
        )

        exit_price = float(
            final_row["close"]
        )

        pnl = (
            calculate_trade_pnl(
                position.entry_price,
                exit_price,
                position.quantity,
            )
        )

        gross_equity += (
            pnl["gross_pnl"]
        )

        total_fees += (
            pnl["fees"]
        )

        net_equity += (
            pnl["net_pnl"]
        )

        trades.append(
            {
                "product":
                    product_id,

                "entry_price":
                    position.entry_price,

                "exit_price":
                    exit_price,

                "gross_pnl":
                    pnl["gross_pnl"],

                "fees":
                    pnl["fees"],

                "pnl":
                    pnl["net_pnl"],

                "net_return_pct":
                    pnl[
                        "net_return_pct"
                    ],

                "reason":
                    "END OF TEST",

                "entry_probability":
                    position
                    .entry_probability,

                "predicted_net_return":
                    position
                    .predicted_net_return,

                "entry_time":
                    position.entry_time,

                "exit_time":
                    int(
                        final_row["time"]
                    ),
            }
        )

        equity_curve.append(
            net_equity
        )

    # =====================================================
    # MARKET PERFORMANCE STATISTICS
    # =====================================================

    wins_pnl = [
        t["pnl"]
        for t in trades
        if t["pnl"] > 0
    ]

    losses_pnl = [
        t["pnl"]
        for t in trades
        if t["pnl"] <= 0
    ]

    wins = len(wins_pnl)
    losses = len(losses_pnl)

    total_trades = len(trades)

    win_rate = (
        wins
        /
        total_trades
        if total_trades
        else 0.0
    )

    average_win = (
        sum(wins_pnl)
        /
        len(wins_pnl)
        if wins_pnl
        else 0.0
    )

    average_loss = (
        sum(losses_pnl)
        /
        len(losses_pnl)
        if losses_pnl
        else 0.0
    )

    winning_pnl = sum(
        wins_pnl
    )

    losing_pnl = abs(
        sum(losses_pnl)
    )

    profit_factor = (
        winning_pnl
        /
        losing_pnl
        if losing_pnl > 0

        else (
            float("inf")
            if winning_pnl > 0
            else 0.0
        )
    )

    expectancy = (
        net_equity
        /
        total_trades
        if total_trades
        else 0.0
    )

    # -----------------------------------------
    # Per-market drawdown
    # -----------------------------------------

    peak = 0.0
    max_drawdown = 0.0

    for value in equity_curve:

        peak = max(
            peak,
            value,
        )

        max_drawdown = min(
            max_drawdown,
            value - peak,
        )

    average_probability = (
        np.mean(
            [
                t[
                    "entry_probability"
                ]
                for t in trades
            ]
        )
        if trades
        else 0.0
    )

    average_predicted_net = (
        np.mean(
            [
                t[
                    "predicted_net_return"
                ]
                for t in trades
            ]
        )
        if trades
        else 0.0
    )

    return {
        "product":
            product_id,

        "days":
            days,

        "candles":
            len(raw),

        "training_rows":
            len(training_data),

        "test_candles":
            len(test),

        "signals_checked":
            signals_checked,

        "classifier_passes":
            classifier_passes,

        "edge_passes":
            edge_passes,

        "buy_signals":
            buy_signals,

        "trades":
            total_trades,

        "wins":
            wins,

        "losses":
            losses,

        "win_rate":
            win_rate,

        "gross_pnl":
            gross_equity,

        "fees":
            total_fees,

        "pnl":
            net_equity,

        "average_win":
            average_win,

        "average_loss":
            average_loss,

        "profit_factor":
            profit_factor,

        "expectancy":
            expectancy,

        "max_drawdown":
            max_drawdown,

        "average_probability":
            average_probability,

        "average_predicted_net_return":
            average_predicted_net,

        "trade_log":
            trades,
    }


# =========================================================
# COMPLETE COINBASE BACKTEST
# =========================================================

def run_backtest(days=7):

    days = int(
        max(
            3,
            min(
                days,
                30,
            ),
        )
    )

    discovered = (
        get_all_coinbase_usd_markets()
    )

    results = []
    errors = []

    print(
        f"{STRATEGY_NAME} BACKTEST: "
        f"discovered "
        f"{len(discovered)} "
        f"Coinbase USD markets."
    )

    print(
        f"Unseen test period: "
        f"{days} days"
    )

    print(
        f"Training lookback: "
        f"{TRAINING_LOOKBACK_DAYS} "
        f"days"
    )

    print(
        f"Buy probability threshold: "
        f"{BUY_THRESHOLD:.1%}"
    )

    print(
        f"Minimum predicted NET return: "
        f"{MIN_PREDICTED_NET_RETURN:.2%}"
    )

    print(
        f"Estimated fee per side: "
        f"{ESTIMATED_FEE_PER_SIDE:.3%}"
    )

    for index, product_id in enumerate(
        discovered,
        start=1,
    ):

        print(
            f"{STRATEGY_NAME} BACKTEST "
            f"{index}/"
            f"{len(discovered)}: "
            f"{product_id}"
        )

        try:

            result = (
                run_product_backtest(
                    product_id,
                    days,
                )
            )

            results.append(result)

            print(
                f"{product_id}: "
                f"{result['trades']} trades | "
                f"{result['win_rate'] * 100:.1f}% WR | "
                f"GBP "
                f"{result['pnl']:+.2f}"
            )

        except Exception as exc:

            errors.append(
                f"{product_id}: "
                f"{exc}"
            )

            print(
                f"BACKTEST SKIP "
                f"{product_id}: "
                f"{exc}"
            )

    if not results:

        raise RuntimeError(
            "Backtest failed for all "
            "Coinbase USD markets."
        )

    # =====================================================
    # COMBINE RESULTS
    # =====================================================

    total_trades = sum(
        x["trades"]
        for x in results
    )

    total_wins = sum(
        x["wins"]
        for x in results
    )

    total_losses = sum(
        x["losses"]
        for x in results
    )

    total_gross_pnl = sum(
        x["gross_pnl"]
        for x in results
    )

    total_fees = sum(
        x["fees"]
        for x in results
    )

    total_pnl = sum(
        x["pnl"]
        for x in results
    )

    combined_win_rate = (
        total_wins
        /
        total_trades
        if total_trades
        else 0.0
    )

    all_trades = []

    for item in results:

        all_trades.extend(
            item["trade_log"]
        )

    wins_pnl = [
        t["pnl"]
        for t in all_trades
        if t["pnl"] > 0
    ]

    losses_pnl = [
        t["pnl"]
        for t in all_trades
        if t["pnl"] <= 0
    ]

    average_win = (
        sum(wins_pnl)
        /
        len(wins_pnl)
        if wins_pnl
        else 0.0
    )

    average_loss = (
        sum(losses_pnl)
        /
        len(losses_pnl)
        if losses_pnl
        else 0.0
    )

    winning_pnl = sum(
        wins_pnl
    )

    losing_pnl = abs(
        sum(losses_pnl)
    )

    profit_factor = (
        winning_pnl
        /
        losing_pnl
        if losing_pnl > 0

        else (
            float("inf")
            if winning_pnl > 0
            else 0.0
        )
    )

    expectancy = (
        total_pnl
        /
        total_trades
        if total_trades
        else 0.0
    )

    trades_per_day = (
        total_trades
        /
        days
        if days
        else 0.0
    )

    ranked = sorted(
        results,
        key=lambda x: x["pnl"],
        reverse=True,
    )

    profitable_markets = sum(
        1
        for x in results
        if x["pnl"] > 0
    )

    losing_markets = sum(
        1
        for x in results
        if x["pnl"] < 0
    )

    flat_markets = (
        len(results)
        -
        profitable_markets
        -
        losing_markets
    )

    average_entry_probability = (
        np.mean(
            [
                t[
                    "entry_probability"
                ]
                for t in all_trades
            ]
        )
        if all_trades
        else 0.0
    )

    average_predicted_net_return = (
        np.mean(
            [
                t[
                    "predicted_net_return"
                ]
                for t in all_trades
            ]
        )
        if all_trades
        else 0.0
    )

    # This remains the worst PER-MARKET drawdown.
    #
    # It is not a true portfolio-level drawdown because
    # every market currently has its own independent
    # equity curve.
    worst_market_drawdown = min(
        x["max_drawdown"]
        for x in results
    )

    return {
        "strategy":
            STRATEGY_NAME,

        "days":
            days,

        "markets_discovered":
            len(discovered),

        "markets":
            len(results),

        "markets_skipped":
            len(errors),

        "profitable_markets":
            profitable_markets,

        "losing_markets":
            losing_markets,

        "flat_markets":
            flat_markets,

        "signals_checked":
            sum(
                x["signals_checked"]
                for x in results
            ),

        "classifier_passes":
            sum(
                x["classifier_passes"]
                for x in results
            ),

        "edge_passes":
            sum(
                x["edge_passes"]
                for x in results
            ),

        "buy_signals":
            sum(
                x["buy_signals"]
                for x in results
            ),

        "trades":
            total_trades,

        "trades_per_day":
            trades_per_day,

        "wins":
            total_wins,

        "losses":
            total_losses,

        "win_rate":
            combined_win_rate,

        "gross_pnl":
            total_gross_pnl,

        "fees":
            total_fees,

        "pnl":
            total_pnl,

        "average_win":
            average_win,

        "average_loss":
            average_loss,

        "profit_factor":
            profit_factor,

        "expectancy":
            expectancy,

        "worst_market_drawdown":
            worst_market_drawdown,

        "average_entry_probability":
            average_entry_probability,

        "average_predicted_net_return":
            average_predicted_net_return,

        "best_market":
            ranked[0]["product"],

        "best_market_pnl":
            ranked[0]["pnl"],

        "worst_market":
            ranked[-1]["product"],

        "worst_market_pnl":
            ranked[-1]["pnl"],

        "fee_per_side":
            ESTIMATED_FEE_PER_SIDE,

        "buy_threshold":
            BUY_THRESHOLD,

        "min_predicted_net_return":
            MIN_PREDICTED_NET_RETURN,

        "stop_loss_pct":
            STOP_LOSS_PCT,

        "take_profit_pct":
            TAKE_PROFIT_PCT,

        "target_horizon_candles":
            TARGET_HORIZON_CANDLES,

        "max_hold_candles":
            MAX_HOLD_CANDLES,

        "training_lookback_days":
            TRAINING_LOOKBACK_DAYS,

        "min_dollar_volume_5m":
            MIN_DOLLAR_VOLUME_5M,

        "min_dollar_volume_24h":
            MIN_DOLLAR_VOLUME_24H,

        "top_markets":
            ranked[:10],

        "bottom_markets":
            list(
                reversed(
                    ranked[-10:]
                )
            ),

        "by_market":
            ranked[:10],

        "errors":
            errors[:10],
    }


# =========================================================
# RUN
# =========================================================

if __name__ == "__main__":

    print(
        json.dumps(
            run_backtest(
                days=7
            ),
            indent=2,
            default=str,
        )
    )