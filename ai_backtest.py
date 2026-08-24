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
# STRATEGY V4
# =========================================================

STRATEGY_NAME = "V4"

GRANULARITY = "FIVE_MINUTE"

CANDLE_SECONDS = 300
PAGE_SIZE = 300


# =========================================================
# MARKET UNIVERSE
# =========================================================

# By default the backtester will test all suitable
# Coinbase USD markets.
#
# If you later want the backtest to match only the six
# live AI markets, we can add that as a Discord option.
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
# V4 ENTRY SETTINGS
# =========================================================

BUY_THRESHOLD = float(
    os.getenv(
        "AI_BUY_THRESHOLD",
        "0.64"
    )
)

BEARISH_THRESHOLD = float(
    os.getenv(
        "AI_BEARISH_THRESHOLD",
        "0.36"
    )
)

# Model predicts whether price will be at least
# +1.8% higher approximately one hour later.
TARGET_RETURN = float(
    os.getenv(
        "AI_TARGET_RETURN",
        "0.018"
    )
)

# 12 x 5 minutes = 60 minutes.
TARGET_CANDLES = int(
    os.getenv(
        "AI_TARGET_CANDLES",
        "12"
    )
)

# Estimated net edge required AFTER round-trip costs.
MIN_EXPECTED_NET_RETURN = float(
    os.getenv(
        "AI_MIN_EXPECTED_NET_RETURN",
        "0.003"
    )
)


# =========================================================
# V4 RISK MANAGEMENT
# =========================================================

STOP_LOSS_PCT = float(
    os.getenv(
        "AI_STOP_LOSS_PCT",
        "0.0125"
    )
)

TAKE_PROFIT_PCT = float(
    os.getenv(
        "AI_TAKE_PROFIT_PCT",
        "0.045"
    )
)

TRADE_SIZE = float(
    os.getenv(
        "AI_TRADE_SIZE",
        "25.0"
    )
)

ESTIMATED_FEE_PER_SIDE = float(
    os.getenv(
        "AI_BACKTEST_FEE_PER_SIDE",
        "0.006"
    )
)

ROUND_TRIP_FEE = (
    ESTIMATED_FEE_PER_SIDE
    * 2
)

# 36 x 5 minutes = 3 hours.
MAX_HOLD_CANDLES = int(
    os.getenv(
        "AI_MAX_HOLD_CANDLES",
        "36"
    )
)

SIGNAL_EXIT_PROBABILITY = float(
    os.getenv(
        "AI_SIGNAL_EXIT_PROBABILITY",
        "0.34"
    )
)

# Don't immediately re-enter after closing a trade.
COOLDOWN_CANDLES = int(
    os.getenv(
        "AI_COOLDOWN_CANDLES",
        "6"
    )
)


# =========================================================
# TRAINING SETTINGS
# =========================================================

# The requested backtest period is kept completely unseen.
#
# We fetch additional history BEFORE the test period
# exclusively for model training.
TRAINING_LOOKBACK_DAYS = int(
    os.getenv(
        "AI_TRAINING_LOOKBACK_DAYS",
        "7"
    )
)

MIN_TRAINING_ROWS = int(
    os.getenv(
        "AI_MIN_TRAINING_ROWS",
        "350"
    )
)

MIN_POSITIVE_TRAINING_EXAMPLES = int(
    os.getenv(
        "AI_MIN_POSITIVE_TRAINING_EXAMPLES",
        "5"
    )
)


# =========================================================
# OPTIONAL MARKET QUALITY FILTERS
# =========================================================

# These stop the AI taking signals from markets where
# extremely poor liquidity makes a paper result unrealistic.

MIN_DOLLAR_VOLUME_5M = float(
    os.getenv(
        "AI_MIN_DOLLAR_VOLUME_5M",
        "25000"
    )
)

MIN_VOLUME_RATIO = float(
    os.getenv(
        "AI_MIN_VOLUME_RATIO",
        "0.70"
    )
)

MIN_ATR_PCT = float(
    os.getenv(
        "AI_MIN_ATR_PCT",
        "0.0015"
    )
)

MAX_ATR_PCT = float(
    os.getenv(
        "AI_MAX_ATR_PCT",
        "0.08"
    )
)


# =========================================================
# CACHE / COINBASE SETTINGS
# =========================================================

CACHE_DIR = os.getenv(
    "AI_BACKTEST_CACHE_DIR",
    "backtest_cache_v4"
)

REQUEST_DELAY_SECONDS = float(
    os.getenv(
        "AI_BACKTEST_REQUEST_DELAY",
        "0.35"
    )
)

MAX_RETRIES = int(
    os.getenv(
        "AI_BACKTEST_MAX_RETRIES",
        "6"
    )
)

BACKOFF_START_SECONDS = float(
    os.getenv(
        "AI_BACKTEST_BACKOFF_START",
        "2"
    )
)


client = RESTClient()

http = requests.Session()

http.headers.update({
    "User-Agent":
        "Alpha-Alerts-Backtest-V4/1.0"
})

os.makedirs(
    CACHE_DIR,
    exist_ok=True
)


# =========================================================
# FEATURES
# =========================================================

FEATURE_COLUMNS = [
    "return_1",
    "return_3",
    "return_6",
    "return_12",
    "ma_ratio_5_20",
    "ma_ratio_20_50",
    "ema_ratio_9_21",
    "volatility_12",
    "volatility_24",
    "volume_ratio",
    "volume_zscore",
    "rsi",
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
        timeout=30
    )

    response.raise_for_status()

    products = response.json()

    markets = []

    for product in products:

        if (
            product.get("status")
            != "online"
        ):
            continue

        if (
            product.get("quote_currency")
            != "USD"
        ):
            continue

        product_id = product.get(
            "id"
        )

        base = product.get(
            "base_currency"
        )

        if not product_id or not base:
            continue

        if base in IGNORED_BASE_ASSETS:
            continue

        if not product_id.endswith(
            "-USD"
        ):
            continue

        markets.append(
            product_id
        )

    return sorted(
        set(markets)
    )


# =========================================================
# CACHE
# =========================================================

def cache_path(
    product_id,
    requested_days
):

    safe = (
        product_id
        .replace(
            "/",
            "_"
        )
        .replace(
            "-",
            "_"
        )
    )

    total_days = (
        requested_days
        + TRAINING_LOOKBACK_DAYS
    )

    return os.path.join(
        CACHE_DIR,
        (
            f"{safe}_"
            f"{total_days}d_v4.json"
        )
    )


def load_cache(
    product_id,
    requested_days
):

    path = cache_path(
        product_id,
        requested_days
    )

    if not os.path.exists(
        path
    ):
        return None

    try:

        age = (
            time.time()
            - os.path.getmtime(
                path
            )
        )

        # Historical data can be reused for six hours.
        if age > 6 * 60 * 60:
            return None

        with open(
            path,
            "r",
            encoding="utf-8"
        ) as f:

            rows = json.load(
                f
            )

        if not rows:
            return None

        return pd.DataFrame(
            rows
        )

    except Exception:
        return None


def save_cache(
    product_id,
    requested_days,
    df
):

    path = cache_path(
        product_id,
        requested_days
    )

    rows = df.to_dict(
        orient="records"
    )

    with open(
        path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            rows,
            f
        )


# =========================================================
# RATE-LIMIT-SAFE COINBASE REQUEST
# =========================================================

def get_candle_page(
    product_id,
    start,
    end
):

    delay = (
        BACKOFF_START_SECONDS
    )

    last_error = None

    for attempt in range(
        MAX_RETRIES
    ):

        try:

            response = (
                client.get_public_candles(
                    product_id=product_id,
                    start=str(
                        int(start)
                    ),
                    end=str(
                        int(end)
                    ),
                    granularity=GRANULARITY,
                    limit=PAGE_SIZE
                )
            )

            time.sleep(
                REQUEST_DELAY_SECONDS
            )

            return response

        except Exception as exc:

            last_error = exc

            text = str(
                exc
            )

            if (
                "429" in text
                or
                "Too Many Requests"
                in text
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

                time.sleep(
                    delay
                )

                delay = min(
                    delay * 2,
                    60
                )

                continue

            raise

    raise RuntimeError(
        (
            "Coinbase rate limit "
            f"persisted for "
            f"{product_id}: "
            f"{last_error}"
        )
    )


# =========================================================
# HISTORICAL DATA
# =========================================================

def get_historical_candles(
    product_id,
    days
):

    cached = load_cache(
        product_id,
        days
    )

    if cached is not None:

        print(
            f"CACHE HIT: "
            f"{product_id}"
        )

        return cached

    end_time = int(
        time.time()
    )

    total_days = (
        days
        + TRAINING_LOOKBACK_DAYS
    )

    start_time = (
        end_time
        - (
            total_days
            * 24
            * 60
            * 60
        )
    )

    rows = []

    cursor = (
        start_time
    )

    while cursor < end_time:

        page_end = min(
            cursor
            + (
                PAGE_SIZE
                * CANDLE_SECONDS
            ),
            end_time
        )

        response = (
            get_candle_page(
                product_id,
                cursor,
                page_end
            )
        )

        for candle in (
            response.candles
        ):

            rows.append({
                "time":
                    int(
                        candle.start
                    ),
                "open":
                    float(
                        candle.open
                    ),
                "high":
                    float(
                        candle.high
                    ),
                "low":
                    float(
                        candle.low
                    ),
                "close":
                    float(
                        candle.close
                    ),
                "volume":
                    float(
                        candle.volume
                    ),
            })

        cursor = (
            page_end
            + CANDLE_SECONDS
        )

    if not rows:

        raise RuntimeError(
            (
                "No historical candles "
                f"returned for "
                f"{product_id}"
            )
        )

    df = (
        pd.DataFrame(
            rows
        )
        .sort_values(
            "time"
        )
        .drop_duplicates(
            subset=["time"]
        )
        .reset_index(
            drop=True
        )
    )

    save_cache(
        product_id,
        days,
        df
    )

    return df


# =========================================================
# RSI
# =========================================================

def calculate_rsi(
    series,
    period=14
):

    delta = (
        series.diff()
    )

    gain = (
        delta.clip(
            lower=0
        )
    )

    loss = (
        -delta.clip(
            upper=0
        )
    )

    avg_gain = (
        gain
        .rolling(
            period
        )
        .mean()
    )

    avg_loss = (
        loss
        .rolling(
            period
        )
        .mean()
    )

    rs = (
        avg_gain
        / avg_loss.replace(
            0,
            np.nan
        )
    )

    rsi = (
        100
        - (
            100
            / (
                1
                + rs
            )
        )
    )

    return rsi.fillna(
        50
    )


# =========================================================
# ATR
# =========================================================

def calculate_atr(
    data,
    period=14
):

    previous_close = (
        data["close"]
        .shift(1)
    )

    high_low = (
        data["high"]
        - data["low"]
    )

    high_previous = (
        data["high"]
        - previous_close
    ).abs()

    low_previous = (
        data["low"]
        - previous_close
    ).abs()

    true_range = (
        pd.concat(
            [
                high_low,
                high_previous,
                low_previous,
            ],
            axis=1
        )
        .max(
            axis=1
        )
    )

    return (
        true_range
        .rolling(
            period
        )
        .mean()
    )


# =========================================================
# FEATURE FRAME
# =========================================================

def build_feature_frame(
    df
):

    data = df.copy()

    data["return_1"] = (
        data["close"]
        .pct_change()
    )

    data["return_3"] = (
        data["close"]
        .pct_change(3)
    )

    data["return_6"] = (
        data["close"]
        .pct_change(6)
    )

    data["return_12"] = (
        data["close"]
        .pct_change(12)
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
        / data["ma_20"]
    )

    data["ma_ratio_20_50"] = (
        data["ma_20"]
        / data["ma_50"]
    )

    data["ema_9"] = (
        data["close"]
        .ewm(
            span=9,
            adjust=False
        )
        .mean()
    )

    data["ema_21"] = (
        data["close"]
        .ewm(
            span=21,
            adjust=False
        )
        .mean()
    )

    data["ema_ratio_9_21"] = (
        data["ema_9"]
        / data["ema_21"]
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
        / data[
            "volume_ma_20"
        ].replace(
            0,
            np.nan
        )
    )

    data["volume_zscore"] = (
        (
            data["volume"]
            - data["volume_ma_20"]
        )
        / data[
            "volume_std_20"
        ].replace(
            0,
            np.nan
        )
    )

    data["rsi"] = (
        calculate_rsi(
            data["close"]
        )
    )

    data["atr"] = (
        calculate_atr(
            data
        )
    )

    data["atr_pct"] = (
        data["atr"]
        / data["close"]
        .replace(
            0,
            np.nan
        )
    )

    candle_range = (
        data["high"]
        - data["low"]
    )

    data["range_pct"] = (
        candle_range
        / data["close"]
        .replace(
            0,
            np.nan
        )
    )

    data["body_pct"] = (
        (
            data["close"]
            - data["open"]
        )
        / data["open"]
        .replace(
            0,
            np.nan
        )
    )

    data["close_position"] = (
        (
            data["close"]
            - data["low"]
        )
        / candle_range.replace(
            0,
            np.nan
        )
    )

    data["dollar_volume"] = (
        data["close"]
        * data["volume"]
    )

    data["dollar_volume_ma20"] = (
        data["dollar_volume"]
        .rolling(20)
        .mean()
    )

    data.replace(
        [
            np.inf,
            -np.inf
        ],
        np.nan,
        inplace=True
    )

    return data


# =========================================================
# TRAINING LABEL
# =========================================================

def build_training_data(
    df
):

    data = (
        build_feature_frame(
            df
        )
    )

    future_price = (
        data["close"]
        .shift(
            -TARGET_CANDLES
        )
    )

    data["future_return"] = (
        (
            future_price
            / data["close"]
        )
        - 1
    )

    # Crucially, unknown future results remain NaN.
    data["target"] = np.where(
        data[
            "future_return"
        ].notna(),
        (
            data[
                "future_return"
            ]
            >= TARGET_RETURN
        ).astype(
            float
        ),
        np.nan
    )

    required = (
        FEATURE_COLUMNS
        + [
            "future_return",
            "target",
        ]
    )

    data = (
        data
        .dropna(
            subset=required
        )
        .reset_index(
            drop=True
        )
    )

    data["target"] = (
        data["target"]
        .astype(
            int
        )
    )

    return data


# =========================================================
# MODEL TRAINING
# =========================================================

def train_models(
    training_data
):

    if (
        len(training_data)
        < MIN_TRAINING_ROWS
    ):

        raise RuntimeError(
            (
                "Not enough training "
                f"history: "
                f"{len(training_data)} rows"
            )
        )

    X_train = (
        training_data[
            FEATURE_COLUMNS
        ]
    )

    y_class = (
        training_data[
            "target"
        ]
    )

    y_return = (
        training_data[
            "future_return"
        ]
    )

    if (
        y_class.nunique()
        < 2
    ):

        raise RuntimeError(
            (
                "Training sample "
                "contains only one "
                "target class."
            )
        )

    positive_count = int(
        (
            y_class == 1
        ).sum()
    )

    if (
        positive_count
        < MIN_POSITIVE_TRAINING_EXAMPLES
    ):

        raise RuntimeError(
            (
                "Not enough successful "
                "historical setups."
            )
        )

    classifier = (
        RandomForestClassifier(
            n_estimators=450,
            max_depth=8,
            min_samples_leaf=6,
            random_state=42,
            class_weight="balanced",
            n_jobs=-1,
        )
    )

    regressor = (
        RandomForestRegressor(
            n_estimators=400,
            max_depth=8,
            min_samples_leaf=6,
            random_state=42,
            n_jobs=-1,
        )
    )

    classifier.fit(
        X_train,
        y_class
    )

    regressor.fit(
        X_train,
        y_return
    )

    return (
        classifier,
        regressor
    )


# =========================================================
# MODEL PREDICTIONS
# =========================================================

def probability_up(
    model,
    row
):

    X = (
        row[
            FEATURE_COLUMNS
        ]
        .to_frame()
        .T
    )

    probabilities = (
        model.predict_proba(
            X
        )[0]
    )

    mapping = {
        int(
            model.classes_[
                index
            ]
        ):
        float(
            probabilities[
                index
            ]
        )
        for index in range(
            len(
                model.classes_
            )
        )
    }

    return mapping.get(
        1,
        0.0
    )


def predict_return(
    model,
    row
):

    X = (
        row[
            FEATURE_COLUMNS
        ]
        .to_frame()
        .T
    )

    return float(
        model.predict(
            X
        )[0]
    )


# =========================================================
# BASIC LIQUIDITY FILTER
# =========================================================

def passes_market_filter(
    row
):

    dollar_volume = float(
        row[
            "dollar_volume_ma20"
        ]
    )

    volume_ratio = float(
        row[
            "volume_ratio"
        ]
    )

    atr_pct = float(
        row[
            "atr_pct"
        ]
    )

    if (
        dollar_volume
        < MIN_DOLLAR_VOLUME_5M
    ):
        return False

    if (
        volume_ratio
        < MIN_VOLUME_RATIO
    ):
        return False

    if (
        atr_pct
        < MIN_ATR_PCT
    ):
        return False

    if (
        atr_pct
        > MAX_ATR_PCT
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

    expected_return: float

    expected_net_return: float

    stop_loss: float

    take_profit: float


# =========================================================
# P&L
# =========================================================

def calculate_trade_pnl(
    entry_price,
    exit_price,
    quantity
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
        - entry_value
    )

    entry_fee = (
        entry_value
        * ESTIMATED_FEE_PER_SIDE
    )

    exit_fee = (
        exit_value
        * ESTIMATED_FEE_PER_SIDE
    )

    total_fees = (
        entry_fee
        + exit_fee
    )

    net_pnl = (
        gross_pnl
        - total_fees
    )

    return {
        "gross_pnl":
            gross_pnl,

        "entry_fee":
            entry_fee,

        "exit_fee":
            exit_fee,

        "fees":
            total_fees,

        "net_pnl":
            net_pnl,
    }


# =========================================================
# SINGLE MARKET BACKTEST
# =========================================================

def run_product_backtest(
    product_id,
    days=7
):

    raw = (
        get_historical_candles(
            product_id,
            days
        )
    )

    feature_data = (
        build_feature_frame(
            raw
        )
    )

    # -----------------------------------------------------
    # DEFINE TRUE TEST WINDOW
    # -----------------------------------------------------

    test_start_time = (
        int(
            time.time()
        )
        - (
            days
            * 24
            * 60
            * 60
        )
    )

    # Anything before this timestamp is training history.
    raw_training = (
        raw[
            raw["time"]
            < test_start_time
        ]
        .copy()
        .reset_index(
            drop=True
        )
    )

    training_data = (
        build_training_data(
            raw_training
        )
    )

    classifier, regressor = (
        train_models(
            training_data
        )
    )

    # Test data gets indicator history from the complete
    # dataframe, but its rows occur strictly AFTER the
    # training period.
    test = (
        feature_data[
            feature_data["time"]
            >= test_start_time
        ]
        .dropna(
            subset=(
                FEATURE_COLUMNS
                + [
                    "atr_pct",
                    "dollar_volume_ma20",
                ]
            )
        )
        .reset_index(
            drop=True
        )
    )

    if len(test) < 60:

        raise RuntimeError(
            "Not enough unseen test candles"
        )

    position = None

    trades = []

    net_equity = 0.0
    gross_equity = 0.0
    total_fees = 0.0

    equity_curve = [
        0.0
    ]

    cooldown_until = -1

    signals_checked = 0
    buy_signals = 0

    for index, row in (
        test.iterrows()
    ):

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

        current_probability = None
        current_expected_return = None

        # -------------------------------------------------
        # MANAGE OPEN TRADE
        # -------------------------------------------------

        if position is not None:

            stop_hit = (
                low
                <= position.stop_loss
            )

            target_hit = (
                high
                >= position.take_profit
            )

            exit_price = None
            reason = None

            # Conservative assumption if both are hit
            # in the same five-minute candle.
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

            else:

                held_candles = (
                    index
                    - position.entry_index
                )

                if (
                    held_candles
                    >= MAX_HOLD_CANDLES
                ):

                    exit_price = price

                    reason = (
                        "MAX HOLD TIME"
                    )

                elif (
                    held_candles >= 4
                ):

                    current_probability = (
                        probability_up(
                            classifier,
                            row
                        )
                    )

                    if (
                        current_probability
                        <= SIGNAL_EXIT_PROBABILITY
                    ):

                        exit_price = price

                        reason = (
                            "AI SIGNAL REVERSAL"
                        )

            if (
                exit_price
                is not None
            ):

                pnl_result = (
                    calculate_trade_pnl(
                        position.entry_price,
                        exit_price,
                        position.quantity
                    )
                )

                gross_equity += (
                    pnl_result[
                        "gross_pnl"
                    ]
                )

                total_fees += (
                    pnl_result[
                        "fees"
                    ]
                )

                net_equity += (
                    pnl_result[
                        "net_pnl"
                    ]
                )

                trades.append({
                    "product":
                        product_id,

                    "entry_price":
                        position.entry_price,

                    "exit_price":
                        exit_price,

                    "gross_pnl":
                        pnl_result[
                            "gross_pnl"
                        ],

                    "fees":
                        pnl_result[
                            "fees"
                        ],

                    "pnl":
                        pnl_result[
                            "net_pnl"
                        ],

                    "reason":
                        reason,

                    "entry_probability":
                        position.entry_probability,

                    "expected_return":
                        position.expected_return,

                    "expected_net_return":
                        position.expected_net_return,

                    "entry_time":
                        position.entry_time,

                    "exit_time":
                        timestamp,
                })

                position = None

                cooldown_until = (
                    index
                    + COOLDOWN_CANDLES
                )

        # -------------------------------------------------
        # LOOK FOR A NEW ENTRY
        # -------------------------------------------------

        if position is None:

            if (
                index
                < cooldown_until
            ):

                equity_curve.append(
                    net_equity
                )

                continue

            if not passes_market_filter(
                row
            ):

                equity_curve.append(
                    net_equity
                )

                continue

            signals_checked += 1

            if (
                current_probability
                is None
            ):

                current_probability = (
                    probability_up(
                        classifier,
                        row
                    )
                )

            current_expected_return = (
                predict_return(
                    regressor,
                    row
                )
            )

            expected_net_return = (
                current_expected_return
                - ROUND_TRIP_FEE
            )

            if (
                current_probability
                >= BUY_THRESHOLD
                and
                expected_net_return
                >= MIN_EXPECTED_NET_RETURN
            ):

                buy_signals += 1

                quantity = (
                    TRADE_SIZE
                    / price
                )

                position = Position(
                    entry_price=price,

                    quantity=quantity,

                    entry_time=timestamp,

                    entry_index=index,

                    entry_probability=(
                        current_probability
                    ),

                    expected_return=(
                        current_expected_return
                    ),

                    expected_net_return=(
                        expected_net_return
                    ),

                    stop_loss=(
                        price
                        * (
                            1
                            - STOP_LOSS_PCT
                        )
                    ),

                    take_profit=(
                        price
                        * (
                            1
                            + TAKE_PROFIT_PCT
                        )
                    )
                )

        equity_curve.append(
            net_equity
        )

    # =====================================================
    # CLOSE REMAINING TRADE
    # =====================================================

    if (
        position is not None
        and len(test) > 0
    ):

        final_row = (
            test.iloc[
                -1
            ]
        )

        exit_price = float(
            final_row[
                "close"
            ]
        )

        pnl_result = (
            calculate_trade_pnl(
                position.entry_price,
                exit_price,
                position.quantity
            )
        )

        gross_equity += (
            pnl_result[
                "gross_pnl"
            ]
        )

        total_fees += (
            pnl_result[
                "fees"
            ]
        )

        net_equity += (
            pnl_result[
                "net_pnl"
            ]
        )

        trades.append({
            "product":
                product_id,

            "entry_price":
                position.entry_price,

            "exit_price":
                exit_price,

            "gross_pnl":
                pnl_result[
                    "gross_pnl"
                ],

            "fees":
                pnl_result[
                    "fees"
                ],

            "pnl":
                pnl_result[
                    "net_pnl"
                ],

            "reason":
                "END OF TEST",

            "entry_probability":
                position.entry_probability,

            "expected_return":
                position.expected_return,

            "expected_net_return":
                position.expected_net_return,

            "entry_time":
                position.entry_time,

            "exit_time":
                int(
                    final_row[
                        "time"
                    ]
                ),
        })

        equity_curve.append(
            net_equity
        )

    # =====================================================
    # RESULTS
    # =====================================================

    winning_trades = [
        trade["pnl"]
        for trade in trades
        if trade["pnl"] > 0
    ]

    losing_trades = [
        trade["pnl"]
        for trade in trades
        if trade["pnl"] <= 0
    ]

    wins = len(
        winning_trades
    )

    losses = len(
        losing_trades
    )

    total_trades = len(
        trades
    )

    win_rate = (
        wins
        / total_trades
        if total_trades
        else 0.0
    )

    average_win = (
        sum(
            winning_trades
        )
        / len(
            winning_trades
        )
        if winning_trades
        else 0.0
    )

    average_loss = (
        sum(
            losing_trades
        )
        / len(
            losing_trades
        )
        if losing_trades
        else 0.0
    )

    winning_pnl = sum(
        winning_trades
    )

    losing_pnl = abs(
        sum(
            losing_trades
        )
    )

    profit_factor = (
        winning_pnl
        / losing_pnl
        if losing_pnl > 0
        else (
            float(
                "inf"
            )
            if winning_pnl > 0
            else 0.0
        )
    )

    expectancy = (
        net_equity
        / total_trades
        if total_trades
        else 0.0
    )

    peak = 0.0

    max_drawdown = 0.0

    for value in equity_curve:

        peak = max(
            peak,
            value
        )

        drawdown = (
            value
            - peak
        )

        max_drawdown = min(
            max_drawdown,
            drawdown
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

        "trade_log":
            trades,
    }


# =========================================================
# ALL MARKET BACKTEST
# =========================================================

def run_backtest(
    days=7
):

    days = int(
        max(
            3,
            min(
                days,
                30
            )
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
        f"{TRAINING_LOOKBACK_DAYS} days"
    )

    for index, product_id in (
        enumerate(
            discovered,
            start=1
        )
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
                    days=days
                )
            )

            results.append(
                result
            )

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
            (
                "Backtest failed for "
                "all Coinbase USD markets."
            )
        )

    total_trades = sum(
        item["trades"]
        for item in results
    )

    total_wins = sum(
        item["wins"]
        for item in results
    )

    total_losses = sum(
        item["losses"]
        for item in results
    )

    total_gross_pnl = sum(
        item["gross_pnl"]
        for item in results
    )

    total_fees = sum(
        item["fees"]
        for item in results
    )

    total_pnl = sum(
        item["pnl"]
        for item in results
    )

    combined_win_rate = (
        total_wins
        / total_trades
        if total_trades
        else 0.0
    )

    all_trades = []

    for item in results:

        all_trades.extend(
            item[
                "trade_log"
            ]
        )

    wins_pnl = [
        trade["pnl"]
        for trade in all_trades
        if trade["pnl"] > 0
    ]

    losses_pnl = [
        trade["pnl"]
        for trade in all_trades
        if trade["pnl"] <= 0
    ]

    average_win = (
        sum(
            wins_pnl
        )
        / len(
            wins_pnl
        )
        if wins_pnl
        else 0.0
    )

    average_loss = (
        sum(
            losses_pnl
        )
        / len(
            losses_pnl
        )
        if losses_pnl
        else 0.0
    )

    winning_pnl = sum(
        wins_pnl
    )

    losing_pnl = abs(
        sum(
            losses_pnl
        )
    )

    profit_factor = (
        winning_pnl
        / losing_pnl
        if losing_pnl > 0
        else (
            float(
                "inf"
            )
            if winning_pnl > 0
            else 0.0
        )
    )

    expectancy = (
        total_pnl
        / total_trades
        if total_trades
        else 0.0
    )

    trades_per_day = (
        total_trades
        / days
        if days
        else 0.0
    )

    ranked = sorted(
        results,
        key=lambda item:
            item[
                "pnl"
            ],
        reverse=True
    )

    best = (
        ranked[
            0
        ]
    )

    worst = (
        ranked[
            -1
        ]
    )

    worst_drawdown = min(
        item[
            "max_drawdown"
        ]
        for item in results
    )

    profitable_markets = sum(
        1
        for item in results
        if item["pnl"] > 0
    )

    losing_markets = sum(
        1
        for item in results
        if item["pnl"] < 0
    )

    flat_markets = (
        len(results)
        - profitable_markets
        - losing_markets
    )

    return {
        "strategy":
            STRATEGY_NAME,

        "days":
            days,

        "markets_discovered":
            len(
                discovered
            ),

        "markets":
            len(
                results
            ),

        "markets_skipped":
            len(
                errors
            ),

        "profitable_markets":
            profitable_markets,

        "losing_markets":
            losing_markets,

        "flat_markets":
            flat_markets,

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

        "max_drawdown":
            worst_drawdown,

        "best_market":
            best[
                "product"
            ],

        "best_market_pnl":
            best[
                "pnl"
            ],

        "worst_market":
            worst[
                "product"
            ],

        "worst_market_pnl":
            worst[
                "pnl"
            ],

        "fee_per_side":
            ESTIMATED_FEE_PER_SIDE,

        "buy_threshold":
            BUY_THRESHOLD,

        "target_return":
            TARGET_RETURN,

        "minimum_expected_net_return":
            MIN_EXPECTED_NET_RETURN,

        "stop_loss_pct":
            STOP_LOSS_PCT,

        "take_profit_pct":
            TAKE_PROFIT_PCT,

        "target_horizon_candles":
            TARGET_CANDLES,

        "training_lookback_days":
            TRAINING_LOOKBACK_DAYS,

        "top_markets":
            ranked[
                :10
            ],

        "bottom_markets":
            list(
                reversed(
                    ranked[
                        -10:
                    ]
                )
            ),

        "by_market":
            ranked[
                :10
            ],

        "errors":
            errors[
                :10
            ],
    }


# =========================================================
# START
# =========================================================

if __name__ == "__main__":

    result = (
        run_backtest(
            days=7
        )
    )

    print(
        json.dumps(
            result,
            indent=2,
            default=str
        )
    )