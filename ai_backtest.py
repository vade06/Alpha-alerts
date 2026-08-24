import os
import time
import json
from dataclasses import dataclass

import numpy as np
import pandas as pd
import requests

from coinbase.rest import RESTClient
from sklearn.ensemble import RandomForestClassifier


# =========================================================
# STRATEGY V2 SETTINGS
# =========================================================

GRANULARITY = "FIVE_MINUTE"
CANDLE_SECONDS = 300
PAGE_SIZE = 300


# ---------------------------------------------------------
# ENTRY SETTINGS
# ---------------------------------------------------------

BUY_THRESHOLD = float(
    os.getenv(
        "AI_BUY_THRESHOLD",
        "0.70"
    )
)

TRADE_SIZE = float(
    os.getenv(
        "AI_TRADE_SIZE",
        "25.0"
    )
)


# ---------------------------------------------------------
# RISK MANAGEMENT
# ---------------------------------------------------------

STOP_LOSS_PCT = float(
    os.getenv(
        "AI_STOP_LOSS_PCT",
        "0.02"
    )
)

TAKE_PROFIT_PCT = float(
    os.getenv(
        "AI_TAKE_PROFIT_PCT",
        "0.05"
    )
)


# ---------------------------------------------------------
# MODEL TARGET
#
# 24 x 5-minute candles = 2 hours
#
# The model learns:
#
# "Does +5% get hit before -2%
# during the next two hours?"
# ---------------------------------------------------------

TARGET_HORIZON_CANDLES = int(
    os.getenv(
        "AI_TARGET_HORIZON_CANDLES",
        "24"
    )
)


# ---------------------------------------------------------
# ENTRY FILTERS
# ---------------------------------------------------------

MIN_RSI = float(
    os.getenv(
        "AI_MIN_RSI",
        "52"
    )
)

MAX_RSI = float(
    os.getenv(
        "AI_MAX_RSI",
        "70"
    )
)

MIN_VOLUME_RATIO = float(
    os.getenv(
        "AI_MIN_VOLUME_RATIO",
        "1.20"
    )
)

MIN_ATR_PCT = float(
    os.getenv(
        "AI_MIN_ATR_PCT",
        "0.004"
    )
)

MAX_ATR_PCT = float(
    os.getenv(
        "AI_MAX_ATR_PCT",
        "0.060"
    )
)

MIN_DOLLAR_VOLUME_5M = float(
    os.getenv(
        "AI_MIN_DOLLAR_VOLUME_5M",
        "25000"
    )
)

MIN_CLOSE_POSITION = float(
    os.getenv(
        "AI_MIN_CLOSE_POSITION",
        "0.55"
    )
)

COOLDOWN_CANDLES = int(
    os.getenv(
        "AI_COOLDOWN_CANDLES",
        "12"
    )
)


# ---------------------------------------------------------
# FEES
# ---------------------------------------------------------

ESTIMATED_FEE_PER_SIDE = float(
    os.getenv(
        "AI_BACKTEST_FEE_PER_SIDE",
        "0.006"
    )
)


# =========================================================
# COINBASE
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


CACHE_DIR = os.getenv(
    "AI_BACKTEST_CACHE_DIR",
    "backtest_cache"
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
    "User-Agent": "Alpha-Alerts-Backtest/4.0"
})


os.makedirs(
    CACHE_DIR,
    exist_ok=True
)


# =========================================================
# MODEL FEATURES
# =========================================================

FEATURE_COLUMNS = [
    "return_1",
    "return_3",
    "return_6",
    "return_12",

    "ema_ratio_20_50",
    "ema_ratio_50_200",

    "price_vs_ema20",
    "price_vs_ema50",
    "price_vs_ema200",

    "volatility_12",
    "volatility_24",

    "volume_ratio",
    "dollar_volume_ratio",

    "rsi",

    "atr_pct",

    "range_pct",
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
    days
):

    safe = product_id.replace(
        "/",
        "_"
    )

    return os.path.join(
        CACHE_DIR,
        f"{safe}_{days}d.json"
    )


def load_cache(
    product_id,
    days
):

    path = cache_path(
        product_id,
        days
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
            encoding="utf-8"
        ) as f:

            rows = json.load(f)

        if not rows:
            return None

        return pd.DataFrame(
            rows
        )

    except Exception:
        return None


def save_cache(
    product_id,
    days,
    df
):

    path = cache_path(
        product_id,
        days
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
# COINBASE RATE LIMIT HANDLING
# =========================================================

def get_candle_page(
    product_id,
    start,
    end
):

    delay = BACKOFF_START_SECONDS

    last_error = None

    for attempt in range(
        MAX_RETRIES
    ):

        try:

            response = (
                client.get_public_candles(
                    product_id=product_id,
                    start=str(start),
                    end=str(end),
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

            text = str(exc)

            if (
                "429" in text
                or
                "Too Many Requests" in text
            ):

                print(
                    f"RATE LIMIT {product_id}: "
                    f"waiting {delay:.0f}s "
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
        f"Coinbase rate limit persisted "
        f"for {product_id}: "
        f"{last_error}"
    )


# =========================================================
# HISTORICAL DATA
# =========================================================

def get_historical_candles(
    product_id,
    days=7
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

    start_time = (
        end_time
        - int(
            days
            * 24
            * 60
            * 60
        )
    )

    rows = []

    cursor = start_time

    while cursor < end_time:

        page_end = min(
            cursor
            + PAGE_SIZE
            * CANDLE_SECONDS,
            end_time
        )

        response = get_candle_page(
            product_id,
            cursor,
            page_end
        )

        for candle in response.candles:

            rows.append({
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
                    float(candle.volume)
            })

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
# INDICATORS
# =========================================================

def calculate_rsi(
    series,
    period=14
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
        adjust=False
    ).mean()

    avg_loss = loss.ewm(
        alpha=1 / period,
        adjust=False
    ).mean()

    rs = (
        avg_gain
        /
        avg_loss.replace(
            0,
            np.nan
        )
    )

    rsi = (
        100
        -
        (
            100
            /
            (
                1 + rs
            )
        )
    )

    return rsi.fillna(
        50
    )


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

    high_close = (
        data["high"]
        - previous_close
    ).abs()

    low_close = (
        data["low"]
        - previous_close
    ).abs()

    true_range = pd.concat(
        [
            high_low,
            high_close,
            low_close
        ],
        axis=1
    ).max(
        axis=1
    )

    return (
        true_range
        .ewm(
            alpha=1 / period,
            adjust=False
        )
        .mean()
    )


# =========================================================
# TARGET
#
# Target = 1 if TP gets hit BEFORE SL
# within TARGET_HORIZON_CANDLES.
#
# This aligns AI training with the actual strategy.
# =========================================================

def create_trade_target(
    data
):

    targets = []

    total = len(data)

    for index in range(total):

        if (
            index
            + TARGET_HORIZON_CANDLES
            >= total
        ):

            targets.append(
                np.nan
            )

            continue

        entry = float(
            data.iloc[index]["close"]
        )

        stop = (
            entry
            * (
                1
                - STOP_LOSS_PCT
            )
        )

        target = (
            entry
            * (
                1
                + TAKE_PROFIT_PCT
            )
        )

        outcome = 0

        future = data.iloc[
            index + 1:
            index
            + TARGET_HORIZON_CANDLES
            + 1
        ]

        for _, candle in future.iterrows():

            low = float(
                candle["low"]
            )

            high = float(
                candle["high"]
            )

            stop_hit = (
                low <= stop
            )

            target_hit = (
                high >= target
            )

            # If both happen in the same
            # 5-minute candle, assume stop first.
            # This is deliberately conservative.
            if stop_hit:
                outcome = 0
                break

            if target_hit:
                outcome = 1
                break

        targets.append(
            outcome
        )

    return targets


# =========================================================
# FEATURES
# =========================================================

def build_features(
    df
):

    data = df.copy()

    data["return_1"] = (
        data["close"]
        .pct_change(1)
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


    # -----------------------------------------------------
    # EMAs
    # -----------------------------------------------------

    data["ema_20"] = (
        data["close"]
        .ewm(
            span=20,
            adjust=False
        )
        .mean()
    )

    data["ema_50"] = (
        data["close"]
        .ewm(
            span=50,
            adjust=False
        )
        .mean()
    )

    data["ema_200"] = (
        data["close"]
        .ewm(
            span=200,
            adjust=False
        )
        .mean()
    )


    data["ema_ratio_20_50"] = (
        data["ema_20"]
        /
        data["ema_50"]
    )

    data["ema_ratio_50_200"] = (
        data["ema_50"]
        /
        data["ema_200"]
    )


    data["price_vs_ema20"] = (
        data["close"]
        /
        data["ema_20"]
    )

    data["price_vs_ema50"] = (
        data["close"]
        /
        data["ema_50"]
    )

    data["price_vs_ema200"] = (
        data["close"]
        /
        data["ema_200"]
    )


    # -----------------------------------------------------
    # VOLATILITY
    # -----------------------------------------------------

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


    # -----------------------------------------------------
    # VOLUME
    # -----------------------------------------------------

    data["volume_ma_20"] = (
        data["volume"]
        .rolling(20)
        .mean()
    )

    data["volume_ratio"] = (
        data["volume"]
        /
        data["volume_ma_20"]
    )


    data["dollar_volume"] = (
        data["volume"]
        * data["close"]
    )

    data["dollar_volume_ma20"] = (
        data["dollar_volume"]
        .rolling(20)
        .mean()
    )

    data["dollar_volume_ma50"] = (
        data["dollar_volume"]
        .rolling(50)
        .mean()
    )

    data["dollar_volume_ratio"] = (
        data["dollar_volume_ma20"]
        /
        data[
            "dollar_volume_ma50"
        ].replace(
            0,
            np.nan
        )
    )


    # -----------------------------------------------------
    # RSI
    # -----------------------------------------------------

    data["rsi"] = calculate_rsi(
        data["close"]
    )


    # -----------------------------------------------------
    # ATR
    # -----------------------------------------------------

    data["atr"] = calculate_atr(
        data
    )

    data["atr_pct"] = (
        data["atr"]
        /
        data["close"]
    )


    # -----------------------------------------------------
    # CANDLE STRUCTURE
    # -----------------------------------------------------

    data["range_pct"] = (
        (
            data["high"]
            - data["low"]
        )
        /
        data["close"]
    )


    candle_range = (
        data["high"]
        - data["low"]
    ).replace(
        0,
        np.nan
    )


    data["close_position"] = (
        (
            data["close"]
            - data["low"]
        )
        /
        candle_range
    )


    # -----------------------------------------------------
    # MODEL TARGET
    # -----------------------------------------------------

    data["target"] = (
        create_trade_target(
            data
        )
    )

    return data


# =========================================================
# MODEL
# =========================================================

def train_model(
    train
):

    X_train = train[
        FEATURE_COLUMNS
    ]

    y_train = (
        train["target"]
        .astype(int)
    )

    if y_train.nunique() < 2:

        raise RuntimeError(
            "Training sample contains "
            "only one target class."
        )


    positive_count = int(
        (
            y_train == 1
        ).sum()
    )


    if positive_count < 10:

        raise RuntimeError(
            "Not enough profitable "
            "historical setups."
        )


    model = RandomForestClassifier(
        n_estimators=250,
        max_depth=8,
        min_samples_leaf=8,
        max_features="sqrt",
        random_state=42,
        class_weight="balanced_subsample",
        n_jobs=1
    )


    model.fit(
        X_train,
        y_train
    )

    return model


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
            model.classes_[index]
        ):
            float(
                probabilities[index]
            )
        for index
        in range(
            len(
                model.classes_
            )
        )
    }

    return mapping.get(
        1,
        0.0
    )


# =========================================================
# ENTRY FILTER
# =========================================================

def passes_entry_filters(
    row
):

    price = float(
        row["close"]
    )

    ema20 = float(
        row["ema_20"]
    )

    ema50 = float(
        row["ema_50"]
    )

    ema200 = float(
        row["ema_200"]
    )

    rsi = float(
        row["rsi"]
    )

    volume_ratio = float(
        row["volume_ratio"]
    )

    atr_pct = float(
        row["atr_pct"]
    )

    dollar_volume = float(
        row["dollar_volume_ma20"]
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


    # Strong bullish structure
    if not (
        price > ema20
        > ema50
        > ema200
    ):
        return False


    # Momentum should be bullish,
    # but not extremely overbought.
    if not (
        MIN_RSI
        <= rsi
        <= MAX_RSI
    ):
        return False


    # Require increased volume.
    if (
        volume_ratio
        < MIN_VOLUME_RATIO
    ):
        return False


    # Avoid dead markets.
    if (
        dollar_volume
        < MIN_DOLLAR_VOLUME_5M
    ):
        return False


    # Avoid markets too quiet
    # to realistically cover fees.
    if (
        atr_pct
        < MIN_ATR_PCT
    ):
        return False


    # Avoid insane volatility.
    if (
        atr_pct
        > MAX_ATR_PCT
    ):
        return False


    # Require recent momentum.
    if return_6 <= 0:
        return False

    if return_12 <= 0:
        return False


    # Prefer candles closing
    # in upper part of their range.
    if (
        close_position
        < MIN_CLOSE_POSITION
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

    entry_probability: float

    stop_loss: float

    take_profit: float


# =========================================================
# TRADE P&L
# =========================================================

def calculate_trade_pnl(
    entry_price,
    exit_price,
    quantity
):

    gross_entry = (
        entry_price
        * quantity
    )

    gross_exit = (
        exit_price
        * quantity
    )

    gross_pnl = (
        gross_exit
        - gross_entry
    )

    entry_fee = (
        gross_entry
        * ESTIMATED_FEE_PER_SIDE
    )

    exit_fee = (
        gross_exit
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
            net_pnl
    }


# =========================================================
# SINGLE-MARKET BACKTEST
# =========================================================

def run_product_backtest(
    product_id,
    days=7
):

    raw = get_historical_candles(
        product_id,
        days=days
    )


    data = build_features(
        raw
    )


    labelled = (
        data.dropna(
            subset=(
                FEATURE_COLUMNS
                + [
                    "target",
                    "ema_200",
                    "atr_pct",
                    "dollar_volume_ma20"
                ]
            )
        )
        .reset_index(
            drop=True
        )
    )


    if len(labelled) < 400:

        raise RuntimeError(
            "Not enough usable history"
        )


    # -----------------------------------------------------
    # TRAIN / TEST SPLIT
    # -----------------------------------------------------

    split = int(
        len(labelled)
        * 0.70
    )


    # Purge target horizon from
    # the end of training data.
    #
    # Prevents training labels from
    # looking into test candles.
    train_end = (
        split
        - TARGET_HORIZON_CANDLES
    )


    if train_end <= 0:

        raise RuntimeError(
            "Not enough training history"
        )


    train = labelled.iloc[
        :train_end
    ].copy()


    test = (
        labelled.iloc[
            split:
        ]
        .reset_index(
            drop=True
        )
    )


    if len(test) < 75:

        raise RuntimeError(
            "Not enough unseen "
            "test candles"
        )


    model = train_model(
        train
    )


    # -----------------------------------------------------
    # SIMULATION
    # -----------------------------------------------------

    position = None

    trades = []

    equity = 0.0

    gross_equity = 0.0

    total_fees = 0.0

    equity_curve = [
        0.0
    ]


    cooldown_until_index = -1


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


        # -------------------------------------------------
        # MANAGE POSITION
        # -------------------------------------------------

        if position is not None:

            exit_price = None
            reason = None


            stop_hit = (
                low
                <= position.stop_loss
            )

            target_hit = (
                high
                >= position.take_profit
            )


            # Conservative assumption:
            # if both happen during same
            # five-minute candle, stop first.
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


            if exit_price is not None:

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


                equity += (
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

                    "entry_time":
                        position.entry_time,

                    "exit_time":
                        timestamp
                })


                position = None


                cooldown_until_index = (
                    index
                    + COOLDOWN_CANDLES
                )


        # -------------------------------------------------
        # LOOK FOR NEW ENTRY
        # -------------------------------------------------

        if position is None:

            if index < cooldown_until_index:

                equity_curve.append(
                    equity
                )

                continue


            if not passes_entry_filters(
                row
            ):

                equity_curve.append(
                    equity
                )

                continue


            up_probability = (
                probability_up(
                    model,
                    row
                )
            )


            if (
                up_probability
                >= BUY_THRESHOLD
            ):

                quantity = (
                    TRADE_SIZE
                    / price
                )


                position = Position(
                    entry_price=price,

                    quantity=quantity,

                    entry_time=timestamp,

                    entry_probability=(
                        up_probability
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
            equity
        )


    # -----------------------------------------------------
    # CLOSE OPEN POSITION AT END
    # -----------------------------------------------------

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


        equity += (
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

            "entry_time":
                position.entry_time,

            "exit_time":
                int(
                    final_row["time"]
                )
        })


        equity_curve.append(
            equity
        )


    # =====================================================
    # STATISTICS
    # =====================================================

    wins_list = [
        trade["pnl"]
        for trade in trades
        if trade["pnl"] > 0
    ]


    losses_list = [
        trade["pnl"]
        for trade in trades
        if trade["pnl"] <= 0
    ]


    wins = len(
        wins_list
    )


    losses = len(
        losses_list
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
        sum(wins_list)
        / len(wins_list)
        if wins_list
        else 0.0
    )


    average_loss = (
        sum(losses_list)
        / len(losses_list)
        if losses_list
        else 0.0
    )


    gross_profit = sum(
        wins_list
    )


    gross_loss = abs(
        sum(
            losses_list
        )
    )


    profit_factor = (
        gross_profit
        / gross_loss
        if gross_loss > 0
        else (
            float("inf")
            if gross_profit > 0
            else 0.0
        )
    )


    expectancy = (
        equity
        / total_trades
        if total_trades
        else 0.0
    )


    # -----------------------------------------------------
    # DRAWDOWN
    # -----------------------------------------------------

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

        "test_candles":
            len(test),

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
            equity,

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
            trades
    }


# =========================================================
# ALL-MARKET BACKTEST
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
        f"V2 BACKTEST: discovered "
        f"{len(discovered)} "
        f"Coinbase USD markets."
    )


    for index, product_id in enumerate(
        discovered,
        start=1
    ):

        print(
            f"V2 BACKTEST "
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
                f"{product_id}: {exc}"
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
    # COMBINED STATISTICS
    # =====================================================

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
            item["trade_log"]
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
        sum(wins_pnl)
        / len(wins_pnl)
        if wins_pnl
        else 0.0
    )


    average_loss = (
        sum(losses_pnl)
        / len(losses_pnl)
        if losses_pnl
        else 0.0
    )


    gross_profit = sum(
        wins_pnl
    )


    gross_loss = abs(
        sum(
            losses_pnl
        )
    )


    profit_factor = (
        gross_profit
        / gross_loss
        if gross_loss > 0
        else (
            float("inf")
            if gross_profit > 0
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
            item["pnl"],
        reverse=True
    )


    best = ranked[0]

    worst = ranked[-1]


    worst_drawdown = min(
        item["max_drawdown"]
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
            "V2",

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
            best["product"],

        "best_market_pnl":
            best["pnl"],

        "worst_market":
            worst["product"],

        "worst_market_pnl":
            worst["pnl"],

        "fee_per_side":
            ESTIMATED_FEE_PER_SIDE,

        "buy_threshold":
            BUY_THRESHOLD,

        "stop_loss_pct":
            STOP_LOSS_PCT,

        "take_profit_pct":
            TAKE_PROFIT_PCT,

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
            errors[:10]
    }


# =========================================================
# MANUAL TEST
# =========================================================

if __name__ == "__main__":

    result = run_backtest(
        days=7
    )

    print(
        json.dumps(
            result,
            indent=2,
            default=str
        )
    )