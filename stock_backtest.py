import json
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
import yfinance as yf

from sklearn.ensemble import (
    RandomForestClassifier,
    RandomForestRegressor,
)


# =========================================================
# STOCK AI BACKTEST V2
# =========================================================

STRATEGY_NAME = "STOCK_V2"

# V2 moves from 5-minute bars to hourly bars so the model
# can train on substantially more history and trade less.
INTERVAL = "1h"

DEFAULT_TEST_DAYS = int(
    os.getenv(
        "STOCK_BACKTEST_DAYS",
        "30"
    )
)

TRAINING_LOOKBACK_DAYS = int(
    os.getenv(
        "STOCK_TRAINING_LOOKBACK_DAYS",
        "180"
    )
)

MAX_TOTAL_DAYS = int(
    os.getenv(
        "STOCK_MAX_TOTAL_DAYS",
        "365"
    )
)


# =========================================================
# MARKET UNIVERSE
# =========================================================

SYMBOLS = [
    symbol.strip().upper()
    for symbol in os.getenv(
        "STOCK_AI_SYMBOLS",
        (
            "AAPL,MSFT,NVDA,AMZN,META,GOOGL,TSLA,"
            "AMD,AVGO,NFLX,PLTR,COIN,MSTR,"
            "JPM,BAC,GS,XOM,CVX,LLY,UNH,WMT,COST,"
            "SPY,QQQ,IWM"
        )
    ).split(",")
    if symbol.strip()
]

BENCHMARK_SYMBOL = os.getenv(
    "STOCK_BENCHMARK_SYMBOL",
    "SPY"
).upper()


# =========================================================
# CORE ENTRY / RISK SETTINGS
# =========================================================

BUY_THRESHOLD = float(
    os.getenv(
        "STOCK_BUY_THRESHOLD",
        "0.64"
    )
)

# V1 was 1.0% / 2.25%.
# V2 gives winners more room while still controlling risk.
STOP_LOSS_PCT = float(
    os.getenv(
        "STOCK_STOP_LOSS_PCT",
        "0.0125"
    )
)

TAKE_PROFIT_PCT = float(
    os.getenv(
        "STOCK_TAKE_PROFIT_PCT",
        "0.030"
    )
)

TRADE_SIZE = float(
    os.getenv(
        "STOCK_TRADE_SIZE",
        "100.0"
    )
)

# Costs are modelled explicitly.
COMMISSION_PER_SIDE = float(
    os.getenv(
        "STOCK_COMMISSION_PER_SIDE",
        "0.0"
    )
)

SLIPPAGE_PER_SIDE = float(
    os.getenv(
        "STOCK_SLIPPAGE_PER_SIDE",
        "0.0005"
    )
)


# =========================================================
# HORIZON / TURNOVER CONTROL
# =========================================================

# 8 hourly bars = roughly one normal US trading day.
TARGET_HORIZON_BARS = int(
    os.getenv(
        "STOCK_TARGET_HORIZON_BARS",
        "8"
    )
)

MAX_HOLD_BARS = int(
    os.getenv(
        "STOCK_MAX_HOLD_BARS",
        "10"
    )
)

# Reduce repeat entries.
COOLDOWN_BARS = int(
    os.getenv(
        "STOCK_COOLDOWN_BARS",
        "4"
    )
)


# =========================================================
# FEE-AWARE EDGE REQUIREMENTS
# =========================================================

MIN_PREDICTED_NET_RETURN = float(
    os.getenv(
        "STOCK_MIN_PREDICTED_NET_RETURN",
        "0.0025"
    )
)

# The predicted return must exceed estimated round-trip
# execution costs by this multiple.
MIN_COST_EDGE_MULTIPLIER = float(
    os.getenv(
        "STOCK_MIN_COST_EDGE_MULTIPLIER",
        "2.0"
    )
)


# =========================================================
# TRAINING REQUIREMENTS
# =========================================================

MIN_TRAINING_ROWS = int(
    os.getenv(
        "STOCK_MIN_TRAINING_ROWS",
        "700"
    )
)

MIN_POSITIVE_EXAMPLES = int(
    os.getenv(
        "STOCK_MIN_POSITIVE_EXAMPLES",
        "30"
    )
)


# =========================================================
# MARKET / ENTRY FILTERS
# =========================================================

MIN_DOLLAR_VOLUME_1H = float(
    os.getenv(
        "STOCK_MIN_DOLLAR_VOLUME_1H",
        "10000000"
    )
)

MIN_VOLUME_RATIO = float(
    os.getenv(
        "STOCK_MIN_VOLUME_RATIO",
        "0.70"
    )
)

MIN_ATR_PCT = float(
    os.getenv(
        "STOCK_MIN_ATR_PCT",
        "0.002"
    )
)

MAX_ATR_PCT = float(
    os.getenv(
        "STOCK_MAX_ATR_PCT",
        "0.060"
    )
)

MIN_RSI = float(
    os.getenv(
        "STOCK_MIN_RSI",
        "42"
    )
)

MAX_RSI = float(
    os.getenv(
        "STOCK_MAX_RSI",
        "74"
    )
)

MIN_RELATIVE_RETURN_8 = float(
    os.getenv(
        "STOCK_MIN_RELATIVE_RETURN_8",
        "-0.015"
    )
)


# =========================================================
# OPTIONAL SYMBOL QUALITY GATE
# =========================================================

MIN_SYMBOL_QUALITY_SAMPLES = int(
    os.getenv(
        "STOCK_MIN_SYMBOL_QUALITY_SAMPLES",
        "30"
    )
)

MIN_SYMBOL_NET_SUCCESS_RATE = float(
    os.getenv(
        "STOCK_MIN_SYMBOL_NET_SUCCESS_RATE",
        "0.40"
    )
)


# =========================================================
# FEATURES
# =========================================================

FEATURE_COLUMNS = [
    "return_1",
    "return_2",
    "return_4",
    "return_8",
    "return_16",

    "ema_ratio_5_10",
    "ema_ratio_10_20",
    "price_vs_ema10",
    "price_vs_ema20",

    "volatility_4",
    "volatility_8",
    "volatility_16",

    "volume_ratio",
    "volume_zscore",

    "rsi",
    "atr_pct",

    "range_pct",
    "body_pct",
    "close_position",

    "benchmark_return_4",
    "benchmark_return_8",
    "relative_return_4",
    "relative_return_8",

    "market_above_ema20",
    "stock_above_ema20",
]


# =========================================================
# DATA DOWNLOAD
# =========================================================

def download_intraday(
    symbol,
    total_days
):

    period_days = min(
        int(total_days),
        MAX_TOTAL_DAYS
    )

    df = yf.download(
        symbol,
        period=f"{period_days}d",
        interval=INTERVAL,
        auto_adjust=False,
        progress=False,
        prepost=False,
        threads=False,
    )

    if df is None or df.empty:
        raise RuntimeError(
            f"No hourly data returned for {symbol}"
        )

    if isinstance(
        df.columns,
        pd.MultiIndex
    ):

        if symbol in df.columns.get_level_values(-1):

            df = df.xs(
                symbol,
                axis=1,
                level=-1
            )

        else:

            df.columns = [
                col[0]
                if isinstance(col, tuple)
                else col
                for col in df.columns
            ]

    df = df.rename(
        columns={
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Adj Close": "adj_close",
            "Volume": "volume",
        }
    )

    required = [
        "open",
        "high",
        "low",
        "close",
        "volume",
    ]

    missing = [
        column
        for column in required
        if column not in df.columns
    ]

    if missing:
        raise RuntimeError(
            f"{symbol} missing columns: {missing}"
        )

    df = (
        df[
            required
        ]
        .dropna()
        .copy()
        .reset_index()
    )

    time_column = (
        "Datetime"
        if "Datetime" in df.columns
        else "Date"
    )

    if time_column not in df.columns:
        time_column = df.columns[0]

    df["time"] = pd.to_datetime(
        df[time_column],
        utc=True
    )

    return (
        df[
            [
                "time",
                "open",
                "high",
                "low",
                "close",
                "volume",
            ]
        ]
        .sort_values("time")
        .drop_duplicates(
            subset=["time"]
        )
        .reset_index(drop=True)
    )


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
        / avg_loss.replace(
            0,
            np.nan
        )
    )

    return (
        100
        - 100
        / (
            1 + rs
        )
    ).fillna(50)


def calculate_atr(
    data,
    period=14
):

    previous_close = (
        data["close"]
        .shift(1)
    )

    true_range = pd.concat(
        [
            data["high"] - data["low"],
            (
                data["high"]
                - previous_close
            ).abs(),
            (
                data["low"]
                - previous_close
            ).abs(),
        ],
        axis=1
    ).max(axis=1)

    return true_range.ewm(
        alpha=1 / period,
        adjust=False
    ).mean()


# =========================================================
# FEATURE ENGINEERING
# =========================================================

def build_feature_frame(
    stock_df,
    benchmark_df
):

    data = stock_df.copy()

    benchmark = (
        benchmark_df[
            [
                "time",
                "close",
            ]
        ]
        .rename(
            columns={
                "close":
                    "benchmark_close"
            }
        )
    )

    data = pd.merge_asof(
        data.sort_values("time"),
        benchmark.sort_values("time"),
        on="time",
        direction="backward",
        tolerance=pd.Timedelta(
            hours=2
        )
    )

    for n in (
        1,
        2,
        4,
        8,
        16
    ):

        data[
            f"return_{n}"
        ] = (
            data["close"]
            .pct_change(n)
        )

    data["benchmark_return_4"] = (
        data["benchmark_close"]
        .pct_change(4)
    )

    data["benchmark_return_8"] = (
        data["benchmark_close"]
        .pct_change(8)
    )

    data["relative_return_4"] = (
        data["return_4"]
        - data["benchmark_return_4"]
    )

    data["relative_return_8"] = (
        data["return_8"]
        - data["benchmark_return_8"]
    )

    data["ema_5"] = (
        data["close"]
        .ewm(
            span=5,
            adjust=False
        )
        .mean()
    )

    data["ema_10"] = (
        data["close"]
        .ewm(
            span=10,
            adjust=False
        )
        .mean()
    )

    data["ema_20"] = (
        data["close"]
        .ewm(
            span=20,
            adjust=False
        )
        .mean()
    )

    data["ema_ratio_5_10"] = (
        data["ema_5"]
        / data["ema_10"]
    )

    data["ema_ratio_10_20"] = (
        data["ema_10"]
        / data["ema_20"]
    )

    data["price_vs_ema10"] = (
        data["close"]
        / data["ema_10"]
    )

    data["price_vs_ema20"] = (
        data["close"]
        / data["ema_20"]
    )

    data["benchmark_ema20"] = (
        data["benchmark_close"]
        .ewm(
            span=20,
            adjust=False
        )
        .mean()
    )

    data["market_above_ema20"] = (
        data["benchmark_close"]
        > data["benchmark_ema20"]
    ).astype(float)

    data["stock_above_ema20"] = (
        data["close"]
        > data["ema_20"]
    ).astype(float)

    data["volatility_4"] = (
        data["return_1"]
        .rolling(4)
        .std()
    )

    data["volatility_8"] = (
        data["return_1"]
        .rolling(8)
        .std()
    )

    data["volatility_16"] = (
        data["return_1"]
        .rolling(16)
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
        / data["volume_ma_20"]
        .replace(
            0,
            np.nan
        )
    )

    data["volume_zscore"] = (
        (
            data["volume"]
            - data["volume_ma_20"]
        )
        / data["volume_std_20"]
        .replace(
            0,
            np.nan
        )
    )

    data["rsi"] = calculate_rsi(
        data["close"]
    )

    data["atr"] = calculate_atr(
        data
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
# ENTRY FILTER
# =========================================================

def passes_entry_filter(row):

    if (
        float(
            row[
                "dollar_volume_ma20"
            ]
        )
        < MIN_DOLLAR_VOLUME_1H
    ):
        return False

    if (
        float(
            row[
                "volume_ratio"
            ]
        )
        < MIN_VOLUME_RATIO
    ):
        return False

    atr_pct = float(
        row["atr_pct"]
    )

    if not (
        MIN_ATR_PCT
        <= atr_pct
        <= MAX_ATR_PCT
    ):
        return False

    rsi = float(
        row["rsi"]
    )

    if not (
        MIN_RSI
        <= rsi
        <= MAX_RSI
    ):
        return False

    if (
        float(
            row[
                "relative_return_8"
            ]
        )
        < MIN_RELATIVE_RETURN_8
    ):
        return False

    # Avoid obvious downtrends.
    if (
        float(
            row[
                "price_vs_ema20"
            ]
        )
        < 0.985
    ):
        return False

    return True


# =========================================================
# COST-AWARE FORWARD SIMULATION
# =========================================================

def round_trip_cost_return(
    exit_ratio=1.0
):

    return (
        COMMISSION_PER_SIDE
        + COMMISSION_PER_SIDE
        + SLIPPAGE_PER_SIDE
        + (
            exit_ratio
            * SLIPPAGE_PER_SIDE
        )
    )


def simulate_forward_trade(
    data,
    index
):

    final_index = (
        index
        + TARGET_HORIZON_BARS
    )

    if final_index >= len(data):
        return None

    entry = float(
        data.iloc[
            index
        ]["close"]
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

    exit_price = None

    for future_index in range(
        index + 1,
        final_index + 1
    ):

        future_row = (
            data.iloc[
                future_index
            ]
        )

        low = float(
            future_row["low"]
        )

        high = float(
            future_row["high"]
        )

        # Conservative ordering if both are touched.
        if low <= stop:

            exit_price = stop
            break

        if high >= target:

            exit_price = target
            break

    if exit_price is None:

        exit_price = float(
            data.iloc[
                final_index
            ]["close"]
        )

    exit_ratio = (
        exit_price
        / entry
    )

    gross_return = (
        exit_ratio
        - 1
    )

    costs = round_trip_cost_return(
        exit_ratio
    )

    net_return = (
        gross_return
        - costs
    )

    return {
        "target":
            (
                1
                if net_return > 0
                else 0
            ),

        "net_return":
            net_return,
    }


# =========================================================
# TRAINING DATA
# =========================================================

def build_training_data(
    feature_data
):

    data = feature_data.copy()

    targets = np.full(
        len(data),
        np.nan
    )

    net_returns = np.full(
        len(data),
        np.nan
    )

    for index in range(
        len(data)
    ):

        result = simulate_forward_trade(
            data,
            index
        )

        if result is None:
            continue

        targets[index] = (
            result["target"]
        )

        net_returns[index] = (
            result["net_return"]
        )

    data[
        "target_success"
    ] = targets

    data[
        "target_net_return"
    ] = net_returns

    data = (
        data.dropna(
            subset=(
                FEATURE_COLUMNS
                + [
                    "target_success",
                    "target_net_return",
                ]
            )
        )
        .reset_index(
            drop=True
        )
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
# SYMBOL QUALITY
# =========================================================

def calculate_symbol_quality(
    training_data
):

    filtered = []

    for _, row in (
        training_data.iterrows()
    ):

        try:

            if passes_entry_filter(
                row
            ):

                filtered.append(
                    row
                )

        except Exception:
            continue

    if (
        len(filtered)
        < MIN_SYMBOL_QUALITY_SAMPLES
    ):

        return {
            "samples":
                len(filtered),

            "success_rate":
                None,

            "average_net_return":
                None,

            "passes":
                True,
        }

    filtered_df = pd.DataFrame(
        filtered
    )

    success_rate = float(
        filtered_df[
            "target_success"
        ].mean()
    )

    average_net_return = float(
        filtered_df[
            "target_net_return"
        ].mean()
    )

    return {
        "samples":
            len(filtered_df),

        "success_rate":
            success_rate,

        "average_net_return":
            average_net_return,

        "passes":
            (
                success_rate
                >= MIN_SYMBOL_NET_SUCCESS_RATE
            ),
    }


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
            f"Not enough training rows: "
            f"{len(training_data)}"
        )

    y_class = (
        training_data[
            "target_success"
        ]
    )

    y_return = (
        training_data[
            "target_net_return"
        ]
    )

    if (
        y_class.nunique()
        < 2
    ):

        raise RuntimeError(
            "Training data contains only "
            "one target class."
        )

    positive_count = int(
        (
            y_class == 1
        ).sum()
    )

    if (
        positive_count
        < MIN_POSITIVE_EXAMPLES
    ):

        raise RuntimeError(
            "Not enough positive "
            "training examples."
        )

    X = (
        training_data[
            FEATURE_COLUMNS
        ]
    )

    classifier = (
        RandomForestClassifier(
            n_estimators=500,
            max_depth=9,
            min_samples_leaf=10,
            max_features="sqrt",
            random_state=42,
            class_weight=(
                "balanced_subsample"
            ),
            n_jobs=-1
        )
    )

    regressor = (
        RandomForestRegressor(
            n_estimators=450,
            max_depth=9,
            min_samples_leaf=10,
            max_features="sqrt",
            random_state=43,
            n_jobs=-1
        )
    )

    classifier.fit(
        X,
        y_class
    )

    regressor.fit(
        X,
        y_return
    )

    return (
        classifier,
        regressor
    )


def probability_success(
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


def predict_net_return(
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
# POSITION
# =========================================================

@dataclass
class Position:

    entry_price: float
    quantity: float

    entry_time: str
    entry_index: int

    entry_probability: float
    predicted_net_return: float

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

    commission = (
        entry_value
        * COMMISSION_PER_SIDE
        + exit_value
        * COMMISSION_PER_SIDE
    )

    slippage = (
        entry_value
        * SLIPPAGE_PER_SIDE
        + exit_value
        * SLIPPAGE_PER_SIDE
    )

    costs = (
        commission
        + slippage
    )

    return {
        "gross_pnl":
            gross_pnl,

        "fees":
            costs,

        "net_pnl":
            gross_pnl
            - costs,
    }


# =========================================================
# CONFIDENCE BUCKET
# =========================================================

def confidence_bucket(
    probability
):

    if probability < 0.65:
        return "64-65"

    if probability < 0.70:
        return "65-70"

    if probability < 0.75:
        return "70-75"

    return "75+"


# =========================================================
# SINGLE SYMBOL BACKTEST
# =========================================================

def run_symbol_backtest(
    symbol,
    stock_df,
    benchmark_df,
    test_days
):

    data = build_feature_frame(
        stock_df,
        benchmark_df
    )

    latest_time = (
        data["time"]
        .max()
    )

    test_start = (
        latest_time
        - pd.Timedelta(
            days=test_days
        )
    )

    training_frame = (
        data[
            data["time"]
            < test_start
        ]
        .copy()
        .reset_index(
            drop=True
        )
    )

    training_data = build_training_data(
        training_frame
    )

    symbol_quality = calculate_symbol_quality(
        training_data
    )

    if not symbol_quality[
        "passes"
    ]:

        raise RuntimeError(
            (
                "Symbol quality below threshold: "
                f"{symbol_quality['success_rate']:.1%}"
            )
        )

    (
        classifier,
        regressor,
    ) = train_models(
        training_data
    )

    required_test_columns = (
        FEATURE_COLUMNS
        + [
            "ema_10",
            "ema_20",
            "dollar_volume_ma20",
        ]
    )

    test = (
        data[
            data["time"]
            >= test_start
        ]
        .dropna(
            subset=required_test_columns
        )
        .reset_index(
            drop=True
        )
    )

    if len(test) < 40:

        raise RuntimeError(
            "Not enough unseen test bars."
        )

    position = None
    trades = []

    gross_equity = 0.0
    total_costs = 0.0
    net_equity = 0.0

    equity_curve = [0.0]

    cooldown_until = -1

    signals_checked = 0
    classifier_passes = 0
    edge_passes = 0

    confidence_stats = {
        "64-65": {
            "trades": 0,
            "wins": 0,
            "pnl": 0.0,
        },
        "65-70": {
            "trades": 0,
            "wins": 0,
            "pnl": 0.0,
        },
        "70-75": {
            "trades": 0,
            "wins": 0,
            "pnl": 0.0,
        },
        "75+": {
            "trades": 0,
            "wins": 0,
            "pnl": 0.0,
        },
    }

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

        timestamp = str(
            row["time"]
        )

        # =====================================
        # MANAGE POSITION
        # =====================================

        if position is not None:

            exit_price = None
            reason = None

            if (
                low
                <= position.stop_loss
            ):

                exit_price = (
                    position.stop_loss
                )

                reason = (
                    "STOP LOSS"
                )

            elif (
                high
                >= position.take_profit
            ):

                exit_price = (
                    position.take_profit
                )

                reason = (
                    "TAKE PROFIT"
                )

            elif (
                index
                - position.entry_index
                >= MAX_HOLD_BARS
            ):

                exit_price = price

                reason = (
                    "MAX HOLD"
                )

            if (
                exit_price
                is not None
            ):

                pnl = calculate_trade_pnl(
                    position.entry_price,
                    exit_price,
                    position.quantity
                )

                gross_equity += (
                    pnl["gross_pnl"]
                )

                total_costs += (
                    pnl["fees"]
                )

                net_equity += (
                    pnl["net_pnl"]
                )

                bucket = confidence_bucket(
                    position.entry_probability
                )

                confidence_stats[
                    bucket
                ][
                    "trades"
                ] += 1

                if (
                    pnl["net_pnl"]
                    > 0
                ):

                    confidence_stats[
                        bucket
                    ][
                        "wins"
                    ] += 1

                confidence_stats[
                    bucket
                ][
                    "pnl"
                ] += (
                    pnl["net_pnl"]
                )

                trades.append({
                    "symbol":
                        symbol,

                    "entry_price":
                        position.entry_price,

                    "exit_price":
                        exit_price,

                    "pnl":
                        pnl["net_pnl"],

                    "gross_pnl":
                        pnl["gross_pnl"],

                    "fees":
                        pnl["fees"],

                    "reason":
                        reason,

                    "entry_probability":
                        position.entry_probability,

                    "predicted_net_return":
                        position.predicted_net_return,

                    "entry_time":
                        position.entry_time,

                    "exit_time":
                        timestamp,
                })

                position = None

                cooldown_until = (
                    index
                    + COOLDOWN_BARS
                )

        # =====================================
        # SEARCH FOR ENTRY
        # =====================================

        if position is None:

            if (
                index
                < cooldown_until
            ):

                equity_curve.append(
                    net_equity
                )

                continue

            if not passes_entry_filter(
                row
            ):

                equity_curve.append(
                    net_equity
                )

                continue

            signals_checked += 1

            probability = probability_success(
                classifier,
                row
            )

            if (
                probability
                < BUY_THRESHOLD
            ):

                equity_curve.append(
                    net_equity
                )

                continue

            classifier_passes += 1

            predicted_net = predict_net_return(
                regressor,
                row
            )

            if (
                predicted_net
                < MIN_PREDICTED_NET_RETURN
            ):

                equity_curve.append(
                    net_equity
                )

                continue

            expected_cost = (
                round_trip_cost_return()
            )

            if (
                predicted_net
                < (
                    expected_cost
                    * MIN_COST_EDGE_MULTIPLIER
                )
            ):

                equity_curve.append(
                    net_equity
                )

                continue

            edge_passes += 1

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
                    probability
                ),

                predicted_net_return=(
                    predicted_net
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
                ),
            )

        equity_curve.append(
            net_equity
        )

    # =========================================
    # CLOSE ANY OPEN POSITION
    # =========================================

    if (
        position is not None
        and len(test) > 0
    ):

        final_row = test.iloc[-1]

        exit_price = float(
            final_row["close"]
        )

        pnl = calculate_trade_pnl(
            position.entry_price,
            exit_price,
            position.quantity
        )

        gross_equity += (
            pnl["gross_pnl"]
        )

        total_costs += (
            pnl["fees"]
        )

        net_equity += (
            pnl["net_pnl"]
        )

        bucket = confidence_bucket(
            position.entry_probability
        )

        confidence_stats[
            bucket
        ][
            "trades"
        ] += 1

        if (
            pnl["net_pnl"]
            > 0
        ):

            confidence_stats[
                bucket
            ][
                "wins"
            ] += 1

        confidence_stats[
            bucket
        ][
            "pnl"
        ] += (
            pnl["net_pnl"]
        )

        trades.append({
            "symbol":
                symbol,

            "entry_price":
                position.entry_price,

            "exit_price":
                exit_price,

            "pnl":
                pnl["net_pnl"],

            "gross_pnl":
                pnl["gross_pnl"],

            "fees":
                pnl["fees"],

            "reason":
                "END OF TEST",

            "entry_probability":
                position.entry_probability,

            "predicted_net_return":
                position.predicted_net_return,

            "entry_time":
                position.entry_time,

            "exit_time":
                str(
                    final_row["time"]
                ),
        })

        equity_curve.append(
            net_equity
        )

    wins = [
        trade["pnl"]
        for trade in trades
        if trade["pnl"] > 0
    ]

    losses = [
        trade["pnl"]
        for trade in trades
        if trade["pnl"] <= 0
    ]

    total_trades = len(
        trades
    )

    peak = 0.0
    max_drawdown = 0.0

    for value in equity_curve:

        peak = max(
            peak,
            value
        )

        max_drawdown = min(
            max_drawdown,
            value - peak
        )

    winning_pnl = sum(
        wins
    )

    losing_pnl = abs(
        sum(
            losses
        )
    )

    profit_factor = (
        winning_pnl
        / losing_pnl
        if losing_pnl > 0
        else (
            float("inf")
            if winning_pnl > 0
            else 0.0
        )
    )

    return {
        "symbol":
            symbol,

        "training_rows":
            len(training_data),

        "symbol_quality_samples":
            symbol_quality["samples"],

        "symbol_quality_success_rate":
            symbol_quality["success_rate"],

        "symbol_quality_average_net_return":
            symbol_quality[
                "average_net_return"
            ],

        "test_bars":
            len(test),

        "signals_checked":
            signals_checked,

        "classifier_passes":
            classifier_passes,

        "edge_passes":
            edge_passes,

        "trades":
            total_trades,

        "wins":
            len(wins),

        "losses":
            len(losses),

        "win_rate":
            (
                len(wins)
                / total_trades
                if total_trades
                else 0.0
            ),

        "gross_pnl":
            gross_equity,

        "fees":
            total_costs,

        "pnl":
            net_equity,

        "average_win":
            (
                sum(wins)
                / len(wins)
                if wins
                else 0.0
            ),

        "average_loss":
            (
                sum(losses)
                / len(losses)
                if losses
                else 0.0
            ),

        "profit_factor":
            profit_factor,

        "expectancy":
            (
                net_equity
                / total_trades
                if total_trades
                else 0.0
            ),

        "max_drawdown":
            max_drawdown,

        "confidence_stats":
            confidence_stats,

        "trade_log":
            trades,
    }


# =========================================================
# COMPLETE STOCK BACKTEST
# =========================================================

def run_stock_backtest(
    days=DEFAULT_TEST_DAYS
):

    days = int(
        max(
            10,
            min(
                days,
                60
            )
        )
    )

    total_days = (
        days
        + TRAINING_LOOKBACK_DAYS
    )

    total_days = min(
        total_days,
        MAX_TOTAL_DAYS
    )

    effective_training_days = (
        total_days
        - days
    )

    print(
        f"{STRATEGY_NAME}: "
        f"{days} unseen test days, "
        f"~{effective_training_days} "
        f"training days."
    )

    print(
        f"Interval: {INTERVAL}"
    )

    print(
        f"Buy threshold: "
        f"{BUY_THRESHOLD:.1%}"
    )

    print(
        f"Minimum predicted NET return: "
        f"{MIN_PREDICTED_NET_RETURN:.2%}"
    )

    benchmark_df = download_intraday(
        BENCHMARK_SYMBOL,
        total_days
    )

    results = []
    errors = []

    for index, symbol in enumerate(
        SYMBOLS,
        start=1
    ):

        print(
            f"{STRATEGY_NAME} "
            f"{index}/{len(SYMBOLS)}: "
            f"{symbol}"
        )

        try:

            stock_df = (
                benchmark_df.copy()
                if symbol == BENCHMARK_SYMBOL
                else download_intraday(
                    symbol,
                    total_days
                )
            )

            result = run_symbol_backtest(
                symbol,
                stock_df,
                benchmark_df,
                days
            )

            results.append(
                result
            )

            print(
                f"{symbol}: "
                f"{result['trades']} trades | "
                f"{result['win_rate'] * 100:.1f}% WR | "
                f"GBP {result['pnl']:+.2f}"
            )

        except Exception as exc:

            errors.append(
                f"{symbol}: {exc}"
            )

            print(
                f"SKIP {symbol}: {exc}"
            )

    if not results:

        raise RuntimeError(
            "Stock V2 backtest failed for "
            "every configured symbol."
        )

    all_trades = []

    for item in results:

        all_trades.extend(
            item["trade_log"]
        )

    total_trades = len(
        all_trades
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

    total_costs = sum(
        item["fees"]
        for item in results
    )

    total_pnl = sum(
        item["pnl"]
        for item in results
    )

    wins = [
        trade["pnl"]
        for trade in all_trades
        if trade["pnl"] > 0
    ]

    losses = [
        trade["pnl"]
        for trade in all_trades
        if trade["pnl"] <= 0
    ]

    winning_pnl = sum(
        wins
    )

    losing_pnl = abs(
        sum(
            losses
        )
    )

    profit_factor = (
        winning_pnl
        / losing_pnl
        if losing_pnl > 0
        else (
            float("inf")
            if winning_pnl > 0
            else 0.0
        )
    )

    ranked = sorted(
        results,
        key=lambda item:
            item["pnl"],
        reverse=True
    )

    combined_confidence = {
        "64-65": {
            "trades": 0,
            "wins": 0,
            "pnl": 0.0,
        },
        "65-70": {
            "trades": 0,
            "wins": 0,
            "pnl": 0.0,
        },
        "70-75": {
            "trades": 0,
            "wins": 0,
            "pnl": 0.0,
        },
        "75+": {
            "trades": 0,
            "wins": 0,
            "pnl": 0.0,
        },
    }

    for item in results:

        for bucket, stats in (
            item[
                "confidence_stats"
            ].items()
        ):

            combined_confidence[
                bucket
            ][
                "trades"
            ] += (
                stats["trades"]
            )

            combined_confidence[
                bucket
            ][
                "wins"
            ] += (
                stats["wins"]
            )

            combined_confidence[
                bucket
            ][
                "pnl"
            ] += (
                stats["pnl"]
            )

    confidence_report = []

    for bucket, stats in (
        combined_confidence.items()
    ):

        trades = (
            stats["trades"]
        )

        wins_count = (
            stats["wins"]
        )

        confidence_report.append({
            "bucket":
                bucket,

            "trades":
                trades,

            "wins":
                wins_count,

            "win_rate":
                (
                    wins_count
                    / trades
                    if trades
                    else 0.0
                ),

            "pnl":
                stats["pnl"],
        })

    return {
        "strategy":
            STRATEGY_NAME,

        "interval":
            INTERVAL,

        "days":
            days,

        "training_days":
            effective_training_days,

        "symbols_configured":
            len(SYMBOLS),

        "symbols_completed":
            len(results),

        "symbols_skipped":
            len(errors),

        "signals_checked":
            sum(
                item[
                    "signals_checked"
                ]
                for item in results
            ),

        "classifier_passes":
            sum(
                item[
                    "classifier_passes"
                ]
                for item in results
            ),

        "edge_passes":
            sum(
                item[
                    "edge_passes"
                ]
                for item in results
            ),

        "trades":
            total_trades,

        "trades_per_day":
            (
                total_trades
                / days
                if days
                else 0.0
            ),

        "wins":
            total_wins,

        "losses":
            total_losses,

        "win_rate":
            (
                total_wins
                / total_trades
                if total_trades
                else 0.0
            ),

        "gross_pnl":
            total_gross_pnl,

        "fees":
            total_costs,

        "pnl":
            total_pnl,

        "average_win":
            (
                sum(wins)
                / len(wins)
                if wins
                else 0.0
            ),

        "average_loss":
            (
                sum(losses)
                / len(losses)
                if losses
                else 0.0
            ),

        "profit_factor":
            profit_factor,

        "expectancy":
            (
                total_pnl
                / total_trades
                if total_trades
                else 0.0
            ),

        "max_drawdown":
            min(
                item[
                    "max_drawdown"
                ]
                for item in results
            ),

        "best_symbol":
            ranked[0]["symbol"],

        "best_symbol_pnl":
            ranked[0]["pnl"],

        "worst_symbol":
            ranked[-1]["symbol"],

        "worst_symbol_pnl":
            ranked[-1]["pnl"],

        "buy_threshold":
            BUY_THRESHOLD,

        "min_predicted_net_return":
            MIN_PREDICTED_NET_RETURN,

        "stop_loss_pct":
            STOP_LOSS_PCT,

        "take_profit_pct":
            TAKE_PROFIT_PCT,

        "commission_per_side":
            COMMISSION_PER_SIDE,

        "slippage_per_side":
            SLIPPAGE_PER_SIDE,

        "confidence_report":
            confidence_report,

        "top_symbols":
            ranked[:10],

        "bottom_symbols":
            list(
                reversed(
                    ranked[-10:]
                )
            ),

        "errors":
            errors[:10],
    }


# =========================================================
# RUN
# =========================================================

if __name__ == "__main__":

    print(
        json.dumps(
            run_stock_backtest(
                days=DEFAULT_TEST_DAYS
            ),
            indent=2,
            default=str
        )
    )
