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
from sklearn.isotonic import IsotonicRegression


# =========================================================
# STOCK AI BACKTEST V9
# =========================================================

STRATEGY_NAME = "STOCK_V9_DIAGNOSTIC_180"
INTERVAL = "1h"

DEFAULT_TEST_DAYS = int(
    os.getenv("STOCK_BACKTEST_DAYS", "180")
)

TRAINING_LOOKBACK_DAYS = int(
    os.getenv("STOCK_TRAINING_LOOKBACK_DAYS", "180")
)

MAX_TOTAL_DAYS = int(
    os.getenv("STOCK_MAX_TOTAL_DAYS", "730")
)

RETRAIN_EVERY_BARS = int(
    os.getenv("STOCK_RETRAIN_EVERY_BARS", "42")
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
            "CRM,ORCL,UBER,MU,ARM,TSM,PANW,CRWD,NOW,"
            "CAT,BA,DIS,GE,SHOP,NKE,"
            "SPY,QQQ,IWM"
        ),
    ).split(",")
    if symbol.strip()
]

BENCHMARK_SYMBOL = os.getenv(
    "STOCK_BENCHMARK_SYMBOL",
    "SPY",
).upper()


# =========================================================
# ENTRY / RISK
# =========================================================

BUY_THRESHOLD = float(
    os.getenv("STOCK_BUY_THRESHOLD", "0.64")
)

COMMISSION_PER_SIDE = float(
    os.getenv("STOCK_COMMISSION_PER_SIDE", "0.0")
)

SLIPPAGE_PER_SIDE = float(
    os.getenv("STOCK_SLIPPAGE_PER_SIDE", "0.0005")
)

# V9 gives the model more time to identify genuine swing moves.
TARGET_HORIZON_BARS = int(
    os.getenv("STOCK_TARGET_HORIZON_BARS", "32")
)

# Roughly one trading week of hourly candles.
MAX_HOLD_BARS = int(
    os.getenv("STOCK_MAX_HOLD_BARS", "42")
)

COOLDOWN_BARS = int(
    os.getenv("STOCK_COOLDOWN_BARS", "4")
)

# Diagnostic lookahead does NOT affect live trade decisions.
DIAGNOSTIC_LOOKAHEAD_BARS = int(
    os.getenv("STOCK_DIAGNOSTIC_LOOKAHEAD_BARS", "42")
)


# =========================================================
# ATR-BASED EXITS
# =========================================================

STOP_ATR_MULTIPLIER = float(
    os.getenv("STOCK_STOP_ATR_MULTIPLIER", "1.35")
)

TARGET_ATR_MULTIPLIER = float(
    os.getenv("STOCK_TARGET_ATR_MULTIPLIER", "2.75")
)

MIN_STOP_PCT = float(
    os.getenv("STOCK_MIN_STOP_PCT", "0.006")
)

MAX_STOP_PCT = float(
    os.getenv("STOCK_MAX_STOP_PCT", "0.0225")
)

MIN_TARGET_PCT = float(
    os.getenv("STOCK_MIN_TARGET_PCT", "0.012")
)

MAX_TARGET_PCT = float(
    os.getenv("STOCK_MAX_TARGET_PCT", "0.050")
)


# =========================================================
# V9 ADAPTIVE HOLD / TRAILING EXIT
# =========================================================

TRAIL_ACTIVATION_R = float(
    os.getenv("STOCK_TRAIL_ACTIVATION_R", "1.0")
)

TRAIL_DISTANCE_R = float(
    os.getenv("STOCK_TRAIL_DISTANCE_R", "0.85")
)

# Give the swing setup longer before declaring its trend broken.
TREND_EXIT_AFTER_BARS = int(
    os.getenv("STOCK_TREND_EXIT_AFTER_BARS", "14")
)


# =========================================================
# POSITION SIZING
# =========================================================

MIN_TRADE_SIZE = float(
    os.getenv("STOCK_MIN_TRADE_SIZE", "50.0")
)

BASE_TRADE_SIZE = float(
    os.getenv("STOCK_BASE_TRADE_SIZE", "100.0")
)

MAX_TRADE_SIZE = float(
    os.getenv("STOCK_MAX_TRADE_SIZE", "175.0")
)

# V9 deliberately uses fixed GBP 100 sizing while diagnosing
# whether confidence and predicted edge are actually useful.
POSITION_SIZING_MODE = "FIXED_DIAGNOSTIC_GBP_100"


# =========================================================
# EDGE REQUIREMENTS
# =========================================================

MIN_PREDICTED_NET_RETURN = float(
    os.getenv(
        "STOCK_MIN_PREDICTED_NET_RETURN",
        "0.0030",
    )
)

MIN_COST_EDGE_MULTIPLIER = float(
    os.getenv(
        "STOCK_MIN_COST_EDGE_MULTIPLIER",
        "2.0",
    )
)


# =========================================================
# TRAINING REQUIREMENTS
# =========================================================

MIN_TRAINING_ROWS = int(
    os.getenv("STOCK_MIN_TRAINING_ROWS", "700")
)

MIN_POSITIVE_EXAMPLES = int(
    os.getenv("STOCK_MIN_POSITIVE_EXAMPLES", "30")
)

CALIBRATION_FRACTION = float(
    os.getenv("STOCK_CALIBRATION_FRACTION", "0.20")
)


# =========================================================
# ENTRY FILTERS
# =========================================================

MIN_DOLLAR_VOLUME_1H = float(
    os.getenv("STOCK_MIN_DOLLAR_VOLUME_1H", "10000000")
)

MIN_VOLUME_RATIO = float(
    os.getenv("STOCK_MIN_VOLUME_RATIO", "0.70")
)

MIN_ATR_PCT = float(
    os.getenv("STOCK_MIN_ATR_PCT", "0.002")
)

MAX_ATR_PCT = float(
    os.getenv("STOCK_MAX_ATR_PCT", "0.060")
)

MIN_RSI = float(
    os.getenv("STOCK_MIN_RSI", "42")
)

MAX_RSI = float(
    os.getenv("STOCK_MAX_RSI", "74")
)

MIN_RELATIVE_RETURN_8 = float(
    os.getenv("STOCK_MIN_RELATIVE_RETURN_8", "-0.015")
)


# =========================================================
# SYMBOL QUALITY GATE
# =========================================================

MIN_SYMBOL_QUALITY_SAMPLES = int(
    os.getenv("STOCK_MIN_SYMBOL_QUALITY_SAMPLES", "30")
)

MIN_SYMBOL_NET_SUCCESS_RATE = float(
    os.getenv("STOCK_MIN_SYMBOL_NET_SUCCESS_RATE", "0.40")
)

MIN_SYMBOL_AVG_NET_RETURN = float(
    os.getenv("STOCK_MIN_SYMBOL_AVG_NET_RETURN", "-0.001")
)


# =========================================================
# REGIME SAFETY
# =========================================================

MIN_BENCHMARK_RETURN_8 = float(
    os.getenv("STOCK_MIN_BENCHMARK_RETURN_8", "-0.012")
)

MIN_EMA_RATIO_10_20 = float(
    os.getenv("STOCK_MIN_EMA_RATIO_10_20", "0.995")
)


# =========================================================
# TREND / PULLBACK SETUP
# =========================================================

V6_MIN_PRICE_VS_EMA20 = float(
    os.getenv("V6_MIN_PRICE_VS_EMA20", "0.995")
)

V6_MIN_EMA10_VS_EMA20 = float(
    os.getenv("V6_MIN_EMA10_VS_EMA20", "0.998")
)

V7_MIN_SETUP_SCORE = float(
    os.getenv("V7_MIN_SETUP_SCORE", "0.67")
)

V7_MAX_PRICE_VS_EMA20 = float(
    os.getenv("V7_MAX_PRICE_VS_EMA20", "1.035")
)

V7_MAX_DISTANCE_FROM_EMA10 = float(
    os.getenv("V7_MAX_DISTANCE_FROM_EMA10", "0.020")
)

V7_MIN_RELATIVE_STRENGTH_8 = float(
    os.getenv("V7_MIN_RELATIVE_STRENGTH_8", "-0.005")
)

V7_MIN_BENCHMARK_RETURN_8 = float(
    os.getenv("V7_MIN_BENCHMARK_RETURN_8", "-0.015")
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
    "relative_strength_8",
    "relative_return_4",
    "relative_return_8",
    "market_above_ema20",
    "stock_above_ema20",
]


# =========================================================
# DATA
# =========================================================

def download_intraday(symbol, total_days):

    period_days = min(
        int(total_days),
        MAX_TOTAL_DAYS,
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

    if isinstance(df.columns, pd.MultiIndex):

        if symbol in df.columns.get_level_values(-1):

            df = df.xs(
                symbol,
                axis=1,
                level=-1,
            )

        else:

            df.columns = [
                col[0] if isinstance(col, tuple) else col
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

    needed = [
        "open",
        "high",
        "low",
        "close",
        "volume",
    ]

    missing = [
        col
        for col in needed
        if col not in df.columns
    ]

    if missing:
        raise RuntimeError(
            f"{symbol} missing columns: {missing}"
        )

    df = (
        df[needed]
        .dropna()
        .copy()
        .reset_index()
    )

    time_col = (
        "Datetime"
        if "Datetime" in df.columns
        else "Date"
    )

    if time_col not in df.columns:
        time_col = df.columns[0]

    df["time"] = pd.to_datetime(
        df[time_col],
        utc=True,
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
        .drop_duplicates(subset=["time"])
        .reset_index(drop=True)
    )


# =========================================================
# INDICATORS
# =========================================================

def calculate_rsi(series, period=14):

    delta = series.diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

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
        / avg_loss.replace(0, np.nan)
    )

    return (
        100
        - 100 / (1 + rs)
    ).fillna(50)


def calculate_atr(data, period=14):

    previous_close = (
        data["close"]
        .shift(1)
    )

    true_range = pd.concat(
        [
            data["high"] - data["low"],
            (data["high"] - previous_close).abs(),
            (data["low"] - previous_close).abs(),
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

def build_feature_frame(
    stock_df,
    benchmark_df,
):

    data = stock_df.copy()

    benchmark = (
        benchmark_df[
            ["time", "close"]
        ]
        .rename(
            columns={
                "close": "benchmark_close"
            }
        )
    )

    data = pd.merge_asof(
        data.sort_values("time"),
        benchmark.sort_values("time"),
        on="time",
        direction="backward",
        tolerance=pd.Timedelta(hours=2),
    )

    for n in (
        1,
        2,
        4,
        8,
        16,
    ):

        data[f"return_{n}"] = (
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

    data["relative_strength_8"] = (
        data["relative_return_8"]
    )

    data["ema_5"] = (
        data["close"]
        .ewm(
            span=5,
            adjust=False,
        )
        .mean()
    )

    data["ema_10"] = (
        data["close"]
        .ewm(
            span=10,
            adjust=False,
        )
        .mean()
    )

    data["ema_20"] = (
        data["close"]
        .ewm(
            span=20,
            adjust=False,
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
            adjust=False,
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
        / data["volume_ma_20"].replace(
            0,
            np.nan,
        )
    )

    data["volume_zscore"] = (
        (
            data["volume"]
            - data["volume_ma_20"]
        )
        / data["volume_std_20"].replace(
            0,
            np.nan,
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
        / data["close"].replace(
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
        / data["close"].replace(
            0,
            np.nan,
        )
    )

    data["body_pct"] = (
        (
            data["close"]
            - data["open"]
        )
        / data["open"].replace(
            0,
            np.nan,
        )
    )

    data["close_position"] = (
        (
            data["close"]
            - data["low"]
        )
        / candle_range.replace(
            0,
            np.nan,
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
        [np.inf, -np.inf],
        np.nan,
        inplace=True,
    )

    return data


# =========================================================
# ENTRY FILTER
# =========================================================

def passes_entry_filter(row):

    if (
        float(row["dollar_volume_ma20"])
        < MIN_DOLLAR_VOLUME_1H
    ):
        return False

    if (
        float(row["volume_ratio"])
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
        float(row["relative_return_8"])
        < MIN_RELATIVE_RETURN_8
    ):
        return False

    if (
        float(row["price_vs_ema20"])
        < 0.985
    ):
        return False

    if (
        float(row["benchmark_return_8"])
        < MIN_BENCHMARK_RETURN_8
    ):
        return False

    if (
        float(row["ema_ratio_10_20"])
        < MIN_EMA_RATIO_10_20
    ):
        return False

    return True


# =========================================================
# STOP / TARGET
# =========================================================

def calculate_stop_target_pct(row):

    atr_pct = float(
        row["atr_pct"]
    )

    stop_pct = float(
        np.clip(
            atr_pct
            * STOP_ATR_MULTIPLIER,
            MIN_STOP_PCT,
            MAX_STOP_PCT,
        )
    )

    target_pct = float(
        np.clip(
            atr_pct
            * TARGET_ATR_MULTIPLIER,
            MIN_TARGET_PCT,
            MAX_TARGET_PCT,
        )
    )

    target_pct = max(
        target_pct,
        stop_pct * 1.75,
    )

    return (
        stop_pct,
        target_pct,
    )


# =========================================================
# COSTS
# =========================================================

def round_trip_cost_return(
    exit_ratio=1.0,
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


# =========================================================
# V9 TRAINING TARGET
# =========================================================

def simulate_forward_trade(
    data,
    index,
):

    final_index = (
        index
        + TARGET_HORIZON_BARS
    )

    if final_index >= len(data):
        return None

    row = data.iloc[index]

    entry = float(
        row["close"]
    )

    stop_pct, target_pct = (
        calculate_stop_target_pct(
            row
        )
    )

    stop = (
        entry
        * (
            1.0 - stop_pct
        )
    )

    target = (
        entry
        * (
            1.0 + target_pct
        )
    )

    exit_price = None

    outcome = "HORIZON"

    maximum_high = entry
    minimum_low = entry

    for future_index in range(
        index + 1,
        final_index + 1,
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

        maximum_high = max(
            maximum_high,
            high,
        )

        minimum_low = min(
            minimum_low,
            low,
        )

        # Conservative assumption:
        # if stop and target are touched in the same candle,
        # assume stop happened first.
        if low <= stop:

            exit_price = stop
            outcome = "STOP"

            break

        if high >= target:

            exit_price = target
            outcome = "TARGET"

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
        - 1.0
    )

    costs = round_trip_cost_return(
        exit_ratio
    )

    net_return = (
        gross_return
        - costs
    )

    forward_mfe = (
        maximum_high
        / entry
        - 1.0
    )

    forward_mae = (
        minimum_low
        / entry
        - 1.0
    )

    # V9 target:
    # successful only if take-profit is reached before stop.
    target_success = (
        1
        if outcome == "TARGET"
        else 0
    )

    return {
        "target":
            target_success,

        "net_return":
            net_return,

        "forward_mfe":
            forward_mfe,

        "forward_mae":
            forward_mae,

        "outcome":
            outcome,
    }


def build_training_data(
    feature_data,
):

    data = feature_data.copy()

    targets = np.full(
        len(data),
        np.nan,
    )

    net_returns = np.full(
        len(data),
        np.nan,
    )

    forward_mfe = np.full(
        len(data),
        np.nan,
    )

    forward_mae = np.full(
        len(data),
        np.nan,
    )

    for index in range(
        len(data)
    ):

        result = simulate_forward_trade(
            data,
            index,
        )

        if result is None:
            continue

        targets[index] = (
            result["target"]
        )

        net_returns[index] = (
            result["net_return"]
        )

        forward_mfe[index] = (
            result["forward_mfe"]
        )

        forward_mae[index] = (
            result["forward_mae"]
        )

    data[
        "target_success"
    ] = targets

    data[
        "target_net_return"
    ] = net_returns

    data[
        "target_forward_mfe"
    ] = forward_mfe

    data[
        "target_forward_mae"
    ] = forward_mae

    data = (
        data.dropna(
            subset=(
                FEATURE_COLUMNS
                + [
                    "target_success",
                    "target_net_return",
                    "target_forward_mfe",
                    "target_forward_mae",
                ]
            )
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
# SYMBOL QUALITY
# =========================================================

def calculate_symbol_quality(
    training_data,
):

    filtered = []

    for _, row in (
        training_data.iterrows()
    ):

        try:

            if passes_entry_filter(row):
                filtered.append(row)

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

    passes = (
        success_rate
        >= MIN_SYMBOL_NET_SUCCESS_RATE
        and
        average_net_return
        >= MIN_SYMBOL_AVG_NET_RETURN
    )

    return {
        "samples":
            len(filtered_df),

        "success_rate":
            success_rate,

        "average_net_return":
            average_net_return,

        "passes":
            passes,
    }


# =========================================================
# MODELS
# =========================================================

def build_classifier():

    return RandomForestClassifier(
        n_estimators=500,
        max_depth=9,
        min_samples_leaf=10,
        max_features="sqrt",
        random_state=42,
        class_weight="balanced_subsample",
        n_jobs=-1,
    )


def train_models(
    training_data,
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

    if y_class.nunique() < 2:

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
            "Not enough positive training examples."
        )

    split = int(
        len(training_data)
        * (
            1.0
            - CALIBRATION_FRACTION
        )
    )

    split = max(
        1,
        min(
            split,
            len(training_data) - 1,
        ),
    )

    fit_data = (
        training_data.iloc[
            :split
        ]
    )

    calibration_data = (
        training_data.iloc[
            split:
        ]
    )

    classifier = build_classifier()

    classifier.fit(
        fit_data[
            FEATURE_COLUMNS
        ],
        fit_data[
            "target_success"
        ],
    )

    calibrator = None

    if (
        len(calibration_data) >= 50
        and
        calibration_data[
            "target_success"
        ].nunique() >= 2
    ):

        raw_probs = (
            classifier.predict_proba(
                calibration_data[
                    FEATURE_COLUMNS
                ]
            )[:, 1]
        )

        calibrator = IsotonicRegression(
            y_min=0.0,
            y_max=1.0,
            out_of_bounds="clip",
        )

        calibrator.fit(
            raw_probs,
            calibration_data[
                "target_success"
            ].to_numpy(
                dtype=float
            ),
        )

    classifier = build_classifier()

    classifier.fit(
        training_data[
            FEATURE_COLUMNS
        ],
        training_data[
            "target_success"
        ],
    )

    regressor = RandomForestRegressor(
        n_estimators=450,
        max_depth=9,
        min_samples_leaf=10,
        max_features="sqrt",
        random_state=43,
        n_jobs=-1,
    )

    regressor.fit(
        training_data[
            FEATURE_COLUMNS
        ],
        training_data[
            "target_net_return"
        ],
    )

    return (
        classifier,
        calibrator,
        regressor,
    )


def probability_success(
    model,
    calibrator,
    row,
):

    X = (
        row[
            FEATURE_COLUMNS
        ]
        .to_frame()
        .T
    )

    raw_probability = float(
        model.predict_proba(
            X
        )[0][1]
    )

    calibrated_probability = (
        float(
            calibrator.predict(
                [raw_probability]
            )[0]
        )
        if calibrator is not None
        else raw_probability
    )

    return (
        raw_probability,
        calibrated_probability,
    )


def predict_net_return(
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

    return float(
        model.predict(X)[0]
    )


# =========================================================
# V9 FIXED POSITION SIZE
# =========================================================

def calculate_trade_size(
    probability,
    predicted_net_return,
    atr_pct,
    setup_score=1.0,
):

    return float(
        np.clip(
            BASE_TRADE_SIZE,
            MIN_TRADE_SIZE,
            MAX_TRADE_SIZE,
        )
    )


# =========================================================
# SETUP SCORE
# =========================================================

def v7_setup_score(row):

    price_vs_ema20 = float(
        row["price_vs_ema20"]
    )

    ema_ratio = float(
        row["ema_ratio_10_20"]
    )

    price = float(
        row["close"]
    )

    ema10 = float(
        row["ema_10"]
    )

    relative_strength = float(
        row.get(
            "relative_strength_8",
            0.0,
        )
    )

    benchmark_return_8 = float(
        row.get(
            "benchmark_return_8",
            0.0,
        )
    )

    body_pct = float(
        row["body_pct"]
    )

    close_position = float(
        row["close_position"]
    )

    volume_ratio = float(
        row["volume_ratio"]
    )

    points = 0.0
    possible = 0.0

    # Trend structure.
    possible += 1.0

    if (
        price_vs_ema20
        >= V6_MIN_PRICE_VS_EMA20
        and
        ema_ratio
        >= V6_MIN_EMA10_VS_EMA20
    ):
        points += 1.0

    # Not excessively extended.
    possible += 1.0

    if (
        price_vs_ema20
        <= V7_MAX_PRICE_VS_EMA20
    ):
        points += 1.0

    # Pullback near EMA10.
    possible += 1.0

    distance_from_ema10 = abs(
        price / ema10 - 1.0
    )

    if (
        distance_from_ema10
        <= V7_MAX_DISTANCE_FROM_EMA10
    ):
        points += 1.0

    # Relative strength.
    possible += 1.0

    if (
        relative_strength
        >= V7_MIN_RELATIVE_STRENGTH_8
    ):

        points += 0.5

        if relative_strength >= 0.0:
            points += 0.5

    # Confirmation candle.
    possible += 1.0

    if body_pct > 0:
        points += 0.5

    if close_position >= 0.55:
        points += 0.5

    # Volume participation.
    possible += 1.0

    if volume_ratio >= 0.70:
        points += 0.5

    if volume_ratio >= 1.00:
        points += 0.5

    if (
        benchmark_return_8
        < V7_MIN_BENCHMARK_RETURN_8
    ):
        return 0.0

    if (
        price_vs_ema20
        < 0.985
    ):
        return 0.0

    return float(
        np.clip(
            points / possible,
            0.0,
            1.0,
        )
    )


def passes_v7_setup(row):

    return (
        v7_setup_score(row)
        >= V7_MIN_SETUP_SCORE
    )


# =========================================================
# POSITION
# =========================================================

@dataclass
class Position:

    entry_price: float
    quantity: float
    value: float

    entry_time: str
    entry_index: int

    raw_probability: float
    calibrated_probability: float
    predicted_net_return: float

    stop_loss: float
    take_profit: float

    stop_pct: float
    target_pct: float

    highest_price: float
    lowest_price: float

    highest_price_index: int
    lowest_price_index: int

    trailing_stop: float


# =========================================================
# V9 DIAGNOSTICS
# =========================================================

def diagnose_trade_path(
    data,
    entry_index,
    exit_index,
    entry_price,
    target_price,
):

    diagnostic_end_index = min(
        entry_index
        + DIAGNOSTIC_LOOKAHEAD_BARS,
        len(data) - 1,
    )

    if (
        diagnostic_end_index
        <= entry_index
    ):

        return {
            "diagnostic_mfe_pct":
                0.0,

            "diagnostic_mae_pct":
                0.0,

            "diagnostic_best_bar":
                0,

            "diagnostic_worst_bar":
                0,

            "later_hit_target":
                False,

            "later_recovered_entry":
                False,

            "post_exit_best_pct":
                0.0,

            "post_exit_worst_pct":
                0.0,
        }

    future = data.iloc[
        entry_index + 1:
        diagnostic_end_index + 1
    ]

    highs = (
        future[
            "high"
        ].astype(float)
    )

    lows = (
        future[
            "low"
        ].astype(float)
    )

    maximum_price = float(
        highs.max()
    )

    minimum_price = float(
        lows.min()
    )

    maximum_index = int(
        highs.idxmax()
    )

    minimum_index = int(
        lows.idxmin()
    )

    diagnostic_mfe_pct = (
        maximum_price
        / entry_price
        - 1.0
    )

    diagnostic_mae_pct = (
        minimum_price
        / entry_price
        - 1.0
    )

    diagnostic_best_bar = (
        maximum_index
        - entry_index
    )

    diagnostic_worst_bar = (
        minimum_index
        - entry_index
    )

    later_hit_target = False
    later_recovered_entry = False

    post_exit_best_pct = 0.0
    post_exit_worst_pct = 0.0

    if (
        exit_index is not None
        and
        exit_index < diagnostic_end_index
    ):

        post_exit = data.iloc[
            exit_index + 1:
            diagnostic_end_index + 1
        ]

        if not post_exit.empty:

            post_high = float(
                post_exit[
                    "high"
                ].max()
            )

            post_low = float(
                post_exit[
                    "low"
                ].min()
            )

            post_exit_best_pct = (
                post_high
                / entry_price
                - 1.0
            )

            post_exit_worst_pct = (
                post_low
                / entry_price
                - 1.0
            )

            later_hit_target = bool(
                post_high
                >= target_price
            )

            later_recovered_entry = bool(
                post_high
                >= entry_price
            )

    return {
        "diagnostic_mfe_pct":
            float(
                diagnostic_mfe_pct
            ),

        "diagnostic_mae_pct":
            float(
                diagnostic_mae_pct
            ),

        "diagnostic_best_bar":
            int(
                diagnostic_best_bar
            ),

        "diagnostic_worst_bar":
            int(
                diagnostic_worst_bar
            ),

        "later_hit_target":
            later_hit_target,

        "later_recovered_entry":
            later_recovered_entry,

        "post_exit_best_pct":
            float(
                post_exit_best_pct
            ),

        "post_exit_worst_pct":
            float(
                post_exit_worst_pct
            ),
    }


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
        - entry_value
    )

    commission = (
        entry_value
        * COMMISSION_PER_SIDE
        +
        exit_value
        * COMMISSION_PER_SIDE
    )

    slippage = (
        entry_value
        * SLIPPAGE_PER_SIDE
        +
        exit_value
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


def confidence_bucket(
    probability,
):

    if probability < 0.67:
        return "64-67"

    if probability < 0.70:
        return "67-70"

    if probability < 0.75:
        return "70-75"

    return "75+"


def empty_confidence_stats():

    return {
        "64-67": {
            "trades": 0,
            "wins": 0,
            "pnl": 0.0,
        },

        "67-70": {
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


def empty_exit_reasons():

    return {
        "STOP LOSS": 0,
        "GAP/STOP LOSS": 0,
        "TRAILING STOP": 0,
        "GAP/TRAIL STOP": 0,
        "TAKE PROFIT": 0,
        "GAP/TAKE PROFIT": 0,
        "TREND DETERIORATION": 0,
        "MAX HOLD": 0,
        "END OF TEST": 0,
    }


# =========================================================
# RECORD CLOSED TRADE
# =========================================================

def build_closed_trade(
    symbol,
    position,
    exit_price,
    reason,
    exit_index,
    exit_time,
    test,
):

    pnl = calculate_trade_pnl(
        position.entry_price,
        exit_price,
        position.quantity,
    )

    fixed_quantity = (
        BASE_TRADE_SIZE
        / position.entry_price
    )

    fixed_pnl = calculate_trade_pnl(
        position.entry_price,
        exit_price,
        fixed_quantity,
    )

    diagnostics = diagnose_trade_path(
        test,
        position.entry_index,
        exit_index,
        position.entry_price,
        position.take_profit,
    )

    actual_mfe_pct = (
        position.highest_price
        / position.entry_price
        - 1.0
    )

    actual_mae_pct = (
        position.lowest_price
        / position.entry_price
        - 1.0
    )

    bars_to_best = (
        position.highest_price_index
        - position.entry_index
    )

    bars_to_worst = (
        position.lowest_price_index
        - position.entry_index
    )

    bars_held = (
        exit_index
        - position.entry_index
    )

    return {
        "symbol":
            symbol,

        "entry_price":
            position.entry_price,

        "exit_price":
            exit_price,

        "trade_size":
            position.value,

        "pnl":
            pnl["net_pnl"],

        "fixed_size_pnl":
            fixed_pnl["net_pnl"],

        "gross_pnl":
            pnl["gross_pnl"],

        "fees":
            pnl["fees"],

        "reason":
            reason,

        "raw_probability":
            position.raw_probability,

        "entry_probability":
            position.calibrated_probability,

        "predicted_net_return":
            position.predicted_net_return,

        "stop_pct":
            position.stop_pct,

        "target_pct":
            position.target_pct,

        "entry_time":
            position.entry_time,

        "exit_time":
            exit_time,

        "bars_held":
            bars_held,

        "actual_mfe_pct":
            actual_mfe_pct,

        "actual_mae_pct":
            actual_mae_pct,

        "bars_to_best":
            bars_to_best,

        "bars_to_worst":
            bars_to_worst,

        **diagnostics,
    }


# =========================================================
# SINGLE-SYMBOL WALK-FORWARD BACKTEST
# =========================================================

def run_symbol_backtest(
    symbol,
    stock_df,
    benchmark_df,
    test_days,
):

    full_data = build_feature_frame(
        stock_df,
        benchmark_df,
    )

    latest_time = (
        full_data["time"]
        .max()
    )

    test_start = (
        latest_time
        - pd.Timedelta(
            days=test_days
        )
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
        full_data[
            full_data["time"]
            >= test_start
        ]
        .dropna(
            subset=required_test_columns
        )
        .reset_index(drop=True)
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
    fixed_size_net_equity = 0.0

    equity_curve = [0.0]

    cooldown_until = -1

    signals_checked = 0
    classifier_passes = 0
    edge_passes = 0
    retrain_count = 0

    setup_score_total = 0.0
    setup_score_count = 0

    classifier = None
    calibrator = None
    regressor = None

    next_retrain_index = 0

    confidence_stats = (
        empty_confidence_stats()
    )

    exit_reasons = (
        empty_exit_reasons()
    )

    trade_sizes = []

    for index, row in (
        test.iterrows()
    ):

        current_time = row["time"]

        # =====================================
        # WALK-FORWARD RETRAIN
        # =====================================

        if (
            classifier is None
            or
            index >= next_retrain_index
        ):

            cutoff = (
                current_time
                - pd.Timedelta(
                    days=TRAINING_LOOKBACK_DAYS
                )
            )

            historical = (
                full_data[
                    (
                        full_data["time"]
                        < current_time
                    )
                    &
                    (
                        full_data["time"]
                        >= cutoff
                    )
                ]
                .copy()
                .reset_index(drop=True)
            )

            training_data = (
                build_training_data(
                    historical
                )
            )

            quality = (
                calculate_symbol_quality(
                    training_data
                )
            )

            if not quality["passes"]:

                classifier = None
                calibrator = None
                regressor = None

                next_retrain_index = (
                    index
                    + RETRAIN_EVERY_BARS
                )

                continue

            (
                classifier,
                calibrator,
                regressor,
            ) = train_models(
                training_data
            )

            retrain_count += 1

            next_retrain_index = (
                index
                + RETRAIN_EVERY_BARS
            )

        if (
            classifier is None
            or
            regressor is None
        ):
            continue

        price = float(
            row["close"]
        )

        high = float(
            row["high"]
        )

        low = float(
            row["low"]
        )

        current_open = float(
            row["open"]
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

            bars_held = (
                index
                - position.entry_index
            )

            # Track complete path while position is open.
            if (
                high
                > position.highest_price
            ):

                position.highest_price = high

                position.highest_price_index = (
                    index
                )

            if (
                low
                < position.lowest_price
            ):

                position.lowest_price = low

                position.lowest_price_index = (
                    index
                )

            initial_risk = (
                position.entry_price
                * position.stop_pct
            )

            trail_activation_price = (
                position.entry_price
                +
                initial_risk
                * TRAIL_ACTIVATION_R
            )

            if (
                position.highest_price
                >= trail_activation_price
            ):

                candidate_trail = (
                    position.highest_price
                    -
                    initial_risk
                    * TRAIL_DISTANCE_R
                )

                position.trailing_stop = max(
                    position.trailing_stop,
                    candidate_trail,
                    position.entry_price,
                )

            effective_stop = max(
                position.stop_loss,
                position.trailing_stop,
            )

            # Gap below stop.
            if (
                current_open
                <= effective_stop
            ):

                exit_price = current_open

                reason = (
                    "GAP/TRAIL STOP"
                    if (
                        position.trailing_stop
                        > position.stop_loss
                    )
                    else
                    "GAP/STOP LOSS"
                )

            # Stop touched intrabar.
            elif (
                low
                <= effective_stop
            ):

                exit_price = effective_stop

                reason = (
                    "TRAILING STOP"
                    if (
                        position.trailing_stop
                        > position.stop_loss
                    )
                    else
                    "STOP LOSS"
                )

            # Gap above target.
            elif (
                current_open
                >= position.take_profit
            ):

                exit_price = current_open

                reason = (
                    "GAP/TAKE PROFIT"
                )

            # Target touched.
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

            # Trend deterioration.
            elif (
                bars_held
                >= TREND_EXIT_AFTER_BARS
                and
                float(
                    row["price_vs_ema20"]
                ) < 0.995
                and
                float(
                    row["ema_ratio_10_20"]
                ) < 0.998
            ):

                exit_price = price

                reason = (
                    "TREND DETERIORATION"
                )

            # Maximum hold.
            elif (
                bars_held
                >= MAX_HOLD_BARS
            ):

                exit_price = price

                reason = "MAX HOLD"

            if (
                exit_price is not None
            ):

                trade = build_closed_trade(
                    symbol=symbol,
                    position=position,
                    exit_price=exit_price,
                    reason=reason,
                    exit_index=index,
                    exit_time=timestamp,
                    test=test,
                )

                gross_equity += (
                    trade["gross_pnl"]
                )

                total_costs += (
                    trade["fees"]
                )

                net_equity += (
                    trade["pnl"]
                )

                fixed_size_net_equity += (
                    trade["fixed_size_pnl"]
                )

                bucket = confidence_bucket(
                    position.calibrated_probability
                )

                confidence_stats[
                    bucket
                ][
                    "trades"
                ] += 1

                if (
                    trade["pnl"]
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
                ] += trade["pnl"]

                exit_reasons[
                    reason
                ] = (
                    exit_reasons.get(
                        reason,
                        0,
                    )
                    + 1
                )

                trade_sizes.append(
                    position.value
                )

                trades.append(
                    trade
                )

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

            if not passes_v7_setup(
                row
            ):

                equity_curve.append(
                    net_equity
                )

                continue

            current_setup_score = (
                v7_setup_score(
                    row
                )
            )

            setup_score_total += (
                current_setup_score
            )

            setup_score_count += 1

            signals_checked += 1

            (
                raw_probability,
                calibrated_probability,
            ) = probability_success(
                classifier,
                calibrator,
                row,
            )

            if (
                calibrated_probability
                < BUY_THRESHOLD
            ):

                equity_curve.append(
                    net_equity
                )

                continue

            classifier_passes += 1

            predicted_net = (
                predict_net_return(
                    regressor,
                    row,
                )
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
                <
                (
                    expected_cost
                    * MIN_COST_EDGE_MULTIPLIER
                )
            ):

                equity_curve.append(
                    net_equity
                )

                continue

            edge_passes += 1

            atr_pct = float(
                row["atr_pct"]
            )

            trade_size = (
                calculate_trade_size(
                    calibrated_probability,
                    predicted_net,
                    atr_pct,
                    current_setup_score,
                )
            )

            quantity = (
                trade_size
                / price
            )

            stop_pct, target_pct = (
                calculate_stop_target_pct(
                    row
                )
            )

            position = Position(
                entry_price=price,
                quantity=quantity,
                value=trade_size,

                entry_time=timestamp,
                entry_index=index,

                raw_probability=(
                    raw_probability
                ),

                calibrated_probability=(
                    calibrated_probability
                ),

                predicted_net_return=(
                    predicted_net
                ),

                stop_loss=(
                    price
                    * (
                        1.0 - stop_pct
                    )
                ),

                take_profit=(
                    price
                    * (
                        1.0 + target_pct
                    )
                ),

                stop_pct=stop_pct,
                target_pct=target_pct,

                highest_price=price,
                lowest_price=price,

                highest_price_index=index,
                lowest_price_index=index,

                trailing_stop=0.0,
            )

        equity_curve.append(
            net_equity
        )

    # =========================================
    # CLOSE OPEN POSITION
    # =========================================

    if (
        position is not None
        and
        len(test) > 0
    ):

        final_index = (
            len(test) - 1
        )

        final_row = (
            test.iloc[
                final_index
            ]
        )

        final_high = float(
            final_row["high"]
        )

        final_low = float(
            final_row["low"]
        )

        if (
            final_high
            > position.highest_price
        ):

            position.highest_price = (
                final_high
            )

            position.highest_price_index = (
                final_index
            )

        if (
            final_low
            < position.lowest_price
        ):

            position.lowest_price = (
                final_low
            )

            position.lowest_price_index = (
                final_index
            )

        exit_price = float(
            final_row["close"]
        )

        trade = build_closed_trade(
            symbol=symbol,
            position=position,
            exit_price=exit_price,
            reason="END OF TEST",
            exit_index=final_index,
            exit_time=str(
                final_row["time"]
            ),
            test=test,
        )

        gross_equity += (
            trade["gross_pnl"]
        )

        total_costs += (
            trade["fees"]
        )

        net_equity += (
            trade["pnl"]
        )

        fixed_size_net_equity += (
            trade["fixed_size_pnl"]
        )

        bucket = confidence_bucket(
            position.calibrated_probability
        )

        confidence_stats[
            bucket
        ][
            "trades"
        ] += 1

        if (
            trade["pnl"]
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
        ] += trade["pnl"]

        exit_reasons[
            "END OF TEST"
        ] += 1

        trade_sizes.append(
            position.value
        )

        trades.append(
            trade
        )

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
            value,
        )

        max_drawdown = min(
            max_drawdown,
            value - peak,
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

        "retrain_count":
            retrain_count,

        "signals_checked":
            signals_checked,

        "classifier_passes":
            classifier_passes,

        "edge_passes":
            edge_passes,

        "average_setup_score":
            (
                setup_score_total
                / setup_score_count
                if setup_score_count
                else 0.0
            ),

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

        "fixed_size_pnl":
            fixed_size_net_equity,

        "average_trade_size":
            (
                float(
                    np.mean(
                        trade_sizes
                    )
                )
                if trade_sizes
                else 0.0
            ),

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

        "exit_reasons":
            exit_reasons,

        "trade_log":
            trades,
    }


# =========================================================
# PORTFOLIO DRAWDOWN
# =========================================================

def calculate_portfolio_drawdown(
    all_trades,
):

    if not all_trades:
        return 0.0

    events = []

    for trade in all_trades:

        try:

            exit_time = pd.Timestamp(
                trade["exit_time"]
            )

        except Exception:

            continue

        events.append(
            (
                exit_time,
                float(
                    trade["pnl"]
                ),
            )
        )

    events.sort(
        key=lambda item:
            item[0]
    )

    equity = 0.0
    peak = 0.0
    max_drawdown = 0.0

    for _, pnl in events:

        equity += pnl

        peak = max(
            peak,
            equity,
        )

        max_drawdown = min(
            max_drawdown,
            equity - peak,
        )

    return max_drawdown


# =========================================================
# HISTORY PROTECTION
# =========================================================

def resolve_valid_test_days(
    benchmark_df,
    requested_days,
):

    if (
        benchmark_df is None
        or
        benchmark_df.empty
    ):

        raise RuntimeError(
            "Benchmark data is unavailable."
        )

    first_time = pd.Timestamp(
        benchmark_df[
            "time"
        ].min()
    )

    last_time = pd.Timestamp(
        benchmark_df[
            "time"
        ].max()
    )

    available_days = max(
        1,
        int(
            (
                last_time
                - first_time
            ).total_seconds()
            / 86400
        ),
    )

    minimum_training_days = 60

    maximum_valid_test_days = max(
        10,
        available_days
        - minimum_training_days,
    )

    resolved_days = min(
        int(requested_days),
        maximum_valid_test_days,
    )

    return {
        "requested_days":
            int(
                requested_days
            ),

        "resolved_days":
            int(
                resolved_days
            ),

        "available_days":
            int(
                available_days
            ),

        "estimated_training_days":
            int(
                max(
                    0,
                    available_days
                    - resolved_days,
                )
            ),
    }


# =========================================================
# V9 DIAGNOSIS
# =========================================================

def build_v9_diagnosis(
    total_trades,
    total_wins,
    stopped_trades,
    stopped_later_hit_target,
    stopped_later_recovered,
    average_diagnostic_mfe,
    average_diagnostic_mae,
):

    if total_trades == 0:

        return (
            "NO TRADES - entry filters/model are too restrictive "
            "or no qualifying setups appeared."
        )

    if total_trades < 10:

        return (
            "TOO FEW TRADES - strategy remains too selective "
            "for a reliable conclusion."
        )

    stopped_count = len(
        stopped_trades
    )

    later_target_count = len(
        stopped_later_hit_target
    )

    later_recovery_count = len(
        stopped_later_recovered
    )

    later_target_rate = (
        later_target_count
        / stopped_count
        if stopped_count
        else 0.0
    )

    later_recovery_rate = (
        later_recovery_count
        / stopped_count
        if stopped_count
        else 0.0
    )

    win_rate = (
        total_wins
        / total_trades
        if total_trades
        else 0.0
    )

    if (
        later_target_rate
        >= 0.40
    ):

        return (
            "ENTRY MODEL MAY HAVE EDGE - many stopped trades "
            "later reached their original take-profit. "
            "Stop-loss or exit logic appears too aggressive."
        )

    if (
        later_recovery_rate
        >= 0.60
    ):

        return (
            "STOPS MAY BE TOO TIGHT - most stopped positions "
            "subsequently recovered back to their entry price."
        )

    if (
        average_diagnostic_mfe
        <= abs(
            average_diagnostic_mae
        )
    ):

        return (
            "ENTRY MODEL LIKELY WEAK - trades generally move "
            "at least as far against the position as they do "
            "in its favour."
        )

    if (
        win_rate
        < 0.35
    ):

        return (
            "ENTRY QUALITY STILL WEAK - positions show some "
            "favourable movement but not enough to produce "
            "acceptable realised results."
        )

    return (
        "MIXED RESULT - entry and exit logic both require "
        "further evaluation."
    )


# =========================================================
# COMPLETE BACKTEST
# =========================================================

def run_stock_backtest(
    days=DEFAULT_TEST_DAYS,
):

    days = int(
        max(
            10,
            min(
                days,
                180,
            ),
        )
    )

    requested_days = days

    total_days = min(
        requested_days
        + TRAINING_LOOKBACK_DAYS,
        MAX_TOTAL_DAYS,
    )

    benchmark_df = download_intraday(
        BENCHMARK_SYMBOL,
        total_days,
    )

    history_info = (
        resolve_valid_test_days(
            benchmark_df,
            requested_days,
        )
    )

    days = history_info[
        "resolved_days"
    ]

    print(
        f"{STRATEGY_NAME}: "
        f"{days} unseen test days"
    )

    if (
        days
        < requested_days
    ):

        print(
            "Requested unseen period was "
            f"{requested_days} days, but only "
            f"{days} days can be tested without "
            "overlapping training history."
        )

    print(
        f"Available hourly history: "
        f"~{history_info['available_days']} days"
    )

    print(
        f"Estimated pre-test training history: "
        f"~{history_info['estimated_training_days']} days"
    )

    print(
        f"Target horizon: "
        f"{TARGET_HORIZON_BARS} bars"
    )

    print(
        f"Maximum hold: "
        f"{MAX_HOLD_BARS} bars"
    )

    print(
        f"Diagnostic lookahead: "
        f"{DIAGNOSTIC_LOOKAHEAD_BARS} bars"
    )

    print(
        f"Walk-forward retrain every "
        f"{RETRAIN_EVERY_BARS} bars"
    )

    print(
        "Probability calibration: ON"
    )

    print(
        "ATR-based exits: ON"
    )

    print(
        f"Position sizing: "
        f"{POSITION_SIZING_MODE}"
    )

    print(
        f"Minimum predicted edge: "
        f"{MIN_PREDICTED_NET_RETURN:.2%}"
    )

    print(
        f"Setup-score minimum: "
        f"{V7_MIN_SETUP_SCORE:.0%}"
    )

    print(
        f"Symbols configured: "
        f"{len(SYMBOLS)}"
    )

    results = []
    errors = []

    for index, symbol in enumerate(
        SYMBOLS,
        start=1,
    ):

        print(
            f"{STRATEGY_NAME} "
            f"{index}/{len(SYMBOLS)}: "
            f"{symbol}"
        )

        try:

            stock_df = (
                benchmark_df.copy()
                if symbol
                == BENCHMARK_SYMBOL
                else download_intraday(
                    symbol,
                    total_days,
                )
            )

            result = run_symbol_backtest(
                symbol,
                stock_df,
                benchmark_df,
                days,
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
            "Stock V9 backtest failed for "
            "every configured symbol."
        )

    all_trades = []

    for item in results:

        all_trades.extend(
            item[
                "trade_log"
            ]
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

    total_fees = sum(
        item["fees"]
        for item in results
    )

    total_pnl = sum(
        item["pnl"]
        for item in results
    )

    total_fixed_size_pnl = sum(
        item["fixed_size_pnl"]
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
        reverse=True,
    )

    combined_confidence = (
        empty_confidence_stats()
    )

    combined_exit_reasons = (
        empty_exit_reasons()
    )

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
            ] += stats["trades"]

            combined_confidence[
                bucket
            ][
                "wins"
            ] += stats["wins"]

            combined_confidence[
                bucket
            ][
                "pnl"
            ] += stats["pnl"]

        for reason, count in (
            item[
                "exit_reasons"
            ].items()
        ):

            combined_exit_reasons[
                reason
            ] = (
                combined_exit_reasons.get(
                    reason,
                    0,
                )
                + count
            )

    confidence_report = []

    for bucket, stats in (
        combined_confidence.items()
    ):

        trades_count = (
            stats["trades"]
        )

        wins_count = (
            stats["wins"]
        )

        confidence_report.append(
            {
                "bucket":
                    bucket,

                "trades":
                    trades_count,

                "wins":
                    wins_count,

                "win_rate":
                    (
                        wins_count
                        / trades_count
                        if trades_count
                        else 0.0
                    ),

                "pnl":
                    stats["pnl"],
            }
        )

    average_trade_size = (
        float(
            np.mean(
                [
                    trade[
                        "trade_size"
                    ]
                    for trade in all_trades
                ]
            )
        )
        if all_trades
        else 0.0
    )

    portfolio_drawdown = (
        calculate_portfolio_drawdown(
            all_trades
        )
    )

    # =====================================================
    # V9 COMBINED DIAGNOSTICS
    # =====================================================

    stopped_trades = [
        trade
        for trade in all_trades
        if "STOP" in trade[
            "reason"
        ]
    ]

    stopped_later_hit_target = [
        trade
        for trade in stopped_trades
        if trade.get(
            "later_hit_target",
            False,
        )
    ]

    stopped_later_recovered = [
        trade
        for trade in stopped_trades
        if trade.get(
            "later_recovered_entry",
            False,
        )
    ]

    average_actual_mfe = (
        float(
            np.mean(
                [
                    trade.get(
                        "actual_mfe_pct",
                        0.0,
                    )
                    for trade in all_trades
                ]
            )
        )
        if all_trades
        else 0.0
    )

    average_actual_mae = (
        float(
            np.mean(
                [
                    trade.get(
                        "actual_mae_pct",
                        0.0,
                    )
                    for trade in all_trades
                ]
            )
        )
        if all_trades
        else 0.0
    )

    average_diagnostic_mfe = (
        float(
            np.mean(
                [
                    trade.get(
                        "diagnostic_mfe_pct",
                        0.0,
                    )
                    for trade in all_trades
                ]
            )
        )
        if all_trades
        else 0.0
    )

    average_diagnostic_mae = (
        float(
            np.mean(
                [
                    trade.get(
                        "diagnostic_mae_pct",
                        0.0,
                    )
                    for trade in all_trades
                ]
            )
        )
        if all_trades
        else 0.0
    )

    winner_trades = [
        trade
        for trade in all_trades
        if trade["pnl"] > 0
    ]

    loser_trades = [
        trade
        for trade in all_trades
        if trade["pnl"] <= 0
    ]

    winner_average_mfe = (
        float(
            np.mean(
                [
                    trade.get(
                        "diagnostic_mfe_pct",
                        0.0,
                    )
                    for trade in winner_trades
                ]
            )
        )
        if winner_trades
        else 0.0
    )

    loser_average_mfe = (
        float(
            np.mean(
                [
                    trade.get(
                        "diagnostic_mfe_pct",
                        0.0,
                    )
                    for trade in loser_trades
                ]
            )
        )
        if loser_trades
        else 0.0
    )

    average_bars_held = (
        float(
            np.mean(
                [
                    trade.get(
                        "bars_held",
                        0,
                    )
                    for trade in all_trades
                ]
            )
        )
        if all_trades
        else 0.0
    )

    v9_diagnosis = (
        build_v9_diagnosis(
            total_trades=total_trades,
            total_wins=total_wins,
            stopped_trades=stopped_trades,
            stopped_later_hit_target=(
                stopped_later_hit_target
            ),
            stopped_later_recovered=(
                stopped_later_recovered
            ),
            average_diagnostic_mfe=(
                average_diagnostic_mfe
            ),
            average_diagnostic_mae=(
                average_diagnostic_mae
            ),
        )
    )

    return {
        "strategy":
            STRATEGY_NAME,

        "interval":
            INTERVAL,

        "days":
            days,

        "requested_days":
            requested_days,

        "available_history_days":
            history_info[
                "available_days"
            ],

        "estimated_training_days":
            history_info[
                "estimated_training_days"
            ],

        "training_days":
            TRAINING_LOOKBACK_DAYS,

        "target_horizon_bars":
            TARGET_HORIZON_BARS,

        "max_hold_bars":
            MAX_HOLD_BARS,

        "diagnostic_lookahead_bars":
            DIAGNOSTIC_LOOKAHEAD_BARS,

        "symbols_configured":
            len(SYMBOLS),

        "symbols_completed":
            len(results),

        "symbols_skipped":
            len(errors),

        "retrain_count":
            sum(
                item[
                    "retrain_count"
                ]
                for item in results
            ),

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

        "average_setup_score":
            (
                float(
                    np.mean(
                        [
                            item[
                                "average_setup_score"
                            ]
                            for item in results
                            if item[
                                "average_setup_score"
                            ] > 0
                        ]
                    )
                )
                if any(
                    item[
                        "average_setup_score"
                    ] > 0
                    for item in results
                )
                else 0.0
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
            total_fees,

        "pnl":
            total_pnl,

        "fixed_size_pnl":
            total_fixed_size_pnl,

        # Retained so your existing Discord formatter does not
        # break. Since V9 deliberately uses fixed sizing, this
        # should normally be approximately zero.
        "dynamic_sizing_improvement":
            (
                total_pnl
                - total_fixed_size_pnl
            ),

        "average_trade_size":
            average_trade_size,

        "min_trade_size":
            MIN_TRADE_SIZE,

        "max_trade_size":
            MAX_TRADE_SIZE,

        "position_sizing_mode":
            POSITION_SIZING_MODE,

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
            portfolio_drawdown,

        "worst_symbol_drawdown":
            min(
                item[
                    "max_drawdown"
                ]
                for item in results
            ),

        "best_symbol":
            ranked[0][
                "symbol"
            ],

        "best_symbol_pnl":
            ranked[0][
                "pnl"
            ],

        "worst_symbol":
            ranked[-1][
                "symbol"
            ],

        "worst_symbol_pnl":
            ranked[-1][
                "pnl"
            ],

        "buy_threshold":
            BUY_THRESHOLD,

        "min_benchmark_return_8":
            MIN_BENCHMARK_RETURN_8,

        "min_ema_ratio_10_20":
            MIN_EMA_RATIO_10_20,

        "min_predicted_net_return":
            MIN_PREDICTED_NET_RETURN,

        "commission_per_side":
            COMMISSION_PER_SIDE,

        "slippage_per_side":
            SLIPPAGE_PER_SIDE,

        "confidence_report":
            confidence_report,

        "exit_reasons":
            combined_exit_reasons,

        "v9_diagnostics": {
            "stopped_trades":
                len(
                    stopped_trades
                ),

            "stopped_later_hit_target":
                len(
                    stopped_later_hit_target
                ),

            "stopped_later_hit_target_rate":
                (
                    len(
                        stopped_later_hit_target
                    )
                    / len(
                        stopped_trades
                    )
                    if stopped_trades
                    else 0.0
                ),

            "stopped_later_recovered_entry":
                len(
                    stopped_later_recovered
                ),

            "stopped_later_recovered_rate":
                (
                    len(
                        stopped_later_recovered
                    )
                    / len(
                        stopped_trades
                    )
                    if stopped_trades
                    else 0.0
                ),

            "average_actual_mfe_pct":
                average_actual_mfe,

            "average_actual_mae_pct":
                average_actual_mae,

            "average_diagnostic_mfe_pct":
                average_diagnostic_mfe,

            "average_diagnostic_mae_pct":
                average_diagnostic_mae,

            "winner_average_mfe_pct":
                winner_average_mfe,

            "loser_average_mfe_pct":
                loser_average_mfe,

            "average_bars_held":
                average_bars_held,

            "diagnosis":
                v9_diagnosis,
        },

        "top_symbols":
            ranked[:10],

        "bottom_symbols":
            list(
                reversed(
                    ranked[-10:]
                )
            ),

        "trade_log":
            all_trades,

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
            default=str,
        )
    )