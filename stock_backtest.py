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
# STOCK AI BACKTEST V4
# =========================================================

STRATEGY_NAME = "STOCK_V7"
INTERVAL = "1h"

DEFAULT_TEST_DAYS = int(
    os.getenv("STOCK_BACKTEST_DAYS", "40")
)

TRAINING_LOOKBACK_DAYS = int(
    os.getenv("STOCK_TRAINING_LOOKBACK_DAYS", "180")
)

MAX_TOTAL_DAYS = int(
    os.getenv("STOCK_MAX_TOTAL_DAYS", "365")
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
        )
    ).split(",")
    if symbol.strip()
]

BENCHMARK_SYMBOL = os.getenv(
    "STOCK_BENCHMARK_SYMBOL",
    "SPY"
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

TARGET_HORIZON_BARS = int(
    os.getenv("STOCK_TARGET_HORIZON_BARS", "8")
)

MAX_HOLD_BARS = int(
    os.getenv("STOCK_MAX_HOLD_BARS", "10")
)

COOLDOWN_BARS = int(
    os.getenv("STOCK_COOLDOWN_BARS", "4")
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
# DYNAMIC POSITION SIZING
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

EDGE_SIZE_WEIGHT = float(
    os.getenv("STOCK_EDGE_SIZE_WEIGHT", "0.50")
)

CONFIDENCE_SIZE_WEIGHT = float(
    os.getenv("STOCK_CONFIDENCE_SIZE_WEIGHT", "0.10")
)

VOLATILITY_SIZE_WEIGHT = float(
    os.getenv("STOCK_VOLATILITY_SIZE_WEIGHT", "0.15")
)


# =========================================================
# EDGE REQUIREMENTS
# =========================================================

MIN_PREDICTED_NET_RETURN = float(
    os.getenv(
        "STOCK_MIN_PREDICTED_NET_RETURN",
        "0.0030"
    )
)

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
# V5 REGIME / CONFIDENCE SAFETY
# =========================================================

# Do not let an extreme model probability alone create an
# extreme position. V4's 75%+ bucket was profitable, but its
# win rate was not actually higher, so V5 shrinks confidence
# toward neutral before using it for position sizing.
CONFIDENCE_SHRINKAGE = float(
    os.getenv("STOCK_CONFIDENCE_SHRINKAGE", "0.35")
)

# Require the broad market not to be in a clearly weak
# short-term regime. This is deliberately mild: it blocks
# obvious risk-off conditions without demanding a bull market.
MIN_BENCHMARK_RETURN_8 = float(
    os.getenv("STOCK_MIN_BENCHMARK_RETURN_8", "-0.012")
)

# Require the stock's 10/20 EMA structure to avoid being
# materially bearish.
MIN_EMA_RATIO_10_20 = float(
    os.getenv("STOCK_MIN_EMA_RATIO_10_20", "0.995")
)


# =========================================================
# V7: PROFIT-FOCUSED TREND-PULLBACK + RELATIVE STRENGTH
# =========================================================

V6_MIN_PRICE_VS_EMA20 = float(os.getenv("V6_MIN_PRICE_VS_EMA20", "0.995"))
V6_MAX_PRICE_VS_EMA20 = float(os.getenv("V6_MAX_PRICE_VS_EMA20", "1.025"))
V6_MIN_EMA10_VS_EMA20 = float(os.getenv("V6_MIN_EMA10_VS_EMA20", "0.998"))
V6_MAX_DISTANCE_FROM_EMA10 = float(os.getenv("V6_MAX_DISTANCE_FROM_EMA10", "0.015"))
V6_MIN_RELATIVE_STRENGTH_8 = float(os.getenv("V6_MIN_RELATIVE_STRENGTH_8", "-0.003"))
V6_MIN_BODY_PCT = float(os.getenv("V6_MIN_BODY_PCT", "0.0005"))
V6_MIN_CLOSE_POSITION = float(os.getenv("V6_MIN_CLOSE_POSITION", "0.55"))
V6_MIN_VOLUME_RATIO = float(os.getenv("V6_MIN_VOLUME_RATIO", "0.80"))


# =========================================================
# V7 PROFIT-FOCUSED SETUP SCORING
# =========================================================

# Instead of demanding every trend/pullback rule be perfect,
# V7 scores the setup. This increases opportunity count while
# still requiring multiple pieces of evidence to agree.

V7_MIN_SETUP_SCORE = float(
    os.getenv("V7_MIN_SETUP_SCORE", "0.67")
)

V7_SETUP_SIZE_WEIGHT = float(
    os.getenv("V7_SETUP_SIZE_WEIGHT", "0.25")
)

# Slightly wider acceptable pullback zone than V6.
V7_MAX_PRICE_VS_EMA20 = float(
    os.getenv("V7_MAX_PRICE_VS_EMA20", "1.035")
)

V7_MAX_DISTANCE_FROM_EMA10 = float(
    os.getenv("V7_MAX_DISTANCE_FROM_EMA10", "0.020")
)

# Relative strength can be mildly negative if the rest of the
# setup is strong; strong relative strength earns a higher score.
V7_MIN_RELATIVE_STRENGTH_8 = float(
    os.getenv("V7_MIN_RELATIVE_STRENGTH_8", "-0.005")
)

# Keep a mild market-regime guard rather than forcing a bull market.
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

    if isinstance(df.columns, pd.MultiIndex):

        if symbol in df.columns.get_level_values(-1):
            df = df.xs(
                symbol,
                axis=1,
                level=-1
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
        adjust=False
    ).mean()

    avg_loss = loss.ewm(
        alpha=1 / period,
        adjust=False
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
        tolerance=pd.Timedelta(hours=2)
    )

    for n in (
        1,
        2,
        4,
        8,
        16
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

    data["ema_5"] = (
        data["close"]
        .ewm(span=5, adjust=False)
        .mean()
    )

    data["ema_10"] = (
        data["close"]
        .ewm(span=10, adjust=False)
        .mean()
    )

    data["ema_20"] = (
        data["close"]
        .ewm(span=20, adjust=False)
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
        .ewm(span=20, adjust=False)
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
        / data["volume_ma_20"].replace(0, np.nan)
    )

    data["volume_zscore"] = (
        (
            data["volume"]
            - data["volume_ma_20"]
        )
        / data["volume_std_20"].replace(0, np.nan)
    )

    data["rsi"] = calculate_rsi(
        data["close"]
    )

    data["atr"] = calculate_atr(
        data
    )

    data["atr_pct"] = (
        data["atr"]
        / data["close"].replace(0, np.nan)
    )

    candle_range = (
        data["high"]
        - data["low"]
    )

    data["range_pct"] = (
        candle_range
        / data["close"].replace(0, np.nan)
    )

    data["body_pct"] = (
        (
            data["close"]
            - data["open"]
        )
        / data["open"].replace(0, np.nan)
    )

    data["close_position"] = (
        (
            data["close"]
            - data["low"]
        )
        / candle_range.replace(0, np.nan)
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

    # V6 relative strength versus SPY.
    if "benchmark_return_8" in data.columns:
        if "return_8" in data.columns:
            data["relative_strength_8"] = data["return_8"] - data["benchmark_return_8"]
        elif "return_6" in data.columns:
            data["relative_strength_8"] = data["return_6"] - data["benchmark_return_8"]
        else:
            data["relative_strength_8"] = -data["benchmark_return_8"]

    data.replace(
        [np.inf, -np.inf],
        np.nan,
        inplace=True
    )

    return data


# =========================================================
# FILTER
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

    # V5: avoid clearly weak broad-market regimes.
    if (
        float(row["benchmark_return_8"])
        < MIN_BENCHMARK_RETURN_8
    ):
        return False

    # V5: reject materially bearish 10/20 EMA structure.
    if (
        float(row["ema_ratio_10_20"])
        < MIN_EMA_RATIO_10_20
    ):
        return False

    return True


# =========================================================
# ADAPTIVE STOP / TARGET
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
            MAX_STOP_PCT
        )
    )

    target_pct = float(
        np.clip(
            atr_pct
            * TARGET_ATR_MULTIPLIER,
            MIN_TARGET_PCT,
            MAX_TARGET_PCT
        )
    )

    # Preserve a worthwhile payoff ratio.
    target_pct = max(
        target_pct,
        stop_pct * 1.75
    )

    return (
        stop_pct,
        target_pct
    )


# =========================================================
# COSTS / TRAINING TARGETS
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
            1 - stop_pct
        )
    )

    target = (
        entry
        * (
            1 + target_pct
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
        exit_ratio - 1
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
    training_data
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
# MODELS + PROBABILITY CALIBRATION
# =========================================================

def build_classifier():

    return RandomForestClassifier(
        n_estimators=500,
        max_depth=9,
        min_samples_leaf=10,
        max_features="sqrt",
        random_state=42,
        class_weight="balanced_subsample",
        n_jobs=-1
    )


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
            "Not enough positive "
            "training examples."
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
            len(training_data) - 1
        )
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
        ]
    )

    calibrator = None

    if (
        len(calibration_data) >= 50
        and calibration_data[
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
            out_of_bounds="clip"
        )

        calibrator.fit(
            raw_probs,
            calibration_data[
                "target_success"
            ].to_numpy(
                dtype=float
            )
        )

    # Refit classifier on all historical rows after
    # learning the calibration mapping.
    classifier = build_classifier()

    classifier.fit(
        training_data[
            FEATURE_COLUMNS
        ],
        training_data[
            "target_success"
        ]
    )

    regressor = RandomForestRegressor(
        n_estimators=450,
        max_depth=9,
        min_samples_leaf=10,
        max_features="sqrt",
        random_state=43,
        n_jobs=-1
    )

    regressor.fit(
        training_data[
            FEATURE_COLUMNS
        ],
        training_data[
            "target_net_return"
        ]
    )

    return (
        classifier,
        calibrator,
        regressor
    )


def probability_success(
    model,
    calibrator,
    row
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
        calibrated_probability
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
        model.predict(X)[0]
    )


# =========================================================
# DYNAMIC SIZING
# =========================================================

def calculate_trade_size(
    probability,
    predicted_net_return,
    atr_pct,
    setup_score=1.0
):

    # V5: shrink calibrated confidence toward 50% before it
    # affects position size. This prevents a 75-90% model
    # probability from automatically receiving the largest bet.
    sizing_probability = (
        0.50
        + (
            probability - 0.50
        )
        * (
            1.0 - CONFIDENCE_SHRINKAGE
        )
    )

    confidence_span = max(
        0.85 - BUY_THRESHOLD,
        0.01
    )

    confidence_score = (
        sizing_probability
        - BUY_THRESHOLD
    ) / confidence_span

    confidence_score = float(
        np.clip(
            confidence_score,
            0.0,
            1.0
        )
    )

    edge_score = (
        predicted_net_return
        - MIN_PREDICTED_NET_RETURN
    ) / max(
        0.015
        - MIN_PREDICTED_NET_RETURN,
        0.001
    )

    edge_score = float(
        np.clip(
            edge_score,
            0.0,
            1.0
        )
    )

    # Lower volatility gets a modest sizing bonus;
    # high volatility automatically reduces size.
    volatility_score = (
        MAX_ATR_PCT
        - atr_pct
    ) / max(
        MAX_ATR_PCT
        - MIN_ATR_PCT,
        0.001
    )

    volatility_score = float(
        np.clip(
            volatility_score,
            0.0,
            1.0
        )
    )

    setup_component = float(
        np.clip(
            setup_score,
            0.0,
            1.0
        )
    )

    combined_score = (
        confidence_score
        * CONFIDENCE_SIZE_WEIGHT
        +
        edge_score
        * EDGE_SIZE_WEIGHT
        +
        volatility_score
        * VOLATILITY_SIZE_WEIGHT
        +
        setup_component
        * V7_SETUP_SIZE_WEIGHT
    )

    total_weight = (
        CONFIDENCE_SIZE_WEIGHT
        +
        EDGE_SIZE_WEIGHT
        +
        VOLATILITY_SIZE_WEIGHT
        +
        V7_SETUP_SIZE_WEIGHT
    )

    if total_weight > 0:
        combined_score = (
            combined_score
            / total_weight
        )

    size = (
        MIN_TRADE_SIZE
        + combined_score
        * (
            MAX_TRADE_SIZE
            - MIN_TRADE_SIZE
        )
    )

    if (
        probability
        >= BUY_THRESHOLD
        and predicted_net_return
        >= MIN_PREDICTED_NET_RETURN
    ):
        size = max(
            size,
            BASE_TRADE_SIZE
        )

    return float(
        np.clip(
            size,
            MIN_TRADE_SIZE,
            MAX_TRADE_SIZE
        )
    )


# =========================================================
# V6 QUALIFIED SETUP
# =========================================================

def v7_setup_score(row):
    """
    Score the intraday trend-pullback setup from 0.0 to 1.0.

    The ML model still decides whether to trade. This function
    only makes sure the candle is a sensible candidate.
    """

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
            0.0
        )
    )

    benchmark_return_8 = float(
        row.get(
            "benchmark_return_8",
            0.0
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

    # 1. Trend structure.
    possible += 1.0

    if (
        price_vs_ema20
        >= V6_MIN_PRICE_VS_EMA20
        and
        ema_ratio
        >= V6_MIN_EMA10_VS_EMA20
    ):
        points += 1.0

    # 2. Not excessively extended.
    possible += 1.0

    if (
        price_vs_ema20
        <= V7_MAX_PRICE_VS_EMA20
    ):
        points += 1.0

    # 3. Pullback is still close enough to EMA10.
    possible += 1.0

    distance_from_ema10 = abs(
        price / ema10 - 1.0
    )

    if (
        distance_from_ema10
        <= V7_MAX_DISTANCE_FROM_EMA10
    ):
        points += 1.0

    # 4. Relative strength versus SPY.
    possible += 1.0

    if (
        relative_strength
        >= V7_MIN_RELATIVE_STRENGTH_8
    ):
        points += 0.5

        if relative_strength >= 0.0:
            points += 0.5

    # 5. Confirmation candle quality.
    possible += 1.0

    candle_quality = 0.0

    if body_pct > 0:
        candle_quality += 0.5

    if close_position >= 0.55:
        candle_quality += 0.5

    points += candle_quality

    # 6. Participation / volume.
    possible += 1.0

    if volume_ratio >= 0.70:
        points += 0.5

    if volume_ratio >= 1.00:
        points += 0.5

    # Hard safety checks.
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
            1.0
        )
    )


def passes_v7_setup(row):

    return (
        v7_setup_score(row)
        >= V7_MIN_SETUP_SCORE
    )


# =========================================================
# POSITION + PNL
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


def confidence_bucket(
    probability
):

    if probability < 0.67:
        return "64-67"

    if probability < 0.70:
        return "67-70"

    if probability < 0.75:
        return "70-75"

    return "75+"


# =========================================================
# SINGLE-SYMBOL WALK-FORWARD BACKTEST
# =========================================================

def run_symbol_backtest(
    symbol,
    stock_df,
    benchmark_df,
    test_days
):

    full_data = build_feature_frame(
        stock_df,
        benchmark_df
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

    confidence_stats = {
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

    exit_reasons = {
        "STOP LOSS": 0,
        "TAKE PROFIT": 0,
        "MAX HOLD": 0,
        "END OF TEST": 0,
    }

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
            or index
            >= next_retrain_index
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

            quality = calculate_symbol_quality(
                training_data
            )

            if not quality[
                "passes"
            ]:

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
            or regressor is None
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
                reason = "STOP LOSS"

            elif (
                high
                >= position.take_profit
            ):

                exit_price = (
                    position.take_profit
                )
                reason = "TAKE PROFIT"

            elif (
                index
                - position.entry_index
                >= MAX_HOLD_BARS
            ):

                exit_price = price
                reason = "MAX HOLD"

            if exit_price is not None:

                pnl = calculate_trade_pnl(
                    position.entry_price,
                    exit_price,
                    position.quantity
                )

                gross_equity += pnl[
                    "gross_pnl"
                ]

                total_costs += pnl[
                    "fees"
                ]

                net_equity += pnl[
                    "net_pnl"
                ]

                fixed_quantity = (
                    BASE_TRADE_SIZE
                    / position.entry_price
                )

                fixed_pnl = calculate_trade_pnl(
                    position.entry_price,
                    exit_price,
                    fixed_quantity
                )

                fixed_size_net_equity += (
                    fixed_pnl[
                        "net_pnl"
                    ]
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

                exit_reasons[
                    reason
                ] += 1

                trade_sizes.append(
                    position.value
                )

                trades.append({
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


            # V7: ML only sees sufficiently strong scored setups.
            if not passes_v7_setup(row):
                equity_curve.append(net_equity)
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
                row
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
                    row
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

            atr_pct = float(
                row["atr_pct"]
            )

            trade_size = (
                calculate_trade_size(
                    calibrated_probability,
                    predicted_net,
                    atr_pct,
                    current_setup_score
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
                        1
                        - stop_pct
                    )
                ),
                take_profit=(
                    price
                    * (
                        1
                        + target_pct
                    )
                ),
                stop_pct=stop_pct,
                target_pct=target_pct,
            )

        equity_curve.append(
            net_equity
        )

    # =========================================
    # CLOSE OPEN POSITION
    # =========================================

    if (
        position is not None
        and len(test) > 0
    ):

        final_row = (
            test.iloc[-1]
        )

        exit_price = float(
            final_row["close"]
        )

        pnl = calculate_trade_pnl(
            position.entry_price,
            exit_price,
            position.quantity
        )

        gross_equity += pnl[
            "gross_pnl"
        ]

        total_costs += pnl[
            "fees"
        ]

        net_equity += pnl[
            "net_pnl"
        ]

        fixed_quantity = (
            BASE_TRADE_SIZE
            / position.entry_price
        )

        fixed_pnl = calculate_trade_pnl(
            position.entry_price,
            exit_price,
            fixed_quantity
        )

        fixed_size_net_equity += (
            fixed_pnl[
                "net_pnl"
            ]
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

        exit_reasons[
            "END OF TEST"
        ] += 1

        trade_sizes.append(
            position.value
        )

        trades.append({
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
                "END OF TEST",

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
    all_trades
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
                )
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
            equity
        )

        max_drawdown = min(
            max_drawdown,
            equity - peak
        )

    return max_drawdown


# =========================================================
# COMPLETE BACKTEST
# =========================================================

def run_stock_backtest(
    days=DEFAULT_TEST_DAYS
):

    days = int(
        max(
            10,
            min(
                days,
                40
            )
        )
    )

    total_days = min(
        days
        + TRAINING_LOOKBACK_DAYS,
        MAX_TOTAL_DAYS
    )

    print(
        f"{STRATEGY_NAME}: "
        f"{days} unseen test days"
    )

    print(
        f"Walk-forward retrain every "
        f"{RETRAIN_EVERY_BARS} bars"
    )

    print(
        f"Probability calibration: ON"
    )

    print(
        f"ATR-based exits: ON"
    )

    print(
        f"Dynamic size: "
        f"GBP {MIN_TRADE_SIZE:.0f}"
        f"-{MAX_TRADE_SIZE:.0f}"
    )

    print(
        f"V5 confidence shrinkage: "
        f"{CONFIDENCE_SHRINKAGE:.0%}"
    )

    print(
        f"V5 predicted edge minimum: "
        f"{MIN_PREDICTED_NET_RETURN:.2%}"
    )

    print(
        f"V7 setup-score minimum: "
        f"{V7_MIN_SETUP_SCORE:.0%}"
    )

    print(
        f"Symbols configured: "
        f"{len(SYMBOLS)}"
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
                if symbol
                == BENCHMARK_SYMBOL
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
                f"Dynamic GBP {result['pnl']:+.2f} | "
                f"Fixed GBP {result['fixed_size_pnl']:+.2f}"
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
            "Stock V5 backtest failed for "
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
        reverse=True
    )

    combined_confidence = {
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

    combined_exit_reasons = {
        "STOP LOSS": 0,
        "TAKE PROFIT": 0,
        "MAX HOLD": 0,
        "END OF TEST": 0,
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
            ] += count

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

        confidence_report.append({
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
        })

    average_trade_size = (
        float(
            np.mean(
                [
                    trade["trade_size"]
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

    return {
        "strategy":
            STRATEGY_NAME,

        "interval":
            INTERVAL,

        "days":
            days,

        "training_days":
            TRAINING_LOOKBACK_DAYS,

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
            ranked[0]["symbol"],

        "best_symbol_pnl":
            ranked[0]["pnl"],

        "worst_symbol":
            ranked[-1]["symbol"],

        "worst_symbol_pnl":
            ranked[-1]["pnl"],

        "buy_threshold":
            BUY_THRESHOLD,

        "confidence_shrinkage":
            CONFIDENCE_SHRINKAGE,

        "v7_strategy":
            "trend_pullback_relative_strength_setup_score",

        "v7_min_setup_score":
            V7_MIN_SETUP_SCORE,

        "v7_setup_size_weight":
            V7_SETUP_SIZE_WEIGHT,

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
