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
# STOCK AI BACKTEST V10
# =========================================================

STRATEGY_NAME = "STOCK_V10_VCP_SWING_180"
INTERVAL = "1h"

DEFAULT_TEST_DAYS = int(os.getenv("STOCK_BACKTEST_DAYS", "180"))
TRAINING_LOOKBACK_DAYS = int(os.getenv("STOCK_TRAINING_LOOKBACK_DAYS", "180"))
MAX_TOTAL_DAYS = int(os.getenv("STOCK_MAX_TOTAL_DAYS", "730"))

# Warmup allows daily 150/200/252-day averages to exist before
# the unseen test period begins.
INDICATOR_WARMUP_DAYS = int(os.getenv("STOCK_INDICATOR_WARMUP_DAYS", "370"))

RETRAIN_EVERY_BARS = int(os.getenv("STOCK_RETRAIN_EVERY_BARS", "42"))


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

BENCHMARK_SYMBOL = os.getenv("STOCK_BENCHMARK_SYMBOL", "SPY").upper()


# =========================================================
# MODEL / ENTRY
# =========================================================

BUY_THRESHOLD = float(os.getenv("STOCK_BUY_THRESHOLD", "0.64"))

MIN_PREDICTED_NET_RETURN = float(
    os.getenv("STOCK_MIN_PREDICTED_NET_RETURN", "0.0040")
)

MIN_PREDICTED_MFE = float(
    os.getenv("STOCK_MIN_PREDICTED_MFE", "0.012")
)

MAX_PREDICTED_MAE = float(
    os.getenv("STOCK_MAX_PREDICTED_MAE", "0.045")
)

MIN_COST_EDGE_MULTIPLIER = float(
    os.getenv("STOCK_MIN_COST_EDGE_MULTIPLIER", "2.0")
)


# =========================================================
# TRANSACTION COSTS
# =========================================================

COMMISSION_PER_SIDE = float(os.getenv("STOCK_COMMISSION_PER_SIDE", "0.0"))
SLIPPAGE_PER_SIDE = float(os.getenv("STOCK_SLIPPAGE_PER_SIDE", "0.0005"))


# =========================================================
# SWING HORIZON
# =========================================================

# ~1 trading week on regular-session hourly bars.
TARGET_HORIZON_BARS = int(os.getenv("STOCK_TARGET_HORIZON_BARS", "42"))

# ~3 trading weeks.
MAX_HOLD_BARS = int(os.getenv("STOCK_MAX_HOLD_BARS", "126"))

COOLDOWN_BARS = int(os.getenv("STOCK_COOLDOWN_BARS", "6"))

DIAGNOSTIC_LOOKAHEAD_BARS = int(
    os.getenv("STOCK_DIAGNOSTIC_LOOKAHEAD_BARS", "126")
)


# =========================================================
# ATR RISK
# =========================================================

STOP_ATR_MULTIPLIER = float(
    os.getenv("STOCK_STOP_ATR_MULTIPLIER", "1.75")
)

TARGET_ATR_MULTIPLIER = float(
    os.getenv("STOCK_TARGET_ATR_MULTIPLIER", "4.0")
)

MIN_STOP_PCT = float(os.getenv("STOCK_MIN_STOP_PCT", "0.012"))
MAX_STOP_PCT = float(os.getenv("STOCK_MAX_STOP_PCT", "0.060"))

MIN_TARGET_PCT = float(os.getenv("STOCK_MIN_TARGET_PCT", "0.025"))
MAX_TARGET_PCT = float(os.getenv("STOCK_MAX_TARGET_PCT", "0.150"))

MIN_RISK_REWARD = float(os.getenv("STOCK_MIN_RISK_REWARD", "2.0"))


# =========================================================
# TRAILING / HOLDING
# =========================================================

TRAIL_ACTIVATION_R = float(os.getenv("STOCK_TRAIL_ACTIVATION_R", "1.50"))
TRAIL_DISTANCE_R = float(os.getenv("STOCK_TRAIL_DISTANCE_R", "1.20"))

TREND_EXIT_AFTER_BARS = int(
    os.getenv("STOCK_TREND_EXIT_AFTER_BARS", "28")
)

BREAK_EVEN_ACTIVATION_R = float(
    os.getenv("STOCK_BREAK_EVEN_ACTIVATION_R", "1.0")
)


# =========================================================
# POSITION SIZING
# =========================================================

MIN_TRADE_SIZE = float(os.getenv("STOCK_MIN_TRADE_SIZE", "50.0"))
BASE_TRADE_SIZE = float(os.getenv("STOCK_BASE_TRADE_SIZE", "100.0"))
MAX_TRADE_SIZE = float(os.getenv("STOCK_MAX_TRADE_SIZE", "175.0"))

POSITION_SIZING_MODE = (
    os.getenv("STOCK_POSITION_SIZING_MODE", "ADAPTIVE_V10")
    .strip()
    .upper()
)


# =========================================================
# BASIC ENTRY FILTERS
# =========================================================

MIN_PRICE = float(os.getenv("STOCK_MIN_PRICE", "10.0"))

MIN_DOLLAR_VOLUME_1H = float(
    os.getenv("STOCK_MIN_DOLLAR_VOLUME_1H", "10000000")
)

MIN_VOLUME_RATIO = float(os.getenv("STOCK_MIN_VOLUME_RATIO", "0.45"))

MIN_ATR_PCT = float(os.getenv("STOCK_MIN_ATR_PCT", "0.002"))
MAX_ATR_PCT = float(os.getenv("STOCK_MAX_ATR_PCT", "0.060"))

MIN_RSI = float(os.getenv("STOCK_MIN_RSI", "40"))
MAX_RSI = float(os.getenv("STOCK_MAX_RSI", "80"))

MIN_BENCHMARK_RETURN_8 = float(
    os.getenv("STOCK_MIN_BENCHMARK_RETURN_8", "-0.025")
)

MIN_RELATIVE_RETURN_8 = float(
    os.getenv("STOCK_MIN_RELATIVE_RETURN_8", "-0.020")
)


# =========================================================
# VCP SETTINGS
# =========================================================

MIN_VCP_SCORE = float(os.getenv("STOCK_MIN_VCP_SCORE", "0.62"))

HIGH_QUALITY_VCP_SCORE = float(
    os.getenv("STOCK_HIGH_QUALITY_VCP_SCORE", "0.78")
)

BREAKOUT_VCP_SCORE = float(
    os.getenv("STOCK_BREAKOUT_VCP_SCORE", "0.70")
)

RESISTANCE_LOOKBACK_BARS = int(
    os.getenv("STOCK_RESISTANCE_LOOKBACK_BARS", "84")
)

CONTRACTION_SHORT_BARS = int(
    os.getenv("STOCK_CONTRACTION_SHORT_BARS", "10")
)

CONTRACTION_LONG_BARS = int(
    os.getenv("STOCK_CONTRACTION_LONG_BARS", "40")
)

HIGHER_LOW_LOOKBACK = int(
    os.getenv("STOCK_HIGHER_LOW_LOOKBACK", "60")
)

MAX_DISTANCE_BELOW_RESISTANCE = float(
    os.getenv("STOCK_MAX_DISTANCE_BELOW_RESISTANCE", "0.075")
)

BREAKOUT_BUFFER = float(os.getenv("STOCK_BREAKOUT_BUFFER", "0.002"))

BREAKOUT_VOLUME_RATIO = float(
    os.getenv("STOCK_BREAKOUT_VOLUME_RATIO", "1.15")
)

PARABOLIC_RETURN_20 = float(
    os.getenv("STOCK_PARABOLIC_RETURN_20", "0.25")
)

PARABOLIC_RETURN_40 = float(
    os.getenv("STOCK_PARABOLIC_RETURN_40", "0.45")
)

MAX_DISTANCE_FROM_200DMA = float(
    os.getenv("STOCK_MAX_DISTANCE_FROM_200DMA", "0.55")
)


# =========================================================
# VCP WEIGHTS
# =========================================================

VCP_WEIGHTS = {
    "higher_lows": 0.16,
    "volatility_contraction": 0.16,
    "range_contraction": 0.12,
    "atr_contraction": 0.10,
    "volume_contraction": 0.10,
    "near_resistance": 0.10,
    "above_150dma": 0.07,
    "above_200dma": 0.07,
    "above_252dma": 0.05,
    "relative_strength": 0.04,
    "not_parabolic": 0.03,
}


# =========================================================
# TRAINING
# =========================================================

MIN_TRAINING_ROWS = int(os.getenv("STOCK_MIN_TRAINING_ROWS", "650"))

MIN_POSITIVE_EXAMPLES = int(
    os.getenv("STOCK_MIN_POSITIVE_EXAMPLES", "25")
)

CALIBRATION_FRACTION = float(
    os.getenv("STOCK_CALIBRATION_FRACTION", "0.20")
)


# =========================================================
# SYMBOL QUALITY
# =========================================================

MIN_SYMBOL_QUALITY_SAMPLES = int(
    os.getenv("STOCK_MIN_SYMBOL_QUALITY_SAMPLES", "20")
)

MIN_SYMBOL_SUCCESS_RATE = float(
    os.getenv("STOCK_MIN_SYMBOL_SUCCESS_RATE", "0.30")
)

MIN_SYMBOL_AVG_NET_RETURN = float(
    os.getenv("STOCK_MIN_SYMBOL_AVG_NET_RETURN", "-0.003")
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
    "return_20",
    "return_40",
    "ema_ratio_5_10",
    "ema_ratio_10_20",
    "ema_ratio_20_50",
    "price_vs_ema10",
    "price_vs_ema20",
    "price_vs_ema50",
    "price_vs_sma150",
    "price_vs_sma200",
    "price_vs_sma252",
    "sma150_vs_sma200",
    "sma200_vs_sma252",
    "volatility_4",
    "volatility_8",
    "volatility_16",
    "volatility_40",
    "volatility_contraction_ratio",
    "range_contraction_ratio",
    "atr_contraction_ratio",
    "volume_contraction_ratio",
    "volume_ratio",
    "volume_zscore",
    "rsi",
    "atr_pct",
    "range_pct",
    "body_pct",
    "close_position",
    "benchmark_return_4",
    "benchmark_return_8",
    "benchmark_return_20",
    "relative_return_4",
    "relative_return_8",
    "relative_return_20",
    "distance_to_resistance",
    "resistance_breakout_pct",
    "higher_low_score",
    "vcp_score",
    "market_above_ema20",
    "stock_above_ema20",
    "is_breakout",
    "is_parabolic",
]


# =========================================================
# DATA
# =========================================================

def download_intraday(symbol, total_days):
    total_days = min(int(total_days), MAX_TOTAL_DAYS)

    df = yf.download(
        symbol,
        period=f"{total_days}d",
        interval=INTERVAL,
        auto_adjust=False,
        progress=False,
        prepost=False,
        threads=False,
    )

    if df is None or df.empty:
        raise RuntimeError(f"No hourly data returned for {symbol}")

    if isinstance(df.columns, pd.MultiIndex):
        try:
            df = df.xs(symbol, axis=1, level=-1)
        except Exception:
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

    required = ["open", "high", "low", "close", "volume"]

    missing = [column for column in required if column not in df.columns]

    if missing:
        raise RuntimeError(f"{symbol} missing columns {missing}")

    df = df[required].dropna().reset_index()

    time_column = "Datetime" if "Datetime" in df.columns else "Date"

    if time_column not in df.columns:
        time_column = df.columns[0]

    df["time"] = pd.to_datetime(df[time_column], utc=True)

    return (
        df[["time", "open", "high", "low", "close", "volume"]]
        .sort_values("time")
        .drop_duplicates("time")
        .reset_index(drop=True)
    )


# =========================================================
# INDICATORS
# =========================================================

def calculate_rsi(series, period=14):
    delta = series.diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)

    return (100 - 100 / (1 + rs)).fillna(50)


def calculate_atr(data, period=14):
    previous_close = data["close"].shift(1)

    true_range = pd.concat(
        [
            data["high"] - data["low"],
            (data["high"] - previous_close).abs(),
            (data["low"] - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    return true_range.ewm(alpha=1 / period, adjust=False).mean()


# =========================================================
# DAILY LONG-TERM TREND
# =========================================================

def build_daily_trend_frame(stock_df):
    daily = (
        stock_df.set_index("time")
        .resample("1D")
        .agg(
            {
                "open": "first",
                "high": "max",
                "low": "min",
                "close": "last",
                "volume": "sum",
            }
        )
        .dropna(subset=["close"])
        .reset_index()
    )

    # These are rolling TRADING-DAY averages because non-trading
    # calendar rows were removed before rolling.
    daily["sma150"] = daily["close"].rolling(150, min_periods=150).mean()
    daily["sma200"] = daily["close"].rolling(200, min_periods=200).mean()
    daily["sma252"] = daily["close"].rolling(252, min_periods=252).mean()

    return daily[["time", "sma150", "sma200", "sma252"]]


# =========================================================
# HIGHER-LOW SCORE
# =========================================================

def calculate_higher_low_score(low_series, lookback=60):
    values = low_series.astype(float).to_numpy()
    output = np.zeros(len(values), dtype=float)

    segment_size = max(5, lookback // 3)

    for i in range(lookback, len(values)):
        window = values[i - lookback : i + 1]

        first = window[:segment_size]
        middle = window[segment_size : segment_size * 2]
        last = window[segment_size * 2 :]

        if len(first) == 0 or len(middle) == 0 or len(last) == 0:
            continue

        low1 = float(np.nanmin(first))
        low2 = float(np.nanmin(middle))
        low3 = float(np.nanmin(last))

        score = 0.0

        if low2 > low1:
            score += 0.5

        if low3 > low2:
            score += 0.5

        output[i] = score

    return pd.Series(output, index=low_series.index)


# =========================================================
# VCP SCORE
# =========================================================

def calculate_vcp_score(row):
    score = 0.0

    higher_low_score = float(row.get("higher_low_score", 0.0))

    if higher_low_score >= 0.5:
        score += VCP_WEIGHTS["higher_lows"] * higher_low_score

    volatility_ratio = float(
        row.get("volatility_contraction_ratio", np.nan)
    )

    if np.isfinite(volatility_ratio):
        if volatility_ratio <= 0.65:
            factor = 1.0
        elif volatility_ratio <= 0.80:
            factor = 0.75
        elif volatility_ratio <= 0.95:
            factor = 0.40
        else:
            factor = 0.0

        score += VCP_WEIGHTS["volatility_contraction"] * factor

    range_ratio = float(row.get("range_contraction_ratio", np.nan))

    if np.isfinite(range_ratio):
        if range_ratio <= 0.70:
            factor = 1.0
        elif range_ratio <= 0.85:
            factor = 0.70
        elif range_ratio <= 1.0:
            factor = 0.35
        else:
            factor = 0.0

        score += VCP_WEIGHTS["range_contraction"] * factor

    atr_ratio = float(row.get("atr_contraction_ratio", np.nan))

    if np.isfinite(atr_ratio):
        if atr_ratio <= 0.75:
            factor = 1.0
        elif atr_ratio <= 0.90:
            factor = 0.65
        elif atr_ratio <= 1.0:
            factor = 0.30
        else:
            factor = 0.0

        score += VCP_WEIGHTS["atr_contraction"] * factor

    volume_contraction = float(
        row.get("volume_contraction_ratio", np.nan)
    )

    if np.isfinite(volume_contraction):
        if volume_contraction <= 0.75:
            factor = 1.0
        elif volume_contraction <= 0.90:
            factor = 0.70
        elif volume_contraction <= 1.0:
            factor = 0.35
        else:
            factor = 0.0

        score += VCP_WEIGHTS["volume_contraction"] * factor

    distance = float(row.get("distance_to_resistance", np.nan))

    if np.isfinite(distance):
        if -0.03 <= distance <= 0.01:
            factor = 1.0
        elif -0.05 <= distance < -0.03:
            factor = 0.75
        elif -MAX_DISTANCE_BELOW_RESISTANCE <= distance < -0.05:
            factor = 0.40
        elif 0.01 < distance <= 0.03:
            factor = 0.50
        else:
            factor = 0.0

        score += VCP_WEIGHTS["near_resistance"] * factor

    if float(row.get("price_vs_sma150", 0.0)) > 1.0:
        score += VCP_WEIGHTS["above_150dma"]

    if float(row.get("price_vs_sma200", 0.0)) > 1.0:
        score += VCP_WEIGHTS["above_200dma"]

    if float(row.get("price_vs_sma252", 0.0)) > 1.0:
        score += VCP_WEIGHTS["above_252dma"]

    relative_return = float(row.get("relative_return_20", 0.0))

    if relative_return >= 0.03:
        factor = 1.0
    elif relative_return >= 0.0:
        factor = 0.70
    elif relative_return >= -0.02:
        factor = 0.30
    else:
        factor = 0.0

    score += VCP_WEIGHTS["relative_strength"] * factor

    if float(row.get("is_parabolic", 1.0)) < 0.5:
        score += VCP_WEIGHTS["not_parabolic"]

    return float(np.clip(score, 0.0, 1.0))


# =========================================================
# FEATURE ENGINEERING
# =========================================================

def build_feature_frame(stock_df, benchmark_df):
    data = stock_df.copy()

    benchmark = benchmark_df[["time", "close"]].rename(
        columns={"close": "benchmark_close"}
    )

    data = pd.merge_asof(
        data.sort_values("time"),
        benchmark.sort_values("time"),
        on="time",
        direction="backward",
        tolerance=pd.Timedelta(hours=2),
    )

    daily_trend = build_daily_trend_frame(stock_df)

    data = pd.merge_asof(
        data.sort_values("time"),
        daily_trend.sort_values("time"),
        on="time",
        direction="backward",
    )

    for n in (1, 2, 4, 8, 16, 20, 40):
        data[f"return_{n}"] = data["close"].pct_change(n)

    data["benchmark_return_4"] = data["benchmark_close"].pct_change(4)
    data["benchmark_return_8"] = data["benchmark_close"].pct_change(8)
    data["benchmark_return_20"] = data["benchmark_close"].pct_change(20)

    data["relative_return_4"] = (
        data["return_4"] - data["benchmark_return_4"]
    )

    data["relative_return_8"] = (
        data["return_8"] - data["benchmark_return_8"]
    )

    data["relative_return_20"] = (
        data["return_20"] - data["benchmark_return_20"]
    )

    for span in (5, 10, 20, 50):
        data[f"ema_{span}"] = data["close"].ewm(
            span=span,
            adjust=False,
        ).mean()

    data["ema_ratio_5_10"] = data["ema_5"] / data["ema_10"]
    data["ema_ratio_10_20"] = data["ema_10"] / data["ema_20"]
    data["ema_ratio_20_50"] = data["ema_20"] / data["ema_50"]

    data["price_vs_ema10"] = data["close"] / data["ema_10"]
    data["price_vs_ema20"] = data["close"] / data["ema_20"]
    data["price_vs_ema50"] = data["close"] / data["ema_50"]

    data["price_vs_sma150"] = data["close"] / data["sma150"]
    data["price_vs_sma200"] = data["close"] / data["sma200"]
    data["price_vs_sma252"] = data["close"] / data["sma252"]

    data["sma150_vs_sma200"] = data["sma150"] / data["sma200"]
    data["sma200_vs_sma252"] = data["sma200"] / data["sma252"]

    for n in (4, 8, 16, 40):
        data[f"volatility_{n}"] = data["return_1"].rolling(n).std()

    short_vol = data["return_1"].rolling(CONTRACTION_SHORT_BARS).std()
    long_vol = data["return_1"].rolling(CONTRACTION_LONG_BARS).std()

    data["volatility_contraction_ratio"] = (
        short_vol / long_vol.replace(0, np.nan)
    )

    data["volume_ma_20"] = data["volume"].rolling(20).mean()
    data["volume_ma_40"] = data["volume"].rolling(40).mean()
    data["volume_std_20"] = data["volume"].rolling(20).std()

    data["volume_ratio"] = (
        data["volume"] / data["volume_ma_20"].replace(0, np.nan)
    )

    data["volume_zscore"] = (
        (data["volume"] - data["volume_ma_20"])
        / data["volume_std_20"].replace(0, np.nan)
    )

    data["volume_contraction_ratio"] = (
        data["volume_ma_20"] / data["volume_ma_40"].replace(0, np.nan)
    )

    data["rsi"] = calculate_rsi(data["close"])
    data["atr"] = calculate_atr(data)

    data["atr_pct"] = (
        data["atr"] / data["close"].replace(0, np.nan)
    )

    data["atr_ma_short"] = (
        data["atr_pct"].rolling(CONTRACTION_SHORT_BARS).mean()
    )

    data["atr_ma_long"] = (
        data["atr_pct"].rolling(CONTRACTION_LONG_BARS).mean()
    )

    data["atr_contraction_ratio"] = (
        data["atr_ma_short"] / data["atr_ma_long"].replace(0, np.nan)
    )

    candle_range = data["high"] - data["low"]

    data["range_pct"] = (
        candle_range / data["close"].replace(0, np.nan)
    )

    data["range_ma_short"] = (
        data["range_pct"].rolling(CONTRACTION_SHORT_BARS).mean()
    )

    data["range_ma_long"] = (
        data["range_pct"].rolling(CONTRACTION_LONG_BARS).mean()
    )

    data["range_contraction_ratio"] = (
        data["range_ma_short"] / data["range_ma_long"].replace(0, np.nan)
    )

    data["body_pct"] = (
        (data["close"] - data["open"])
        / data["open"].replace(0, np.nan)
    )

    data["close_position"] = (
        (data["close"] - data["low"])
        / candle_range.replace(0, np.nan)
    )

    data["dollar_volume"] = data["close"] * data["volume"]

    data["dollar_volume_ma20"] = (
        data["dollar_volume"].rolling(20).mean()
    )

    data["benchmark_ema20"] = (
        data["benchmark_close"].ewm(span=20, adjust=False).mean()
    )

    data["market_above_ema20"] = (
        data["benchmark_close"] > data["benchmark_ema20"]
    ).astype(float)

    data["stock_above_ema20"] = (
        data["close"] > data["ema_20"]
    ).astype(float)

    # Shift prevents the current candle from creating its own
    # "previous resistance" and therefore avoids look-ahead.
    data["resistance"] = (
        data["high"]
        .shift(1)
        .rolling(RESISTANCE_LOOKBACK_BARS)
        .max()
    )

    data["distance_to_resistance"] = (
        data["close"] / data["resistance"] - 1.0
    )

    data["resistance_breakout_pct"] = (
        data["high"] / data["resistance"] - 1.0
    )

    data["higher_low_score"] = calculate_higher_low_score(
        data["low"],
        HIGHER_LOW_LOOKBACK,
    )

    parabolic = (
        (data["return_20"] >= PARABOLIC_RETURN_20)
        | (data["return_40"] >= PARABOLIC_RETURN_40)
        | (
            data["price_vs_sma200"]
            >= (1.0 + MAX_DISTANCE_FROM_200DMA)
        )
    )

    data["is_parabolic"] = (
        parabolic.fillna(False).astype(float)
    )

    data["is_breakout"] = (
        (
            data["close"]
            >= data["resistance"] * (1.0 + BREAKOUT_BUFFER)
        )
        & (
            data["volume_ratio"]
            >= BREAKOUT_VOLUME_RATIO
        )
    ).astype(float)

    data["vcp_score"] = data.apply(
        calculate_vcp_score,
        axis=1,
    )

    data.replace([np.inf, -np.inf], np.nan, inplace=True)

    return data


# =========================================================
# SETUP CLASSIFICATION
# =========================================================

def classify_vcp_setup(row):
    score = float(row["vcp_score"])

    breakout = bool(float(row["is_breakout"]) >= 0.5)

    distance = float(row["distance_to_resistance"])

    if breakout:
        if score >= BREAKOUT_VCP_SCORE:
            return "CONFIRMED BREAKOUT"

        return "BREAKOUT"

    if score >= HIGH_QUALITY_VCP_SCORE:
        if -0.035 <= distance <= 0.005:
            return "BREAKOUT READY"

        return "HIGH QUALITY VCP"

    if score >= MIN_VCP_SCORE:
        return "GOOD VCP"

    if score >= 0.45:
        return "FORMING VCP"

    return "NO VCP"


# =========================================================
# ENTRY FILTER
# =========================================================

def passes_entry_filter(row):
    price = float(row["close"])

    if price < MIN_PRICE:
        return False

    if float(row["dollar_volume_ma20"]) < MIN_DOLLAR_VOLUME_1H:
        return False

    if float(row["volume_ratio"]) < MIN_VOLUME_RATIO:
        return False

    atr_pct = float(row["atr_pct"])

    if not MIN_ATR_PCT <= atr_pct <= MAX_ATR_PCT:
        return False

    rsi = float(row["rsi"])

    if not MIN_RSI <= rsi <= MAX_RSI:
        return False

    if float(row["benchmark_return_8"]) < MIN_BENCHMARK_RETURN_8:
        return False

    if float(row["relative_return_8"]) < MIN_RELATIVE_RETURN_8:
        return False

    # Long-term trend filter.
    if float(row["price_vs_sma150"]) <= 1.0:
        return False

    if float(row["price_vs_sma200"]) <= 1.0:
        return False

    if float(row["price_vs_sma252"]) <= 1.0:
        return False

    if float(row["is_parabolic"]) >= 0.5:
        return False

    if float(row["vcp_score"]) < MIN_VCP_SCORE:
        return False

    return True


# =========================================================
# STOP / TARGET
# =========================================================

def calculate_stop_target_pct(row):
    atr_pct = float(row["atr_pct"])

    stop_pct = float(
        np.clip(
            atr_pct * STOP_ATR_MULTIPLIER,
            MIN_STOP_PCT,
            MAX_STOP_PCT,
        )
    )

    target_pct = float(
        np.clip(
            atr_pct * TARGET_ATR_MULTIPLIER,
            MIN_TARGET_PCT,
            MAX_TARGET_PCT,
        )
    )

    target_pct = max(
        target_pct,
        stop_pct * MIN_RISK_REWARD,
    )

    target_pct = min(
        target_pct,
        MAX_TARGET_PCT,
    )

    return stop_pct, target_pct


# =========================================================
# COSTS
# =========================================================

def round_trip_cost_return(exit_ratio=1.0):
    return (
        COMMISSION_PER_SIDE
        + COMMISSION_PER_SIDE
        + SLIPPAGE_PER_SIDE
        + exit_ratio * SLIPPAGE_PER_SIDE
    )


# =========================================================
# TRAINING TARGET
# =========================================================

def simulate_forward_trade(data, index):
    final_index = index + TARGET_HORIZON_BARS

    if final_index >= len(data):
        return None

    row = data.iloc[index]
    entry = float(row["close"])

    stop_pct, target_pct = calculate_stop_target_pct(row)

    stop = entry * (1.0 - stop_pct)
    target = entry * (1.0 + target_pct)

    maximum_high = entry
    minimum_low = entry

    outcome = "HORIZON"
    exit_price = None

    for future_index in range(index + 1, final_index + 1):
        future = data.iloc[future_index]

        low = float(future["low"])
        high = float(future["high"])

        maximum_high = max(maximum_high, high)
        minimum_low = min(minimum_low, low)

        # Conservative same-candle assumption.
        if low <= stop:
            exit_price = stop
            outcome = "STOP"
            break

        if high >= target:
            exit_price = target
            outcome = "TARGET"
            break

    if exit_price is None:
        exit_price = float(data.iloc[final_index]["close"])

    exit_ratio = exit_price / entry

    gross_return = exit_ratio - 1.0
    costs = round_trip_cost_return(exit_ratio)
    net_return = gross_return - costs

    forward_mfe = maximum_high / entry - 1.0
    forward_mae = minimum_low / entry - 1.0

    target_success = 1 if outcome == "TARGET" else 0

    return {
        "target": target_success,
        "net_return": net_return,
        "forward_mfe": forward_mfe,
        "forward_mae": forward_mae,
        "outcome": outcome,
    }


def build_training_data(feature_data):
    data = feature_data.copy().reset_index(drop=True)

    targets = np.full(len(data), np.nan)
    net_returns = np.full(len(data), np.nan)
    mfes = np.full(len(data), np.nan)
    maes = np.full(len(data), np.nan)

    for index in range(len(data)):
        result = simulate_forward_trade(data, index)

        if result is None:
            continue

        targets[index] = result["target"]
        net_returns[index] = result["net_return"]
        mfes[index] = result["forward_mfe"]
        maes[index] = result["forward_mae"]

    data["target_success"] = targets
    data["target_net_return"] = net_returns
    data["target_forward_mfe"] = mfes
    data["target_forward_mae"] = maes

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

    data["target_success"] = data["target_success"].astype(int)

    return data


# =========================================================
# SYMBOL QUALITY
# =========================================================

def calculate_symbol_quality(training_data):
    eligible = []

    for _, row in training_data.iterrows():
        try:
            if passes_entry_filter(row):
                eligible.append(row)
        except Exception:
            continue

    if len(eligible) < MIN_SYMBOL_QUALITY_SAMPLES:
        return {
            "samples": len(eligible),
            "success_rate": None,
            "average_net_return": None,
            "passes": True,
        }

    eligible = pd.DataFrame(eligible)

    success_rate = float(eligible["target_success"].mean())
    average_net_return = float(eligible["target_net_return"].mean())

    passes = (
        success_rate >= MIN_SYMBOL_SUCCESS_RATE
        and average_net_return >= MIN_SYMBOL_AVG_NET_RETURN
    )

    return {
        "samples": len(eligible),
        "success_rate": success_rate,
        "average_net_return": average_net_return,
        "passes": passes,
    }


# =========================================================
# MODELS
# =========================================================

def build_classifier():
    return RandomForestClassifier(
        n_estimators=550,
        max_depth=9,
        min_samples_leaf=10,
        max_features="sqrt",
        class_weight="balanced_subsample",
        random_state=42,
        n_jobs=-1,
    )


def build_regressor(random_state):
    return RandomForestRegressor(
        n_estimators=450,
        max_depth=9,
        min_samples_leaf=10,
        max_features="sqrt",
        random_state=random_state,
        n_jobs=-1,
    )


def train_models(training_data):
    if len(training_data) < MIN_TRAINING_ROWS:
        raise RuntimeError(
            f"Not enough training rows: {len(training_data)}"
        )

    y_class = training_data["target_success"]

    if y_class.nunique() < 2:
        raise RuntimeError(
            "Training data contains only one target class."
        )

    positive_count = int((y_class == 1).sum())

    if positive_count < MIN_POSITIVE_EXAMPLES:
        raise RuntimeError(
            "Not enough positive training examples."
        )

    split = int(
        len(training_data)
        * (1.0 - CALIBRATION_FRACTION)
    )

    split = max(
        1,
        min(
            split,
            len(training_data) - 1,
        ),
    )

    fit_data = training_data.iloc[:split]
    calibration_data = training_data.iloc[split:]

    # IMPORTANT:
    # Keep the same classifier after fitting the calibrator.
    # V9 fitted a calibrator to one RF and then replaced the RF,
    # which made the calibration mapping inconsistent.
    classifier = build_classifier()

    classifier.fit(
        fit_data[FEATURE_COLUMNS],
        fit_data["target_success"],
    )

    calibrator = None

    if (
        len(calibration_data) >= 50
        and calibration_data["target_success"].nunique() >= 2
    ):
        raw_probs = classifier.predict_proba(
            calibration_data[FEATURE_COLUMNS]
        )[:, 1]

        calibrator = IsotonicRegression(
            y_min=0.0,
            y_max=1.0,
            out_of_bounds="clip",
        )

        calibrator.fit(
            raw_probs,
            calibration_data["target_success"].to_numpy(dtype=float),
        )

    return_model = build_regressor(43)
    return_model.fit(
        training_data[FEATURE_COLUMNS],
        training_data["target_net_return"],
    )

    mfe_model = build_regressor(44)
    mfe_model.fit(
        training_data[FEATURE_COLUMNS],
        training_data["target_forward_mfe"],
    )

    mae_model = build_regressor(45)
    mae_model.fit(
        training_data[FEATURE_COLUMNS],
        training_data["target_forward_mae"],
    )

    return (
        classifier,
        calibrator,
        return_model,
        mfe_model,
        mae_model,
    )


def probability_success(classifier, calibrator, row):
    X = row[FEATURE_COLUMNS].to_frame().T

    raw_probability = float(
        classifier.predict_proba(X)[0][1]
    )

    if calibrator is None:
        return raw_probability, raw_probability

    calibrated_probability = float(
        calibrator.predict([raw_probability])[0]
    )

    return raw_probability, calibrated_probability


def regression_prediction(model, row):
    X = row[FEATURE_COLUMNS].to_frame().T
    return float(model.predict(X)[0])


# =========================================================
# POSITION SIZING
# =========================================================

def calculate_trade_size(
    probability,
    predicted_return,
    predicted_mfe,
    predicted_mae,
    vcp_score,
):
    if POSITION_SIZING_MODE == "FIXED":
        return float(
            np.clip(
                BASE_TRADE_SIZE,
                MIN_TRADE_SIZE,
                MAX_TRADE_SIZE,
            )
        )

    probability_score = np.clip(
        (probability - BUY_THRESHOLD)
        / max(0.01, 0.85 - BUY_THRESHOLD),
        0.0,
        1.0,
    )

    return_score = np.clip(
        predicted_return / 0.06,
        0.0,
        1.0,
    )

    mfe_score = np.clip(
        predicted_mfe / 0.10,
        0.0,
        1.0,
    )

    risk_score = np.clip(
        1.0 - abs(predicted_mae) / 0.06,
        0.0,
        1.0,
    )

    setup_score = np.clip(
        (vcp_score - MIN_VCP_SCORE)
        / max(0.01, 1.0 - MIN_VCP_SCORE),
        0.0,
        1.0,
    )

    conviction = (
        0.30 * probability_score
        + 0.25 * return_score
        + 0.15 * mfe_score
        + 0.10 * risk_score
        + 0.20 * setup_score
    )

    trade_size = (
        MIN_TRADE_SIZE
        + conviction
        * (
            MAX_TRADE_SIZE
            - MIN_TRADE_SIZE
        )
    )

    return float(
        np.clip(
            trade_size,
            MIN_TRADE_SIZE,
            MAX_TRADE_SIZE,
        )
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
    predicted_mfe: float
    predicted_mae: float

    vcp_score: float
    setup_type: str

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
# PNL
# =========================================================

def calculate_trade_pnl(
    entry_price,
    exit_price,
    quantity,
):
    entry_value = entry_price * quantity
    exit_value = exit_price * quantity

    gross_pnl = exit_value - entry_value

    commission = (
        entry_value * COMMISSION_PER_SIDE
        + exit_value * COMMISSION_PER_SIDE
    )

    slippage = (
        entry_value * SLIPPAGE_PER_SIDE
        + exit_value * SLIPPAGE_PER_SIDE
    )

    fees = commission + slippage

    return {
        "gross_pnl": gross_pnl,
        "fees": fees,
        "net_pnl": gross_pnl - fees,
    }


# =========================================================
# DIAGNOSTICS
# =========================================================

def diagnose_trade_path(
    data,
    entry_index,
    exit_index,
    entry_price,
    target_price,
):
    end_index = min(
        entry_index + DIAGNOSTIC_LOOKAHEAD_BARS,
        len(data) - 1,
    )

    future = data.iloc[
        entry_index + 1 : end_index + 1
    ]

    if future.empty:
        return {
            "diagnostic_mfe_pct": 0.0,
            "diagnostic_mae_pct": 0.0,
            "later_hit_target": False,
            "later_recovered_entry": False,
        }

    maximum_price = float(future["high"].max())
    minimum_price = float(future["low"].min())

    mfe = maximum_price / entry_price - 1.0
    mae = minimum_price / entry_price - 1.0

    later_hit_target = False
    later_recovered = False

    if (
        exit_index is not None
        and exit_index < end_index
    ):
        post_exit = data.iloc[
            exit_index + 1 : end_index + 1
        ]

        if not post_exit.empty:
            post_high = float(post_exit["high"].max())

            later_hit_target = post_high >= target_price
            later_recovered = post_high >= entry_price

    return {
        "diagnostic_mfe_pct": float(mfe),
        "diagnostic_mae_pct": float(mae),
        "later_hit_target": bool(later_hit_target),
        "later_recovered_entry": bool(later_recovered),
    }


# =========================================================
# CLOSED TRADE
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

    bars_held = (
        exit_index
        - position.entry_index
    )

    actual_mfe = (
        position.highest_price
        / position.entry_price
        - 1.0
    )

    actual_mae = (
        position.lowest_price
        / position.entry_price
        - 1.0
    )

    return {
        "symbol": symbol,
        "entry_price": position.entry_price,
        "exit_price": exit_price,
        "trade_size": position.value,
        "pnl": pnl["net_pnl"],
        "fixed_size_pnl": fixed_pnl["net_pnl"],
        "gross_pnl": pnl["gross_pnl"],
        "fees": pnl["fees"],
        "reason": reason,
        "raw_probability": position.raw_probability,
        "entry_probability": position.calibrated_probability,
        "predicted_net_return": position.predicted_net_return,
        "predicted_mfe": position.predicted_mfe,
        "predicted_mae": position.predicted_mae,
        "vcp_score": position.vcp_score,
        "setup_type": position.setup_type,
        "stop_pct": position.stop_pct,
        "target_pct": position.target_pct,
        "entry_time": position.entry_time,
        "exit_time": exit_time,
        "bars_held": bars_held,
        "actual_mfe_pct": actual_mfe,
        "actual_mae_pct": actual_mae,
        **diagnostics,
    }


# =========================================================
# REPORT HELPERS
# =========================================================

def confidence_bucket(probability):
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
        "VCP FAILURE": 0,
        "TREND DETERIORATION": 0,
        "MAX HOLD": 0,
        "END OF TEST": 0,
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

    latest_time = full_data["time"].max()

    test_start = (
        latest_time
        - pd.Timedelta(days=test_days)
    )

    required_columns = (
        FEATURE_COLUMNS
        + [
            "ema_10",
            "ema_20",
            "ema_50",
            "sma150",
            "sma200",
            "sma252",
            "dollar_volume_ma20",
            "resistance",
        ]
    )

    test = (
        full_data[
            full_data["time"]
            >= test_start
        ]
        .dropna(
            subset=required_columns
        )
        .reset_index(drop=True)
    )

    if len(test) < 40:
        raise RuntimeError(
            "Not enough unseen test bars."
        )

    position = None
    trades = []

    net_equity = 0.0
    gross_equity = 0.0
    total_costs = 0.0
    fixed_size_equity = 0.0

    equity_curve = [0.0]

    cooldown_until = -1

    classifier = None
    calibrator = None
    return_model = None
    mfe_model = None
    mae_model = None

    next_retrain_index = 0
    retrain_count = 0

    signals_checked = 0
    classifier_passes = 0
    edge_passes = 0

    confidence_stats = empty_confidence_stats()
    exit_reasons = empty_exit_reasons()

    trade_sizes = []
    vcp_scores = []

    for index, row in test.iterrows():
        current_time = row["time"]

        # =====================================
        # WALK-FORWARD RETRAIN
        # =====================================

        if (
            classifier is None
            or index >= next_retrain_index
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
                    & (
                        full_data["time"]
                        >= cutoff
                    )
                ]
                .copy()
                .reset_index(drop=True)
            )

            try:
                training_data = build_training_data(
                    historical
                )

                quality = calculate_symbol_quality(
                    training_data
                )

                if not quality["passes"]:
                    raise RuntimeError(
                        "Symbol quality gate failed"
                    )

                (
                    classifier,
                    calibrator,
                    return_model,
                    mfe_model,
                    mae_model,
                ) = train_models(
                    training_data
                )

                retrain_count += 1

            except Exception:
                classifier = None
                calibrator = None
                return_model = None
                mfe_model = None
                mae_model = None

            next_retrain_index = (
                index
                + RETRAIN_EVERY_BARS
            )

        if classifier is None:
            equity_curve.append(net_equity)
            continue

        price = float(row["close"])
        high = float(row["high"])
        low = float(row["low"])
        current_open = float(row["open"])
        timestamp = str(row["time"])

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

            # IMPORTANT:
            # Track the bar's high/low for diagnostics.
            if high > position.highest_price:
                position.highest_price = high
                position.highest_price_index = index

            if low < position.lowest_price:
                position.lowest_price = low
                position.lowest_price_index = index

            initial_risk = (
                position.entry_price
                * position.stop_pct
            )

            previous_trailing_stop = (
                position.trailing_stop
            )

            # Calculate a new trailing stop from this bar's high,
            # but do NOT let that newly raised stop retrospectively
            # trigger on the same bar's earlier low. This avoids
            # impossible intrabar ordering.
            new_trailing_stop = previous_trailing_stop

            break_even_activation = (
                position.entry_price
                + initial_risk
                * BREAK_EVEN_ACTIVATION_R
            )

            if (
                position.highest_price
                >= break_even_activation
            ):
                new_trailing_stop = max(
                    new_trailing_stop,
                    position.entry_price,
                )

            trail_activation = (
                position.entry_price
                + initial_risk
                * TRAIL_ACTIVATION_R
            )

            if (
                position.highest_price
                >= trail_activation
            ):
                trail_candidate = (
                    position.highest_price
                    - initial_risk
                    * TRAIL_DISTANCE_R
                )

                new_trailing_stop = max(
                    new_trailing_stop,
                    trail_candidate,
                    position.entry_price,
                )

            # Existing stop is what can be hit during this bar.
            effective_stop_this_bar = max(
                position.stop_loss,
                previous_trailing_stop,
            )

            # Gap below stop.
            if (
                current_open
                <= effective_stop_this_bar
            ):
                exit_price = current_open

                reason = (
                    "GAP/TRAIL STOP"
                    if (
                        previous_trailing_stop
                        > position.stop_loss
                    )
                    else "GAP/STOP LOSS"
                )

            # Stop touched intrabar.
            elif (
                low
                <= effective_stop_this_bar
            ):
                exit_price = (
                    effective_stop_this_bar
                )

                reason = (
                    "TRAILING STOP"
                    if (
                        previous_trailing_stop
                        > position.stop_loss
                    )
                    else "STOP LOSS"
                )

            # Gap above target.
            elif (
                current_open
                >= position.take_profit
            ):
                exit_price = current_open
                reason = "GAP/TAKE PROFIT"

            # Target touched intrabar.
            elif (
                high
                >= position.take_profit
            ):
                exit_price = (
                    position.take_profit
                )

                reason = "TAKE PROFIT"

            elif (
                bars_held >= 16
                and float(
                    row["price_vs_ema50"]
                ) < 0.985
                and float(
                    row["relative_return_20"]
                ) < -0.04
            ):
                exit_price = price
                reason = "VCP FAILURE"

            elif (
                bars_held
                >= TREND_EXIT_AFTER_BARS
                and float(
                    row["price_vs_sma150"]
                ) < 0.99
                and float(
                    row["price_vs_ema50"]
                ) < 0.99
            ):
                exit_price = price
                reason = "TREND DETERIORATION"

            elif (
                bars_held
                >= MAX_HOLD_BARS
            ):
                exit_price = price
                reason = "MAX HOLD"

            # Only carry forward newly raised trailing level if the
            # position survived the current bar.
            if exit_price is None:
                position.trailing_stop = (
                    new_trailing_stop
                )

            if exit_price is not None:
                trade = build_closed_trade(
                    symbol=symbol,
                    position=position,
                    exit_price=exit_price,
                    reason=reason,
                    exit_index=index,
                    exit_time=timestamp,
                    test=test,
                )

                gross_equity += trade["gross_pnl"]
                total_costs += trade["fees"]
                net_equity += trade["pnl"]
                fixed_size_equity += trade["fixed_size_pnl"]

                bucket = confidence_bucket(
                    position.calibrated_probability
                )

                confidence_stats[bucket]["trades"] += 1

                if trade["pnl"] > 0:
                    confidence_stats[bucket]["wins"] += 1

                confidence_stats[bucket]["pnl"] += trade["pnl"]

                exit_reasons[reason] = (
                    exit_reasons.get(reason, 0)
                    + 1
                )

                trade_sizes.append(
                    position.value
                )

                vcp_scores.append(
                    position.vcp_score
                )

                trades.append(trade)

                position = None

                cooldown_until = (
                    index
                    + COOLDOWN_BARS
                )

        # =====================================
        # SEARCH FOR ENTRY
        # =====================================

        if position is None:
            if index < cooldown_until:
                equity_curve.append(net_equity)
                continue

            try:
                if not passes_entry_filter(row):
                    equity_curve.append(net_equity)
                    continue
            except Exception:
                equity_curve.append(net_equity)
                continue

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
                equity_curve.append(net_equity)
                continue

            classifier_passes += 1

            predicted_return = regression_prediction(
                return_model,
                row,
            )

            predicted_mfe = regression_prediction(
                mfe_model,
                row,
            )

            predicted_mae = regression_prediction(
                mae_model,
                row,
            )

            if (
                predicted_return
                < MIN_PREDICTED_NET_RETURN
            ):
                equity_curve.append(net_equity)
                continue

            if predicted_mfe < MIN_PREDICTED_MFE:
                equity_curve.append(net_equity)
                continue

            if predicted_mae < -MAX_PREDICTED_MAE:
                equity_curve.append(net_equity)
                continue

            expected_cost = round_trip_cost_return()

            if (
                predicted_return
                < (
                    expected_cost
                    * MIN_COST_EDGE_MULTIPLIER
                )
            ):
                equity_curve.append(net_equity)
                continue

            edge_passes += 1

            vcp_score = float(row["vcp_score"])
            setup_type = classify_vcp_setup(row)

            trade_size = calculate_trade_size(
                probability=calibrated_probability,
                predicted_return=predicted_return,
                predicted_mfe=predicted_mfe,
                predicted_mae=predicted_mae,
                vcp_score=vcp_score,
            )

            quantity = trade_size / price

            stop_pct, target_pct = (
                calculate_stop_target_pct(row)
            )

            position = Position(
                entry_price=price,
                quantity=quantity,
                value=trade_size,
                entry_time=timestamp,
                entry_index=index,
                raw_probability=raw_probability,
                calibrated_probability=calibrated_probability,
                predicted_net_return=predicted_return,
                predicted_mfe=predicted_mfe,
                predicted_mae=predicted_mae,
                vcp_score=vcp_score,
                setup_type=setup_type,
                stop_loss=price * (1.0 - stop_pct),
                take_profit=price * (1.0 + target_pct),
                stop_pct=stop_pct,
                target_pct=target_pct,
                highest_price=price,
                lowest_price=price,
                highest_price_index=index,
                lowest_price_index=index,
                trailing_stop=0.0,
            )

        equity_curve.append(net_equity)

    # =========================================
    # CLOSE OPEN POSITION
    # =========================================

    if (
        position is not None
        and len(test) > 0
    ):
        final_index = len(test) - 1
        final_row = test.iloc[final_index]

        final_high = float(final_row["high"])
        final_low = float(final_row["low"])

        if final_high > position.highest_price:
            position.highest_price = final_high
            position.highest_price_index = final_index

        if final_low < position.lowest_price:
            position.lowest_price = final_low
            position.lowest_price_index = final_index

        trade = build_closed_trade(
            symbol=symbol,
            position=position,
            exit_price=float(final_row["close"]),
            reason="END OF TEST",
            exit_index=final_index,
            exit_time=str(final_row["time"]),
            test=test,
        )

        trades.append(trade)

        gross_equity += trade["gross_pnl"]
        total_costs += trade["fees"]
        net_equity += trade["pnl"]
        fixed_size_equity += trade["fixed_size_pnl"]

        bucket = confidence_bucket(
            position.calibrated_probability
        )

        confidence_stats[bucket]["trades"] += 1

        if trade["pnl"] > 0:
            confidence_stats[bucket]["wins"] += 1

        confidence_stats[bucket]["pnl"] += trade["pnl"]

        exit_reasons["END OF TEST"] += 1

        trade_sizes.append(position.value)
        vcp_scores.append(position.vcp_score)

        equity_curve.append(net_equity)

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

    winning_pnl = sum(wins)
    losing_pnl = abs(sum(losses))

    profit_factor = (
        winning_pnl / losing_pnl
        if losing_pnl > 0
        else (
            float("inf")
            if winning_pnl > 0
            else 0.0
        )
    )

    peak = 0.0
    max_drawdown = 0.0

    for value in equity_curve:
        peak = max(peak, value)
        max_drawdown = min(
            max_drawdown,
            value - peak,
        )

    total_trades = len(trades)

    return {
        "symbol": symbol,
        "retrain_count": retrain_count,
        "signals_checked": signals_checked,
        "classifier_passes": classifier_passes,
        "edge_passes": edge_passes,
        "average_setup_score": (
            float(np.mean(vcp_scores))
            if vcp_scores
            else 0.0
        ),
        "trades": total_trades,
        "wins": len(wins),
        "losses": len(losses),
        "win_rate": (
            len(wins) / total_trades
            if total_trades
            else 0.0
        ),
        "gross_pnl": gross_equity,
        "fees": total_costs,
        "pnl": net_equity,
        "fixed_size_pnl": fixed_size_equity,
        "average_trade_size": (
            float(np.mean(trade_sizes))
            if trade_sizes
            else 0.0
        ),
        "average_win": (
            float(np.mean(wins))
            if wins
            else 0.0
        ),
        "average_loss": (
            float(np.mean(losses))
            if losses
            else 0.0
        ),
        "profit_factor": profit_factor,
        "expectancy": (
            net_equity / total_trades
            if total_trades
            else 0.0
        ),
        "max_drawdown": max_drawdown,
        "confidence_stats": confidence_stats,
        "exit_reasons": exit_reasons,
        "trade_log": trades,
    }


# =========================================================
# PORTFOLIO DRAWDOWN
# =========================================================

def calculate_portfolio_drawdown(all_trades):
    if not all_trades:
        return 0.0

    events = []

    for trade in all_trades:
        try:
            exit_time = pd.Timestamp(
                trade["exit_time"]
            )

            events.append(
                (
                    exit_time,
                    float(trade["pnl"]),
                )
            )
        except Exception:
            continue

    events.sort(key=lambda item: item[0])

    equity = 0.0
    peak = 0.0
    max_drawdown = 0.0

    for _, pnl in events:
        equity += pnl
        peak = max(peak, equity)
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
        or benchmark_df.empty
    ):
        raise RuntimeError(
            "Benchmark data is unavailable."
        )

    first_time = pd.Timestamp(
        benchmark_df["time"].min()
    )

    last_time = pd.Timestamp(
        benchmark_df["time"].max()
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

    minimum_pretest_history = (
        TRAINING_LOOKBACK_DAYS
        + 30
    )

    maximum_valid_test_days = max(
        10,
        available_days
        - minimum_pretest_history,
    )

    resolved_days = min(
        int(requested_days),
        maximum_valid_test_days,
    )

    return {
        "requested_days": int(requested_days),
        "resolved_days": int(resolved_days),
        "available_days": int(available_days),
        "estimated_training_days": int(
            max(
                0,
                available_days
                - resolved_days,
            )
        ),
    }


# =========================================================
# V10 DIAGNOSIS
# =========================================================

def build_v10_diagnosis(trades):
    if not trades:
        return (
            "NO TRADES - VCP/trend filters or model "
            "thresholds were too restrictive."
        )

    if len(trades) < 10:
        return (
            "TOO FEW TRADES - insufficient sample size "
            "for a strong conclusion."
        )

    wins = [
        trade
        for trade in trades
        if trade["pnl"] > 0
    ]

    stopped = [
        trade
        for trade in trades
        if "STOP" in trade["reason"]
    ]

    recovered = [
        trade
        for trade in stopped
        if trade.get(
            "later_recovered_entry",
            False,
        )
    ]

    later_targets = [
        trade
        for trade in stopped
        if trade.get(
            "later_hit_target",
            False,
        )
    ]

    win_rate = len(wins) / len(trades)

    avg_mfe = float(
        np.mean(
            [
                trade.get(
                    "diagnostic_mfe_pct",
                    0.0,
                )
                for trade in trades
            ]
        )
    )

    avg_mae = float(
        np.mean(
            [
                trade.get(
                    "diagnostic_mae_pct",
                    0.0,
                )
                for trade in trades
            ]
        )
    )

    later_target_rate = (
        len(later_targets)
        / len(stopped)
        if stopped
        else 0.0
    )

    recovery_rate = (
        len(recovered)
        / len(stopped)
        if stopped
        else 0.0
    )

    if later_target_rate >= 0.40:
        return (
            "ENTRY EDGE LOOKS PROMISING - many stopped "
            "VCP trades later reached their original target. "
            "Exit logic may still be too aggressive."
        )

    if recovery_rate >= 0.60:
        return (
            "STOPS MAY STILL BE TOO TIGHT - most stopped "
            "positions later recovered to entry."
        )

    if avg_mfe <= abs(avg_mae):
        return (
            "ENTRY QUALITY LOOKS WEAK - average adverse "
            "movement is comparable to or greater than "
            "favourable movement."
        )

    if (
        win_rate >= 0.50
        and avg_mfe > abs(avg_mae)
    ):
        return (
            "PROMISING V10 RESULT - entries show favourable "
            "movement and acceptable realised win rate."
        )

    return (
        "MIXED V10 RESULT - inspect expectancy, profit "
        "factor, drawdown, setup quality and exit reasons."
    )


# =========================================================
# COMPLETE BACKTEST
# =========================================================

def run_stock_backtest(days=DEFAULT_TEST_DAYS):
    days = int(
        max(
            10,
            min(
                int(days),
                180,
            ),
        )
    )

    requested_days = days

    total_days = min(
        requested_days
        + TRAINING_LOOKBACK_DAYS
        + INDICATOR_WARMUP_DAYS,
        MAX_TOTAL_DAYS,
    )

    benchmark_df = download_intraday(
        BENCHMARK_SYMBOL,
        total_days,
    )

    history_info = resolve_valid_test_days(
        benchmark_df,
        requested_days,
    )

    days = history_info["resolved_days"]

    print("=" * 60)
    print(STRATEGY_NAME)
    print("=" * 60)

    print(
        f"Requested unseen days: "
        f"{requested_days}"
    )

    print(
        f"Resolved unseen days: "
        f"{days}"
    )

    print(
        f"Available history: "
        f"~{history_info['available_days']} days"
    )

    print(
        f"Training lookback: "
        f"{TRAINING_LOOKBACK_DAYS} days"
    )

    print(
        f"Indicator warmup: "
        f"{INDICATOR_WARMUP_DAYS} days"
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
        f"Minimum VCP score: "
        f"{MIN_VCP_SCORE:.0%}"
    )

    print(
        f"Buy threshold: "
        f"{BUY_THRESHOLD:.0%}"
    )

    print(
        f"Position sizing: "
        f"{POSITION_SIZING_MODE}"
    )

    print(
        f"Symbols: "
        f"{len(SYMBOLS)}"
    )

    print("=" * 60)

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

            results.append(result)

            print(
                f"{symbol}: "
                f"{result['trades']} trades | "
                f"{result['win_rate'] * 100:.1f}% WR | "
                f"GBP {result['pnl']:+.2f} | "
                f"VCP {result['average_setup_score']:.2f}"
            )

        except Exception as exc:
            message = f"{symbol}: {exc}"

            errors.append(message)

            print(
                f"SKIP {message}"
            )

    if not results:
        raise RuntimeError(
            "Stock V10 backtest failed for "
            "every configured symbol."
        )

    all_trades = []

    for item in results:
        all_trades.extend(
            item["trade_log"]
        )

    total_trades = len(all_trades)

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

    winning_pnl = sum(wins)
    losing_pnl = abs(sum(losses))

    profit_factor = (
        winning_pnl / losing_pnl
        if losing_pnl > 0
        else (
            float("inf")
            if winning_pnl > 0
            else 0.0
        )
    )

    ranked = sorted(
        results,
        key=lambda item: item["pnl"],
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
            item["confidence_stats"].items()
        ):
            combined_confidence[bucket]["trades"] += stats["trades"]
            combined_confidence[bucket]["wins"] += stats["wins"]
            combined_confidence[bucket]["pnl"] += stats["pnl"]

        for reason, count in (
            item["exit_reasons"].items()
        ):
            combined_exit_reasons[reason] = (
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
        trades_count = stats["trades"]
        wins_count = stats["wins"]

        confidence_report.append(
            {
                "bucket": bucket,
                "trades": trades_count,
                "wins": wins_count,
                "win_rate": (
                    wins_count / trades_count
                    if trades_count
                    else 0.0
                ),
                "pnl": stats["pnl"],
            }
        )

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

    stopped_trades = [
        trade
        for trade in all_trades
        if "STOP" in trade["reason"]
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

    average_vcp_score = (
        float(
            np.mean(
                [
                    trade.get(
                        "vcp_score",
                        0.0,
                    )
                    for trade in all_trades
                ]
            )
        )
        if all_trades
        else 0.0
    )

    average_predicted_return = (
        float(
            np.mean(
                [
                    trade.get(
                        "predicted_net_return",
                        0.0,
                    )
                    for trade in all_trades
                ]
            )
        )
        if all_trades
        else 0.0
    )

    average_predicted_mfe = (
        float(
            np.mean(
                [
                    trade.get(
                        "predicted_mfe",
                        0.0,
                    )
                    for trade in all_trades
                ]
            )
        )
        if all_trades
        else 0.0
    )

    average_predicted_mae = (
        float(
            np.mean(
                [
                    trade.get(
                        "predicted_mae",
                        0.0,
                    )
                    for trade in all_trades
                ]
            )
        )
        if all_trades
        else 0.0
    )

    v10_diagnosis = (
        build_v10_diagnosis(
            all_trades
        )
    )

    return {
        "strategy": STRATEGY_NAME,
        "interval": INTERVAL,
        "days": days,
        "requested_days": requested_days,
        "available_history_days": history_info["available_days"],
        "estimated_training_days": history_info["estimated_training_days"],
        "training_days": TRAINING_LOOKBACK_DAYS,
        "target_horizon_bars": TARGET_HORIZON_BARS,
        "max_hold_bars": MAX_HOLD_BARS,
        "diagnostic_lookahead_bars": DIAGNOSTIC_LOOKAHEAD_BARS,
        "symbols_configured": len(SYMBOLS),
        "symbols_completed": len(results),
        "symbols_skipped": len(errors),
        "retrain_count": sum(
            item["retrain_count"]
            for item in results
        ),
        "signals_checked": sum(
            item["signals_checked"]
            for item in results
        ),
        "classifier_passes": sum(
            item["classifier_passes"]
            for item in results
        ),
        "edge_passes": sum(
            item["edge_passes"]
            for item in results
        ),
        "average_setup_score": average_vcp_score,
        "average_vcp_score": average_vcp_score,
        "trades": total_trades,
        "trades_per_day": (
            total_trades / days
            if days
            else 0.0
        ),
        "wins": total_wins,
        "losses": total_losses,
        "win_rate": (
            total_wins / total_trades
            if total_trades
            else 0.0
        ),
        "gross_pnl": total_gross_pnl,
        "fees": total_fees,
        "pnl": total_pnl,
        "fixed_size_pnl": total_fixed_size_pnl,
        "dynamic_sizing_improvement": (
            total_pnl
            - total_fixed_size_pnl
        ),
        "average_trade_size": average_trade_size,
        "min_trade_size": MIN_TRADE_SIZE,
        "max_trade_size": MAX_TRADE_SIZE,
        "position_sizing_mode": POSITION_SIZING_MODE,
        "average_win": (
            sum(wins) / len(wins)
            if wins
            else 0.0
        ),
        "average_loss": (
            sum(losses) / len(losses)
            if losses
            else 0.0
        ),
        "profit_factor": profit_factor,
        "expectancy": (
            total_pnl / total_trades
            if total_trades
            else 0.0
        ),
        "max_drawdown": portfolio_drawdown,
        "worst_symbol_drawdown": min(
            item["max_drawdown"]
            for item in results
        ),
        "best_symbol": ranked[0]["symbol"],
        "best_symbol_pnl": ranked[0]["pnl"],
        "worst_symbol": ranked[-1]["symbol"],
        "worst_symbol_pnl": ranked[-1]["pnl"],
        "buy_threshold": BUY_THRESHOLD,
        "min_benchmark_return_8": MIN_BENCHMARK_RETURN_8,
        "min_predicted_net_return": MIN_PREDICTED_NET_RETURN,
        "min_vcp_score": MIN_VCP_SCORE,
        "commission_per_side": COMMISSION_PER_SIDE,
        "slippage_per_side": SLIPPAGE_PER_SIDE,
        "confidence_report": confidence_report,
        "exit_reasons": combined_exit_reasons,
        "average_predicted_return": average_predicted_return,
        "average_predicted_mfe": average_predicted_mfe,
        "average_predicted_mae": average_predicted_mae,

        # Retained so an existing Discord formatter using the V9
        # diagnostics key does not immediately break.
        "v9_diagnostics": {
            "stopped_trades": len(stopped_trades),
            "stopped_later_hit_target": len(stopped_later_hit_target),
            "stopped_later_hit_target_rate": (
                len(stopped_later_hit_target)
                / len(stopped_trades)
                if stopped_trades
                else 0.0
            ),
            "stopped_later_recovered_entry": len(stopped_later_recovered),
            "stopped_later_recovered_rate": (
                len(stopped_later_recovered)
                / len(stopped_trades)
                if stopped_trades
                else 0.0
            ),
            "average_actual_mfe_pct": average_actual_mfe,
            "average_actual_mae_pct": average_actual_mae,
            "average_diagnostic_mfe_pct": average_diagnostic_mfe,
            "average_diagnostic_mae_pct": average_diagnostic_mae,
            "winner_average_mfe_pct": winner_average_mfe,
            "loser_average_mfe_pct": loser_average_mfe,
            "average_bars_held": average_bars_held,
            "diagnosis": v10_diagnosis,
        },

        "v10_diagnostics": {
            "average_vcp_score": average_vcp_score,
            "average_predicted_return": average_predicted_return,
            "average_predicted_mfe": average_predicted_mfe,
            "average_predicted_mae": average_predicted_mae,
            "average_bars_held": average_bars_held,
            "diagnosis": v10_diagnosis,
        },

        "top_symbols": ranked[:10],
        "bottom_symbols": list(reversed(ranked[-10:])),
        "trade_log": all_trades,
        "errors": errors[:20],
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
