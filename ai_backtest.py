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
# SETTINGS
# =========================================================

GRANULARITY = "FIVE_MINUTE"
CANDLE_SECONDS = 300
PAGE_SIZE = 300

BUY_THRESHOLD = 0.62
STOP_LOSS_PCT = 0.02
TAKE_PROFIT_PCT = 0.04
TRADE_SIZE = 25.0

ESTIMATED_FEE_PER_SIDE = float(
    os.getenv(
        "AI_BACKTEST_FEE_PER_SIDE",
        "0.006"
    )
)

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
    "User-Agent": "Alpha-Alerts-Backtest/3.0"
})

os.makedirs(
    CACHE_DIR,
    exist_ok=True
)


FEATURE_COLUMNS = [
    "return_1",
    "return_3",
    "return_6",
    "return_12",
    "ma_ratio_5_20",
    "ma_ratio_20_50",
    "volatility_12",
    "volatility_24",
    "volume_ratio",
    "rsi",
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

        if product.get(
            "status"
        ) != "online":
            continue

        if product.get(
            "quote_currency"
        ) != "USD":
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
        set(
            markets
        )
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

        # Reuse same-day cache.
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
    days,
    df
):
    path = cache_path(
        product_id,
        days
    )

    rows = (
        df.to_dict(
            orient="records"
        )
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
# RATE-LIMIT-SAFE API CALL
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

            text = str(
                exc
            )

            if (
                "429"
                in text
                or "Too Many Requests"
                in text
            ):

                print(
                    f"RATE LIMIT {product_id}: "
                    f"waiting {delay:.0f}s "
                    f"(attempt {attempt + 1}/{MAX_RETRIES})"
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
        f"for {product_id}: {last_error}"
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
            + (
                PAGE_SIZE
                * CANDLE_SECONDS
            ),
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
                    )
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
        pd.DataFrame(
            rows
        )
        .sort_values(
            "time"
        )
        .drop_duplicates(
            subset=[
                "time"
            ]
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
# FEATURES
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

    avg_gain = gain.rolling(
        period
    ).mean()

    avg_loss = loss.rolling(
        period
    ).mean()

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


def build_features(
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

    data[
        "ma_ratio_5_20"
    ] = (
        data["ma_5"]
        / data["ma_20"]
    )

    data[
        "ma_ratio_20_50"
    ] = (
        data["ma_20"]
        / data["ma_50"]
    )

    data[
        "volatility_12"
    ] = (
        data["return_1"]
        .rolling(12)
        .std()
    )

    data[
        "volatility_24"
    ] = (
        data["return_1"]
        .rolling(24)
        .std()
    )

    data[
        "volume_ma_20"
    ] = (
        data["volume"]
        .rolling(20)
        .mean()
    )

    data[
        "volume_ratio"
    ] = (
        data["volume"]
        / data["volume_ma_20"]
    )

    data[
        "rsi"
    ] = (
        calculate_rsi(
            data["close"]
        )
    )

    data[
        "range_pct"
    ] = (
        (
            data["high"]
            - data["low"]
        )
        / data["close"]
    )

    candle_range = (
        data["high"]
        - data["low"]
    ).replace(
        0,
        np.nan
    )

    data[
        "close_position"
    ] = (
        (
            data["close"]
            - data["low"]
        )
        / candle_range
    )

    future_price = (
        data["close"]
        .shift(-3)
    )

    future_return = (
        (
            future_price
            / data["close"]
        )
        - 1
    )

    data[
        "target"
    ] = np.where(
        future_price.notna(),
        (
            future_return
            > 0.0025
        ).astype(int),
        np.nan
    )

    return data


# =========================================================
# MODEL
# =========================================================

def train_model(
    train
):

    X_train = (
        train[
            FEATURE_COLUMNS
        ]
    )

    y_train = (
        train[
            "target"
        ]
        .astype(int)
    )

    if (
        y_train.nunique()
        < 2
    ):

        raise RuntimeError(
            "Training sample contains "
            "only one target class."
        )

    model = (
        RandomForestClassifier(
            n_estimators=160,
            max_depth=7,
            min_samples_leaf=5,
            random_state=42,
            class_weight="balanced",
            n_jobs=1
        )
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
            model.classes_[
                index
            ]
        ):
            float(
                probabilities[
                    index
                ]
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
# BACKTEST ENGINE
# =========================================================

@dataclass
class Position:
    entry_price: float
    quantity: float
    entry_time: int
    entry_probability: float
    stop_loss: float
    take_profit: float


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

    entry_fee = (
        gross_entry
        * ESTIMATED_FEE_PER_SIDE
    )

    exit_fee = (
        gross_exit
        * ESTIMATED_FEE_PER_SIDE
    )

    return (
        gross_exit
        - gross_entry
        - entry_fee
        - exit_fee
    )


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
                    "target"
                ]
            )
        )
        .reset_index(
            drop=True
        )
    )

    if len(
        labelled
    ) < 300:

        raise RuntimeError(
            "Not enough usable history"
        )

    split = int(
        len(
            labelled
        )
        * 0.70
    )

    train = labelled.iloc[
        :split
    ]

    test = (
        labelled.iloc[
            split:
        ]
        .reset_index(
            drop=True
        )
    )

    if len(
        test
    ) < 50:

        raise RuntimeError(
            "Not enough unseen test candles"
        )

    model = train_model(
        train
    )

    position = None

    trades = []

    equity = 0.0

    equity_curve = [
        0.0
    ]

    for _, row in test.iterrows():

        price = float(
            row[
                "close"
            ]
        )

        high = float(
            row[
                "high"
            ]
        )

        low = float(
            row[
                "low"
            ]
        )

        timestamp = int(
            row[
                "time"
            ]
        )

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

            if (
                exit_price
                is not None
            ):

                pnl = (
                    calculate_trade_pnl(
                        position.entry_price,
                        exit_price,
                        position.quantity
                    )
                )

                equity += pnl

                trades.append({
                    "product":
                        product_id,
                    "entry_price":
                        position.entry_price,
                    "exit_price":
                        exit_price,
                    "pnl":
                        pnl,
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

        if position is None:

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
                    entry_probability=up_probability,
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

    if (
        position is not None
        and len(
            test
        ) > 0
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

        pnl = (
            calculate_trade_pnl(
                position.entry_price,
                exit_price,
                position.quantity
            )
        )

        equity += pnl

        trades.append({
            "product":
                product_id,
            "entry_price":
                position.entry_price,
            "exit_price":
                exit_price,
            "pnl":
                pnl,
            "reason":
                "END OF TEST",
            "entry_probability":
                position.entry_probability,
            "entry_time":
                position.entry_time,
            "exit_time":
                int(
                    final_row[
                        "time"
                    ]
                )
        })

        equity_curve.append(
            equity
        )

    wins = sum(
        1
        for trade in trades
        if trade[
            "pnl"
        ] > 0
    )

    losses = (
        len(
            trades
        )
        - wins
    )

    win_rate = (
        wins
        / len(
            trades
        )
        if trades
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
            len(
                raw
            ),
        "test_candles":
            len(
                test
            ),
        "trades":
            len(
                trades
            ),
        "wins":
            wins,
        "losses":
            losses,
        "win_rate":
            win_rate,
        "pnl":
            equity,
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
        f"BACKTEST: discovered "
        f"{len(discovered)} "
        f"Coinbase USD markets."
    )

    for index, product_id in enumerate(
        discovered,
        start=1
    ):

        print(
            f"BACKTEST "
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

    total_trades = sum(
        item[
            "trades"
        ]
        for item in results
    )

    total_wins = sum(
        item[
            "wins"
        ]
        for item in results
    )

    total_losses = sum(
        item[
            "losses"
        ]
        for item in results
    )

    total_pnl = sum(
        item[
            "pnl"
        ]
        for item in results
    )

    combined_win_rate = (
        total_wins
        / total_trades
        if total_trades
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

    best = ranked[
        0
    ]

    worst = ranked[
        -1
    ]

    worst_drawdown = min(
        item[
            "max_drawdown"
        ]
        for item in results
    )

    profitable_markets = sum(
        1
        for item in results
        if item[
            "pnl"
        ] > 0
    )

    losing_markets = sum(
        1
        for item in results
        if item[
            "pnl"
        ] < 0
    )

    flat_markets = (
        len(
            results
        )
        - profitable_markets
        - losing_markets
    )

    return {
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
        "wins":
            total_wins,
        "losses":
            total_losses,
        "win_rate":
            combined_win_rate,
        "pnl":
            total_pnl,
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
            ]
    }


if __name__ == "__main__":

    print(
        run_backtest(
            days=7
        )
    )
