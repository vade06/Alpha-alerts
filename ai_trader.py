import json
import os
import time
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd

from coinbase.rest import RESTClient
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


# =========================================================
# AI TRADER SETTINGS
# =========================================================

PRODUCTS = [
    "BTC-USD",
    "ETH-USD",
    "SOL-USD",
    "XRP-USD",
    "LINK-USD",
    "ADA-USD",
]

GRANULARITY = "FIVE_MINUTE"

STARTING_BALANCE = 1000.0
TRADE_SIZE = 25.0

# The AI will only open a paper trade when its estimated
# probability of the target upward move is at least this high.
BUY_THRESHOLD = 0.62

# Below this probability, the market is labelled bearish.
BEARISH_THRESHOLD = 0.38

STOP_LOSS_PCT = 0.02
TAKE_PROFIT_PCT = 0.04

STATE_FILE = "paper_state.json"
MODEL_DIR = "models"

# Number of 5-minute candles requested per market.
CANDLE_LIMIT = 300

# Target:
# price must rise by at least 0.25% over the next 3 candles
# (approximately 15 minutes).
TARGET_RETURN = 0.0025
TARGET_CANDLES = 3


client = RESTClient()

os.makedirs(
    MODEL_DIR,
    exist_ok=True
)


# =========================================================
# STATE
# =========================================================

def default_state():

    return {
        "cash": STARTING_BALANCE,
        "position": None,
        "trades": [],
        "wins": 0,
        "losses": 0,
        "realized_pnl": 0.0
    }


def load_state():

    if not os.path.exists(
        STATE_FILE
    ):

        return default_state()

    try:

        with open(
            STATE_FILE,
            "r"
        ) as f:

            state = json.load(f)

    except Exception:

        return default_state()

    # Backwards compatibility if an older state file
    # is missing any newer fields.
    baseline = default_state()

    for key, value in baseline.items():

        if key not in state:
            state[key] = value

    return state


def save_state(
    state
):

    with open(
        STATE_FILE,
        "w"
    ) as f:

        json.dump(
            state,
            f,
            indent=2
        )


# =========================================================
# COINBASE CANDLES
# =========================================================

def get_candles(
    product_id,
    limit=CANDLE_LIMIT
):

    now = int(
        time.time()
    )

    start = (
        now
        - (
            limit
            * 300
        )
    )

    response = (
        client.get_public_candles(
            product_id=product_id,
            start=str(start),
            end=str(now),
            granularity=GRANULARITY,
            limit=limit
        )
    )

    candles = []

    for candle in response.candles:

        candles.append({
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

    if not candles:

        raise RuntimeError(
            f"No Coinbase candles returned "
            f"for {product_id}"
        )

    df = pd.DataFrame(
        candles
    )

    df = (
        df.sort_values(
            "time"
        )
        .drop_duplicates(
            subset=["time"]
        )
        .reset_index(
            drop=True
        )
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
        gain.rolling(
            period
        )
        .mean()
    )

    avg_loss = (
        loss.rolling(
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

    data["ma_ratio_5_20"] = (
        data["ma_5"]
        / data["ma_20"]
    )

    data["ma_ratio_20_50"] = (
        data["ma_20"]
        / data["ma_50"]
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

    data["volume_ratio"] = (
        data["volume"]
        / data["volume_ma_20"]
    )

    data["rsi"] = (
        calculate_rsi(
            data["close"]
        )
    )

    data["range_pct"] = (
        (
            data["high"]
            - data["low"]
        )
        / data["close"]
    )

    data["close_position"] = (
        (
            data["close"]
            - data["low"]
        )
        / (
            (
                data["high"]
                - data["low"]
            )
            .replace(
                0,
                np.nan
            )
        )
    )

    future_price = (
        data["close"]
        .shift(
            -TARGET_CANDLES
        )
    )

    future_return = (
        (
            future_price
            / data["close"]
        )
        - 1
    )

    data["target"] = (
        future_return
        > TARGET_RETURN
    ).astype(
        int
    )

    data = (
        data.dropna()
        .reset_index(
            drop=True
        )
    )

    return data


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
# MODEL
# =========================================================

def model_path(
    product_id
):

    safe_name = (
        product_id
        .replace(
            "-",
            "_"
        )
    )

    return os.path.join(
        MODEL_DIR,
        f"{safe_name}.joblib"
    )


def train_model(
    product_id,
    data
):

    if len(data) < 100:

        raise RuntimeError(
            f"Not enough usable candles "
            f"for {product_id}"
        )

    split = int(
        len(data)
        * 0.80
    )

    train = data.iloc[
        :split
    ]

    test = data.iloc[
        split:
    ]

    X_train = (
        train[
            FEATURE_COLUMNS
        ]
    )

    y_train = (
        train[
            "target"
        ]
    )

    X_test = (
        test[
            FEATURE_COLUMNS
        ]
    )

    y_test = (
        test[
            "target"
        ]
    )

    # A classifier needs both target classes.
    if (
        y_train.nunique()
        < 2
    ):

        raise RuntimeError(
            f"Training data for "
            f"{product_id} only contains "
            f"one target class"
        )

    model = (
        RandomForestClassifier(
            n_estimators=350,
            max_depth=7,
            min_samples_leaf=5,
            random_state=42,
            class_weight="balanced",
            n_jobs=-1
        )
    )

    model.fit(
        X_train,
        y_train
    )

    accuracy = None

    if (
        len(X_test) > 0
        and y_test.nunique() >= 1
    ):

        predictions = (
            model.predict(
                X_test
            )
        )

        accuracy = (
            accuracy_score(
                y_test,
                predictions
            )
        )

    joblib.dump(
        model,
        model_path(
            product_id
        )
    )

    return (
        model,
        accuracy
    )


def load_or_train_model(
    product_id,
    data
):

    path = model_path(
        product_id
    )

    if os.path.exists(
        path
    ):

        try:

            model = joblib.load(
                path
            )

            return (
                model,
                None
            )

        except Exception:

            pass

    return train_model(
        product_id,
        data
    )


def probability_up(
    model,
    latest_row
):

    X = (
        latest_row[
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

    class_map = {
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

    return class_map.get(
        1,
        0.0
    )


# =========================================================
# MARKET SCANNER
# =========================================================

def classify_market(
    up_probability
):

    if (
        up_probability
        >= BUY_THRESHOLD
    ):

        return (
            "BUY",
            up_probability
        )

    if (
        up_probability
        <= BEARISH_THRESHOLD
    ):

        return (
            "BEARISH",
            1.0
            - up_probability
        )

    # HOLD confidence measures how strongly the probability
    # sits inside the neutral zone.
    neutral_distance = abs(
        up_probability
        - 0.50
    )

    hold_confidence = max(
        0.50,
        1.0
        - (
            neutral_distance
            * 2
        )
    )

    return (
        "HOLD",
        hold_confidence
    )


def analyse_product(
    product_id
):

    df = get_candles(
        product_id
    )

    data = build_features(
        df
    )

    model, accuracy = (
        load_or_train_model(
            product_id,
            data
        )
    )

    latest = data.iloc[
        -1
    ]

    price = float(
        latest[
            "close"
        ]
    )

    up_probability = (
        probability_up(
            model,
            latest
        )
    )

    decision, confidence = (
        classify_market(
            up_probability
        )
    )

    return {
        "product":
            product_id,
        "price":
            price,
        "decision":
            decision,
        "confidence":
            confidence,
        "probability_up":
            up_probability,
        "model_accuracy":
            accuracy,
        "rsi":
            float(
                latest[
                    "rsi"
                ]
            ),
        "volume_ratio":
            float(
                latest[
                    "volume_ratio"
                ]
            ),
        "return_3":
            float(
                latest[
                    "return_3"
                ]
            ),
        "return_12":
            float(
                latest[
                    "return_12"
                ]
            ),
    }


def scan_markets():

    results = []

    errors = []

    for product_id in PRODUCTS:

        try:

            result = analyse_product(
                product_id
            )

            results.append(
                result
            )

        except Exception as exc:

            errors.append(
                f"{product_id}: {exc}"
            )

            print(
                f"AI scan error "
                f"{product_id}: {exc}"
            )

    if not results:

        raise RuntimeError(
            "AI could not analyse "
            "any configured markets"
        )

    # Highest estimated upside probability first.
    results.sort(
        key=lambda item:
            item[
                "probability_up"
            ],
        reverse=True
    )

    return (
        results,
        errors
    )


# =========================================================
# PAPER TRADING
# =========================================================

def open_position(
    state,
    market
):

    if (
        state[
            "position"
        ]
        is not None
    ):

        return None

    if (
        state[
            "cash"
        ]
        < TRADE_SIZE
    ):

        return None

    price = market[
        "price"
    ]

    quantity = (
        TRADE_SIZE
        / price
    )

    state[
        "cash"
    ] -= TRADE_SIZE

    state[
        "position"
    ] = {
        "product":
            market[
                "product"
            ],
        "entry_price":
            price,
        "quantity":
            quantity,
        "value":
            TRADE_SIZE,
        "confidence":
            market[
                "confidence"
            ],
        "probability_up":
            market[
                "probability_up"
            ],
        "stop_loss":
            price
            * (
                1
                - STOP_LOSS_PCT
            ),
        "take_profit":
            price
            * (
                1
                + TAKE_PROFIT_PCT
            ),
        "opened_at":
            datetime.now(
                timezone.utc
            ).isoformat()
    }

    save_state(
        state
    )

    return state[
        "position"
    ]


def close_position(
    state,
    price,
    reason
):

    position = state[
        "position"
    ]

    if position is None:
        return None

    value_now = (
        position[
            "quantity"
        ]
        * price
    )

    pnl = (
        value_now
        - position[
            "value"
        ]
    )

    state[
        "cash"
    ] += value_now

    state[
        "realized_pnl"
    ] += pnl

    if pnl > 0:

        state[
            "wins"
        ] += 1

    else:

        state[
            "losses"
        ] += 1

    trade = {
        "product":
            position[
                "product"
            ],
        "entry_price":
            position[
                "entry_price"
            ],
        "exit_price":
            price,
        "quantity":
            position[
                "quantity"
            ],
        "pnl":
            pnl,
        "reason":
            reason,
        "confidence":
            position[
                "confidence"
            ],
        "probability_up":
            position.get(
                "probability_up"
            ),
        "opened_at":
            position[
                "opened_at"
            ],
        "closed_at":
            datetime.now(
                timezone.utc
            ).isoformat()
    }

    state[
        "trades"
    ].append(
        trade
    )

    state[
        "position"
    ] = None

    save_state(
        state
    )

    return trade


def find_market(
    scan_results,
    product_id
):

    for market in scan_results:

        if (
            market[
                "product"
            ]
            == product_id
        ):

            return market

    return None


def check_position(
    state,
    scan_results
):

    position = state[
        "position"
    ]

    if position is None:

        return None

    market = find_market(
        scan_results,
        position[
            "product"
        ]
    )

    if market is None:

        return None

    price = market[
        "price"
    ]

    if (
        price
        <= position[
            "stop_loss"
        ]
    ):

        return close_position(
            state,
            price,
            "STOP LOSS"
        )

    if (
        price
        >= position[
            "take_profit"
        ]
    ):

        return close_position(
            state,
            price,
            "TAKE PROFIT"
        )

    return None


def get_portfolio_value(
    state,
    scan_results
):

    value = state[
        "cash"
    ]

    position = state[
        "position"
    ]

    if position is None:

        return value

    market = find_market(
        scan_results,
        position[
            "product"
        ]
    )

    if market is None:

        # Fall back to entry value if current market
        # data is unavailable.
        return (
            value
            + position[
                "value"
            ]
        )

    value += (
        position[
            "quantity"
        ]
        * market[
            "price"
        ]
    )

    return value


# =========================================================
# MAIN AI CYCLE
# =========================================================

def run_ai_cycle():

    state = load_state()

    scan_results, scan_errors = (
        scan_markets()
    )

    closed_trade = (
        check_position(
            state,
            scan_results
        )
    )

    opened_position = None

    # The strongest estimated upside setup.
    best_market = (
        scan_results[
            0
        ]
    )

    # Only open a new position if:
    # 1. there is currently no paper position
    # 2. the strongest market actually passes BUY threshold
    if (
        state[
            "position"
        ]
        is None
        and best_market[
            "decision"
        ]
        == "BUY"
    ):

        opened_position = (
            open_position(
                state,
                best_market
            )
        )

    # Reload because open/close operations may have changed it.
    state = load_state()

    portfolio_value = (
        get_portfolio_value(
            state,
            scan_results
        )
    )

    display_market = best_market

    # If a position is open, display that coin as the primary
    # market so Discord shows the market currently being traded.
    if (
        state[
            "position"
        ]
        is not None
    ):

        position_market = (
            find_market(
                scan_results,
                state[
                    "position"
                ][
                    "product"
                ]
            )
        )

        if (
            position_market
            is not None
        ):

            display_market = (
                position_market
            )

    result = {
        "product":
            display_market[
                "product"
            ],
        "price":
            display_market[
                "price"
            ],
        "decision":
            display_market[
                "decision"
            ],
        "confidence":
            display_market[
                "confidence"
            ],
        "probability_up":
            display_market[
                "probability_up"
            ],
        "cash":
            state[
                "cash"
            ],
        "portfolio_value":
            portfolio_value,
        "realized_pnl":
            state[
                "realized_pnl"
            ],
        "wins":
            state[
                "wins"
            ],
        "losses":
            state[
                "losses"
            ],
        "position":
            state[
                "position"
            ],
        "opened_position":
            opened_position,
        "closed_trade":
            closed_trade,
        "model_accuracy":
            display_market[
                "model_accuracy"
            ],

        # New fields for future Discord commands/upgrades.
        "best_opportunity":
            best_market,
        "market_rankings":
            scan_results,
        "scan_errors":
            scan_errors,
        "markets_scanned":
            len(
                scan_results
            )
    }

    return result


if __name__ == "__main__":

    result = run_ai_cycle()

    print(
        json.dumps(
            result,
            indent=2
        )
    )
