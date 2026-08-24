import json

import os

import time

from datetime import datetime, timezone

import joblib

import numpy as np

import pandas as pd

from coinbase.rest import RESTClient

from sklearn.ensemble import (

    RandomForestClassifier,

    RandomForestRegressor,

)

from sklearn.metrics import (

    accuracy_score,

    mean_absolute_error,

)

# =========================================================

# AI TRADER V4 SETTINGS

# =========================================================

STRATEGY_VERSION = "V4"

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

# =========================================================

# TRADING COSTS

# =========================================================

# Estimated Coinbase cost per side.

#

# 0.006 = 0.60%

#

# Entry + exit:

# approximately 1.20% round trip.

#

# This deliberately matches the conservative assumption

# being used by the backtester.

FEE_PER_SIDE = 0.006

ROUND_TRIP_FEE = (

    FEE_PER_SIDE

    * 2

)

# =========================================================

# AI TARGET

# =========================================================

# V3:

# +0.25% within 15 minutes.

#

# That was far too small relative to fees.

#

# V4:

# Look approximately 60 minutes into the future.

TARGET_CANDLES = 12

# The classifier learns whether price achieves at least

# approximately +1.8% over that future period.

TARGET_RETURN = 0.018

# =========================================================

# ENTRY FILTERS

# =========================================================

# Probability that TARGET_RETURN will be achieved.

BUY_THRESHOLD = 0.64

BEARISH_THRESHOLD = 0.36

# Even if probability is high, the regression model must

# estimate that the trade still has an edge AFTER estimated

# round-trip fees.

#

# 0.003 = +0.30% estimated net edge.

MIN_EXPECTED_NET_RETURN = 0.003

# =========================================================

# RISK MANAGEMENT

# =========================================================

STOP_LOSS_PCT = 0.0125

TAKE_PROFIT_PCT = 0.045

# Do not allow a trade to sit open indefinitely.

MAX_HOLD_MINUTES = 180

# If the AI becomes strongly bearish while holding

# a position, it may close early.

SIGNAL_EXIT_PROBABILITY = 0.34

# =========================================================

# DATA / MODEL SETTINGS

# =========================================================

STATE_FILE = "paper_state.json"

MODEL_DIR = "models"

# Only about 300 candles are needed for each live scan.

LIVE_CANDLE_LIMIT = 300

# When retraining, V4 attempts to obtain substantially

# more history.

TRAIN_CANDLE_LIMIT = 2000

# Retrain each model periodically.

MODEL_RETRAIN_HOURS = 6

MIN_TRAINING_ROWS = 350

MODEL_VERSION = 4

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

        "realized_pnl": 0.0,

        "fees_paid": 0.0,

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

def request_candle_chunk(

    product_id,

    start,

    end,

    limit=300

):

    response = (

        client.get_public_candles(

            product_id=product_id,

            start=str(int(start)),

            end=str(int(end)),

            granularity=GRANULARITY,

            limit=limit

        )

    )

    candles = []

    for candle in response.candles:

        candles.append({

            "time": int(

                candle.start

            ),

            "open": float(

                candle.open

            ),

            "high": float(

                candle.high

            ),

            "low": float(

                candle.low

            ),

            "close": float(

                candle.close

            ),

            "volume": float(

                candle.volume

            ),

        })

    return candles

def get_candles(

    product_id,

    limit=LIVE_CANDLE_LIMIT

):

    now = int(

        time.time()

    )

    candles = []

    remaining = int(

        limit

    )

    chunk_end = now

    while remaining > 0:

        chunk_limit = min(

            300,

            remaining

        )

        chunk_seconds = (

            chunk_limit

            * 300

        )

        chunk_start = (

            chunk_end

            - chunk_seconds

        )

        try:

            chunk = request_candle_chunk(

                product_id,

                chunk_start,

                chunk_end,

                chunk_limit

            )

        except Exception as exc:

            if candles:

                break

            raise RuntimeError(

                f"Coinbase candle error "

                f"{product_id}: {exc}"

            )

        if not chunk:

            break

        candles.extend(

            chunk

        )

        oldest_time = min(

            candle["time"]

            for candle in chunk

        )

        chunk_end = (

            oldest_time

            - 1

        )

        remaining -= len(

            chunk

        )

        if len(chunk) < chunk_limit:

            break

        # Small pause to avoid hammering Coinbase

        # during model retraining.

        if limit > 300:

            time.sleep(

                0.08

            )

    if not candles:

        raise RuntimeError(

            f"No Coinbase candles returned "

            f"for {product_id}"

        )

    df = pd.DataFrame(

        candles

    )

    df = (

        df

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

    return df

# =========================================================

# INDICATORS

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

    data.replace(

        [np.inf, -np.inf],

        np.nan,

        inplace=True

    )

    return data

def build_training_data(

    df

):

    data = build_feature_frame(

        df

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

    # IMPORTANT:

    #

    # The final TARGET_CANDLES rows do not yet have a known

    # future result.

    #

    # V3 effectively allowed those unresolved observations

    # to behave like negative labels.

    #

    # V4 explicitly leaves them unresolved.

    data["target"] = np.where(

        data[

            "future_return"

        ].notna(),

        (

            data[

                "future_return"

            ]

            >= TARGET_RETURN

        ).astype(float),

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

        .astype(int)

    )

    return data

# =========================================================

# MODEL FILES

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

        (

            f"{safe_name}_"

            f"v{MODEL_VERSION}.joblib"

        )

    )

def model_is_fresh(

    bundle

):

    if not isinstance(

        bundle,

        dict

    ):

        return False

    if (

        bundle.get(

            "model_version"

        )

        != MODEL_VERSION

    ):

        return False

    trained_at = bundle.get(

        "trained_at"

    )

    if trained_at is None:

        return False

    age_seconds = (

        time.time()

        - float(

            trained_at

        )

    )

    max_age = (

        MODEL_RETRAIN_HOURS

        * 3600

    )

    return (

        age_seconds

        < max_age

    )

# =========================================================

# MODEL TRAINING

# =========================================================

def train_model(

    product_id,

    data

):

    if len(data) < MIN_TRAINING_ROWS:

        raise RuntimeError(

            f"Not enough usable candles "

            f"for {product_id}: "

            f"{len(data)} rows"

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

    y_train_class = (

        train[

            "target"

        ]

    )

    y_train_return = (

        train[

            "future_return"

        ]

    )

    if (

        y_train_class.nunique()

        < 2

    ):

        raise RuntimeError(

            f"Training data for "

            f"{product_id} only contains "

            f"one target class"

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

        y_train_class

    )

    regressor.fit(

        X_train,

        y_train_return

    )

    accuracy = None

    return_mae = None

    if len(test) > 0:

        X_test = (

            test[

                FEATURE_COLUMNS

            ]

        )

        y_test_class = (

            test[

                "target"

            ]

        )

        y_test_return = (

            test[

                "future_return"

            ]

        )

        class_predictions = (

            classifier.predict(

                X_test

            )

        )

        return_predictions = (

            regressor.predict(

                X_test

            )

        )

        accuracy = float(

            accuracy_score(

                y_test_class,

                class_predictions

            )

        )

        return_mae = float(

            mean_absolute_error(

                y_test_return,

                return_predictions

            )

        )

    bundle = {

        "model_version":

            MODEL_VERSION,

        "strategy":

            STRATEGY_VERSION,

        "trained_at":

            time.time(),

        "classifier":

            classifier,

        "regressor":

            regressor,

        "accuracy":

            accuracy,

        "return_mae":

            return_mae,

        "training_rows":

            len(data),

        "positive_rate":

            float(

                data[

                    "target"

                ].mean()

            ),

    }

    joblib.dump(

        bundle,

        model_path(

            product_id

        )

    )

    print(

        f"AI MODEL TRAINED: "

        f"{product_id} | "

        f"{len(data)} rows | "

        f"positive rate "

        f"{bundle['positive_rate'] * 100:.1f}%"

    )

    return bundle

def load_model_bundle(

    product_id

):

    path = model_path(

        product_id

    )

    if not os.path.exists(

        path

    ):

        return None

    try:

        bundle = joblib.load(

            path

        )

    except Exception:

        return None

    if not model_is_fresh(

        bundle

    ):

        return None

    return bundle

def get_or_train_model(

    product_id

):

    bundle = load_model_bundle(

        product_id

    )

    if bundle is not None:

        return bundle

    print(

        f"Retraining AI model: "

        f"{product_id}"

    )

    historical_df = (

        get_candles(

            product_id,

            TRAIN_CANDLE_LIMIT

        )

    )

    training_data = (

        build_training_data(

            historical_df

        )

    )

    return train_model(

        product_id,

        training_data

    )

# =========================================================

# MODEL PREDICTIONS

# =========================================================

def probability_target(

    classifier,

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

        classifier.predict_proba(

            X

        )[0]

    )

    class_map = {

        int(

            classifier.classes_[

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

                classifier.classes_

            )

        )

    }

    return class_map.get(

        1,

        0.0

    )

def predicted_return(

    regressor,

    latest_row

):

    X = (

        latest_row[

            FEATURE_COLUMNS

        ]

        .to_frame()

        .T

    )

    prediction = (

        regressor.predict(

            X

        )[0]

    )

    return float(

        prediction

    )

# =========================================================

# MARKET CLASSIFICATION

# =========================================================

def classify_market(

    up_probability,

    expected_return,

    expected_net_return

):

    if (

        up_probability

        >= BUY_THRESHOLD

        and

        expected_net_return

        >= MIN_EXPECTED_NET_RETURN

    ):

        return (

            "BUY",

            up_probability

        )

    if (

        up_probability

        <= BEARISH_THRESHOLD

        or expected_return < 0

    ):

        confidence = max(

            1.0

            - up_probability,

            0.50

        )

        return (

            "BEARISH",

            confidence

        )

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

# =========================================================

# MARKET ANALYSIS

# =========================================================

def analyse_product(

    product_id

):

    recent_df = (

        get_candles(

            product_id,

            LIVE_CANDLE_LIMIT

        )

    )

    live_data = (

        build_feature_frame(

            recent_df

        )

    )

    live_data = (

        live_data

        .dropna(

            subset=FEATURE_COLUMNS

        )

        .reset_index(

            drop=True

        )

    )

    if len(live_data) < 1:

        raise RuntimeError(

            f"No usable live features "

            f"for {product_id}"

        )

    latest = (

        live_data.iloc[

            -1

        ]

    )

    bundle = (

        get_or_train_model(

            product_id

        )

    )

    classifier = bundle[

        "classifier"

    ]

    regressor = bundle[

        "regressor"

    ]

    price = float(

        latest[

            "close"

        ]

    )

    up_probability = (

        probability_target(

            classifier,

            latest

        )

    )

    expected_return = (

        predicted_return(

            regressor,

            latest

        )

    )

    expected_net_return = (

        expected_return

        - ROUND_TRIP_FEE

    )

    decision, confidence = (

        classify_market(

            up_probability,

            expected_return,

            expected_net_return

        )

    )

    # Opportunity score rewards BOTH:

    #

    # 1. probability

    # 2. expected profitability after costs

    #

    # This is superior to ranking purely by probability.

    opportunity_score = (

        up_probability

        * max(

            expected_net_return,

            0

        )

    )

    return {

        "strategy":

            STRATEGY_VERSION,

        "product":

            product_id,

        "price":

            price,

        "decision":

            decision,

        "confidence":

            confidence,

        # Kept with the same name so your current bot.py

        # remains compatible.

        "probability_up":

            up_probability,

        "expected_return":

            expected_return,

        "expected_net_return":

            expected_net_return,

        "opportunity_score":

            opportunity_score,

        "model_accuracy":

            bundle.get(

                "accuracy"

            ),

        "return_mae":

            bundle.get(

                "return_mae"

            ),

        "training_rows":

            bundle.get(

                "training_rows",

                0

            ),

        "positive_rate":

            bundle.get(

                "positive_rate"

            ),

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

# =========================================================

# MARKET SCANNER

# =========================================================

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

    # V4 ranks by estimated financial opportunity,

    # not merely raw classification probability.

    results.sort(

        key=lambda item: (

            item[

                "decision"

            ]

            == "BUY",

            item[

                "opportunity_score"

            ],

            item[

                "probability_up"

            ],

        ),

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

    entry_fee = (

        TRADE_SIZE

        * FEE_PER_SIDE

    )

    total_cash_required = (

        TRADE_SIZE

        + entry_fee

    )

    if (

        state[

            "cash"

        ]

        < total_cash_required

    ):

        return None

    price = float(

        market[

            "price"

        ]

    )

    quantity = (

        TRADE_SIZE

        / price

    )

    state[

        "cash"

    ] -= total_cash_required

    state[

        "fees_paid"

    ] += entry_fee

    state[

        "position"

    ] = {

        "strategy":

            STRATEGY_VERSION,

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

        "entry_fee":

            entry_fee,

        "confidence":

            market[

                "confidence"

            ],

        "probability_up":

            market[

                "probability_up"

            ],

        "expected_return":

            market.get(

                "expected_return"

            ),

        "expected_net_return":

            market.get(

                "expected_net_return"

            ),

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

    price = float(

        price

    )

    gross_value_now = (

        position[

            "quantity"

        ]

        * price

    )

    exit_fee = (

        gross_value_now

        * FEE_PER_SIDE

    )

    entry_fee = float(

        position.get(

            "entry_fee",

            0

        )

    )

    net_value_now = (

        gross_value_now

        - exit_fee

    )

    pnl = (

        net_value_now

        - position[

            "value"

        ]

        - entry_fee

    )

    gross_pnl = (

        gross_value_now

        - position[

            "value"

        ]

    )

    state[

        "cash"

    ] += net_value_now

    state[

        "fees_paid"

    ] += exit_fee

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

        "strategy":

            position.get(

                "strategy",

                STRATEGY_VERSION

            ),

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

        "gross_pnl":

            gross_pnl,

        "entry_fee":

            entry_fee,

        "exit_fee":

            exit_fee,

        "fees":

            entry_fee

            + exit_fee,

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

        "expected_return":

            position.get(

                "expected_return"

            ),

        "expected_net_return":

            position.get(

                "expected_net_return"

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

# =========================================================

# POSITION HELPERS

# =========================================================

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

def position_age_minutes(

    position

):

    try:

        opened_at = (

            datetime.fromisoformat(

                position[

                    "opened_at"

                ]

            )

        )

        if (

            opened_at.tzinfo

            is None

        ):

            opened_at = (

                opened_at.replace(

                    tzinfo=timezone.utc

                )

            )

        age = (

            datetime.now(

                timezone.utc

            )

            - opened_at

        )

        return (

            age.total_seconds()

            / 60

        )

    except Exception:

        return 0

# =========================================================

# POSITION MANAGEMENT

# =========================================================

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

    price = float(

        market[

            "price"

        ]

    )

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

    age_minutes = (

        position_age_minutes(

            position

        )

    )

    if (

        age_minutes

        >= MAX_HOLD_MINUTES

    ):

        return close_position(

            state,

            price,

            "MAX HOLD TIME"

        )

    # Give the original trade a little time before

    # allowing the AI to reverse its opinion.

    if (

        age_minutes >= 20

        and

        market[

            "probability_up"

        ]

        <= SIGNAL_EXIT_PROBABILITY

    ):

        return close_position(

            state,

            price,

            "AI SIGNAL REVERSAL"

        )

    return None

# =========================================================

# PORTFOLIO VALUE

# =========================================================

def get_portfolio_value(

    state,

    scan_results

):

    value = float(

        state[

            "cash"

        ]

    )

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

        return (

            value

            + float(

                position[

                    "value"

                ]

            )

        )

    gross_value = (

        position[

            "quantity"

        ]

        * market[

            "price"

        ]

    )

    estimated_exit_fee = (

        gross_value

        * FEE_PER_SIDE

    )

    value += (

        gross_value

        - estimated_exit_fee

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

    # check_position can modify state.

    state = load_state()

    opened_position = None

    best_market = (

        scan_results[

            0

        ]

    )

    if (

        state[

            "position"

        ]

        is None

        and

        best_market[

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

    state = load_state()

    portfolio_value = (

        get_portfolio_value(

            state,

            scan_results

        )

    )

    display_market = (

        best_market

    )

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

        "strategy":

            STRATEGY_VERSION,

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

        "expected_return":

            display_market.get(

                "expected_return",

                0

            ),

        "expected_net_return":

            display_market.get(

                "expected_net_return",

                0

            ),

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

        "fees_paid":

            state.get(

                "fees_paid",

                0

            ),

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

            display_market.get(

                "model_accuracy"

            ),

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

# =========================================================

# LEARNING / PERFORMANCE STATS

# =========================================================

def get_learning_stats():

    state = load_state()

    trades = state.get(

        "trades",

        []

    )

    resolved = len(

        trades

    )

    wins = sum(

        1

        for trade in trades

        if float(

            trade.get(

                "pnl",

                0

            )

            or 0

        ) > 0

    )

    buy_signals = resolved

    buy_correct = wins

    buy_accuracy = (

        buy_correct

        / buy_signals

        if buy_signals > 0

        else None

    )

    by_product = {}

    for trade in trades:

        product = trade.get(

            "product",

            "Unknown"

        )

        pnl = float(

            trade.get(

                "pnl",

                0

            )

            or 0

        )

        if product not in by_product:

            by_product[

                product

            ] = {

                "product":

                    product,

                "resolved":

                    0,

                "correct":

                    0,

                "pnl":

                    0.0,

                "fees":

                    0.0,

            }

        item = by_product[

            product

        ]

        item[

            "resolved"

        ] += 1

        item[

            "pnl"

        ] += pnl

        item[

            "fees"

        ] += float(

            trade.get(

                "fees",

                0

            )

            or 0

        )

        if pnl > 0:

            item[

                "correct"

            ] += 1

    products = []

    for item in by_product.values():

        count = item[

            "resolved"

        ]

        item[

            "directional_accuracy"

        ] = (

            item[

                "correct"

            ]

            / count

            if count > 0

            else None

        )

        products.append(

            item

        )

    products.sort(

        key=lambda item: (

            item[

                "pnl"

            ],

            item[

                "resolved"

            ]

        ),

        reverse=True

    )

    return {

        "strategy":

            STRATEGY_VERSION,

        "resolved_predictions":

            resolved,

        "buy_signals":

            buy_signals,

        "buy_correct":

            buy_correct,

        "buy_accuracy":

            buy_accuracy,

        "bearish_signals":

            0,

        "bearish_correct":

            0,

        "bearish_accuracy":

            None,

        "fees_paid":

            float(

                state.get(

                    "fees_paid",

                    0

                )

            ),

        "realized_pnl":

            float(

                state.get(

                    "realized_pnl",

                    0

                )

            ),

        "products":

            products,

    }

# =========================================================

# START

# =========================================================

if __name__ == "__main__":

    result = run_ai_cycle()

    print(

        json.dumps(

            result,

            indent=2,

            default=str

        )

    )