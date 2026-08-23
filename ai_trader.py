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


PRODUCT_ID = "BTC-USD"

GRANULARITY = "FIVE_MINUTE"

STARTING_BALANCE = 1000.0
TRADE_SIZE = 25.0

BUY_THRESHOLD = 0.62

STOP_LOSS_PCT = 0.02
TAKE_PROFIT_PCT = 0.04

STATE_FILE = "paper_state.json"
MODEL_FILE = "trading_model.joblib"


client = RESTClient()


def load_state():

    if not os.path.exists(STATE_FILE):

        return {
            "cash": STARTING_BALANCE,
            "position": None,
            "trades": [],
            "wins": 0,
            "losses": 0,
            "realized_pnl": 0.0
        }

    with open(STATE_FILE, "r") as f:
        return json.load(f)


def save_state(state):

    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)


def get_candles(limit=300):

    now = int(time.time())

    # 300 x 5-minute candles
    start = now - (limit * 300)

    response = client.get_public_candles(
        product_id=PRODUCT_ID,
        start=str(start),
        end=str(now),
        granularity=GRANULARITY,
        limit=limit
    )

    candles = []

    for candle in response.candles:

        candles.append({
            "time": int(candle.start),
            "open": float(candle.open),
            "high": float(candle.high),
            "low": float(candle.low),
            "close": float(candle.close),
            "volume": float(candle.volume)
        })

    df = pd.DataFrame(candles)

    df = df.sort_values("time").reset_index(drop=True)

    return df


def calculate_rsi(series, period=14):

    delta = series.diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)

    rsi = 100 - (100 / (1 + rs))

    return rsi.fillna(50)


def build_features(df):

    data = df.copy()

    data["return_1"] = data["close"].pct_change()

    data["return_3"] = data["close"].pct_change(3)

    data["return_6"] = data["close"].pct_change(6)

    data["ma_5"] = data["close"].rolling(5).mean()

    data["ma_20"] = data["close"].rolling(20).mean()

    data["ma_ratio"] = data["ma_5"] / data["ma_20"]

    data["volatility"] = (
        data["return_1"]
        .rolling(12)
        .std()
    )

    data["volume_ma"] = (
        data["volume"]
        .rolling(20)
        .mean()
    )

    data["volume_ratio"] = (
        data["volume"] /
        data["volume_ma"]
    )

    data["rsi"] = calculate_rsi(data["close"])

    data["range_pct"] = (
        (data["high"] - data["low"]) /
        data["close"]
    )

    #
    # TARGET
    #
    # Did price rise at least 0.25%
    # over the next 3 candles?
    #

    future_price = data["close"].shift(-3)

    future_return = (
        future_price /
        data["close"]
    ) - 1

    data["target"] = (
        future_return > 0.0025
    ).astype(int)

    data = data.dropna().reset_index(drop=True)

    return data


FEATURE_COLUMNS = [
    "return_1",
    "return_3",
    "return_6",
    "ma_ratio",
    "volatility",
    "volume_ratio",
    "rsi",
    "range_pct"
]


def train_model(data):

    #
    # Never randomly shuffle financial time-series.
    #

    split = int(len(data) * 0.8)

    train = data.iloc[:split]

    test = data.iloc[split:]

    X_train = train[FEATURE_COLUMNS]

    y_train = train["target"]

    X_test = test[FEATURE_COLUMNS]

    y_test = test["target"]

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=7,
        min_samples_leaf=5,
        random_state=42,
        class_weight="balanced"
    )

    model.fit(
        X_train,
        y_train
    )

    predictions = model.predict(X_test)

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    joblib.dump(
        model,
        MODEL_FILE
    )

    return model, accuracy


def load_or_train_model(data):

    if os.path.exists(MODEL_FILE):

        try:

            model = joblib.load(MODEL_FILE)

            return model, None

        except Exception:

            pass

    return train_model(data)


def get_ai_signal(model, latest_row):

    X = latest_row[FEATURE_COLUMNS].to_frame().T

    probabilities = model.predict_proba(X)[0]

    probability_up = float(
        probabilities[1]
    )

    if probability_up >= BUY_THRESHOLD:

        decision = "BUY"

    elif probability_up <= 0.40:

        decision = "BEARISH"

    else:

        decision = "HOLD"

    return decision, probability_up


def open_position(state, price, confidence):

    if state["position"] is not None:

        return None

    if state["cash"] < TRADE_SIZE:

        return None

    quantity = TRADE_SIZE / price

    state["cash"] -= TRADE_SIZE

    state["position"] = {
        "product": PRODUCT_ID,
        "entry_price": price,
        "quantity": quantity,
        "value": TRADE_SIZE,
        "confidence": confidence,
        "stop_loss": price * (1 - STOP_LOSS_PCT),
        "take_profit": price * (1 + TAKE_PROFIT_PCT),
        "opened_at": datetime.now(
            timezone.utc
        ).isoformat()
    }

    save_state(state)

    return state["position"]


def close_position(state, price, reason):

    position = state["position"]

    if position is None:

        return None

    value_now = (
        position["quantity"] *
        price
    )

    pnl = (
        value_now -
        position["value"]
    )

    state["cash"] += value_now

    state["realized_pnl"] += pnl

    if pnl > 0:

        state["wins"] += 1

    else:

        state["losses"] += 1

    trade = {
        "product": PRODUCT_ID,
        "entry_price": position["entry_price"],
        "exit_price": price,
        "quantity": position["quantity"],
        "pnl": pnl,
        "reason": reason,
        "confidence": position["confidence"],
        "opened_at": position["opened_at"],
        "closed_at": datetime.now(
            timezone.utc
        ).isoformat()
    }

    state["trades"].append(trade)

    state["position"] = None

    save_state(state)

    return trade


def check_position(state, price):

    position = state["position"]

    if position is None:

        return None

    if price <= position["stop_loss"]:

        return close_position(
            state,
            price,
            "STOP LOSS"
        )

    if price >= position["take_profit"]:

        return close_position(
            state,
            price,
            "TAKE PROFIT"
        )

    return None


def get_portfolio_value(state, current_price=None):

    value = state["cash"]

    position = state["position"]

    if (
        position is not None
        and current_price is not None
    ):

        value += (
            position["quantity"]
            * current_price
        )

    return value


def run_ai_cycle():

    state = load_state()

    df = get_candles()

    data = build_features(df)

    model, accuracy = load_or_train_model(
        data
    )

    latest = data.iloc[-1]

    current_price = float(
        latest["close"]
    )

    closed_trade = check_position(
        state,
        current_price
    )

    decision, confidence = get_ai_signal(
        model,
        latest
    )

    opened_position = None

    if (
        decision == "BUY"
        and state["position"] is None
    ):

        opened_position = open_position(
            state,
            current_price,
            confidence
        )

    portfolio_value = get_portfolio_value(
        state,
        current_price
    )

    result = {
        "product": PRODUCT_ID,
        "price": current_price,
        "decision": decision,
        "confidence": confidence,
        "cash": state["cash"],
        "portfolio_value": portfolio_value,
        "realized_pnl": state[
            "realized_pnl"
        ],
        "wins": state["wins"],
        "losses": state["losses"],
        "position": state["position"],
        "opened_position": opened_position,
        "closed_trade": closed_trade,
        "model_accuracy": accuracy
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