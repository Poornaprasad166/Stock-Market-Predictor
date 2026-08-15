from flask import Flask, request, jsonify
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from flask_cors import CORS

app = Flask(__name__)
CORS(app)


@app.route("/")
def home():
    return jsonify({"message": "Stock Price Prediction API is running"})


@app.route("/api/predict", methods=["POST"])
def predict():
    data = request.get_json()

    ticker = data.get("ticker", "AAPL")
    start = data.get("start", "2020-01-01")
    end = data.get("end", "2024-01-01")
    days = int(data.get("days", 30))

    stock = yf.download(
        ticker,
        start=start,
        end=end,
        auto_adjust=True
    )

    stock = stock.dropna()

    if stock.empty:
        return jsonify({"error": "No stock data found"}), 400

    stock["DateOrdinal"] = stock.index.map(pd.Timestamp.toordinal)

    X = stock[["DateOrdinal"]]
    y = stock["Close"].squeeze()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    model = LinearRegression()
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    last_date = int(X.iloc[-1, 0])

    future_dates = np.arange(
        last_date + 1,
        last_date + days + 1
    ).reshape(-1, 1)

    future_predictions = model.predict(future_dates)

    return jsonify({
        "ticker": ticker,
        "mse": float(mse),
        "r2": float(r2),
        "future_dates": [
            pd.Timestamp.fromordinal(int(date)).strftime("%Y-%m-%d")
            for date in future_dates.flatten()
        ],
        "future_prices": [
    float(np.asarray(price).item())
    for price in future_predictions
]
    })


if __name__ == "__main__":
    app.run(debug=True, port=5000)