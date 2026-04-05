import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Fetch historical stock data
def fetch_data(ticker, start, end):
    stock = yf.download(ticker, start=start, end=end , auto_adjust=True)
    stock = stock.dropna()  # Ensure no missing values
    return stock

# Prepare data
def prepare_data(stock):
    stock = stock.copy()
    stock['Date'] = stock.index.map(pd.Timestamp.toordinal)
    X = stock[['Date']]
    y = stock['Close']
    return X, y

# Train and evaluate model
def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    
    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    
    print(f'Mean Squared Error: {mse:.4f}')
    print(f'R2 Score: {r2:.4f}')
    return model

# Predict future stock prices
def predict_future(model, last_date, days=30):
    future_dates = np.arange(last_date + 1, last_date + days + 1).reshape(-1, 1)
    future_predictions = model.predict(future_dates)
    return future_dates, future_predictions

# Main execution
ticker = 'AAPL'  # Change ticker as needed
start_date = '2020-01-01'
end_date = '2024-01-01'

stock_data = fetch_data(ticker, start_date, end_date)
X, y = prepare_data(stock_data)
model = train_model(X, y)

# Predict future stock prices
last_date = int(X.iloc[-1, 0]) if not X.empty else pd.Timestamp(end_date).toordinal()
future_dates, future_prices = predict_future(model, last_date)

# Plot results
plt.figure(figsize=(10,5))
plt.scatter(X, y, label='Historical Prices', color='blue', alpha=0.5)
plt.plot(future_dates, future_prices, label='Predicted Prices', color='red')
plt.xlabel('Date (Ordinal)')
plt.ylabel('Stock Price')
plt.title(f'Stock Price Prediction for {ticker}')
plt.legend()
plt.show()
