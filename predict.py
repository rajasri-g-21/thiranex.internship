"""
Task 3: Predictive Analytics Using Historical Data
Forecasts future sales trends using Linear Regression with time features.

Steps:
1. Load data (a sample dataset is created automatically if sales_data.csv is missing)
2. Clean and preprocess
3. Train a regression model
4. Evaluate accuracy (MAE, RMSE, R2)
5. Forecast the next 12 months and visualize
"""

import os

import matplotlib
matplotlib.use("Agg")  # works without a display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

DATA_FILE = "sales_data.csv"
FORECAST_MONTHS = 12
TEST_MONTHS = 12


# ---------------------------------------------------------------- 1. DATA
def create_sample_data(path):
    """Create a realistic monthly sales dataset (trend + seasonality + noise)."""
    rng = np.random.default_rng(42)
    dates = pd.date_range("2018-01-01", "2025-12-01", freq="MS")
    t = np.arange(len(dates))
    sales = 200 + 3.5 * t + 30 * np.sin(2 * np.pi * dates.month / 12)
    sales = sales + rng.normal(0, 10, len(dates))
    df = pd.DataFrame({"date": dates, "sales": sales.round(2)})
    df.loc[[10, 25, 40], "sales"] = np.nan  # simulate missing values
    df.to_csv(path, index=False)
    print(f"Sample dataset created: {path}")


def load_data(path):
    if not os.path.exists(path):
        create_sample_data(path)
    df = pd.read_csv(path)
    print("\n--- Raw data (first 5 rows) ---")
    print(df.head())
    print(f"\nShape: {df.shape}")
    print(f"Missing values:\n{df.isna().sum()}")
    return df


# ------------------------------------------------------------ 2. CLEANING
def preprocess(df):
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.drop_duplicates(subset="date").sort_values("date")
    df = df.set_index("date")
    df["sales"] = df["sales"].interpolate(method="time")  # fill gaps
    df = df.dropna()
    print(f"\nAfter cleaning -> rows: {len(df)}, missing: {int(df['sales'].isna().sum())}")
    return df


def make_features(dates, start_date):
    """Time index + seasonal (month) features."""
    dates = pd.DatetimeIndex(dates)
    t = (dates.year - start_date.year) * 12 + (dates.month - start_date.month)
    return pd.DataFrame(
        {
            "t": t,
            "month_sin": np.sin(2 * np.pi * dates.month / 12),
            "month_cos": np.cos(2 * np.pi * dates.month / 12),
        },
        index=dates,
    )


# ------------------------------------------------------------- 3-4. MODEL
def main():
    df = preprocess(load_data(DATA_FILE))
    start_date = df.index[0]

    X = make_features(df.index, start_date)
    y = df["sales"]

    # Time-based split: last 12 months are the test set
    X_train, X_test = X.iloc[:-TEST_MONTHS], X.iloc[-TEST_MONTHS:]
    y_train, y_test = y.iloc[:-TEST_MONTHS], y.iloc[-TEST_MONTHS:]

    model = LinearRegression()
    model.fit(X_train, y_train)
    pred_test = model.predict(X_test)

    mae = mean_absolute_error(y_test, pred_test)
    rmse = np.sqrt(mean_squared_error(y_test, pred_test))
    r2 = r2_score(y_test, pred_test)
    print("\n--- Model evaluation (test set) ---")
    print(f"MAE : {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"R2  : {r2:.3f}")

    # --------------------------------------------------------- 5. FORECAST
    # Refit on all data, then predict the future
    model.fit(X, y)
    future_dates = pd.date_range(
        df.index[-1] + pd.offsets.MonthBegin(1), periods=FORECAST_MONTHS, freq="MS"
    )
    future_pred = model.predict(make_features(future_dates, start_date))
    forecast = pd.DataFrame({"date": future_dates, "predicted_sales": future_pred.round(2)})
    forecast.to_csv("forecast.csv", index=False)
    print("\n--- Forecast for next 12 months ---")
    print(forecast.to_string(index=False))

    # ------------------------------------------------------ VISUALIZATION
    plt.figure(figsize=(12, 6))
    plt.plot(df.index, y, label="Historical sales", color="steelblue")
    plt.plot(y_test.index, pred_test, label="Predicted (test period)",
             color="orange", linestyle="--")
    plt.plot(future_dates, future_pred, label="Future forecast",
             color="green", linestyle="--", marker="o")
    plt.axvline(y_test.index[0], color="gray", linestyle=":", label="Train/Test split")
    plt.title("Sales Forecast using Linear Regression")
    plt.xlabel("Date")
    plt.ylabel("Sales")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("forecast_plot.png", dpi=150)
    print("\nSaved: forecast_plot.png, forecast.csv")


if __name__ == "__main__":
    main()
