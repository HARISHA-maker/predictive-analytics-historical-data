"""
Predictive Analytics Using Historical Data
------------------------------------------
Pipeline: load -> clean -> feature engineering -> train/test split (time-based)
          -> models -> evaluation -> visualisation -> future forecast

Usage:
    python predictive_analytics.py                      # runs on airline_passengers.csv (real data)
    python predictive_analytics.py data.csv Date Sales  # your CSV, date column, target column
"""
import os
import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

warnings.filterwarnings("ignore")
FORECAST_STEPS = 12      # how many future periods to predict
LAGS = [1, 2, 3, 6, 12]  # lag features
TEST_FRACTION = 0.2


# ---------------------------------------------------------------- 1. LOAD
def load_data():
    if len(sys.argv) >= 4:
        path, date_col, target_col = sys.argv[1:4]
        df = pd.read_csv(path)
    else:
        # Real dataset: monthly international airline passengers (thousands), Mar 1949 - Dec 1960
        here = os.path.dirname(os.path.abspath(__file__))
        df = pd.read_csv(os.path.join(here, "airline_passengers.csv"))
        date_col, target_col = "Month", "Passengers"
        print("Using real dataset: airline_passengers.csv (monthly passengers, 1949-1960)")
    return df, date_col, target_col


# ---------------------------------------------------------------- 2. CLEAN
def clean(df, date_col, target_col):
    print(f"\n[Clean] raw shape: {df.shape}")
    df = df[[date_col, target_col]].copy()
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col]).drop_duplicates(subset=[date_col])
    df = df.sort_values(date_col).set_index(date_col)

    # Regular frequency so lags are meaningful
    freq = pd.infer_freq(df.index) or "MS"
    df = df.asfreq(freq)

    # Outliers: clip values beyond 3 * IQR-based fences
    s = df[target_col]
    q1, q3 = s.quantile([0.25, 0.75])
    iqr = q3 - q1
    lo, hi = q1 - 3 * iqr, q3 + 3 * iqr
    n_out = int(((s < lo) | (s > hi)).sum())
    df[target_col] = s.where((s >= lo) & (s <= hi))  # outliers -> NaN

    n_missing = int(df[target_col].isna().sum())
    df[target_col] = df[target_col].interpolate(method="time").bfill().ffill()
    print(f"[Clean] outliers removed: {n_out}, missing values filled: {n_missing}")
    print(f"[Clean] clean shape: {df.shape}, frequency: {freq}")
    return df, freq


# ---------------------------------------------------------------- 3. FEATURES
def add_features(df, target):
    f = pd.DataFrame(index=df.index)
    f["y"] = df[target]
    f["t"] = np.arange(len(df))
    f["month"] = df.index.month
    f["month_sin"] = np.sin(2 * np.pi * df.index.month / 12)
    f["month_cos"] = np.cos(2 * np.pi * df.index.month / 12)
    for lag in LAGS:
        f[f"lag_{lag}"] = f["y"].shift(lag)
    f["roll_mean_3"] = f["y"].shift(1).rolling(3).mean()
    f["roll_mean_12"] = f["y"].shift(1).rolling(12).mean()
    return f.dropna()


# ---------------------------------------------------------------- 4. METRICS
def evaluate(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    r2 = r2_score(y_true, y_pred)
    return {"MAE": mae, "RMSE": rmse, "MAPE_%": mape, "R2": r2}


# ---------------------------------------------------------------- 5. MAIN
def main():
    raw, date_col, target_col = load_data()
    df, freq = clean(raw, date_col, target_col)

    # Quick trend view
    ax = df[target_col].plot(figsize=(10, 4), title="Historical data (cleaned)")
    ax.set_ylabel(target_col)
    plt.tight_layout(); plt.savefig("01_history.png", dpi=130); plt.close()

    feats = add_features(df, target_col)
    X, y = feats.drop(columns="y"), feats["y"]

    # Time-based split (never shuffle time series!)
    split = int(len(feats) * (1 - TEST_FRACTION))
    X_tr, X_te, y_tr, y_te = X.iloc[:split], X.iloc[split:], y.iloc[:split], y.iloc[split:]
    print(f"\n[Split] train: {len(X_tr)} rows, test: {len(X_te)} rows")

    models = {
        "Linear Regression": LinearRegression(),
        "Ridge": Ridge(alpha=1.0),
        "Random Forest": RandomForestRegressor(n_estimators=300, random_state=42),
        "Gradient Boosting": GradientBoostingRegressor(random_state=42),
    }
    # Naive baseline: predict last month's value
    results = {"Naive (lag_1)": evaluate(y_te.values, X_te["lag_1"].values)}
    preds = {"Naive (lag_1)": X_te["lag_1"].values}

    for name, m in models.items():
        m.fit(X_tr, y_tr)
        p = m.predict(X_te)
        preds[name] = p
        results[name] = evaluate(y_te.values, p)

    # Optional: Holt-Winters time-series model if statsmodels is installed
    try:
        from statsmodels.tsa.holtwinters import ExponentialSmoothing
        hw = ExponentialSmoothing(y_tr, trend="add", seasonal="add", seasonal_periods=12).fit()
        p = hw.forecast(len(y_te)).values
        preds["Holt-Winters"] = p
        results["Holt-Winters"] = evaluate(y_te.values, p)
    except ImportError:
        print("[Info] statsmodels not installed -> skipping Holt-Winters (pip install statsmodels)")

    res = pd.DataFrame(results).T.sort_values("RMSE")
    print("\n[Evaluation] test-set results (lower error is better):")
    print(res.round(3).to_string())
    res.round(3).to_csv("model_comparison.csv")

    best = res.index[0]
    print(f"\n[Best model] {best}")

    # Plot: actual vs predicted on test set
    plt.figure(figsize=(10, 4.5))
    plt.plot(y.index, y.values, color="lightgray", label="Actual (all)")
    plt.plot(y_te.index, y_te.values, color="black", label="Actual (test)")
    for name in res.index[:3]:
        plt.plot(y_te.index, preds[name], "--", label=name)
    plt.title("Actual vs Predicted (test period)")
    plt.legend(); plt.tight_layout(); plt.savefig("02_actual_vs_predicted.png", dpi=130); plt.close()

    # Bar chart of errors
    res[["MAE", "RMSE"]].plot(kind="bar", figsize=(8, 4), title="Model error comparison")
    plt.xticks(rotation=30, ha="right"); plt.tight_layout()
    plt.savefig("03_model_comparison.png", dpi=130); plt.close()

    # ------------------------------------------------ 6. FUTURE FORECAST
    # Recursive forecast using the best sklearn model (retrained on all data)
    sk_names = [n for n in res.index if n in models]
    final_name = sk_names[0]
    final = models[final_name].fit(X, y)
    history = df[target_col].copy()
    future_idx = pd.date_range(history.index[-1], periods=FORECAST_STEPS + 1, freq=freq)[1:]
    for d in future_idx:
        history.loc[d] = np.nan
        tmp = add_features(history.to_frame(target_col).ffill(), target_col)
        row = tmp.loc[[d]].drop(columns="y")
        history.loc[d] = final.predict(row)[0]
    forecast = history.loc[future_idx]

    plt.figure(figsize=(10, 4.5))
    plt.plot(df.index, df[target_col], label="Historical")
    plt.plot(forecast.index, forecast.values, "r--o", label=f"Forecast ({final_name})")
    plt.title(f"{FORECAST_STEPS}-period forecast")
    plt.legend(); plt.tight_layout(); plt.savefig("04_future_forecast.png", dpi=130); plt.close()
    forecast.rename("forecast").to_csv("future_forecast.csv")
    print(f"\n[Forecast] next {FORECAST_STEPS} periods using {final_name}:")
    print(forecast.round(1).to_string())
    print("\nSaved: 01_history.png, 02_actual_vs_predicted.png, "
          "03_model_comparison.png, 04_future_forecast.png, model_comparison.csv, future_forecast.csv")


if __name__ == "__main__":
    main()
