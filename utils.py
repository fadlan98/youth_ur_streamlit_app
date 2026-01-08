import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm

from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.model_selection import LeaveOneOut
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def mape(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = np.where(y_true == 0, np.nan, y_true)
    return np.nanmean(np.abs((y_true - y_pred) / denom)) * 100


def load_monthly_csv(uploaded_file) -> pd.DataFrame:
    df = pd.read_csv(uploaded_file)
    required = {"date", "u_rate_15_24"}
    if not required.issubset(df.columns):
        raise ValueError("Monthly CSV must contain: date, u_rate_15_24")

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date")

    for c in ["u_rate_15_24", "u_rate", "p_rate", "youth_gap_pp"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(subset=["u_rate_15_24"])
    return df


def load_annual_csv(uploaded_file) -> pd.DataFrame:
    df = pd.read_csv(uploaded_file)
    required = {
        "year", "youth_ur",
        "gdp_growth_yoy", "share_uppersec", "neet_15_24",
        "inflation_yoy", "fdi_net"
    }
    if not required.issubset(df.columns):
        raise ValueError(f"Annual CSV must contain: {sorted(required)}")

    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    for c in ["youth_ur","gdp_growth_yoy","share_uppersec","neet_15_24","inflation_yoy","fdi_net"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(subset=["year"]).sort_values("year")
    df["year"] = df["year"].astype(int)
    return df


def plot_monthly_overview(df: pd.DataFrame):
    mco_date = pd.Timestamp("2020-03-31")
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(df["date"], df["u_rate_15_24"], label="Youth UR (15–24)")
    if "u_rate" in df.columns:
        ax.plot(df["date"], df["u_rate"], label="Overall UR")
    ax.axvline(mco_date, linestyle="--",color="red", label="COVID-19 (MCO)")
    ax.set_title("Monthly Unemployment Rate in Malaysia")
    ax.set_xlabel("Date")
    ax.set_ylabel("Unemployment rate (%)")
    ax.legend()
    return fig


def run_ets(y: pd.Series, H: int, fh: int):
    train = y.iloc[:-H]
    test  = y.iloc[-H:]

    ets = ExponentialSmoothing(
        train,
        trend="add",
        damped_trend=True,
        seasonal=None,
        initialization_method="estimated"
    ).fit(optimized=True)

    pred_hold = ets.forecast(H)
    mae = mean_absolute_error(test, pred_hold)
    rmse = np.sqrt(mean_squared_error(test, pred_hold))
    mape_val = mape(test.values, pred_hold.values)

    ets_full = ExponentialSmoothing(
        y,
        trend="add",
        damped_trend=True,
        seasonal=None,
        initialization_method="estimated"
    ).fit(optimized=True)

    fc = ets_full.forecast(fh)
    return pred_hold, fc, mae, rmse, mape_val


def run_arima_like_notebook(y: pd.Series, H: int, fh: int):
    train = y.iloc[:-H]
    test  = y.iloc[-H:]

    best = None
    best_fit = None

    for p in range(0, 3):
        for d in range(0, 2):
            for q in range(0, 3):
                try:
                    mod = sm.tsa.statespace.SARIMAX(
                        train,
                        order=(p, d, q),
                        enforce_stationarity=False,
                        enforce_invertibility=False
                    )
                    fit = mod.fit(disp=False)
                    fc = fit.forecast(H)
                    mae = mean_absolute_error(test, fc)
                    if (best is None) or (mae < best[0]):
                        best = (mae, p, d, q)
                        best_fit = fit
                except Exception:
                    pass

    if best_fit is None:
        return None

    fc_hold = best_fit.forecast(H)
    mae_a = mean_absolute_error(test, fc_hold)
    rmse_a = np.sqrt(mean_squared_error(test, fc_hold))
    mape_a = (
        (np.abs((test - fc_hold) / test))
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
        .mean() * 100
    )

    mod_full = sm.tsa.statespace.SARIMAX(
        y,
        order=(best[1], best[2], best[3]),
        enforce_stationarity=False,
        enforce_invertibility=False
    )
    arima_full = mod_full.fit(disp=False)
    fc_future = arima_full.forecast(fh)

    metrics = {"name": f"ARIMA(p={best[1]}, d={best[2]}, q={best[3]})",
               "mae": mae_a, "rmse": rmse_a, "mape": mape_a}

    return metrics, fc_hold, fc_future, test


def mlr_loocv_report(df_clean, feature_cols, target_col="youth_ur"):
    X = df_clean[feature_cols]
    y = df_clean[target_col]

    loo = LeaveOneOut()
    y_true, y_pred = [], []

    for tr_idx, te_idx in loo.split(X):
        X_tr, X_te = X.iloc[tr_idx], X.iloc[te_idx]
        y_tr, y_te = y.iloc[tr_idx], y.iloc[te_idx]

        model = LinearRegression()
        model.fit(X_tr, y_tr)
        y_pred.append(model.predict(X_te)[0])
        y_true.append(y_te.values[0])

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    metrics = pd.DataFrame({
        "Metric": ["RMSE", "MAE", "R-squared"],
        "Value": [
            np.sqrt(mean_squared_error(y_true, y_pred)),
            mean_absolute_error(y_true, y_pred),
            r2_score(y_true, y_pred)
        ]
    })

    final_model = LinearRegression()
    final_model.fit(X, y)

    coef_df = pd.DataFrame({"Feature": feature_cols, "Coefficient": final_model.coef_})
    coef_df["abs"] = coef_df["Coefficient"].abs()
    coef_df = coef_df.sort_values("abs", ascending=False).drop(columns=["abs"])

    return metrics, coef_df


def rf_loocv_report(df_clean, feature_cols, target_col="youth_ur", n_estimators=100, seed=42):
    X = df_clean[feature_cols]
    y = df_clean[target_col]

    loo = LeaveOneOut()
    y_true, y_pred = [], []

    for tr_idx, te_idx in loo.split(X):
        X_tr, X_te = X.iloc[tr_idx], X.iloc[te_idx]
        y_tr, y_te = y.iloc[tr_idx], y.iloc[te_idx]

        model = RandomForestRegressor(n_estimators=n_estimators, random_state=seed)
        model.fit(X_tr, y_tr)
        y_pred.append(model.predict(X_te)[0])
        y_true.append(y_te.values[0])

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    metrics = pd.DataFrame({
        "Metric": ["RMSE", "MAE", "R-squared"],
        "Value": [
            np.sqrt(mean_squared_error(y_true, y_pred)),
            mean_absolute_error(y_true, y_pred),
            r2_score(y_true, y_pred)
        ]
    })

    final_model = RandomForestRegressor(n_estimators=n_estimators, random_state=seed)
    final_model.fit(X, y)

    imp_df = pd.DataFrame({
        "Feature": feature_cols,
        "Importance": final_model.feature_importances_
    }).sort_values("Importance", ascending=False)

    return metrics, imp_df
