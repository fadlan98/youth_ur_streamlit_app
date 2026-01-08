import streamlit as st
import matplotlib.pyplot as plt
from utils import load_annual_csv, mlr_loocv_report

st.title("Multiple Linear Regression")

st.sidebar.header("Annual input (Drivers)")
annual_file = st.sidebar.file_uploader(
    "Upload annual_youth_drivers_wide.csv",
    type=["csv"],
    key="annual_mlr"
)

if annual_file is None:
    st.info("Upload annual_youth_drivers_wide.csv to run MLR.")
    st.stop()

dfA = load_annual_csv(annual_file)

feature_cols = ["gdp_growth_yoy", "share_uppersec", "neet_15_24", "inflation_yoy", "fdi_net"]
target_col = "youth_ur"
df_clean = dfA.dropna(subset=[target_col] + feature_cols).copy()

metrics, coef_df = mlr_loocv_report(df_clean, feature_cols, target_col=target_col)

# ---- LOOCV report: ETS/ARIMA-style metrics row ----
rmse = float(metrics.loc[metrics["Metric"] == "RMSE", "Value"].iloc[0])
mae  = float(metrics.loc[metrics["Metric"] == "MAE", "Value"].iloc[0])
r2   = float(metrics.loc[metrics["Metric"] == "R-squared", "Value"].iloc[0])

c1, c2, c3 = st.columns(3)
c1.metric("MAE (pp)", f"{mae:.3f}")
c2.metric("RMSE (pp)", f"{rmse:.3f}")
c3.metric("R² (LOOCV)", f"{r2:.3f}")

# ---- Top drivers table (keep notebook-style table) ----
st.subheader("Top drivers (MLR coefficients)")
show = coef_df.copy()
show["Direction"] = show["Coefficient"].apply(lambda x: "↓ (negative)" if x < 0 else "↑ (positive)")
show["Coefficient"] = show["Coefficient"].round(4)
show = show.rename(columns={"Feature": "Driver"})
st.dataframe(show[["Driver","Coefficient","Direction"]], use_container_width=True, hide_index=True)
st.write(
    f"The LOOCV results indicate weak out-of-sample performance (RMSE ≈ {rmse:.2f} pp, MAE ≈ {mae:.2f} pp, R² ≈ {r2:.2f}), "
    "so MLR should be treated as an exploratory association check rather than a reliable predictive model for the small annual sample."
)

# ---- Coefficient plot ----
vals = coef_df["Coefficient"].values
colors = ["#ff7f0e" if v > 0 else "#1f77b4" for v in vals]  # positive vs negative

fig, ax = plt.subplots(figsize=(10, 5))
ax.barh(coef_df["Feature"][::-1], coef_df["Coefficient"][::-1], color=colors[::-1])
ax.axvline(0, linestyle="--", alpha=0.6)
ax.set_title("Regression Coefficients (Multiple Linear Regression)")
ax.set_xlabel("Coefficient Value")
ax.grid(axis="x", linestyle="--", alpha=0.4)
st.pyplot(fig)



st.subheader("Insights (MLR)")
st.write(
    "In the coefficient chart, the largest linear associations are NEET (15–24) and upper-secondary completion share, "
    "but the NEET sign is counterintuitive. This suggests the coefficient directions may be influenced by small-sample "
    "effects, multicollinearity, annual aggregation, or structural breaks; therefore, stakeholders should avoid causal "
    "interpretation and instead use MLR mainly to flag which variables tend to move together with youth unemployment."
)
