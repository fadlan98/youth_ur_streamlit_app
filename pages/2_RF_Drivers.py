import streamlit as st
import matplotlib.pyplot as plt
from utils import load_annual_csv, rf_loocv_report

st.title("Random Forest")

st.sidebar.header("Annual input (Drivers)")
annual_file = st.sidebar.file_uploader(
    "Upload annual_youth_drivers_wide.csv",
    type=["csv"],
    key="annual_rf"
)

if annual_file is None:
    st.info("Upload annual_youth_drivers_wide.csv to run RF.")
    st.stop()

try:
    dfA = load_annual_csv(annual_file)
except ValueError as e:
    st.error("Wrong dataset uploaded. Please upload the annual drivers CSV (year, youth_ur, gdp_growth_yoy, share_uppersec, neet_15_24, inflation_yoy, fdi_net).")
    st.caption(f"Details: {e}")
    st.stop()


feature_cols = ["gdp_growth_yoy", "share_uppersec", "neet_15_24", "inflation_yoy", "fdi_net"]
target_col = "youth_ur"
df_clean = dfA.dropna(subset=[target_col] + feature_cols).copy()

metrics, imp_df = rf_loocv_report(df_clean, feature_cols, target_col=target_col, n_estimators=100, seed=42)

# ---- LOOCV report: ETS/ARIMA-style metrics row ----
rmse = float(metrics.loc[metrics["Metric"] == "RMSE", "Value"].iloc[0])
mae  = float(metrics.loc[metrics["Metric"] == "MAE", "Value"].iloc[0])
r2   = float(metrics.loc[metrics["Metric"] == "R-squared", "Value"].iloc[0])

c1, c2, c3 = st.columns(3)
c1.metric("MAE (pp)", f"{mae:.3f}")
c2.metric("RMSE (pp)", f"{rmse:.3f}")
c3.metric("R² (LOOCV)", f"{r2:.3f}")

# ---- Top drivers table ----
st.subheader("Top drivers (RF feature importance)")
show = imp_df.copy().reset_index(drop=True)
show["Rank"] = show.index + 1
show["Importance"] = show["Importance"].round(4)
show = show.rename(columns={"Feature": "Driver"})
show = show[["Rank", "Driver", "Importance"]]

st.dataframe(
    show,
    use_container_width=True,
    hide_index=True,
    column_config={
        "Rank": st.column_config.NumberColumn("Rank", width=30),
        "Driver": st.column_config.TextColumn("Driver", width="large"),
        "Importance": st.column_config.NumberColumn("Importance", format="%.4f", width="medium"),
    },
)
st.write(
    f"RF achieves lower errors than MLR in LOOCV (RMSE ≈ {rmse:.2f} pp, MAE ≈ {mae:.2f} pp, R² ≈ {r2:.2f}), "
    "but performance is still constrained by the small annual sample. Use RF mainly for driver ranking, not high-confidence prediction."
)

# ---- Importance plot ----
# --- 3-shade blue palette by importance level (high/medium/low) ---
palette = {
    "High": "#1f77b4",     # deep blue
    "Medium": "#5dade2",   # medium blue
    "Low": "#d6eaf8",      # pale blue
}

plot_df = imp_df.copy()

# Assign High/Medium/Low based on tertiles
q1 = plot_df["Importance"].quantile(1/3)
q2 = plot_df["Importance"].quantile(2/3)

def bucket(v):
    if v >= q2:
        return "High"
    elif v >= q1:
        return "Medium"
    return "Low"

plot_df["Level"] = plot_df["Importance"].apply(bucket)
colors = plot_df["Level"].map(palette).tolist()

# Plot (keep your reverse ordering)
fig, ax = plt.subplots(figsize=(10, 5))
ax.barh(plot_df["Feature"][::-1], plot_df["Importance"][::-1], color=colors[::-1])
ax.set_title("Feature Importance (Random Forest)")
ax.set_xlabel("Importance")
ax.grid(axis="x", linestyle="--", alpha=0.4)

# Optional: legend
from matplotlib.patches import Patch
handles = [Patch(color=palette[k], label=f"{k} importance") for k in ["High","Medium","Low"]]
ax.legend(handles=handles, loc="lower right")

st.pyplot(fig)




st.subheader("Insights (RF)")

st.write(
    "Feature importance ranks GDP growth as the strongest signal, followed by NEET (15–24), then FDI net and upper-secondary "
    "completion, with inflation least important. For stakeholders, this supports a monitoring and intervention focus on "
    "macroeconomic conditions (growth) plus youth activation/transition indicators (NEET and education proxy). Note that RF "
    "importance does not indicate direction (increase vs decrease), only predictive usefulness."
)
