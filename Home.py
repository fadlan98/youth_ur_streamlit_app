import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd

from utils import load_monthly_csv, plot_monthly_overview, run_ets, run_arima_like_notebook

st.set_page_config(page_title="Youth UR App", layout="wide")

st.title("Youth Unemployment in Malaysia — Forecasting & Drivers (Prototype)")

st.sidebar.header("Forecasting navigation")
view = st.sidebar.radio("Go to", ["Home", "ETS Forecast", "ARIMA Forecast"])

st.sidebar.header("Monthly input (Forecasting)")
monthly_file = st.sidebar.file_uploader(
    "Upload monthly_unemployment.csv",
    type=["csv"],
    key="monthly_forecast"
)

if monthly_file is None:
    st.info("Upload monthly_unemployment.csv to use Home/ETS/ARIMA.")
    st.stop()

try:
    df = load_monthly_csv(monthly_file)
except ValueError as e:
    st.error("Wrong dataset uploaded. Please upload the monthly unemployment CSV that contains: date, u_rate_15_24.")
    st.caption(f"Details: {e}")
    st.stop()

y = df.set_index("date")["u_rate_15_24"].dropna().sort_index()

if view == "Home":
    st.subheader("In this Page, We display the current statistics of Youth vs Overall Unemployment")
    st.subheader("Preview")
    st.dataframe(df.head(10))
    st.subheader("Monthly Unemployment Rate in Malaysia")
    st.pyplot(plot_monthly_overview(df))
    st.write(
    	"The series shows a clear break around the start of COVID-19 (MCO): youth unemployment rose sharply in 2020 and "
    	"remained elevated longer than overall unemployment, indicating young workers were disproportionately affected and "
    	"recovered more slowly. From 2022 onward, the youth rate declines steadily and stabilizes near ~10% by 2024–2025, "
    	"suggesting the labour market has largely normalized but with a persistent youth unemployment gap that likely requires "
    	"targeted school-to-work and entry-level employment interventions."
    )

elif view == "ETS Forecast":
    st.subheader("ETS — Hold-out and Forecast")
    H  = st.sidebar.slider("Hold-out months (H)", 6, 24, 12, 1)
    fh = st.sidebar.slider("Forecast horizon (months)", 6, 24, 12, 1)

    pred_hold, fc, mae, rmse, mape_val = run_ets(y, H, fh)

    c1, c2, c3 = st.columns(3)
    c1.metric("MAE (pp)", f"{mae:.3f}")
    c2.metric("RMSE (pp)", f"{rmse:.3f}")
    c3.metric("MAPE (%)", f"{mape_val:.2f}")

    fig1, ax1 = plt.subplots(figsize=(10, 4))
    y.plot(ax=ax1, label="actual")
    y.iloc[-H:].plot(ax=ax1, style=":", label="hold-out")
    pred_hold.plot(ax=ax1, style="--", label="ETS prediction (hold-out)")
    ax1.set_title("ETS hold-out performance")
    ax1.set_ylabel("%")
    ax1.legend()
    st.pyplot(fig1)

    st.write(
        "Based on the ETS hold-out performance, the prediction line closely overlaps the hold-out observations, "
        "suggesting ETS captures the overall level of youth unemployment but may miss short-term fluctuations."
    )

    fig2, ax2 = plt.subplots(figsize=(10, 4))
    y.plot(ax=ax2, label="actual")
    fc.plot(ax=ax2, style="--", label="ETS forecast")
    ax2.set_title(f"ETS {fh}-month youth unemployment forecast")
    ax2.set_ylabel("%")
    ax2.legend()
    st.pyplot(fig2)

    st.write(
    	"ETS projects a stability scenario: youth unemployment is expected to stay broadly flat over the next 12 months, "
    	"hovering just above 10%. For planning, this implies no major deterioration is anticipated under current conditions, "
   	 "but also that youth unemployment may not decline quickly without additional policy or programme support."
    )

elif view == "ARIMA Forecast":
    st.subheader("ARIMA — Hold-out and Forecast")
    H  = st.sidebar.slider("Hold-out months (H)", 6, 24, 12, 1)
    fh = st.sidebar.slider("Forecast horizon (months)", 6, 24, 12, 1)

    res = run_arima_like_notebook(y, H, fh)
    if res is None:
        st.warning("ARIMA search failed; proceed with ETS only.")
        st.stop()

    metrics, fc_hold, fc_future, test = res
    st.write(f"Best **{metrics['name']}** (selected by minimum MAE on hold-out)")

    c1, c2, c3 = st.columns(3)
    c1.metric("MAE (pp)", f"{metrics['mae']:.3f}")
    c2.metric("RMSE (pp)", f"{metrics['rmse']:.3f}")
    c3.metric("MAPE (%)", f"{metrics['mape']:.2f}")

    fig1, ax = plt.subplots(figsize=(10, 4))
    y.plot(ax=ax, label="actual")
    test.plot(ax=ax, style=":", label="hold-out")
    fc_hold.plot(ax=ax, style="--", label="ARIMA prediction (hold-out)")
    ax.set_title("ARIMA hold-out performance")
    ax.set_ylabel("%")
    ax.legend()
    st.pyplot(fig1)
    st.write(
        "Based on the ARIMA hold-out performance, the green prediction line closely overlaps the orange hold-out observations, "
        "which indicates that the ARIMA model is able to track short-term movements in youth unemployment very accurately on the last 12 months of data."
    )

    fig2, ax2 = plt.subplots(figsize=(10, 4))
    y.plot(ax=ax2, label="actual")
    fc_future.plot(ax=ax2, style="--", label="ARIMA forecast")
    ax2.set_title(f"{metrics['name']} - {fh}-month youth unemployment forecast")
    ax2.set_ylabel("%")
    ax2.legend()
    st.pyplot(fig2)
    st.write(
    	"ARIMA suggests a modest improvement scenario: the youth unemployment rate is projected to edge down gradually over "
    	"the next 12 months rather than rebound. For stakeholders, this supports maintaining current labour-market measures "
    	"while prioritizing structural actions (skills matching, apprenticeships, entry-level hiring incentives) to reduce the "
    	"remaining youth unemployment gap."
    )