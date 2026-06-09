import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import joblib

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------

st.set_page_config(
    page_title="Energy Consumption Forecasting",
    page_icon="⚡",
    layout="wide"
)

# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------

@st.cache_data
def load_data():
    df = pd.read_csv("data/cleaned_energy_dataset.csv")
    df["Datetime"] = pd.to_datetime(df["Datetime"])
    return df

df = load_data()

# --------------------------------------------------
# LOAD MODEL
# --------------------------------------------------

try:
    model = joblib.load("model/tuned_xgboost_model.pkl")
    model_loaded = True
except:
    model_loaded = False

# --------------------------------------------------
# HEADER
# --------------------------------------------------

st.title("⚡ Energy Consumption Forecasting System")

st.markdown("""
Forecast future electricity demand using historical load patterns,
weather conditions, and machine learning models.
""")

# --------------------------------------------------
# KPI CARDS
# --------------------------------------------------

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Records",
        f"{len(df):,}"
    )

with col2:
    st.metric(
        "Average Load",
        f"{df['Load_MW'].mean():.0f} MW"
    )

with col3:
    st.metric(
        "Maximum Load",
        f"{df['Load_MW'].max():.0f} MW"
    )

with col4:
    st.metric(
        "Average Temperature",
        f"{df['Temperature_C'].mean():.1f} °C"
    )

# --------------------------------------------------
# TABS
# --------------------------------------------------

tab1, tab2, tab3, tab4 = st.tabs(
    [
        "📊 Dataset Overview",
        "🔮 Forecasting",
        "📈 Trend Analysis",
        "ℹ️ Project Information"
    ]
)

# ==================================================
# TAB 1 - DATASET OVERVIEW
# ==================================================

with tab1:

    st.subheader("Dataset Preview")

    st.dataframe(df.head(20))

    st.subheader("Dataset Statistics")

    st.dataframe(df.describe())

    st.subheader("Missing Values")

    st.dataframe(
        df.isnull().sum().reset_index(
            name="Missing Values"
        )
    )

# ==================================================
# TAB 2 - FORECASTING
# ==================================================

with tab2:

    st.subheader("Energy Demand Forecasting")

    col1, col2 = st.columns(2)

    with col1:
        forecast_date = st.date_input(
            "Select Forecast Date"
        )

    with col2:
        horizon = st.selectbox(
            "Forecast Horizon",
            [
                "24 Hours",
                "48 Hours",
                "7 Days"
            ]
        )

    temperature = st.slider(
        "Expected Temperature (°C)",
        -10,
        45,
        25
    )

    if st.button("Generate Forecast"):

        if horizon == "24 Hours":
            periods = 24

        elif horizon == "48 Hours":
            periods = 48

        else:
            periods = 168

        future_dates = pd.date_range(
            start=pd.Timestamp(forecast_date),
            periods=periods,
            freq="h"
        )

        # Placeholder Forecast
        # Replace later with real XGBoost prediction

        forecast_values = np.random.normal(
            loc=df["Load_MW"].mean(),
            scale=100,
            size=periods
        )

        forecast_df = pd.DataFrame(
            {
                "Datetime": future_dates,
                "Forecasted_Load_MW": forecast_values
            }
        )

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                "Peak Load",
                f"{forecast_values.max():.0f} MW"
            )

        with col2:
            st.metric(
                "Average Load",
                f"{forecast_values.mean():.0f} MW"
            )

        with col3:
            st.metric(
                "Minimum Load",
                f"{forecast_values.min():.0f} MW"
            )

        st.subheader("Forecast Curve")

        fig = px.line(
            forecast_df,
            x="Datetime",
            y="Forecasted_Load_MW",
            markers=True
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        st.subheader("Forecast Data")

        forecast_df["Datetime"] = forecast_df["Datetime"].astype(str)
        st.dataframe(forecast_df)

        csv = forecast_df.to_csv(
            index=False
        ).encode("utf-8")

        st.download_button(
            "📥 Download Forecast CSV",
            csv,
            "forecast.csv",
            "text/csv"
        )

# ==================================================
# TAB 3 - TREND ANALYSIS
# ==================================================

with tab3:

    st.subheader("Historical Energy Consumption")

    fig1 = px.line(
        df.tail(1000),
        x="Datetime",
        y="Load_MW",
        title="Load Trend"
    )

    st.plotly_chart(
        fig1,
        use_container_width=True
    )

    st.subheader("Temperature vs Load")

    sample_df = df.sample(
        min(3000, len(df))
    )

    fig2 = px.scatter(
        sample_df,
        x="Temperature_C",
        y="Load_MW",
        title="Temperature vs Energy Consumption"
    )

    st.plotly_chart(
        fig2,
        use_container_width=True
    )

# ==================================================
# TAB 4 - PROJECT INFORMATION
# ==================================================

with tab4:

    st.subheader("Project Overview")

    st.markdown("""
### Objective

Forecast future electricity demand using historical load
patterns and weather information.

### Dataset Features

- Datetime
- Load_MW
- Temperature_C

### Engineered Features

- Hour
- DayOfWeek
- Month
- IsWeekend
- Lag_1
- Lag_24
- Lag_168
- Rolling_Mean_24
- Temp_Squared

### Models Used

- SARIMAX
- Random Forest Regressor
- XGBoost Regressor

### Current Model Performance

- MAE: 62.88
- RMSE: 81.12

### Technology Stack

- Python
- Pandas
- NumPy
- XGBoost
- Streamlit
- Plotly
""")