import streamlit as st

st.set_page_config(
    page_title="Energy Consumption Forecasting",
    page_icon="⚡",
    layout="wide"
)

st.title("⚡ Energy Consumption Forecasting System")

st.markdown("""
### Project Overview

This system forecasts future electricity demand using:

- Historical Load Patterns
- Weather Information
- Time-Based Features
- Machine Learning Models

### Algorithms Used

- SARIMAX
- Random Forest Regressor
- XGBoost Regressor

Use the sidebar to navigate through the application.
""")