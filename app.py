# -*- coding: utf-8 -*-
"""
Created on Sat Sep 19 21:13:20 2026

@author: id
"""

import streamlit as st
import pandas as pd
import numpy as np
from pickle import load
from pandas.tseries.holiday import USFederalHolidayCalendar

import requests
import joblib
import os

MODEL_URL = "https://huggingface.co/Ash-1185/energy-forecasting-model/blob/main/random_forest_model.pkl"
MODEL_PATH = "random_forest_model.pkl"

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        response = requests.get(MODEL_URL)
        response.raise_for_status()

        with open(MODEL_PATH, "wb") as f:
            f.write(response.content)

    return joblib.load(MODEL_PATH)

model = load_model()



df = pd.read_csv("PJMW_hourly.csv")

df["Datetime"] = pd.to_datetime(df["Datetime"])

df = df.sort_values("Datetime").reset_index(drop=True)


# --------------------------------------------------
# Streamlit title
# --------------------------------------------------

st.title("Hourly Energy Consumption Forecast")

st.write(
    "30-day hourly energy consumption forecast "
    "using a trained Random Forest model."
)


# --------------------------------------------------
# Generate forecast when button is clicked
# --------------------------------------------------

if st.button("Generate 30-Day Forecast"):

    # Last available datetime in the historical data
    last_datetime = df["Datetime"].max()

    # Create the next 720 hourly timestamps
    # 30 days × 24 hours = 720 hours
    future_dates = pd.date_range(
        start=last_datetime + pd.Timedelta(hours=1),
        periods=720,
        freq="h"
    )


    # --------------------------------------------------
    # Create US federal holidays
    # --------------------------------------------------

    calendar = USFederalHolidayCalendar()

    holidays = calendar.holidays(
        start=df["Datetime"].min(),
        end=future_dates.max()
    )


    # --------------------------------------------------
    # Store historical consumption values
    # --------------------------------------------------

    history = df["PJMW_MW"].tolist()

    predictions = []


    # --------------------------------------------------
    # Recursive forecasting
    # --------------------------------------------------

    for current_datetime in future_dates:

        # Calendar features
        hour = current_datetime.hour
        day_of_week = current_datetime.dayofweek
        month = current_datetime.month
        year = current_datetime.year
        week = current_datetime.isocalendar().week

        # Holiday
        holiday = current_datetime.normalize() in holidays

        # Weekend
        is_weekend = day_of_week >= 5


        # Lag features
        lag_1 = history[-1]
        lag_24 = history[-24]
        lag_168 = history[-168]


        # Create input features
        input_data = pd.DataFrame({
            "Hour": [hour],
            "DayOfWeek": [day_of_week],
            "Month": [month],
            "Holiday": [holiday],
            "IsWeekend": [is_weekend],
            "Week": [week],
            "Year": [year],
            "lag_1": [lag_1],
            "lag_24": [lag_24],
            "lag_168": [lag_168]
        })


        # Use the trained model to predict
        prediction = model.predict(input_data)[0]


        # Store prediction
        predictions.append(prediction)


        # Add prediction to history
        # This is needed for the next hour's lag features
        history.append(prediction)


    # --------------------------------------------------
    # Create forecast DataFrame
    # --------------------------------------------------

    forecast_df = pd.DataFrame({
        "Datetime": future_dates,
        "Predicted_PJMW_MW": predictions
    })


    # --------------------------------------------------
    # Display forecast
    # --------------------------------------------------

    st.subheader("30-Day Forecast")

    st.line_chart(
        forecast_df.set_index("Datetime")[
            "Predicted_PJMW_MW"
        ]
    )


    # --------------------------------------------------
    # Display hourly predictions
    # --------------------------------------------------

    st.subheader("Hourly Forecast")

    st.dataframe(
        forecast_df,
        use_container_width=True
    )


    # --------------------------------------------------
    # Daily summary
    # --------------------------------------------------

    forecast_summary = forecast_df.copy()

    forecast_summary["Forecast_Day"] = (
        np.arange(len(forecast_summary)) // 24
    ) + 1


    daily_forecast = forecast_summary.groupby(
        "Forecast_Day"
    ).agg(
        Average_Consumption=(
            "Predicted_PJMW_MW",
            "mean"
        ),
        Minimum_Consumption=(
            "Predicted_PJMW_MW",
            "min"
        ),
        Maximum_Consumption=(
            "Predicted_PJMW_MW",
            "max"
        )
    ).reset_index()


    # --------------------------------------------------
    # Display daily summary
    # --------------------------------------------------

    st.subheader("Daily Forecast Summary")

    st.dataframe(
        daily_forecast,
        use_container_width=True
    )


    # --------------------------------------------------
    # Download forecast
    # --------------------------------------------------

    csv = forecast_df.to_csv(index=False)

    st.download_button(
        label="Download Forecast CSV",
        data=csv,
        file_name="30_day_energy_forecast.csv",
        mime="text/csv"
    )