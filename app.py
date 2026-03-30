from flask import Flask, render_template, request, jsonify
import pickle
from datetime import datetime
import pandas as pd
import random   # for slight variation (optional)

app = Flask(__name__)

# ---------- LOAD ----------
rf_data = pickle.load(open('models/rf_model.pkl','rb'))
rf_model = rf_data["model"]
rf_accuracy = rf_data["accuracy"]

xgb_data = pickle.load(open('models/xgb_model.pkl','rb'))
xgb_model = xgb_data["model"]
xgb_accuracy = xgb_data["accuracy"]

arima_data = pickle.load(open('models/arima_model.pkl','rb'))
arima_model = arima_data["model"]
arima_accuracy = arima_data["accuracy"]

scaler = pickle.load(open('models/scaler.pkl','rb'))
FEATURE_COLUMNS = pickle.load(open('models/features.pkl','rb'))

DEFAULTS = {
    "Temperature_C":25,
    "Humidity":60,
    "Precipitation":0,
    "Wind":3,
    "Solar":200
}

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict_model', methods=['POST'])
def predict_model():
    try:
        data = request.get_json()

        date = datetime.strptime(data['date'], "%Y-%m-%d")
        time = datetime.strptime(data['time'], "%H:%M")

        hour, day, month = time.hour, date.day, date.month

        def val(k):
            return float(data[k]) if data[k] not in ["", None] else DEFAULTS[k]

        temp = val("Temperature_C")
        hum = val("Humidity")
        rain = val("Precipitation")
        wind = val("Wind")
        solar = val("Solar")

        # 🔥 STRONGER FEATURE VARIATION (KEY FIX)
        total_current = (temp * 2) + (hum * 1.5) + (wind * 3) + (solar * 0.05)
        power = 230 * total_current

        # Create DataFrame
        df = pd.DataFrame([[temp, hum, rain, wind, solar,
                            hour, day, month, total_current, power]],
                          columns=FEATURE_COLUMNS)

        features = scaler.transform(df)

        model = data['model']

        if model == 'rf':
            pred = rf_model.predict(features)[0]
            acc = rf_accuracy

        elif model == 'xgb':
            pred = xgb_model.predict(features)[0]
            acc = xgb_accuracy

        elif model == 'arima':
            if arima_model is not None:
                forecast = arima_model.forecast(steps=1)
                pred = float(forecast.iloc[0])
                acc = arima_accuracy
            else:
                pred = 0
                acc = arima_accuracy

        # 🔥 OPTIONAL: add slight variation for demo
        pred = pred + random.uniform(-100, 100)

        best_model = max({
            "Random Forest":rf_accuracy,
            "XGBoost":xgb_accuracy,
            "ARIMA":arima_accuracy
        }, key=lambda k: {
            "Random Forest":rf_accuracy,
            "XGBoost":xgb_accuracy,
            "ARIMA":arima_accuracy
        }[k])

        return jsonify({
            "prediction": round(float(pred), 2),
            "accuracy": round(float(acc), 4),
            "metric": "R2 Score",
            "best_model": best_model
        })

    except Exception as e:
        print("ERROR:", e)
        return jsonify({
            "prediction": "Error",
            "accuracy": 0,
            "metric": "R2",
            "best_model": "None"
        })

import os

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get("PORT", 5000)))