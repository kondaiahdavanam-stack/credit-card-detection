from flask import Flask, render_template, request
import joblib
import pandas as pd
from datetime import datetime

app = Flask(__name__)

model = joblib.load("fraud_model.pkl")
scaler = joblib.load("scaler.pkl")
label_encoder = joblib.load("label_encoder.pkl")


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    # Read input
    amount = float(request.form["amount"])
    transaction_hour = int(request.form["transaction_hour"])
    device_trust_score = float(request.form["device_trust_score"])
    velocity_last_24h = float(request.form["velocity_last_24h"])
    cardholder_age = int(request.form["cardholder_age"])
    current_time = datetime.now().strftime("%d-%m-%Y %I:%M:%S %p")
    merchant = request.form["merchant_category"].title()
    merchant = label_encoder.transform([merchant])[0]

    foreign_transaction = 1 if "foreign_transaction" in request.form else 0
    location_mismatch = 1 if "location_mismatch" in request.form else 0

    # ---------- Feature Engineering ----------
    high_amount = int(amount > 5000)          # We'll improve this later
    night_transaction = int(transaction_hour >= 22 or transaction_hour <= 5)
    high_velocity = int(velocity_last_24h > 5)
    geo_risk = foreign_transaction + location_mismatch
    low_trust_device = int(device_trust_score < 50)

    # ---------- Scale Numerical Features ----------
    numerical = scaler.transform([[
        amount,
        transaction_hour,
        device_trust_score,
        velocity_last_24h,
        cardholder_age
    ]])[0]

    amount = numerical[0]
    transaction_hour = numerical[1]
    device_trust_score = numerical[2]
    velocity_last_24h = numerical[3]
    cardholder_age = numerical[4]

    # ---------- Create Input ----------
    input_data = pd.DataFrame([[
        amount,
        transaction_hour,
        merchant,
        foreign_transaction,
        location_mismatch,
        device_trust_score,
        velocity_last_24h,
        cardholder_age,
        high_amount,
        night_transaction,
        high_velocity,
        geo_risk,
        low_trust_device
    ]], columns=[
        "amount",
        "transaction_hour",
        "merchant_category",
        "foreign_transaction",
        "location_mismatch",
        "device_trust_score",
        "velocity_last_24h",
        "cardholder_age",
        "high_amount",
        "night_transcation",
        "high_velocity",
        "geo_risk",
        "low_trust_device"
    ])

    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    if probability >= 0.90:
        risk = "HIGH 🔴"
    elif probability >= 0.70:
        risk = "MEDIUM 🟠"
    else:
        risk = "LOW 🟢"

    if prediction == 1:

        status = "⚠ FRAUD DETECTED"

        recommendation = "❌ Block the transaction immediately and notify the customer."

    else:

        status = "✅ LEGITIMATE TRANSACTION"

        recommendation = "✔ Transaction approved. No suspicious activity detected."

    return render_template(
    "index.html",

    prediction=status,
    probability=f"{probability*100:.2f}",
    risk=risk,
    recommendation=recommendation,
    time=current_time,

    amount=request.form["amount"],
    hour=request.form["transaction_hour"],
    merchant=request.form["merchant_category"],
    foreign=foreign_transaction,
    mismatch=location_mismatch,
    trust=request.form["device_trust_score"],
    velocity=request.form["velocity_last_24h"],
    age=request.form["cardholder_age"],

    model_accuracy="99.70"
)


if __name__ == "__main__":
    app.run(debug=True)