import joblib
import pandas as pd
import numpy as np
import os

MODEL_PATH = "models/model.pkl"
SCALER_PATH = "models/scaler.pkl"
ENCODER_PATH = "models/label_encoder.pkl"
KMEANS_PATH = "models/kmeans.pkl"
FEATURE_COLUMNS_PATH = "models/feature_columns.pkl"
THRESHOLD_PATH = "models/churn_threshold.pkl"


def load_artifacts():
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    encoder = joblib.load(ENCODER_PATH)
    kmeans = joblib.load(KMEANS_PATH)
    feature_columns = joblib.load(FEATURE_COLUMNS_PATH)

    # fallback safety: if threshold missing, default to 0.25
    if os.path.exists(THRESHOLD_PATH):
        churn_threshold = joblib.load(THRESHOLD_PATH)
    else:
        churn_threshold = 0.25

    return model, scaler, encoder, kmeans, feature_columns, churn_threshold


def preprocess_new_record(input_data, scaler, kmeans, feature_columns):
    df = pd.DataFrame([input_data])

    # geographic clustering
    if "Latitude" in df.columns and "Longitude" in df.columns:
        df["GeoCluster"] = kmeans.predict(df[["Latitude", "Longitude"]])

    # drop unused columns
    df = df.drop(
        ["Customer ID", "City", "Zip Code", "Latitude", "Longitude"],
        axis=1,
        errors="ignore"
    )

    # one-hot encoding
    df = pd.get_dummies(df, drop_first=True)

    # align with training columns
    df = df.reindex(columns=feature_columns, fill_value=0)

    return scaler.transform(df)


def predict_customer_status(input_data):
    (
        model,
        scaler,
        encoder,
        kmeans,
        feature_columns,
        churn_threshold
    ) = load_artifacts()

    customer_id = input_data.get("Customer ID")

    processed = preprocess_new_record(
        input_data, scaler, kmeans, feature_columns
    )

    pred_encoded = model.predict(processed)[0]
    pred_label = encoder.inverse_transform([pred_encoded])[0]

    probs = model.predict_proba(processed)[0]

    # churn probability (class name "Churned")
    churn_index = list(encoder.classes_).index("Churned")
    churn_prob = float(probs[churn_index])

    churn_flag = "YES" if churn_prob >= churn_threshold else "NO"

    return {
        "customer_id": customer_id,
        "prediction": pred_label,
        "churn_flag": churn_flag,
        "churn_probability": churn_prob,
        "threshold_used": float(churn_threshold),
        "probabilities": {
            class_name: float(probs[i])
            for i, class_name in enumerate(encoder.classes_)
        }
    }


if __name__ == "__main__":
    example_customer = {
        "Customer ID": "0001-BGFD",
        "Monthly Charge": 75,
        "Total Revenue": 2800,
        "Tenure Months": 24,
        "Latitude": 40.7,
        "Longitude": -73.9,
        "Gender": "Male",
        "Senior Citizen": "No",
        "Internet Service": "Fiber Optic",
        "Contract": "Month-to-Month",
        "Payment Method": "Credit Card"
    }

    print(predict_customer_status(example_customer))
