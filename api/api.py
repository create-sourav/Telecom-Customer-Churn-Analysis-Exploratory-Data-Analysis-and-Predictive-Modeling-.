from fastapi import FastAPI
import joblib
import pandas as pd

print("🚀 API starting up...")   # <--- visible in HuggingFace logs

MODEL_PATH = "models/model.pkl"
SCALER_PATH = "models/scaler.pkl"
ENCODER_PATH = "models/label_encoder.pkl"
KMEANS_PATH = "models/kmeans.pkl"
FEATURE_COLUMNS_PATH = "models/feature_columns.pkl"

app = FastAPI(title="Customer Churn Prediction API")


def load_artifacts():
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    encoder = joblib.load(ENCODER_PATH)
    kmeans = joblib.load(KMEANS_PATH)
    feature_columns = joblib.load(FEATURE_COLUMNS_PATH)

    return model, scaler, encoder, kmeans, feature_columns


def preprocess_input(data, scaler, kmeans, feature_columns):
    df = pd.DataFrame([data])

    # Geo cluster
    df["GeoCluster"] = kmeans.predict(df[["Latitude", "Longitude"]])

    # Drop fields not used by the model
    df = df.drop(
        ["Customer ID", "City", "Zip Code", "Latitude", "Longitude"],
        axis=1,
        errors="ignore"
    )

    # One-hot encode
    df = pd.get_dummies(df, drop_first=True)

    # Align with training columns
    df = df.reindex(columns=feature_columns, fill_value=0)

    df_scaled = scaler.transform(df)

    return df_scaled


@app.get("/")
def root():
    # HuggingFace uses this for health checks
    return {"status": "ok", "message": "Customer Churn Prediction API is running"}


@app.get("/health")
def health():
    return {"healthy": True}


@app.post("/predict")
def predict(customer: dict):
    model, scaler, encoder, kmeans, feature_columns = load_artifacts()

    processed = preprocess_input(customer, scaler, kmeans, feature_columns)

    pred_encoded = model.predict(processed)[0]
    pred_label = encoder.inverse_transform([pred_encoded])[0]

    probs = model.predict_proba(processed)[0]

    return {
        "prediction": pred_label,
        "probabilities": {
            class_name: float(probs[i])
            for i, class_name in enumerate(encoder.classes_)
        }
    }
