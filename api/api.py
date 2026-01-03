from fastapi import FastAPI
import pandas as pd
import joblib

from huggingface_hub import hf_hub_download

print(" API starting up...")

REPO_ID = "souravmondal619/churn-mlops-model"   # HF model repo

app = FastAPI(title="Customer Churn Prediction API")


def load_artifacts():
    model_path = hf_hub_download(REPO_ID, "model.pkl")
    scaler_path = hf_hub_download(REPO_ID, "scaler.pkl")
    encoder_path = hf_hub_download(REPO_ID, "label_encoder.pkl")
    kmeans_path = hf_hub_download(REPO_ID, "kmeans.pkl")
    feature_cols_path = hf_hub_download(REPO_ID, "feature_columns.pkl")

    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    encoder = joblib.load(encoder_path)
    kmeans = joblib.load(kmeans_path)
    feature_columns = joblib.load(feature_cols_path)

    return model, scaler, encoder, kmeans, feature_columns


def preprocess_input(data, scaler, kmeans, feature_columns):
    df = pd.DataFrame([data])

    # Geo cluster
    df["GeoCluster"] = kmeans.predict(df[["Latitude", "Longitude"]])

    df = df.drop(
        ["Customer ID", "City", "Zip Code", "Latitude", "Longitude"],
        axis=1,
        errors="ignore"
    )

    df = pd.get_dummies(df, drop_first=True)

    df = df.reindex(columns=feature_columns, fill_value=0)

    df_scaled = scaler.transform(df)

    return df_scaled


@app.get("/")
def root():
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
