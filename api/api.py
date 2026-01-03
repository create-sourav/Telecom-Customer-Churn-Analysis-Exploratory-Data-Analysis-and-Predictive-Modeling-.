from fastapi import FastAPI
import joblib
import pandas as pd
from huggingface_hub import hf_hub_download

REPO_ID = "souravmondal619/churn-mlops-model"

app = FastAPI(title="Customer Churn Prediction API")


def load_artifacts():
    model = joblib.load(hf_hub_download(REPO_ID, "model.pkl"))
    scaler = joblib.load(hf_hub_download(REPO_ID, "scaler.pkl"))
    encoder = joblib.load(hf_hub_download(REPO_ID, "label_encoder.pkl"))
    kmeans = joblib.load(hf_hub_download(REPO_ID, "kmeans.pkl"))
    feature_columns = joblib.load(hf_hub_download(REPO_ID, "feature_columns.pkl"))
    churn_threshold = joblib.load(hf_hub_download(REPO_ID, "churn_threshold.pkl"))

    return model, scaler, encoder, kmeans, feature_columns, churn_threshold


def preprocess_input(data, scaler, kmeans, feature_columns):
    df = pd.DataFrame([data])

    if "Latitude" in df and "Longitude" in df:
        df["GeoCluster"] = kmeans.predict(df[["Latitude", "Longitude"]])

    df = df.drop(
        ["Customer ID", "City", "Zip Code", "Latitude", "Longitude"],
        axis=1,
        errors="ignore",
    )

    df = pd.get_dummies(df, drop_first=True)
    df = df.reindex(columns=feature_columns, fill_value=0)

    return scaler.transform(df)


@app.get("/")
def root():
    return {"status": "ok", "message": "API online"}


@app.post("/predict")
def predict(customer: dict):

    (
        model,
        scaler,
        encoder,
        kmeans,
        feature_columns,
        churn_threshold,
    ) = load_artifacts()

    processed = preprocess_input(customer, scaler, kmeans, feature_columns)

    probs = model.predict_proba(processed)[0]

    # 🔒 safer lookup (class name instead of numeric index)
    churn_index = list(encoder.classes_).index("Churned")
    churn_prob = float(probs[churn_index])

    churn_flag = "YES" if churn_prob >= churn_threshold else "NO"

    pred_encoded = model.predict(processed)[0]
    pred_label = encoder.inverse_transform([pred_encoded])[0]

    return {
        "customer_id": customer.get("Customer ID"),
        "prediction": pred_label,
        "churn_flag": churn_flag,
        "churn_probability": churn_prob,
        "threshold_used": float(churn_threshold),
        "probabilities": {
            class_name: float(probs[i])
            for i, class_name in enumerate(encoder.classes_)
        },
    }
