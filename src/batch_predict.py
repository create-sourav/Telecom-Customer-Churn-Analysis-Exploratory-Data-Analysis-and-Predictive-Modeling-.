import joblib
import pandas as pd
import os

MODEL_PATH = "models/model.pkl"
SCALER_PATH = "models/scaler.pkl"
ENCODER_PATH = "models/label_encoder.pkl"
KMEANS_PATH = "models/kmeans.pkl"
FEATURE_COLUMNS_PATH = "models/feature_columns.pkl"
THRESHOLD_PATH = "models/churn_threshold.pkl"

INPUT_FILE = "new_test_data/new_data.xlsx"
OUTPUT_FILE = "new_test_data/predictions_output.csv"


def load_artifacts():
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    encoder = joblib.load(ENCODER_PATH)
    kmeans = joblib.load(KMEANS_PATH)
    feature_columns = joblib.load(FEATURE_COLUMNS_PATH)

    churn_threshold = (
        joblib.load(THRESHOLD_PATH)
        if os.path.exists(THRESHOLD_PATH)
        else 0.25
    )

    return model, scaler, encoder, kmeans, feature_columns, churn_threshold


def preprocess(df, scaler, kmeans, feature_columns):
    if {"Latitude", "Longitude"}.issubset(df.columns):
        df["GeoCluster"] = kmeans.predict(df[["Latitude", "Longitude"]])

    df = df.drop(
        ["City", "Zip Code", "Latitude", "Longitude"],
        axis=1,
        errors="ignore"
    )

    df = pd.get_dummies(df, drop_first=True)
    df = df.reindex(columns=feature_columns, fill_value=0)

    return scaler.transform(df)


def run_batch_predictions():
    (
        model,
        scaler,
        encoder,
        kmeans,
        feature_columns,
        churn_threshold
    ) = load_artifacts()

    df = pd.read_excel(INPUT_FILE)

    X = preprocess(df.copy(), scaler, kmeans, feature_columns)

    probs = model.predict_proba(X)
    churn_index = list(encoder.classes_).index("Churned")

    df["Prediction"] = encoder.inverse_transform(model.predict(X))
    df["Churn_Probability"] = probs[:, churn_index]
    df["Churn_Flag"] = (df["Churn_Probability"] >= churn_threshold).map(
        {True: "YES", False: "NO"}
    )
    df["Threshold_Used"] = float(churn_threshold)

    df_out = df[["Customer ID", "Prediction", "Churn_Flag",
                 "Churn_Probability", "Threshold_Used"]]

    df_out.to_csv(OUTPUT_FILE, index=False)

    print(f"\nSaved predictions → {OUTPUT_FILE}")
    print(df_out.head())


if __name__ == "__main__":
    run_batch_predictions()
