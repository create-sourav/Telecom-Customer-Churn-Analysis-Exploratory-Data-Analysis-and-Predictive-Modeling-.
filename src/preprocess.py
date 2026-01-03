import os
import joblib
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import KMeans
from imblearn.over_sampling import BorderlineSMOTE


DATA_FILE_PATH = "data/churn_clean.xlsx"
ARTIFACT_DIR = "models"


df = pd.read_excel(DATA_FILE_PATH)
print("total null values:", df.isnull().sum().sum())
print("duplicated rows:", df.duplicated(subset=["Customer ID"]).sum())


def load_data():
    df = pd.read_excel(DATA_FILE_PATH)

    y = df["Customer Status"].astype(str)

    kmeans = KMeans(n_clusters=6, random_state=0)
    df["GeoCluster"] = kmeans.fit_predict(df[["Latitude", "Longitude"]])

    X = df.drop(
        ["Customer ID", "Customer Status", "City",
         "Zip Code", "Latitude", "Longitude"],
        axis=1
    )

    X = pd.get_dummies(X, drop_first=True).astype(int)

    return X, y, kmeans


def split_data(X, y):
    return train_test_split(
        X,
        y,
        test_size=0.20,
        stratify=y,
        random_state=0
    )


def preprocess_training_data(x_train, x_test, y_train, y_test):
    encoder = LabelEncoder()

    y_train_enc = encoder.fit_transform(y_train)
    y_test_enc = encoder.transform(y_test)

    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled  = scaler.transform(x_test)

    smote = BorderlineSMOTE(kind="borderline-1", random_state=0)

    x_train_balanced, y_train_balanced = smote.fit_resample(
        x_train_scaled,
        y_train_enc
    )

    return (
        x_train_balanced,
        y_train_balanced,
        x_test_scaled,
        y_test_enc,
        encoder,
        scaler
    )


def save_artifacts(kmeans, encoder, scaler, feature_columns):
    os.makedirs(ARTIFACT_DIR, exist_ok=True)

    joblib.dump(kmeans, f"{ARTIFACT_DIR}/kmeans.pkl")
    joblib.dump(encoder, f"{ARTIFACT_DIR}/label_encoder.pkl")
    joblib.dump(scaler, f"{ARTIFACT_DIR}/scaler.pkl")
    joblib.dump(feature_columns, f"{ARTIFACT_DIR}/feature_columns.pkl")


def run_preprocessing():
    X, y, kmeans = load_data()

    x_train, x_test, y_train, y_test = split_data(X, y)

    (
        x_train_balanced,
        y_train_balanced,
        x_test_scaled,
        y_test_encoded,
        encoder,
        scaler
    ) = preprocess_training_data(x_train, x_test, y_train, y_test)

    feature_columns = X.columns.tolist()

    save_artifacts(kmeans, encoder, scaler, feature_columns)

    return (
        x_train_balanced,
        y_train_balanced,
        x_test_scaled,
        y_test_encoded,
        scaler,
        encoder,
        kmeans,
        feature_columns
    )


if __name__ == "__main__":
    (
        x_train_balanced,
        y_train_balanced,
        x_test_scaled,
        y_test_encoded,
        scaler,
        encoder,
        kmeans,
        feature_columns
    ) = run_preprocessing()

    print("Preprocessing complete.")
    print("Train shape:", x_train_balanced.shape)
    print("Test shape:", x_test_scaled.shape)
    print("Balanced class dist:", np.bincount(y_train_balanced))
