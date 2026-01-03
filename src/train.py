import os
import joblib
import numpy as np

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_curve

from preprocess import run_preprocessing
from huggingface_hub import upload_file


MODEL_PATH = "models/model.pkl"
REPO_ID = "souravmondal619/churn-mlops-model"


def upload_artifacts():
    files = [
        ("models/model.pkl", "model.pkl"),
        ("models/scaler.pkl", "scaler.pkl"),
        ("models/label_encoder.pkl", "label_encoder.pkl"),
        ("models/kmeans.pkl", "kmeans.pkl"),
        ("models/feature_columns.pkl", "feature_columns.pkl"),
        ("models/churn_threshold.pkl", "churn_threshold.pkl"),
    ]

    for local_path, repo_path in files:
        upload_file(
            path_or_fileobj=local_path,
            path_in_repo=repo_path,
            repo_id=REPO_ID,
            repo_type="model"
        )

    print("✔ All artifacts uploaded to HuggingFace Hub")


def train_model():

    (
        x_train,
        y_train,
        x_test,
        y_test,
        scaler,
        label_encoder,
        kmeans,
        feature_columns,
    ) = run_preprocessing()

    model = GradientBoostingClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=3,
        min_samples_split=10,
        min_samples_leaf=5,
        subsample=0.8,
        random_state=0
    )

    model.fit(x_train, y_train)

    os.makedirs("models", exist_ok=True)

    joblib.dump(model, "models/model.pkl")
    joblib.dump(scaler, "models/scaler.pkl")
    joblib.dump(label_encoder, "models/label_encoder.pkl")
    joblib.dump(kmeans, "models/kmeans.pkl")
    joblib.dump(feature_columns, "models/feature_columns.pkl")

    # -----------------------------
    # NOTEBOOK-STYLE THRESHOLD (TEST SET)
    # -----------------------------
    churn_index = list(model.classes_).index(0)      # class 0 = Churned

    probs = model.predict_proba(x_test)[:, churn_index]

    y_test_binary = (y_test == 0).astype(int)

    fpr, tpr, thresholds = roc_curve(y_test_binary, probs)

    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]

    joblib.dump(optimal_threshold, "models/churn_threshold.pkl")
    print(f"Saved optimal churn threshold: {optimal_threshold:.3f}")
    # -----------------------------

    print("Saved model + preprocessing artifacts locally")

    upload_artifacts()

    return model, x_test, y_test


if __name__ == "__main__":
    model, x_test, y_test = train_model()
    print("Training complete.")
