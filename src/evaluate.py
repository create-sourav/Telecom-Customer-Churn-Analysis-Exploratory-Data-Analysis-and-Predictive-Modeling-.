import joblib
import numpy as np

from sklearn.metrics import (
    accuracy_score,
    log_loss,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    roc_curve
)

from train import train_model


MODEL_PATH = "models/model.pkl"
ENCODER_PATH = "models/label_encoder.pkl"


def evaluate_model():
    model, x_test, y_test = train_model()

    label_encoder = joblib.load(ENCODER_PATH)

    y_test_encoded = label_encoder.transform(y_test)

    y_pred = model.predict(x_test)
    y_prob = model.predict_proba(x_test)

    accuracy = accuracy_score(y_test_encoded, y_pred)
    loss = log_loss(y_test_encoded, y_prob)

    print(f"\nAccuracy: {accuracy:.4f}")
    print(f"Log Loss: {loss:.4f}")

    print("\nPer-Class Metrics:")
    for class_index, class_name in enumerate(label_encoder.classes_):

        y_true_binary = (y_test_encoded == class_index).astype(int)
        y_pred_binary = (y_pred == class_index).astype(int)

        precision = precision_score(y_true_binary, y_pred_binary, zero_division=0)
        recall = recall_score(y_true_binary, y_pred_binary, zero_division=0)
        f1 = f1_score(y_true_binary, y_pred_binary, zero_division=0)

        roc_auc = roc_auc_score(
            y_true_binary,
            y_prob[:, class_index]
        )

        print(
            f"{class_name} | "
            f"Precision={precision:.3f}  "
            f"Recall={recall:.3f}  "
            f"F1={f1:.3f}  "
            f"ROC-AUC={roc_auc:.3f}"
        )

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test_encoded, y_pred))

    print("\nClassification Report:")
    print(classification_report(y_test_encoded, y_pred, zero_division=0))

    # ----- Optimal churn threshold calculation -----
    churn_index = list(label_encoder.classes_).index("Churned")

    y_true_binary = (y_test_encoded == churn_index).astype(int)
    y_churn_prob = y_prob[:, churn_index]

    fpr, tpr, thresholds = roc_curve(y_true_binary, y_churn_prob)

    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]

    joblib.dump(optimal_threshold, "models/churn_threshold.pkl")

    print(f"\nOptimal Churn Threshold Saved: {optimal_threshold:.3f}")


if __name__ == "__main__":
    evaluate_model()
