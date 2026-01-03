import os
import joblib

from sklearn.ensemble import GradientBoostingClassifier

from preprocess import run_preprocessing


MODEL_PATH = "models/model.pkl"


def train_model():
    (
        x_train,
        y_train,
        x_test,
        y_test
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
    joblib.dump(model, MODEL_PATH)

    return model, x_test, y_test


if __name__ == "__main__":
    model, x_test, y_test = train_model()
    print("Model training complete. Model saved at:", MODEL_PATH)
