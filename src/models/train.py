"""Training script for the fraud-detection model.

This scaffold is deliberately non-deterministic — running it twice
produces different metrics on the same input data. A reproducibility
fix is required so downstream tooling (checksum-based caching,
experiment comparison, audit trails) can rely on run-to-run
stability.

Every non-reproducibility concern is already handled here — data
loading, split, training, MLflow logging, model persistence, and
metrics serialisation. Edit this file to add the seed discipline
that makes the run reproducible; do not change anything else.
"""
import os
import json
import joblib
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

TRAIN_CSV = "/root/code/fraud-detection/data/train.csv"
MODEL_PATH = "/root/code/fraud-detection/models/model.pkl"
METRICS_OUT = os.environ.get(
    "METRICS_OUT", "/root/code/fraud-detection/reports/last_metrics.json"
)
RUN_NAME = os.environ.get("MLFLOW_RUN_NAME", "repro-run")

SEED = 42


def main():
    df = pd.read_csv(TRAIN_CSV)
    X = df.drop(columns=["is_fraud"])
    y = df["is_fraud"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=SEED
    )

    model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=SEED)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    metrics = {
        "accuracy": round(accuracy_score(y_test, preds), 6),
        "f1_score": round(f1_score(y_test, preds), 6),
    }

    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment("fraud-detection-repro")
    with mlflow.start_run(run_name=RUN_NAME):
        mlflow.log_params({"n_estimators": 100, "max_depth": 5})
        for key, value in metrics.items():
            mlflow.log_metric(key, value)
        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        joblib.dump(model, MODEL_PATH)
        mlflow.sklearn.log_model(model, name="model")

    # Probe payload is a superset of `metrics` plus feature_importances.
    # The list lives here — not in `metrics` — because MLflow's
    # log_metric only accepts scalars. feature_importances_ is an
    # average over 100 trees' bootstrap + feature-subset randomness;
    # two unseeded runs can coincidentally produce the same accuracy
    # / f1 bucket on a 40-row stratified test set, but their importance
    # triplets essentially never match, so the probe's byte-diff can
    # distinguish a real deterministic run from a lucky collision.
    probe_payload = {
        **metrics,
        "feature_importances": model.feature_importances_.tolist(),
    }
    os.makedirs(os.path.dirname(METRICS_OUT), exist_ok=True)
    with open(METRICS_OUT, "w") as f:
        json.dump(probe_payload, f, indent=2, sort_keys=True)

    print(f"{RUN_NAME}: accuracy={metrics['accuracy']}, f1_score={metrics['f1_score']}")


if __name__ == "__main__":
    main()