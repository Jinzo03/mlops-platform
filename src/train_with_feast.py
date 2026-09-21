import os
import pandas as pd
import mlflow
import mlflow.sklearn
from dotenv import load_dotenv
from feast import FeatureStore
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score

# Load AWS credentials securely from .env
load_dotenv()

S3_BUCKET = "s3://mlflow-artifacts-jinzo03-mlops"

# Connect to Feast repository
store = FeatureStore(repo_path="feature_repo")

def main():
    # 1. Read Entity DataFrame (iris_id + event_timestamp)
    df_parquet = pd.read_parquet("feature_repo/data/iris_features.parquet")
    entity_df = df_parquet[["iris_id", "event_timestamp"]]

    # 2. Fetch Historical Features via Point-in-Time Join
    features_to_fetch = [
        "iris_features:sepal_length",
        "iris_features:sepal_width",
        "iris_features:petal_length",
        "iris_features:petal_width",
        "iris_features:target",
    ]

    print("Fetching point-in-time features from Feast Feature Store...")
    training_data = store.get_historical_features(
        entity_df=entity_df,
        features=features_to_fetch
    ).to_df()

    # Prepare features and target label
    X = training_data[
        ["sepal_length", "sepal_width", "petal_length", "petal_width"]
    ]
    y = training_data["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 3. Setup MLflow Experiment
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    experiment_name = "iris-feast-s3-pipeline"
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        mlflow.create_experiment(
            name=experiment_name,
            artifact_location=S3_BUCKET
        )
    mlflow.set_experiment(experiment_name)

    n_estimators = 200
    max_depth = 5

    with mlflow.start_run(run_name="rf_feast_s3_v1"):
        clf = RandomForestClassifier(
            n_estimators=n_estimators, max_depth=max_depth, random_state=42
        )
        clf.fit(X_train, y_train)

        y_pred = clf.predict(X_test)
        y_proba = clf.predict_proba(X_test)

        acc = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_proba, multi_class="ovr")

        # Log Metadata & Model
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_param("data_source", "Feast Feature Store")
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("roc_auc", auc)

        # Log Model with fixed validation rules & clean arguments
        mlflow.sklearn.log_model(
            sk_model=clf,
            name="model",
            registered_model_name="iris-rf-feast-model",
            skops_trusted_types=["sklearn.tree._tree.Tree"]
        )

        print("Training completed using Feast features!")
        print(f"Artifacts pushed to: {S3_BUCKET}")
        print(f"Accuracy: {acc:.4f} | ROC AUC: {auc:.4f}")

if __name__ == "__main__":
    main()
