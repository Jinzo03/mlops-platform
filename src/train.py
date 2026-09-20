import os
import mlflow
import mlflow.sklearn
from dotenv import load_dotenv
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score

# Load AWS credentials securely from .env file
load_dotenv()

S3_BUCKET = "s3://mlflow-artifacts-jinzo03-mlops"

mlflow.set_tracking_uri("sqlite:///mlflow.db")

experiment = mlflow.get_experiment_by_name("iris-classification-s3")
if experiment is None:
    mlflow.create_experiment(
        name="iris-classification-s3",
        artifact_location=S3_BUCKET
    )
mlflow.set_experiment("iris-classification-s3")

def main():
    X, y = load_iris(return_X_y=True, as_frame=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    n_estimators = 150
    max_depth = 6

    with mlflow.start_run(run_name="rf_s3_artifact_v1"):
        clf = RandomForestClassifier(
            n_estimators=n_estimators, max_depth=max_depth, random_state=42
        )
        clf.fit(X_train, y_train)

        y_pred = clf.predict(X_test)
        y_proba = clf.predict_proba(X_test)

        acc = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_proba, multi_class="ovr")

        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("roc_auc", auc)

        mlflow.sklearn.log_model(
            sk_model=clf,
            artifact_path="model",
            registered_model_name="iris-rf-s3-model"
        )

        print("Run completed successfully.")
        print(f"Artifacts pushed to: {S3_BUCKET}")
        print(f"Accuracy: {acc:.4f} | ROC AUC: {auc:.4f}")

if __name__ == "__main__":
    main()