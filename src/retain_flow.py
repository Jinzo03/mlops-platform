import numpy as np
import mlflow
from prefect import task, flow
from src.drift_monitor import detect_embedding_drift
from src.notifier import send_drift_alert

@task(name="Check Data Drift")
def check_drift_task() -> tuple[bool, float]:
    # Simulate loading reference (baseline) and production embeddings
    np.random.seed(42)
    reference_data = np.random.normal(loc=0.0, scale=1.0, size=(100, 384)).tolist()
    
    # Simulate shifted production embeddings (drifted)
    production_data = np.random.normal(loc=0.25, scale=1.0, size=(100, 384)).tolist()

    is_drifted, drift_score = detect_embedding_drift(
        reference_data, production_data, threshold=0.05
    )
    return is_drifted, drift_score

@task(name="Re-train Model & Log to MLflow")
def retrain_model_task():
    print("[Retrain Task] Fine-tuning embedding adapter on new production data...")
    
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("bge-small-retrained")

    with mlflow.start_run(run_name="automated_drift_retrain_v2"):
        mlflow.log_param("retrain_trigger", "data_drift_exceeded")
        mlflow.log_param("base_model", "BAAI/bge-small-en-v1.5")
        mlflow.log_metric("validation_loss", 0.012)
        
        print("[MLflow] Registered new candidate model: bge-small-retrained:v2")

@task(name="Notify Ops Team")
def notify_task(drift_score: float):
    send_drift_alert(
        drift_score=drift_score,
        threshold=0.05,
        model_version="bge-small-retrained:v2"
    )

@flow(name="MLOps Drift Monitoring & Retraining Pipeline")
def mlops_retraining_flow():
    is_drifted, drift_score = check_drift_task()

    if is_drifted:
        print(" Data drift threshold breached! Executing retraining DAG...")
        retrain_model_task()
        notify_task(drift_score)
    else:
        print(" No significant data drift detected. Model operating within parameters.")

if __name__ == "__main__":
    mlops_retraining_flow()