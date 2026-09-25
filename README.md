# MLOps Platform

A production-style MLOps platform combining a **feature store**, an **ONNX-optimized embedding service**, **event-driven autoscaling**, and an **automated drift-detection and retraining pipeline** — deployed on AWS EKS with Terraform and KEDA.

## Architecture

```
Client ──▶ Ingestion Gateway (FastAPI) ──▶ SQS Queue ──▶ KEDA-scaled Worker Pool (0→10 pods)
                                                                    │
                                                          ONNX Runtime Inference
                                                        (BAAI/bge-small-en-v1.5)
                                                                    │
                                                          Embeddings ──▶ S3 Data Lake

Feast Feature Store ──▶ point-in-time training data ──▶ MLflow (tracking + S3 artifacts)

Prefect Flow: Drift Check (Evidently-style cosine distance) ──▶ Retrain ──▶ Discord/Slack Alert
```

## Key Components

- **Feature Store** (`feature_repo/`) — A [Feast](https://feast.dev) repo defining an `iris_features` feature view over a Parquet offline source, with SQLite online/offline stores for both historical (point-in-time) training data and low-latency online lookups.
- **Training** (`src/train.py`, `src/train_with_feast.py`) — Trains a `RandomForestClassifier`, logging params, metrics (accuracy, ROC AUC), and the model artifact to MLflow (S3-backed artifact store). The Feast variant sources training data via `get_historical_features` for a proper offline/online-consistent pipeline.
- **Async Embedding Service** — A decoupled, event-driven inference path:
  - `src/ingest.py` — FastAPI gateway that accepts text batches and enqueues jobs to SQS, returning a job ID immediately (202 Accepted).
  - `src/worker.py` — Long-polls SQS and runs batched ONNX inference, deleting messages on completion.
  - `src/embedding_app.py` — A synchronous FastAPI alternative exposing `/predict` directly, for lower-latency single-request use, instrumented with Prometheus metrics.
  - `src/export_onnx.py` — Exports `BAAI/bge-small-en-v1.5` from Hugging Face to an optimized ONNX graph for CPU inference.
- **Autoscaling** (`k8s/`) — A KEDA `ScaledObject` scales the worker `Deployment` from **0 to 10 replicas** based on SQS queue depth, so compute cost is zero when idle.
- **Drift Monitoring & Retraining** (`src/drift_monitor.py`, `src/retrain_flow.py`, `src/notifier.py`) — A Prefect flow computes centroid cosine distance between reference and production embeddings; if drift exceeds threshold, it triggers a retraining task and posts an alert to a Discord/Slack webhook.
- **Data Logging** (`src/logger.py`) — Streams every inference request/response pair to a date-partitioned S3 data lake for offline analysis and drift baselining.
- **Infrastructure** (`terraform/`) — Provisions the S3 MLflow artifact bucket, an EKS cluster, an SQS queue for async jobs, and installs KEDA via Helm.

## Tech Stack

FastAPI · Feast · MLflow · scikit-learn · ONNX Runtime / Optimum · Hugging Face Transformers · Prefect · AWS (EKS, SQS, S3) · KEDA · Terraform · Prometheus

## Getting Started

### Install dependencies

```bash
git clone https://github.com/Jinzo03/mlops-platform.git
cd mlops-platform
pip install -r requirements.txt
```

### Feature store: generate data and train

```bash
python src/generate_feature_data.py       # builds feature_repo/data/iris_features.parquet
cd feature_repo && feast apply && cd ..    # registers entities/feature views
python src/train_with_feast.py             # trains using point-in-time Feast features
```

### Export and serve the embedding model

```bash
python src/export_onnx.py                  # exports BAAI/bge-small-en-v1.5 to ONNX
uvicorn src.embedding_app:app --host 0.0.0.0 --port 8000
```

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": ["What is MLOps?"]}'
```

### Run the async ingestion + worker path

Requires an SQS queue (see `terraform/sqs.tf`) and AWS credentials in `.env`:

```
SQS_QUEUE_URL=https://sqs.us-east-1.amazonaws.com/<account_id>/embedding-request-queue
AWS_DEFAULT_REGION=us-east-1
```

```bash
uvicorn src.ingest:app --port 8001   # gateway: POST /embed
python src/worker.py                  # worker: polls SQS, runs ONNX inference
```

### Run the drift-monitoring pipeline

```bash
python src/retain_flow.py
```

### Provision infrastructure

```bash
cd terraform
terraform init
terraform apply
```

Creates the S3 artifact bucket, EKS cluster, SQS queue, and installs KEDA. Then deploy the worker and its autoscaler:

```bash
kubectl apply -f k8s/worker-deployment.yaml
kubectl apply -f k8s/scale-object.yaml
```

## Project Structure

```
mlops-platform/
├── feature_repo/                 # Feast feature store (entities, feature views, data)
├── src/
│   ├── train.py                   # Baseline MLflow training run
│   ├── train_with_feast.py        # Training via Feast historical features
│   ├── generate_feature_data.py   # Synthetic Iris feature dataset generator
│   ├── ingest.py                  # FastAPI gateway -> SQS
│   ├── worker.py                  # SQS consumer running ONNX inference
│   ├── embedding_app.py           # Synchronous FastAPI embedding service
│   ├── export_onnx.py             # Hugging Face -> ONNX model export
│   ├── drift_monitor.py           # Embedding drift detection (cosine distance)
│   ├── retain_flow.py             # Prefect flow: check -> retrain -> notify
│   ├── notifier.py                # Discord/Slack webhook alerts
│   ├── logger.py                  # Inference logging to S3 data lake
│   └── test_online_inference.py   # Feast online-store lookup demo
├── k8s/                           # KEDA ScaledObject + worker Deployment
├── terraform/                     # S3, EKS, SQS, KEDA (Helm)
└── Dockerfile                     # ONNX embedding service container
```

## Notes

Several values (AWS account ID in `k8s/worker-deployment.yaml`, the S3 bucket name in `terraform/main.tf` and `src/train.py`, webhook URLs) are specific to the original author's AWS setup and should be replaced with your own before deploying.
