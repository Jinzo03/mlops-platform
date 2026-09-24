import os
import io
import pandas as pd
from datetime import datetime
import boto3
from dotenv import load_dotenv

load_dotenv()

S3_BUCKET = "mlflow-artifacts-jinzo03-mlops"
AWS_REGION = os.getenv("AWS_DEFAULT_REGION", "us-east-1")

s3_client = boto3.client("s3", region_name=AWS_REGION)

def log_prediction_to_s3(text_batch: list[str], embeddings: list[list[float]]):
    """Streams inference payloads directly into S3 Data Lake partitions."""
    df = pd.DataFrame({
        "timestamp": [datetime.utcnow()] * len(text_batch),
        "text": text_batch,
        "embedding": embeddings
    })

    # Partition by YYYY-MM-DD
    date_str = datetime.utcnow().strftime("%Y-%m-%d")
    timestamp_str = datetime.utcnow().strftime("%H-%M-%S-%f")
    s3_key = f"production_logs/dt={date_str}/log_{timestamp_str}.parquet"

    buffer = io.BytesIO()
    df.to_parquet(buffer, index=False)
    buffer.seek(0)

    try:
        s3_client.put_object(
            Bucket=S3_BUCKET,
            Key=s3_key,
            Body=buffer.getvalue()
        )
        print(f"[Data Lake] Logged {len(text_batch)} samples to s3://{S3_BUCKET}/{s3_key}")
    except Exception as e:
        print(f"[Data Lake Error] Failed to upload log to S3: {e}")