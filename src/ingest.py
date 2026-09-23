import os
import json
import uuid
import boto3
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field

load_dotenv()

SQS_QUEUE_URL = os.getenv("SQS_QUEUE_URL")
AWS_REGION = os.getenv("AWS_DEFAULT_REGION", "us-east-1")

sqs_client = boto3.client("sqs", region_name=AWS_REGION)

app = FastAPI(title="MLOps Async Ingestion Gateway")

class IngestRequest(BaseModel):
    text: list[str] = Field(..., example=["Asynchronous inference with KEDA and SQS."])

class IngestResponse(BaseModel):
    job_id: str
    status: str
    queue_url: str

@app.get("/health")
def health_check():
    return {"status": "healthy", "service": "ingestion-gateway"}

@app.post("/embed", status_code=status.HTTP_202_ACCEPTED, response_model=IngestResponse)
def enqueue_embedding_request(payload: IngestRequest):
    if not payload.text:
        raise HTTPException(status_code=400, detail="Text payload cannot be empty")

    job_id = str(uuid.uuid4())
    message_body = {
        "job_id": job_id,
        "text": payload.text
    }

    try:
        sqs_client.send_message(
            QueueUrl=SQS_QUEUE_URL,
            MessageBody=json.dumps(message_body)
        )
        return IngestResponse(
            job_id=job_id,
            status="QUEUED",
            queue_url=SQS_QUEUE_URL
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to push message to SQS: {str(e)}")