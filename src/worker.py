import os
import json
import time
import boto3
import torch
from dotenv import load_dotenv
from optimum.onnxruntime import ORTModelForFeatureExtraction
from transformers import AutoTokenizer

load_dotenv()

SQS_QUEUE_URL = os.getenv("SQS_QUEUE_URL")
AWS_REGION = os.getenv("AWS_DEFAULT_REGION", "us-east-1")
MODEL_DIR = "models/bge-small-onnx"

sqs = boto3.client("sqs", region_name=AWS_REGION)

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

def process_messages():
    print(f"Loading ONNX Model from {MODEL_DIR}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = ORTModelForFeatureExtraction.from_pretrained(
        MODEL_DIR, provider="CPUExecutionProvider"
    )
    print("Worker ready. Polling SQS Queue...")

    while True:
        response = sqs.receive_message(
            QueueUrl=SQS_QUEUE_URL,
            MaxNumberOfMessages=5,
            WaitTimeSeconds=10 # Long polling
        )

        messages = response.get("Messages", [])
        if not messages:
            continue

        for msg in messages:
            receipt_handle = msg["ReceiptHandle"]
            body = json.loads(msg["Body"])
            job_id = body["job_id"]
            text_batch = body["text"]

            start = time.perf_counter()
            
            # Run ONNX inference
            inputs = tokenizer(text_batch, padding=True, truncation=True, return_tensors="pt")
            with torch.no_grad():
                outputs = model(**inputs)

            embeddings = mean_pooling(outputs, inputs["attention_mask"])
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1).numpy().tolist()
            
            latency = (time.perf_counter() - start) * 1000
            print(f"[Job {job_id}] Processed {len(text_batch)} texts in {latency:.2f}ms")

            # Delete processed message from SQS
            sqs.delete_message(QueueUrl=SQS_QUEUE_URL, ReceiptHandle=receipt_handle)

if __name__ == "__main__":
    process_messages()