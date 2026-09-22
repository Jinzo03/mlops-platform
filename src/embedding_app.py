import os
import time
import torch
import numpy as np
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field
from prometheus_fastapi_instrumentator import Instrumentator
from optimum.onnxruntime import ORTModelForFeatureExtraction
from transformers import AutoTokenizer

MODEL_DIR = "models/bge-small-onnx"

# Global state holders
model = None
tokenizer = None

def mean_pooling(model_output, attention_mask):
    """Performs mean pooling over token embeddings weighted by attention mask."""
    token_embeddings = model_output[0] # First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

class EmbeddingRequest(BaseModel):
    text: list[str] = Field(
        ..., 
        example=["What is MLOps?", "Deploying models to Kubernetes using GitOps."],
        description="Batch of sentences/strings to vectorize"
    )

class EmbeddingResponse(BaseModel):
    embeddings: list[list[float]]
    dimensions: int
    batch_size: int
    latency_ms: float

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, tokenizer
    print("Configuring ONNX Runtime multi-threading options...")
    
    # Configure CPU thread-pool parallelism for ONNX execution
    os.environ["OMP_NUM_THREADS"] = "4"
    os.environ["MKL_NUM_THREADS"] = "4"

    print(f"Loading ONNX Model from {MODEL_DIR}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = ORTModelForFeatureExtraction.from_pretrained(
        MODEL_DIR,
        provider="CPUExecutionProvider" # Swap to 'CUDAExecutionProvider' if GPU is available
    )
    print("ONNX Inference Engine ready!")
    yield
    print("Shutting down inference worker...")

app = FastAPI(
    title="High-Performance ONNX Embedding Server",
    version="1.0.0",
    lifespan=lifespan
)

# Expose Prometheus metrics at /metrics
Instrumentator().instrument(app).expose(app)

@app.get("/health", status_code=status.HTTP_200_OK)
def health_check():
    if model is None or tokenizer is None:
        raise HTTPException(status_code=500, detail="ONNX model worker not loaded")
    return {"status": "healthy", "engine": "ONNX Runtime", "model": "bge-small-en-v1.5"}

@app.post("/predict", response_model=EmbeddingResponse)
def generate_embeddings(payload: EmbeddingRequest):
    if not payload.text:
        raise HTTPException(status_code=400, detail="Payload text list cannot be empty")

    start_time = time.perf_counter()

    try:
        # Tokenize input batch
        encoded_input = tokenizer(
            payload.text, 
            padding=True, 
            truncation=True, 
            max_length=512, 
            return_tensors="pt"
        )

        # ONNX Runtime Inference
        with torch.no_grad():
            model_output = model(**encoded_input)

        # Mean pooling & L2 Normalization
        sentence_embeddings = mean_pooling(model_output, encoded_input["attention_mask"])
        sentence_embeddings = torch.nn.functional.normalize(sentence_embeddings, p=2, dim=1)

        embeddings_list = sentence_embeddings.numpy().tolist()
        latency = (time.perf_counter() - start_time) * 1000

        return EmbeddingResponse(
            embeddings=embeddings_list,
            dimensions=len(embeddings_list[0]),
            batch_size=len(payload.text),
            latency_ms=round(latency, 2)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")