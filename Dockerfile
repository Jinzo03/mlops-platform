# Stage 1: Dependency builder
FROM python:3.11-slim as builder

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# Stage 2: Runtime Image
FROM python:3.11-slim

WORKDIR /app

COPY --from=builder /install /usr/local

# Copy exported ONNX model weights and application code
COPY models/bge-small-onnx ./models/bge-small-onnx
COPY src/embedding_app.py ./src/embedding_app.py

# Threading tuning for containerized CPU inference
ENV OMP_NUM_THREADS=4
ENV MKL_NUM_THREADS=4
ENV OPENBLAS_NUM_THREADS=4
ENV PYTHONUNBUFFERED=1

EXPOSE 8000

CMD ["uvicorn", "src.embedding_app:app", "--host", "0.0.0.0", "--port", "8000"]