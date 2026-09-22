FROM python:3.11-slim as builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

FROM python:3.11-slim

WORKDIR /app

# Copy installed dependencies from builder stage
COPY --from=builder /install /usr/local

# Copy application files, Feast configs, and database registries
COPY feature_repo/ ./feature_repo/
COPY src/ ./src/
COPY mlflow.db .

EXPOSE 8000

ENV PYTHONUNBUFFERED=1

CMD ["uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "8000"]