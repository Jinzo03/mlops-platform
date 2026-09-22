import os
import pandas as pd
import mlflow.sklearn
from contextlib import asynccontextmanager
from dotenv import load_dotenv
from feast import FeatureStore
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field
from prometheus_fastapi_instrumentator import Instrumentator

load_dotenv()

# Global state holders
model = None
feast_store = None

# Pydantic Input Schemes
class PredictionRequestByEntity(BaseModel):
    iris_id: int = Field(..., example=1, description="Entity ID to fetch features from Feast")

class PredictionRequestByFeatures(BaseModel):
    sepal_length: float = Field(..., example=5.1)
    sepal_width: float = Field(..., example=3.5)
    petal_length: float = Field(..., example=1.4)
    petal_width: float = Field(..., example=0.2)

class PredictionResponse(BaseModel):
    prediction: int
    probabilities: list[float]
    model_version: str

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, feast_store
    print("Initializing Feast Feature Store...")
    feast_store = FeatureStore(repo_path="feature_repo")

    # Load latest registered model from MLflow tracking server
    model_uri = "models:/iris-rf-feast-model/latest"
    print(f"Loading model from MLflow Registry: {model_uri}...")
    
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    try:
        model = mlflow.sklearn.load_model(model_uri)
        print("Model successfully loaded into memory!")
    except Exception as e:
        print(f"Error loading model: {e}")
        raise e

    yield
    print("Shutting down inference service...")

app = FastAPI(
    title="MLOps Model Serving Service",
    version="1.0.0",
    lifespan=lifespan
)

# Instrument Prometheus metrics at /metrics
Instrumentator().instrument(app).expose(app)

@app.get("/health", status_code=status.HTTP_200_OK)
def health_check():
    if model is None:
        raise HTTPException(status_code=500, detail="Model is not loaded")
    return {"status": "healthy", "service": "mlops-inference-api"}

@app.post("/predict/entity", response_model=PredictionResponse)
def predict_by_entity(payload: PredictionRequestByEntity):
    """Fetches real-time features from Feast for a given iris_id and returns prediction."""
    features_to_fetch = [
        "iris_features:sepal_length",
        "iris_features:sepal_width",
        "iris_features:petal_length",
        "iris_features:petal_width",
    ]

    try:
        response = feast_store.get_online_features(
            features=features_to_fetch,
            entity_rows=[{"iris_id": payload.iris_id}]
        ).to_dict()

        df_features = pd.DataFrame.from_dict(response)
        
        # Verify feature lookup
        if df_features["sepal_length"].iloc[0] is None:
            raise HTTPException(status_code=404, detail=f"iris_id {payload.iris_id} not found in Online Store")

        input_data = df_features[["sepal_length", "sepal_width", "petal_length", "petal_width"]]
        
        prediction = int(model.predict(input_data)[0])
        probabilities = model.predict_proba(input_data)[0].tolist()

        return PredictionResponse(
            prediction=prediction,
            probabilities=probabilities,
            model_version="iris-rf-feast-model:latest"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/features", response_model=PredictionResponse)
def predict_by_features(payload: PredictionRequestByFeatures):
    """Direct prediction using explicit feature payload."""
    try:
        input_data = pd.DataFrame([[
            payload.sepal_length,
            payload.sepal_width,
            payload.petal_length,
            payload.petal_width
        ]], columns=["sepal_length", "sepal_width", "petal_length", "petal_width"])

        prediction = int(model.predict(input_data)[0])
        probabilities = model.predict_proba(input_data)[0].tolist()

        return PredictionResponse(
            prediction=prediction,
            probabilities=probabilities,
            model_version="iris-rf-feast-model:latest"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))