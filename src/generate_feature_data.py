import os
import pandas as pd
from datetime import datetime, timedelta
from sklearn.datasets import load_iris

def generate_data():
    # 1. Load Iris data
    iris = load_iris(as_frame=True)
    df = iris.frame

    # Clean column names to standard feature naming
    df.columns = [
        "sepal_length",
        "sepal_width",
        "petal_length",
        "petal_width",
        "target",
    ]

    # Add primary key (entity_id) and timestamp columns required by Feast
    df["iris_id"] = range(1, len(df) + 1)
    
    # Simulate historical feature creation timestamps over the last 10 days
    now = datetime.now()
    timestamps = [now - timedelta(days=i % 10) for i in range(len(df))]
    df["event_timestamp"] = timestamps
    df["created_timestamp"] = now

    # Ensure feature_repo/data directory exists
    os.makedirs("feature_repo/data", exist_ok=True)

    # Save to Parquet format (Feast standard offline store format)
    parquet_path = "feature_repo/data/iris_features.parquet"
    df.to_parquet(parquet_path, index=False)
    print(f"Successfully generated offline feature dataset: {parquet_path}")

if __name__ == "__main__":
    generate_data()