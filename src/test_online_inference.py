import pandas as pd
from feast import FeatureStore

store = FeatureStore(repo_path="feature_repo")

def fetch_online_features():
    # Simulate receiving incoming entity IDs at prediction time
    entity_rows = [
        {"iris_id": 1},
        {"iris_id": 50},
        {"iris_id": 100},
    ]

    features_to_fetch = [
        "iris_features:sepal_length",
        "iris_features:sepal_width",
        "iris_features:petal_length",
        "iris_features:petal_width",
    ]

    # Sub-10ms lookup from Online Store (SQLite/Redis)
    response = store.get_online_features(
        features=features_to_fetch,
        entity_rows=entity_rows
    ).to_dict()

    df_online = pd.DataFrame.from_dict(response)
    print("--- Online Features Fetched for Real-Time Inference ---")
    print(df_online)

if __name__ == "__main__":
    fetch_online_features()