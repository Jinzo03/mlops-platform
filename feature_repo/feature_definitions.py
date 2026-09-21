from datetime import timedelta
from feast import (
    Entity,
    Field,
    FeatureView,
    FileSource,
    ValueType,
)
from feast.types import Float64, Int64

# 1. Define Offline Data Source
iris_source = FileSource(
    name="iris_source",
    path="data/iris_features.parquet",
    timestamp_field="event_timestamp",
    created_timestamp_column="created_timestamp",
)

# 2. Define Primary Entity (Primary key used to query features)
iris_entity = Entity(
    name="iris_id",
    value_type=ValueType.INT64,
    description="Unique identifier for Iris sample observation",
)

# 3. Define Feature View (Group of related features schema)
iris_feature_view = FeatureView(
    name="iris_features",
    entities=[iris_entity],
    ttl=timedelta(days=30),
    schema=[
        Field(name="sepal_length", dtype=Float64),
        Field(name="sepal_width", dtype=Float64),
        Field(name="petal_length", dtype=Float64),
        Field(name="petal_width", dtype=Float64),
        Field(name="target", dtype=Int64),
    ],
    online=True,
    source=iris_source,
)