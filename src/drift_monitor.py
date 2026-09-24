import numpy as np
from scipy.spatial.distance import cosine

def detect_embedding_drift(
    reference_embeddings: list[list[float]],
    production_embeddings: list[list[float]],
    threshold: float = 0.05
) -> tuple[bool, float]:
    """
    Computes Centroid Cosine Distance between reference baseline embeddings
    and current production embeddings to check for statistical drift.
    """
    ref_arr = np.array(reference_embeddings)
    prod_arr = np.array(production_embeddings)

    # Compute mean centroid vectors
    centroid_ref = np.mean(ref_arr, axis=0)
    centroid_prod = np.mean(prod_arr, axis=0)

    # Calculate Cosine Distance (1 - Cosine Similarity)
    drift_score = float(cosine(centroid_ref, centroid_prod))
    is_drifted = drift_score > threshold

    print(f"[Evidently Drift Check] Distance: {drift_score:.4f} | Threshold: {threshold}")
    return is_drifted, drift_score