import os
from optimum.onnxruntime import ORTModelForFeatureExtraction
from transformers import AutoTokenizer

MODEL_ID = "BAAI/bge-small-en-v1.5"
SAVE_DIR = "models/bge-small-onnx"

def export_model():
    print(f"Loading and exporting {MODEL_ID} to ONNX format...")
    
    # Download and convert PyTorch model to ONNX
    model = ORTModelForFeatureExtraction.from_pretrained(
        MODEL_ID, 
        export=True
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    # Save ONNX model graph and tokenizer locally
    model.save_pretrained(SAVE_DIR)
    tokenizer.save_pretrained(SAVE_DIR)
    print(f"Successfully exported quantized ONNX model to {SAVE_DIR}")

if __name__ == "__main__":
    export_model()