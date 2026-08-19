# src/classifier/predict.py
"""
Utility to load the trained intent classifier and predict intent for a given query.

The function `predict` returns a tuple:
    (predicted_intent: str, confidence: float)

The intent labels are: "factual", "comparative", "definitional", "multihop".
Confidence is the highest class probability (if `predict_proba` is available) or 1.0
as a fallback.
"""

import os
import joblib
from typing import Tuple

# Path to the saved model file (relative to this script location)
MODEL_PATH = os.path.join(
    os.path.dirname(__file__),  # src/classifier
    "..", "..", "configs", "classifier_model.joblib"
)

# Load the model once at import time
# This will raise an error if the file is missing, making the problem obvious early.
model = joblib.load(MODEL_PATH)

def predict(query: str) -> Tuple[str, float]:
    """
    Predict the intent of `query`.

    Args:
        query: Input text to classify.

    Returns:
        predicted_intent: One of the four intent categories.
        confidence:   Highest probability estimated by the model (0.0‑1.0).

    The returned tuple matches the ``predicted_intent`` and ``confidence`` fields
    defined in ``RouterResult`` (``src/utils/schemas.py``).
    """
    # Use predict_proba if available (e.g., scikit-learn classifiers 
    # sklearn Pipelines). If not, fall back to predict.
    if hasattr(model, "predict_proba"):
        # predict_proba returns an array of shape (n_samples, n_classes)
        probs = model.predict_proba([query])[0]  # first (and only) input example
        confidence = float(max(probs))
        label_idx = int(probs.argmax())
    elif hasattr(model, "predict"):
        label_idx = int(model.predict([query])[0])
        confidence = 1.0  # unknown confidence – assume certain
    else:
        raise RuntimeError("Loaded model does not implement 'predict' or 'predict_proba'.")

    # Convert numeric index to the actual intent label
    predicted_intent = model.classes_[label_idx]

    return predicted_intent, confidence


# -------------------------------------------------------------------------------
# Example usage when run directly (optional)
# -------------------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python predict.py <query>")
        sys.exit(1)

    q = sys.argv[1]
    intent, conf = predict(q)
    print(f"Query: {q}")
    print(f"Predicted intent: {intent} (confidence: {conf:.3f})")