"""Test script to evaluate classifier generalization on hand‑written queries."""

import joblib
import os

# Load the trained model
model_path = os.path.join("configs", "classifier_model.joblib")
model = joblib.load(model_path)

# Define queries with their true intent labels
queries = [
    ("What is the boiling point of water?", "factual"),
    ("Explain what photosynthesis means.", "definitional"),
    ("Is Python or Java faster for machine learning?", "comparative"),
    ("What year did the director of Inception release their first film?", "multihop"),
    ("Who wrote Pride and Prejudice?", "factual"),
    ("What does the term algorithm mean in computer science?", "definitional"),
    ("Which is bigger, the Pacific or Atlantic ocean?", "comparative"),
    ("What company did the founder of Tesla start before Tesla?", "multihop")
]

# Evaluate each query
for query, true_intent in queries:
    pred = model.predict([query])[0]
    probs = model.predict_proba([query])[0]
    confidence = probs.max()
    print(f"Query: {query}")
    print(f"  Predicted: {pred} (confidence: {confidence:.3f}) | True: {true_intent}\n")