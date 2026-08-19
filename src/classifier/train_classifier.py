import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score
import joblib
import os

def main():
    # Load the training data
    data_path = "data/processed/intent_training_data.csv"
    df = pd.read_csv(data_path)
    
    # Features and labels
    X = df["query"]
    y = df["intent"]
    
    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Create a pipeline: TF-IDF + Logistic Regression
    model = make_pipeline(
        TfidfVectorizer(),
        LogisticRegression(max_iter=1000, random_state=42)
    )
    
    # Train the model
    model.fit(X_train, y_train)
    
    # Predict and evaluate
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Test accuracy: {accuracy:.4f}")
    
    # Save the trained model
    model_dir = "configs"
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "classifier_model.joblib")
    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")

if __name__ == "__main__":
    main()
