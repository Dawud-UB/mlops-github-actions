from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib

def train():
    # Load dataset
    X, y = load_breast_cancer(return_X_y=True)
    
    # Train model
    model = LogisticRegression(max_iter=500)
    model.fit(X, y)
    
    # Evaluate
    y_pred = model.predict(X)
    acc = accuracy_score(y, y_pred)
    
    # Save model
    joblib.dump(model, "model.pkl")
    
    # Save metrics to file
    with open("metrics.txt", "w") as f:
        f.write(f"Training Accuracy: {acc:.4f}\n")

if __name__ == "__main__":
    train()
