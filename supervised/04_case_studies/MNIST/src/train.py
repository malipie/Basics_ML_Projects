import numpy as np
import pandas as pd
import joblib  
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# --- Configuration ---
np.random.seed(42)
SAMPLE_SIZE = 7000
MODEL_DIR = "models"
TEST_DATA_PATH = "models/test_data.pkl"

def main():
    # 1. Load Data
    print("Fetching MNIST dataset from OpenML...")
    mnist = fetch_openml('mnist_784', version=1, as_frame=True, parser='auto')
    X = mnist['data']
    y = mnist['target'].astype(int)
    print(f"Full dataset loaded: X={X.shape}, y={y.shape}")

    # 2. Prepare Data: Sub-sampling
    print(f"Taking a stratified sample of {SAMPLE_SIZE} images...")
    X_sample, _, y_sample, _ = train_test_split(
        X, y, 
        train_size=SAMPLE_SIZE, 
        random_state=42, 
        stratify=y
    )

    # 3. Prepare Data: Scaling
    X_sample = X_sample / 255.0

    # 4. Split Sampled Data
    X_train, X_test, y_train, y_test = train_test_split(
        X_sample, y_sample, 
        test_size=0.25, 
        random_state=42, 
        stratify=y_sample
    )
    print(f"Train shapes: X={X_train.shape}, y={y_train.shape}")
    print(f"Test shapes: X={X_test.shape}, y={y_test.shape}")

    # 5. Save Test Data for Notebook
    # We save the test set so the notebook can use the exact same data
    print(f"Saving test data to {TEST_DATA_PATH}...")
    joblib.dump((X_test, y_test), TEST_DATA_PATH)

    # --- Model 1: SVM ---
    print("\nTraining SVM model (kernel='rbf')...")
    svm_model = SVC(gamma='scale', kernel='rbf')
    svm_model.fit(X_train, y_train)
    
    # Evaluate and print basic score
    y_pred_svm = svm_model.predict(X_test)
    acc_svm = accuracy_score(y_test, y_pred_svm)
    print(f"SVM Model Accuracy: {acc_svm:.4f}")

    # Save the model
    svm_model_path = f"{MODEL_DIR}/svm_model.pkl"
    print(f"Saving SVM model to {svm_model_path}...")
    joblib.dump(svm_model, svm_model_path)

    # --- Model 2: Logistic Regression ---
    print("\nTraining Logistic Regression model...")
    # 'saga' solver is good for this kind of data, 'lbfgs' is also a default
    lr_model = LogisticRegression(solver='saga', max_iter=100, random_state=42, multi_class='multinomial')
    lr_model.fit(X_train, y_train)

    # Evaluate and print basic score
    y_pred_lr = lr_model.predict(X_test)
    acc_lr = accuracy_score(y_test, y_pred_lr)
    print(f"Logistic Regression Accuracy: {acc_lr:.4f}")

    # Save the model
    lr_model_path = f"{MODEL_DIR}/lr_model.pkl"
    print(f"Saving Logistic Regression model to {lr_model_path}...")
    joblib.dump(lr_model, lr_model_path)

    print("\nTraining complete. Models and test data saved.")

if __name__ == "__main__":
    main()