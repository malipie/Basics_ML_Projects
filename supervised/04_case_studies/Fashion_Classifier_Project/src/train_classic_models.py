import numpy as np
import joblib
from tensorflow.keras.datasets.fashion_mnist import load_data
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# --- Configuration ---
MODEL_DIR = "models"
SVM_MODEL_PATH = f"{MODEL_DIR}/svm_model.pkl"
TEST_DATA_PATH = f"{MODEL_DIR}/test_data.pkl" # We save the test set here
np.random.seed(42)

def main():
    # 1. Load Data
    print("Loading Fashion-MNIST data...")
    (X_train, y_train), (X_test, y_test) = load_data()

    # 2. Prepare Data
    print("Preparing data for classic model...")
    # Normalize pixel values (from 0-255 to 0.0-1.0)
    X_train = X_train / 255.0
    X_test = X_test / 255.0

    # 3. Save Scaled Test Data (as 2D images) for notebook
    # We save the *un-flattened* test data for later visualization
    print(f"Saving scaled test data to {TEST_DATA_PATH}...")
    joblib.dump((X_test, y_test), TEST_DATA_PATH)

    # 4. Flatten data for SVM
    # Classic models cannot see 2D structure, they need flat vectors
    X_train_flat = X_train.reshape(X_train.shape[0], -1)
    X_test_flat = X_test.reshape(X_test.shape[0], -1)

    print(f"Train shapes: X={X_train_flat.shape}, y={y_train.shape}")
    print(f"Test shapes: X={X_test_flat.shape}, y={y_test.shape}")

    # 5. Train SVM Model
    # We use a subset (e.g., 10k) for faster training. SVM is slow on 60k samples.
    print("Training SVM model on a subset of 20,000 samples...")
    sample_size = 20000
    classifier = SVC(gamma='scale', kernel='rbf', verbose=True)
    classifier.fit(X_train_flat[:sample_size], y_train[:sample_size])

    # 6. Evaluate and print basic score
    y_pred_svm = classifier.predict(X_test_flat)
    acc_svm = accuracy_score(y_test, y_pred_svm)
    print(f"SVM Model Accuracy (on full test set): {acc_svm:.4f}")

    # 7. Save the model
    print(f"Saving SVM model to {SVM_MODEL_PATH}...")
    joblib.dump(classifier, SVM_MODEL_PATH)

    print("\nClassic model training complete.")

if __name__ == "__main__":
    main()