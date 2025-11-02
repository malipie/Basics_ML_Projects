import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets.fashion_mnist import load_data
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

# --- Configuration ---
MODEL_DIR = "models"
CNN_MODEL_PATH = f"{MODEL_DIR}/cnn_model.keras" # Keras models use .keras or .h5
np.random.seed(42)
tf.random.set_seed(42)

def main():
    # 1. Load Data
    print("Loading Fashion-MNIST data...")
    (X_train, y_train), (X_test, y_test) = load_data()

    # 2. Prepare Data for CNN
    print("Preparing data for CNN model...")
    # Normalize pixel values
    X_train = X_train / 255.0
    X_test = X_test / 255.0

    # Reshape data to include channel dimension (1 for grayscale)
    # CNNs need data in shape (samples, height, width, channels)
    X_train_cnn = X_train.reshape(X_train.shape[0], 28, 28, 1)
    X_test_cnn = X_test.reshape(X_test.shape[0], 28, 28, 1)

    print(f"Train shapes: X={X_train_cnn.shape}, y={y_train.shape}")
    print(f"Test shapes: X={X_test_cnn.shape}, y={y_test.shape}")

    # 3. Build the CNN Model
    print("Building CNN model architecture...")
    model = Sequential([
        # Layer 1: Convolution + Pooling
        Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        MaxPooling2D((2, 2)),
        
        # Layer 2: Convolution + Pooling
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        
        # Layer 3: Flatten and Dense
        Flatten(),
        Dense(64, activation='relu'),
        Dropout(0.5), # Add dropout for regularization
        Dense(10, activation='softmax') # 10 output classes, softmax for probabilities
    ])

    # 4. Compile the Model
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    
    model.summary() # Print model architecture

    # 5. Train the Model
    print("Training CNN model...")
    model.fit(X_train_cnn, y_train,
              epochs=10, 
              validation_data=(X_test_cnn, y_test),
              batch_size=64)

    # 6. Save the model
    print(f"Saving CNN model to {CNN_MODEL_PATH}...")
    model.save(CNN_MODEL_PATH)

    print("\nNeural network training complete.")

if __name__ == "__main__":
    main()