# Fashion-MNIST Classifier: A Deep Learning Comparison Project

This project demonstrates an advanced machine learning workflow, comparing a classic ML model (SVM) against a Deep Learning model (Convolutional Neural Network) on the Fashion-MNIST dataset.

The core goal is to show *why* and *how* CNNs outperform models like SVM on image classification tasks by leveraging spatial information (pixels, edges, shapes) that classic models ignore when data is flattened.

## 🚀 Key Features

* **Model Comparison:** Directly compares `scikit-learn` (SVM) with `TensorFlow/Keras` (CNN).
* **Professional Structure:** Separates concerns into training scripts (`src/`) and analysis notebooks (`notebooks/`).
* **Reusable Code:** Uses helper functions (`src/utils.py`) for consistent plotting.
* **Deep Learning:** Implements a simple but effective Convolutional Neural Network (CNN).

## 📁 Project Structure

fashion_classifier_project/

├── models/

│   ├── svm_model.pkl       (Saved classic SVM model)

│   └── cnn_model.keras     (Saved Keras CNN model)

│   └── test_data.pkl     (Saved test dataset for evaluation)

├── notebooks/

│   └── evaluate_models.ipynb (Jupyter Notebook for model comparison)

├── src/

│   ├── init.py

│   ├── train_classic_models.py (Trains and saves the SVM)

│   ├── train_neural_network.py (Trains and saves the CNN)

│   └── utils.py                (Helper functions for plotting)

├── requirements.txt

└── README.md

## ⚙️ Setup

1.  **Create a Virtual Environment:**
    ```bash
    python -m venv venv
    ```

2.  **Activate the Environment:**
    * On Mac/Linux: `source venv/bin/activate`
    * On Windows: `venv\Scripts\activate`

3.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## 💡 How to Use: A 2-Step Workflow

### Step 1: Train the Models

Run *both* training scripts from the project's root directory. This only needs to be done once.

**First, train the classic model:**
```bash
# This script will also save the test data
python src/train_classic_models.py

**Second, train the neural network:**
# This script builds and trains the CNN
python src/train_neural_network.py

### Step 2: Analyze the Results

Step 2: Analyze the Results
Now that the models are trained, you can compare them.

Start Jupyter Notebook:

Bash

jupyter notebook
In your browser, navigate to notebooks/ and open evaluate_models.ipynb.

Run the cells in the notebook to:

Load both saved models and the test data.

Generate side-by-side classification reports.

Display confusion matrices for both SVM and CNN.

Visually analyze the types of errors each model makes.