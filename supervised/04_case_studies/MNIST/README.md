# MNIST Digit Classifier: A Structured ML Project

This project demonstrates a professional workflow for a machine learning task (classifying handwritten digits).

It separates the project into two distinct phases:
1.  **Training (`.py` scripts):** Heavy-duty scripts that pre-process data, train multiple models, and save the resulting model files.
2.  **Analysis (`.ipynb` notebooks):** Interactive notebooks that load the pre-trained models for evaluation, visualization, and error analysis.

## 📁 Project Structure
MNIST/ ├── models/ │ ├── lr_model.pkl (Saved Logistic Regression model) │ ├── svm_model.pkl (Saved SVM model) │ └── test_data.pkl (Saved test dataset for consistent evaluation) ├── notebooks/ │ └── evaluate_models.ipynb (Jupyter Notebook for analysis) ├── src/ │ ├── init.py │ ├── train.py (Main script to train and save all models) │ └── utils.py (Helper functions for plotting) ├── requirements.txt └── README.md


## 🚀 Setup

1.  **Create a Virtual Environment**
    (Recommended to isolate project dependencies)
    ```bash
    python -m venv venv
    ```

2.  **Activate the Environment**
    * On Mac/Linux:
        ```bash
        source venv/bin/activate
        ```
    * On Windows:
        ```bash
        venv\Scripts\activate
        ```

3.  **Install Dependencies**
    Ensure your environment is active (you should see `(venv)` at the start of your terminal prompt).
    ```bash
    pip install -r requirements.txt
    ```

## 💡 How to Use: A 2-Step Workflow

This project has a two-step workflow.

### Step 1: Train the Models

Run the main training script from the project's root directory. You only need to do this once or whenever you want to re-train the models.

```bash
# Make sure your (venv) is active!
python src/train.py


### Step 2: Analyze the Results
Once the models are trained, you can interactively analyze them.

Start Jupyter Notebook: (Make sure your (venv) is still active)
In your browser, navigate to the notebooks/ folder and open evaluate_models.ipynb.

Run the cells in the notebook to:

Load the saved models (.pkl) and test data.

Generate classification reports for both models.

Display interactive confusion matrices.

Compare model performance.

Visualize examples of where the models made mistakes.