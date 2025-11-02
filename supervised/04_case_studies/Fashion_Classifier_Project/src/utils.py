import pandas as pd
import matplotlib.pyplot as plt
import plotly.figure_factory as ff
import numpy as np

# This list is a constant for the Fashion MNIST dataset
CLASS_NAMES = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

def plot_confusion_matrix(cm, class_names, title='Confusion Matrix'):
    """
    Plots an interactive confusion matrix using Plotly.
    """
    cm_reversed = cm[::-1]
    class_names_reversed = class_names[::-1]
    
    cm_df = pd.DataFrame(cm_reversed, columns=class_names, index=class_names_reversed)

    fig = ff.create_annotated_heatmap(
        z=cm_df.values, x=list(cm_df.columns), y=list(cm_df.index),
        colorscale='ice', showscale=True, reversescale=True
    )
    fig.update_layout(width=800, height=600, title=title, font_size=16)
    fig.show()

def plot_error_examples(X_test_errors, y_true_errors, y_pred_errors, class_names):
    """
    Displays the first 15 images that the model misclassified.
    Assumes X_test_errors is 2D image data (e.g., shape (N, 28, 28)).
    """
    plt.figure(figsize=(16, 10))
    for idx in range(15):
        if idx >= len(y_true_errors):
            break
            
        image = X_test_errors[idx]
        true_label = class_names[y_true_errors[idx]]
        pred_label = class_names[y_pred_errors[idx]]
        
        plt.subplot(3, 5, idx + 1)
        plt.axis('off')
        plt.imshow(image, cmap='Greys')
        plt.title(f"True: {true_label}\nPred: {pred_label}", fontsize=12)
    
    plt.suptitle("Error Analysis: Misclassified Images", fontsize=20)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()