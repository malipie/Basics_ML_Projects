import pandas as pd
import matplotlib.pyplot as plt
import plotly.figure_factory as ff
from sklearn.metrics import confusion_matrix

def plot_confusion_matrix(cm, title='Confusion Matrix'):
    """
    Plots an interactive confusion matrix using Plotly.
    """
    columns = ['pred_' + str(i) for i in range(10)]
    index = ['true_' + str(i) for i in range(10)]
    
    cm_reversed = cm[::-1]
    cm_df = pd.DataFrame(cm_reversed, columns=columns, index=index[::-1])

    fig = ff.create_annotated_heatmap(
        z=cm_df.values, x=list(cm_df.columns), y=list(cm_df.index),
        colorscale='ice', showscale=True, reversescale=True
    )
    fig.update_layout(width=700, height=500, title=title, font_size=16)
    fig.show()

def plot_error_examples(X_test_errors, y_true_errors, y_pred_errors, image_shape=(28, 28)):
    """
    Displays the first 4 images that the model misclassified.
    """
    plt.figure(figsize=(12, 10))
    for idx in range(4):
        if idx >= len(y_true_errors):
            break
            
        image = X_test_errors.iloc[idx].values.reshape(image_shape)
        true_label = y_true_errors.iloc[idx]
        pred_label = y_pred_errors[idx]
        
        plt.subplot(2, 4, idx + 1)
        plt.axis('off')
        plt.imshow(image, cmap='Greys')
        plt.title(f"True: {true_label}\nPred: {pred_label}")
    
    plt.suptitle("Error Analysis: Misclassified Images")
    plt.show()