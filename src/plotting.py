# plotting.py
"""Functions for generating visualizations."""

import matplotlib.pyplot as plt
import seaborn as sns # Optional, makes plots nicer
from sklearn.metrics import ConfusionMatrixDisplay
import numpy as np

def plot_confusion_matrix(cm, labels, title="Confusion Matrix", filepath=None):
    """
    Plots a confusion matrix using Matplotlib/Seaborn.

    Args:
        cm (np.ndarray): Confusion matrix array.
        labels (list): List of class labels for display.
        title (str): Title for the plot.
        filepath (str, optional): If provided, saves the plot to this path. Defaults to None (show plot).
    """
    try:
        plt.figure(figsize=(8, 6))
        # Use seaborn heatmap for better visuals if available
        if sns:
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                        xticklabels=labels, yticklabels=labels, cbar=False)
        else:
            # Fallback to Matplotlib display
             disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
             disp.plot(cmap=plt.cm.Blues, values_format='d')
             # disp.ax_.set_title(title) # Title set below works for both

        plt.title(title)
        plt.ylabel('Actual Label')
        plt.xlabel('Predicted Label')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout() # Adjust layout

        if filepath:
            plt.savefig(filepath)
            print(f"Confusion matrix saved to {filepath}")
        else:
            plt.show()
        plt.close() # Close the figure to prevent display issues in loops

    except Exception as e:
        print(f"Error during plotting confusion matrix: {type(e).__name__} - {e}")