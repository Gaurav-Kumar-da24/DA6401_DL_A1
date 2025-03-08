# utils/metrics.py
import numpy as np
import matplotlib.pyplot as plt
import wandb
from sklearn.metrics import confusion_matrix

def compute_accuracy(y_pred, y_true):
    """
    Compute classification accuracy
    
    Args:
        y_pred (ndarray): Predicted probabilities
        y_true (ndarray): One-hot encoded true labels
    
    Returns:
        float: Classification accuracy
    """
    # Convert predictions to class labels
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true_classes = np.argmax(y_true, axis=1)
    
    # Compute accuracy
    return np.mean(y_pred_classes == y_true_classes)

def plot_confusion_matrix(y_pred, y_true, dataset_name):
    """
    Plot and log confusion matrix
    
    Args:
        y_pred (ndarray): Predicted probabilities
        y_true (ndarray): One-hot encoded true labels
        dataset_name (str): Name of the dataset
    """
    # Convert predictions to class labels
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true_classes = np.argmax(y_true, axis=1)
    
    # Compute confusion matrix
    cm = confusion_matrix(y_true_classes, y_pred_classes)
    
    # Define class names
    if dataset_name == 'fashion_mnist':
        class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 
                       'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    else:  # mnist
        class_names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    
    # Create figure
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    
    # Add labels
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)
    
    # Add text annotations
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
    
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    
    # Log to wandb
    wandb.log({"confusion_matrix": wandb.Image(plt)})
    
    plt.close()
