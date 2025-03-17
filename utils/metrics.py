import numpy as np

def compute_accuracy(y_pred, y_true):
    # Convert one-hot encoded vectors to class indices
    y_pred_indices = np.argmax(y_pred, axis=1)
    y_true_indices = np.argmax(y_true, axis=1)
    correct = np.sum(y_pred_indices == y_true_indices)
    total = len(y_true_indices)
    return correct/total