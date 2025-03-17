# utils/losses.py
import numpy as np

def mean_squared_error(y_pred, y_true):
    return np.mean(np.sum(np.square(y_pred - y_true), axis=1)) / 2

def mean_squared_error_derivative(y_pred, y_true):
    return y_pred - y_true

def cross_entropy(y_pred, y_true):
    epsilon = 1e-15     # Added small epsilon to avoid log(0)
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return -np.mean(np.sum(y_true * np.log(y_pred), axis=1))

def cross_entropy_derivative(y_pred, y_true):
    # For softmax output and one-hot encoded labels, the derivative simplifies to: y_pred - y_true
    return y_pred - y_true

def get_loss_function(name):
    if name == 'mean_squared_error':
        return mean_squared_error, mean_squared_error_derivative
    elif name == 'cross_entropy':
        return cross_entropy, cross_entropy_derivative
    else:
        raise ValueError(f"Unsupported loss function: {name}")
