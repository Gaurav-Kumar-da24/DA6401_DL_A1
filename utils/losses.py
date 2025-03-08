# utils/losses.py
import numpy as np

def mean_squared_error(y_pred, y_true):
    """
    Mean Squared Error loss function
    
    Args:
        y_pred (ndarray): Predicted values
        y_true (ndarray): True values
    
    Returns:
        float: MSE loss
    """
    return np.mean(np.sum(np.square(y_pred - y_true), axis=1)) / 2

def mean_squared_error_derivative(y_pred, y_true):
    """
    Derivative of Mean Squared Error loss function
    
    Args:
        y_pred (ndarray): Predicted values
        y_true (ndarray): True values
    
    Returns:
        ndarray: Gradient of MSE with respect to predictions
    """
    return y_pred - y_true

def cross_entropy(y_pred, y_true):
    """
    Cross Entropy loss function
    
    Args:
        y_pred (ndarray): Predicted probabilities
        y_true (ndarray): One-hot encoded true labels
    
    Returns:
        float: Cross entropy loss
    """
    # Add small epsilon to avoid log(0)
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return -np.mean(np.sum(y_true * np.log(y_pred), axis=1))

def cross_entropy_derivative(y_pred, y_true):
    """
    Derivative of Cross Entropy loss function
    
    Args:
        y_pred (ndarray): Predicted probabilities
        y_true (ndarray): One-hot encoded true labels
    
    Returns:
        ndarray: Gradient of cross entropy with respect to predictions
    """
    # For softmax output and one-hot encoded labels, the derivative simplifies to:
    return y_pred - y_true

def get_loss_function(name):
    """
    Get loss function and its derivative by name
    
    Args:
        name (str): Name of the loss function
    
    Returns:
        tuple: (loss_function, derivative_function)
    """
    if name == 'mean_squared_error':
        return mean_squared_error, mean_squared_error_derivative
    elif name == 'cross_entropy':
        return cross_entropy, cross_entropy_derivative
    else:
        raise ValueError(f"Unsupported loss function: {name}")
