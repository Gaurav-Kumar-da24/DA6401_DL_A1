# utils/activations.py
import numpy as np

def identity(x):
    """
    Identity activation function: f(x) = x
    
    Args:
        x (ndarray): Input
    
    Returns:
        ndarray: x
    """
    return x

def identity_derivative(x):
    """
    Derivative of identity function: f'(x) = 1
    
    Args:
        x (ndarray): Input
    
    Returns:
        ndarray: Array of ones with same shape as x
    """
    return np.ones_like(x)

def sigmoid(x):
    """
    Sigmoid activation function: f(x) = 1 / (1 + exp(-x))
    
    Args:
        x (ndarray): Input
    
    Returns:
        ndarray: Sigmoid output
    """
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))  # Clip to avoid overflow

def sigmoid_derivative(x):
    """
    Derivative of sigmoid function: f'(x) = f(x) * (1 - f(x))
    
    Args:
        x (ndarray): Input
    
    Returns:
        ndarray: Derivative values
    """
    s = sigmoid(x)
    return s * (1 - s)

def tanh(x):
    """
    Hyperbolic tangent activation function
    
    Args:
        x (ndarray): Input
    
    Returns:
        ndarray: tanh output
    """
    return np.tanh(x)

def tanh_derivative(x):
    """
    Derivative of tanh function: f'(x) = 1 - f(x)^2
    
    Args:
        x (ndarray): Input
    
    Returns:
        ndarray: Derivative values
    """
    return 1 - np.square(np.tanh(x))

def relu(x):
    """
    Rectified Linear Unit activation function: f(x) = max(0, x)
    
    Args:
        x (ndarray): Input
    
    Returns:
        ndarray: ReLU output
    """
    return np.maximum(0, x)

def relu_derivative(x):
    """
    Derivative of ReLU function: f'(x) = 1 if x > 0 else 0
    
    Args:
        x (ndarray): Input
    
    Returns:
        ndarray: Derivative values
    """
    return (x > 0).astype(float)

def softmax(x):
    """
    Softmax activation function
    
    Args:
        x (ndarray): Input
    
    Returns:
        ndarray: Softmax output
    """
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))  # Subtract max for numerical stability
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

def get_activation_function(name):
    """
    Get activation function and its derivative by name
    
    Args:
        name (str): Name of the activation function
    
    Returns:
        tuple: (activation_function, derivative_function)
    """
    if name == 'identity':
        return identity, identity_derivative
    elif name == 'sigmoid':
        return sigmoid, sigmoid_derivative
    elif name == 'tanh':
        return tanh, tanh_derivative
    elif name == 'ReLU':
        return relu, relu_derivative
    else:
        raise ValueError(f"Unsupported activation function: {name}")
