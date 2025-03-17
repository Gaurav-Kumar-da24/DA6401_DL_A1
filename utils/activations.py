# utils/activations.py
import numpy as np

def identity(x):
    # Identity activation function: f(x) = x, Args: x (ndarray), Returns:ndarray input
    return x
def identity_derivative(x):
    return np.ones_like(x)

def sigmoid(x):
    # Sigmoid activation function: f(x) = 1 / (1 + exp(-x))
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))  # Clip to avoid overflow

def sigmoid_derivative(x):
    #Derivative of sigmoid function: f'(x) = f(x) * (1 - f(x))
    s = sigmoid(x)
    return s * (1 - s)

def tanh(x):
    # tanh(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x))
    return np.tanh(x)

def tanh_derivative(x):
    return 1 - np.square(np.tanh(x))

def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return (x > 0).astype(float)

def softmax(x):
    # Softmax activation function Args: x (ndarray): Input and Returns:ndarray: Softmax output
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))  # Subtract max for numerical stability
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

def get_activation_function(name):
    # Get activation function and its derivative
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
