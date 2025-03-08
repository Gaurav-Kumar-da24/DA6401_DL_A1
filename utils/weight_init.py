# utils/weight_init.py
import numpy as np

def initialize_weights(input_dim, output_dim, method='random'):
    """
    Initialize weights for a neural network layer
    
    Args:
        input_dim (int): Input dimension
        output_dim (int): Output dimension
        method (str): Initialization method ('random' or 'Xavier')
    
    Returns:
        ndarray: Initialized weights
    """
    if method == 'random':
        # Small random values
        return np.random.randn(input_dim, output_dim) * 0.01
    
    elif method == 'Xavier':
        # Xavier/Glorot initialization
        limit = np.sqrt(6 / (input_dim + output_dim))
        return np.random.uniform(-limit, limit, (input_dim, output_dim))
    
    else:
        raise ValueError(f"Unsupported weight initialization method: {method}")
