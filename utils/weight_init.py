# utils/weight_init.py
import numpy as np
# Xavier and  Random initialization 
def initialize_weights(input_dim, output_dim, method='random'):
    if method == 'Xavier':
        limit = np.sqrt(6 / (input_dim + output_dim))
        return np.random.uniform(-limit, limit, (input_dim, output_dim))
    if method == 'random':
        return np.random.randn(input_dim, output_dim) * 0.01 
    
