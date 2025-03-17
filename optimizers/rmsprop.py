# optimizers/rmsprop.py
import numpy as np

class RMSprop:
    def __init__(self, learning_rate=0.01, beta=0.9, epsilon=1e-8, weight_decay=0.0):
        """
        RMSprop optimizer
        Args:
            learning_rate (float): Learning rate
            beta (float): Exponential decay rate for the squared gradients
            epsilon (float): Small constant for numerical stability
            weight_decay (float): L2 regularization strength
        """
        self.learning_rate = learning_rate
        self.beta = beta
        self.epsilon = epsilon
        self.weight_decay = weight_decay
        
        self.cache_w = None
        self.cache_b = None
    
    def update(self, model):
        """
        Update the model parameters using RMSprop
        Args:model (FeedForwardNN): Neural network model
        """
        weights, biases = model.get_weights()
        dw, db = model.get_gradients()
        
        # Initialize cache if not already done
        if self.cache_w is None:
            self.cache_w = [np.zeros_like(w) for w in weights]
            self.cache_b = [np.zeros_like(b) for b in biases]
        
        for i in range(len(weights)):
            # Apply L2 regularization to weights
            if self.weight_decay > 0:
                dw[i] += self.weight_decay * weights[i]
            
            # Update cache
            self.cache_w[i] = self.beta * self.cache_w[i] + (1 - self.beta) * np.square(dw[i])
            self.cache_b[i] = self.beta * self.cache_b[i] + (1 - self.beta) * np.square(db[i])
            
            # Update parameters
            weights[i] -= self.learning_rate * dw[i] / (np.sqrt(self.cache_w[i]) + self.epsilon)
            biases[i] -= self.learning_rate * db[i] / (np.sqrt(self.cache_b[i]) + self.epsilon)
        
        model.set_weights(weights, biases)
