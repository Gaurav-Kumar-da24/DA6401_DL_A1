# optimizers/sgd.py
import numpy as np

class SGD:
    def __init__(self, learning_rate=0.01, weight_decay=0.0):
        """
        Stochastic Gradient Descent optimizer
        Args:
            learning_rate (float): Learning rate
            weight_decay (float): L2 regularization strength
        """
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
    
    def update(self, model):
        """
        Update the model parameters using SGD
        Args:
            model (FeedForwardNN): Neural network model
        """
        weights, biases = model.get_weights()
        dw, db = model.get_gradients()
        
        for i in range(len(weights)):
            # Apply L2 regularization to weights
            if self.weight_decay > 0:
                weights[i] -= self.learning_rate * self.weight_decay * weights[i]
            
            # Update weights and biases
            weights[i] -= self.learning_rate * dw[i]
            biases[i] -= self.learning_rate * db[i]
        
        model.set_weights(weights, biases)
