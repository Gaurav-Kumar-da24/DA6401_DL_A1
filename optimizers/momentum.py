# optimizers/momentum.py
import numpy as np

class Momentum:
    def __init__(self, learning_rate=0.01, momentum=0.9, weight_decay=0.0):
        """
        Momentum-based optimizer
        
        Args:
            learning_rate (float): Learning rate
            momentum (float): Momentum coefficient
            weight_decay (float): L2 regularization strength
        """
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.weight_decay = weight_decay
        
        self.velocity_w = None
        self.velocity_b = None
    
    def update(self, model):
        """
        Update the model parameters using momentum
        
        Args:
            model (FeedForwardNN): Neural network model
        """
        weights, biases = model.get_weights()
        dw, db = model.get_gradients()
        
        # Initialize velocity if not already done
        if self.velocity_w is None:
            self.velocity_w = [np.zeros_like(w) for w in weights]
            self.velocity_b = [np.zeros_like(b) for b in biases]
        
        for i in range(len(weights)):
            # Apply L2 regularization to weights
            if self.weight_decay > 0:
                dw[i] += self.weight_decay * weights[i]
            
            # Update velocity
            self.velocity_w[i] = self.momentum * self.velocity_w[i] - self.learning_rate * dw[i]
            self.velocity_b[i] = self.momentum * self.velocity_b[i] - self.learning_rate * db[i]
            
            # Update parameters
            weights[i] += self.velocity_w[i]
            biases[i] += self.velocity_b[i]
        
        model.set_weights(weights, biases)

