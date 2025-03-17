# optimizers/adam.py
import numpy as np

class Adam:
    def __init__(self, learning_rate=0.01, beta1=0.9, beta2=0.999, epsilon=1e-8, weight_decay=0.0):
        """
        Adam optimizer
        
        Args:
            learning_rate (float): Learning rate
            beta1 (float): Exponential decay rate for first moment estimates
            beta2 (float): Exponential decay rate for second moment estimates
            epsilon (float): Small constant for numerical stability
            weight_decay (float): L2 regularization strength
        """
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.weight_decay = weight_decay
        
        self.m_w = None  # First moment of weights
        self.v_w = None  # Second moment of weights
        self.m_b = None  # First moment of biases
        self.v_b = None  # Second moment of biases
        
        self.t = 0  # Time step
    
    def update(self, model):
        """
        Update the model parameters using Adam
        
        Args:
            model (FeedForwardNN): Neural network model
        """
        weights, biases = model.get_weights()
        dw, db = model.get_gradients()
        
        # Initialize moments if not already done
        if self.m_w is None:
            self.m_w = [np.zeros_like(w) for w in weights]
            self.v_w = [np.zeros_like(w) for w in weights]
            self.m_b = [np.zeros_like(b) for b in biases]
            self.v_b = [np.zeros_like(b) for b in biases]
        
        # Increment time step
        self.t += 1
        
        for i in range(len(weights)):
            # Apply L2 regularization to weights
            if self.weight_decay > 0:
                dw[i] += self.weight_decay * weights[i]
            
            # Update biased first moment estimate
            self.m_w[i] = self.beta1 * self.m_w[i] + (1 - self.beta1) * dw[i]
            self.m_b[i] = self.beta1 * self.m_b[i] + (1 - self.beta1) * db[i]
            
            # Update biased second raw moment estimate
            self.v_w[i] = self.beta2 * self.v_w[i] + (1 - self.beta2) * np.square(dw[i])
            self.v_b[i] = self.beta2 * self.v_b[i] + (1 - self.beta2) * np.square(db[i])
            
            # Compute bias-corrected first moment estimate
            m_hat_w = self.m_w[i] / (1 - self.beta1 ** self.t)
            m_hat_b = self.m_b[i] / (1 - self.beta1 ** self.t)
            
            # Compute bias-corrected second raw moment estimate
            v_hat_w = self.v_w[i] / (1 - self.beta2 ** self.t)
            v_hat_b = self.v_b[i] / (1 - self.beta2 ** self.t)
            
            # Update parameters
            weights[i] -= self.learning_rate * m_hat_w / (np.sqrt(v_hat_w) + self.epsilon)
            biases[i] -= self.learning_rate * m_hat_b / (np.sqrt(v_hat_b) + self.epsilon)
        
        model.set_weights(weights, biases)
