# models/neural_network.py
import numpy as np
from utils.weight_init import initialize_weights

class FeedForwardNN:
    def __init__(self, input_dim, hidden_sizes, output_dim, activation_fn, activation_derivative, weight_init='random'):
        """
        Initialize a feedforward neural network
        
        Args:
            input_dim (int): Input dimension
            hidden_sizes (list): List of hidden layer sizes
            output_dim (int): Output dimension
            activation_fn (function): Activation function for hidden layers
            activation_derivative (function): Derivative of activation function
            weight_init (str): Weight initialization method ('random' or 'Xavier')
        """
        self.input_dim = input_dim
        self.hidden_sizes = hidden_sizes
        self.output_dim = output_dim
        self.activation_fn = activation_fn
        self.activation_derivative = activation_derivative
        
        # Initialize weights and biases
        self.weights = []
        self.biases = []
        
        # Input layer to first hidden layer
        if len(hidden_sizes) > 0:
            self.weights.append(initialize_weights(input_dim, hidden_sizes[0], weight_init))
            self.biases.append(np.zeros((1, hidden_sizes[0])))
        
        # Hidden layers
        for i in range(1, len(hidden_sizes)):
            self.weights.append(initialize_weights(hidden_sizes[i-1], hidden_sizes[i], weight_init))
            self.biases.append(np.zeros((1, hidden_sizes[i])))
        
        # Last hidden layer to output layer
        if len(hidden_sizes) > 0:
            self.weights.append(initialize_weights(hidden_sizes[-1], output_dim, weight_init))
        else:
            self.weights.append(initialize_weights(input_dim, output_dim, weight_init))
        
        self.biases.append(np.zeros((1, output_dim)))
        
        # Cached values for backpropagation
        self.z_values = []  # Pre-activation values
        self.a_values = []  # Post-activation values
        self.x = None       # Input to the network
        
        # Gradients
        self.dw = [np.zeros_like(w) for w in self.weights]
        self.db = [np.zeros_like(b) for b in self.biases]
    
    def forward(self, x):
        """
        Forward pass through the network
        
        Args:
            x (ndarray): Input data of shape (batch_size, input_dim)
        
        Returns:
            ndarray: Output predictions of shape (batch_size, output_dim)
        """
        self.x = x
        self.z_values = []
        self.a_values = []
        
        # Input layer to first hidden layer
        a = x
        self.a_values.append(a)
        
        # Hidden layers with activation
        for i in range(len(self.weights) - 1):
            z = np.dot(a, self.weights[i]) + self.biases[i]
            self.z_values.append(z)
            a = self.activation_fn(z)
            self.a_values.append(a)
        
        # Output layer with softmax activation
        z = np.dot(a, self.weights[-1]) + self.biases[-1]
        self.z_values.append(z)
        
        # Apply softmax for classification
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))  # Subtract max for numerical stability
        a = exp_z / np.sum(exp_z, axis=1, keepdims=True)
        self.a_values.append(a)
        
        return a
    
    def backward(self, d_loss):
        """
        Backward pass to compute gradients
        
        Args:
            d_loss (ndarray): Derivative of loss with respect to output
        """
        batch_size = self.x.shape[0]
        
        # Gradient at the output layer
        delta = d_loss
        
        # Backpropagate through layers
        for i in reversed(range(len(self.weights))):
            # Compute weight and bias gradients
            self.dw[i] = np.dot(self.a_values[i].T, delta) / batch_size
            self.db[i] = np.sum(delta, axis=0, keepdims=True) / batch_size
            
            # Skip backpropagation for input layer
            if i > 0:
                # Compute delta for previous layer
                delta = np.dot(delta, self.weights[i].T) * self.activation_derivative(self.z_values[i-1])
    
    def get_weights(self):
        """
        Get all weights and biases
        
        Returns:
            tuple: (weights, biases)
        """
        return self.weights, self.biases
    
    def set_weights(self, weights, biases):
        """
        Set all weights and biases
        
        Args:
            weights (list): List of weight matrices
            biases (list): List of bias vectors
        """
        self.weights = weights
        self.biases = biases
    
    def get_gradients(self):
        """
        Get weight and bias gradients
        
        Returns:
            tuple: (dw, db)
        """
        return self.dw, self.db

