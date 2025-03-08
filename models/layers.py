# models/layers.py
import numpy as np
from utils.weight_init import initialize_weights

class Layer:
    def __init__(self):
        self.params = {}
        self.grads = {}
    
    def forward(self, inputs):
        """
        Forward pass for the layer
        
        Args:
            inputs (ndarray): Input to the layer
        
        Returns:
            ndarray: Output of the layer
        """
        raise NotImplementedError
    
    def backward(self, grad_output):
        """
        Backward pass for the layer
        
        Args:
            grad_output (ndarray): Gradient of the loss with respect to the output
        
        Returns:
            ndarray: Gradient of the loss with respect to the input
        """
        raise NotImplementedError

class DenseLayer(Layer):
    def __init__(self, input_dim, output_dim, activation_fn, activation_derivative, weight_init='random'):
        """
        Fully connected layer
        
        Args:
            input_dim (int): Input dimension
            output_dim (int): Output dimension
            activation_fn (function): Activation function
            activation_derivative (function): Derivative of activation function
            weight_init (str): Weight initialization method
        """
        super().__init__()
        self.params['W'] = initialize_weights(input_dim, output_dim, weight_init)
        self.params['b'] = np.zeros((1, output_dim))
        
        self.activation_fn = activation_fn
        self.activation_derivative = activation_derivative
        
        # Cache for backpropagation
        self.inputs = None
        self.z = None
        self.a = None
    
    def forward(self, inputs):
        """
        Forward pass
        
        Args:
            inputs (ndarray): Input data of shape (batch_size, input_dim)
        
        Returns:
            ndarray: Activated outputs
        """
        self.inputs = inputs
        self.z = np.dot(inputs, self.params['W']) + self.params['b']
        self.a = self.activation_fn(self.z)
        return self.a
    
    def backward(self, grad_output):
        """
        Backward pass
        
        Args:
            grad_output (ndarray): Gradient of loss with respect to layer output
        
        Returns:
            ndarray: Gradient of loss with respect to layer input
        """
        batch_size = self.inputs.shape[0]
        
        # If this is not the output layer, multiply by activation derivative
        if self.activation_fn is not None:
            delta = grad_output * self.activation_derivative(self.z)
        else:
            delta = grad_output
        
        # Compute gradients
        self.grads['W'] = np.dot(self.inputs.T, delta) / batch_size
        self.grads['b'] = np.sum(delta, axis=0, keepdims=True) / batch_size
        
        # Propagate gradient to previous layer
        return np.dot(delta, self.params['W'].T)

