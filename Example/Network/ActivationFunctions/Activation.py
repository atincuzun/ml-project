from Network.Layers.Layer import Layer  # Add this import
import numpy as np


# 3. Activation Layer Base Class
class Activation(Layer):
    """Base class for activation layers"""
    
    def __init__(self, activation_function, activation_derivative):
        """
        Initialize activation layer
        
        Parameters:
        -----------
        activation_function : function
            The activation function
        activation_derivative : function
            The derivative of the activation function
        """
        super().__init__()
        self.activation = activation_function
        self.activation_derivative = activation_derivative
    
    def forward(self, input_data):
        """Forward pass"""
        self.input = input_data
        self.output = self.activation(input_data)
        return self.output
    
    def backward(self, output_gradient, learning_rate):
        """Backward pass"""
        # Chain rule: output_gradient * derivative_of_activation(input)
        return np.multiply(output_gradient, self.activation_derivative(self.input))
