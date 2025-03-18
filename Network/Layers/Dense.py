import numpy as np
from Network.Layers.Layer import Layer  # Fix this import

# 2. Dense (Fully Connected) Layer
class Dense(Layer):
    """
    Fully connected layer implementing: output = input @ weights + bias
    """
    
    def __init__(self, input_size, output_size):
        """
        Initialize Dense layer with random weights and zero biases
        
        Parameters:
        -----------
        input_size : int
            Size of the input features
        output_size : int
            Size of the output features
        """
        super().__init__()
        # Xavier/Glorot initialization for weights
        self.weights = np.random.randn(input_size, output_size) * np.sqrt(2.0 / (input_size + output_size))
        self.bias = np.zeros((1, output_size))
    
    def forward(self, input_data):
        """Forward pass"""
        self.input = input_data
        self.output = np.dot(input_data, self.weights) + self.bias
        return self.output
    
    def backward(self, output_gradient, learning_rate):
        """Backward pass"""
        # Gradient with respect to weights: input^T @ output_gradient
        weights_gradient = np.dot(self.input.T, output_gradient)
        
        # Gradient with respect to bias: sum of output_gradient across batch
        bias_gradient = np.sum(output_gradient, axis=0, keepdims=True)
        
        # Update parameters
        self.weights -= learning_rate * weights_gradient
        self.bias -= learning_rate * bias_gradient
        
        # Gradient with respect to input: output_gradient @ weights^T
        input_gradient = np.dot(output_gradient, self.weights.T)
        return input_gradient