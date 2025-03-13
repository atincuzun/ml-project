# Network/ActivationFunctions/Softmax.py
import numpy as np
from Network.Layers.Layer import Layer

class Softmax(Layer):
    """
    Softmax activation function for multi-class classification
    
    Converts raw network outputs into probabilities that sum to 1
    """
    
    def __init__(self):
        super().__init__()
    
    def forward(self, input_data):
        """Forward pass"""
        self.input = input_data
        # Shift input for numerical stability (prevent overflow)
        shifted_input = input_data - np.max(input_data, axis=1, keepdims=True)
        exp_values = np.exp(shifted_input)
        # Normalize to get probabilities
        self.output = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        return self.output
    
    def backward(self, output_gradient, learning_rate):
        """
        Backward pass for softmax layer
        
        Note: This implementation assumes softmax is followed by 
        categorical cross-entropy loss, which simplifies the gradient.
        """
        # For softmax with cross-entropy, gradient simplifies
        # The output_gradient is typically y_pred - y_true directly
        return output_gradient