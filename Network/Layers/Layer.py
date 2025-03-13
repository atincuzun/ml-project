import numpy as np


# 1. Base Layer Class
class Layer:
    """Base class for all neural network layers"""
    
    def __init__(self):
        self.input = None
        self.output = None
    
    def forward(self, input_data):
        """
        Forward pass - computes output given input
        
        Parameters:
        -----------
        input_data : numpy.ndarray
            Input data to the layer
            
        Returns:
        --------
        numpy.ndarray
            Output of the layer
        """
        raise NotImplementedError
    
    def backward(self, output_gradient, learning_rate):
        """
        Backward pass - computes gradients given output gradient
        
        Parameters:
        -----------
        output_gradient : numpy.ndarray
            Gradient of the loss with respect to the layer's output
        learning_rate : float
            Learning rate for parameter updates
            
        Returns:
        --------
        numpy.ndarray
            Gradient of the loss with respect to the layer's input
        """
        raise NotImplementedError
