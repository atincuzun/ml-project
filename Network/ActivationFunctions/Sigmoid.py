import numpy as np
from Network.ActivationFunctions.Activation import Activation


class Sigmoid(Activation):
    """Sigmoid activation function: f(x) = 1 / (1 + exp(-x))"""
    
    def __init__(self):
        def sigmoid(x):
            return 1 / (1 + np.exp(-np.clip(x, -15, 15)))  # Clip to avoid overflow
            
        def sigmoid_derivative(x):
            s = sigmoid(x)
            return s * (1 - s)
            
        super().__init__(sigmoid, sigmoid_derivative)