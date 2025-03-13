from Network.ActivationFunctions.Activation import Activation
import numpy as np


# 4. ReLU Activation
class ReLU(Activation):
    """ReLU activation function: f(x) = max(0, x)"""
    
    def __init__(self):
        def relu(x):
            return np.maximum(0, x)
            
        def relu_derivative(x):
            return np.where(x > 0, 1, 0)
            
        super().__init__(relu, relu_derivative)
