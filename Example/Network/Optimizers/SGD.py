import numpy as np

class SGD:
    """
    Stochastic Gradient Descent optimizer
    
    Basic implementation of gradient descent algorithm that updates
    parameters based on the gradient and learning rate.
    """
    
    def __init__(self, learning_rate=0.01):
        """
        Initialize SGD optimizer
        
        Parameters:
        -----------
        learning_rate : float
            Learning rate (step size) for parameter updates
        """
        self.learning_rate = learning_rate
    
    def update(self, params, grads):
        """
        Update parameters using gradients
        
        Parameters:
        -----------
        params : list of numpy.ndarray
            List of parameter arrays to update (weights and biases)
        grads : list of numpy.ndarray
            List of gradient arrays corresponding to params
            
        Returns:
        --------
        list of numpy.ndarray
            Updated parameters
        """
        updated_params = []
        
        for param, grad in zip(params, grads):
            # Simple gradient descent update rule:
            # param = param - learning_rate * gradient
            updated_param = param - self.learning_rate * grad
            updated_params.append(updated_param)
            
        return updated_params