class RegularizationBase:
    """Base class for regularization techniques"""
    
    def __init__(self, lambda_param=0.01):
        """
        Initialize regularization with regularization strength parameter.
        
        Parameters:
        -----------
        lambda_param : float
            Regularization strength (default: 0.01)
        """
        self.lambda_param = lambda_param
    
    def loss(self, weights):
        """
        Calculate regularization loss contribution.
        
        Parameters:
        -----------
        weights : numpy.ndarray
            Weights of the neural network layer
            
        Returns:
        --------
        float
            Regularization loss value
        """
        raise NotImplementedError
    
    def gradient(self, weights):
        """
        Calculate gradient of regularization term with respect to weights.
        
        Parameters:
        -----------
        weights : numpy.ndarray
            Weights of the neural network layer
            
        Returns:
        --------
        numpy.ndarray
            Gradient of regularization term
        """
        raise NotImplementedError
