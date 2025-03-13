class L2Regularization(RegularizationBase):
    """
    L2 Regularization (Ridge / Weight Decay)
    
    Adds the sum of squared weights to the loss function:
    reg_loss = lambda * sum(weights^2)
    
    Discourages large weight values and encourages weight values to be 
    small and more evenly distributed.
    """
    
    def loss(self, weights):
        """
        Calculate L2 regularization loss.
        
        Parameters:
        -----------
        weights : numpy.ndarray
            Weights of the neural network layer
            
        Returns:
        --------
        float
            L2 regularization loss value
        """
        return 0.5 * self.lambda_param * np.sum(np.square(weights))
    
    def gradient(self, weights):
        """
        Calculate gradient of L2 regularization term.
        
        Parameters:
        -----------
        weights : numpy.ndarray
            Weights of the neural network layer
            
        Returns:
        --------
        numpy.ndarray
            Gradient of L2 regularization term
        """
        # Gradient of 0.5 * w^2 is w
        return self.lambda_param * weights
