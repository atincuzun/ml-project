class L1Regularization(RegularizationBase):
    """
    L1 Regularization (Lasso)
    
    Adds the sum of absolute values of weights to the loss function:
    reg_loss = lambda * sum(|weights|)
    
    Encourages sparse weight matrices by driving some weights to exactly zero.
    """
    
    def loss(self, weights):
        """
        Calculate L1 regularization loss.
        
        Parameters:
        -----------
        weights : numpy.ndarray
            Weights of the neural network layer
            
        Returns:
        --------
        float
            L1 regularization loss value
        """
        return self.lambda_param * np.sum(np.abs(weights))
    
    def gradient(self, weights):
        """
        Calculate gradient of L1 regularization term.
        
        Parameters:
        -----------
        weights : numpy.ndarray
            Weights of the neural network layer
            
        Returns:
        --------
        numpy.ndarray
            Gradient of L1 regularization term
        """
        # Gradient of |w| is sign(w)
        return self.lambda_param * np.sign(weights)
