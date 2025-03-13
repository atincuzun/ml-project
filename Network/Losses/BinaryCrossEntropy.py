# Network/Losses/BinaryCrossEntropy.py
import numpy as np

class BinaryCrossEntropy:
    """
    Binary Cross-Entropy loss function for binary classification
    """
    
    @staticmethod
    def loss(y_true, y_pred):
        """
        Calculate binary cross-entropy loss
        
        Parameters:
        -----------
        y_true : numpy.ndarray
            True labels (0 or 1)
        y_pred : numpy.ndarray
            Predicted probabilities (from sigmoid)
            
        Returns:
        --------
        float
            Binary cross-entropy loss value
        """
        # Add small epsilon to prevent log(0)
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        
        # Binary cross entropy formula: -y*log(y_pred) - (1-y)*log(1-y_pred)
        batch_size = y_true.shape[0]
        return -np.sum(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred)) / batch_size
    
    @staticmethod
    def gradient(y_true, y_pred):
        """
        Calculate gradient of binary cross-entropy with respect to predictions
        
        Parameters:
        -----------
        y_true : numpy.ndarray
            True labels (0 or 1)
        y_pred : numpy.ndarray
            Predicted probabilities (from sigmoid)
            
        Returns:
        --------
        numpy.ndarray
            Gradient of binary cross-entropy loss
        """
        # Add small epsilon to prevent division by zero
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        
        # Gradient: (y_pred - y_true) / (y_pred * (1 - y_pred))
        # Simplified when used with sigmoid output
        batch_size = y_true.shape[0]
        return (y_pred - y_true) / batch_size