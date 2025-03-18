import numpy as np

class CategoricalCrossEntropy:
    """
    Categorical Cross-Entropy loss for multi-class classification
    
    Works with softmax outputs and one-hot encoded target labels
    """
    
    @staticmethod
    def loss(y_true, y_pred):
        """
        Calculate cross-entropy loss
        
        Parameters:
        -----------
        y_true : numpy.ndarray
            One-hot encoded true labels
        y_pred : numpy.ndarray
            Predicted probabilities (from softmax)
            
        Returns:
        --------
        float
            Cross-entropy loss value
        """
        # Add small epsilon to prevent log(0)
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        
        # If y_true is one-hot encoded
        if len(y_true.shape) == 2:
            # Sum over classes for each sample, then average over samples
            return -np.sum(y_true * np.log(y_pred)) / y_true.shape[0]
        else:
            # If y_true is class indices
            n_samples = y_true.shape[0]
            return -np.sum(np.log(y_pred[np.arange(n_samples), y_true])) / n_samples
    
    @staticmethod
    def gradient(y_true, y_pred):
        """
        Calculate gradient of cross-entropy with respect to predictions
        
        Parameters:
        -----------
        y_true : numpy.ndarray
            One-hot encoded true labels
        y_pred : numpy.ndarray
            Predicted probabilities (from softmax)
            
        Returns:
        --------
        numpy.ndarray
            Gradient of cross-entropy loss
        """
        # Gradient is (y_pred - y_true) / n_samples
        n_samples = y_true.shape[0]
        return (y_pred - y_true) / n_samples