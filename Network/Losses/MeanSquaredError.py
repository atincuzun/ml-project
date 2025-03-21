# Network/Losses/MeanSquaredError.py
import numpy as np

class MeanSquaredError:
    """
    Mean Squared Error loss function
    
    Calculates the average of squared differences between predictions and targets
    """
    
    @staticmethod
    def loss(y_true, y_pred):
        """
        Calculate MSE loss
        
        Parameters:
        -----------
        y_true : numpy.ndarray
            True target values
        y_pred : numpy.ndarray
            Predicted values
            
        Returns:
        --------
        float
            Mean squared error value
        """
        return np.mean(np.square(y_true - y_pred))
    
    @staticmethod
    def gradient(y_true, y_pred):
        """
        Calculate gradient of MSE with respect to predictions
        
        Parameters:
        -----------
        y_true : numpy.ndarray
            True target values
        y_pred : numpy.ndarray
            Predicted values
            
        Returns:
        --------
        numpy.ndarray
            Gradient of MSE loss
        """
        n_samples = y_true.shape[0]
        return 2 * (y_pred - y_true) / n_samples