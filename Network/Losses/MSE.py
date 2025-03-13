import numpy as np


# 6. Loss Function - Mean Squared Error
class MSE:
    """Mean Squared Error loss function"""
    
    @staticmethod
    def loss(y_true, y_pred):
        """Calculate MSE loss"""
        return np.mean(np.power(y_true - y_pred, 2))
    
    @staticmethod
    def gradient(y_true, y_pred):
        """Calculate gradient of MSE with respect to predictions"""
        return 2 * (y_pred - y_true) / y_pred.shape[0]
