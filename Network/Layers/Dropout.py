import numpy as np
from Network.Layers.Layer import Layer

class Dropout(Layer):
    """
    Dropout layer for regularization
    Randomly sets a fraction of inputs to zero during training
    Scales other values to maintain expected value
    """
    
    def __init__(self, dropout_rate=0.2):
        """
        Initialize Dropout layer
        
        Parameters:
        -----------
        dropout_rate : float
            Fraction of the input units to drop (0 to 1)
        """
        super().__init__()
        # Ensure dropout rate is not too high
        self.dropout_rate = min(max(0.0, dropout_rate), 0.8)  # Cap at 80% for safety
        self.mask = None
        self.training_mode = True  # Set to False during inference
    
    def forward(self, input_data):
        """Forward pass with dropout"""
        self.input = input_data
        
        # Apply dropout only during training
        if self.training_mode and self.dropout_rate > 0:
            # Create binary mask: 1 for keep, 0 for drop
            self.mask = np.random.binomial(1, 1 - self.dropout_rate, size=input_data.shape)
            
            # Make sure at least one neuron is active in each sample
            # This prevents having all zeros in a sample
            if self.mask.shape[1] > 1:  # Only if we have more than one neuron
                zero_rows = np.where(np.sum(self.mask, axis=1) == 0)[0]
                for row in zero_rows:
                    # Randomly activate one neuron if all are dropped
                    random_col = np.random.randint(0, self.mask.shape[1])
                    self.mask[row, random_col] = 1
                    
            # Scale by 1/(1-rate) to maintain expected sum
            if self.dropout_rate < 1.0:  # Avoid division by zero
                scale = 1.0 / (1.0 - self.dropout_rate)
            else:
                scale = 0.0
                
            output = input_data * self.mask * scale
        else:
            # During testing/inference, no dropout is applied
            output = input_data
            
        return output
    
    def backward(self, output_gradient, learning_rate):
        """Backward pass with dropout"""
        # Apply same mask to gradients
        if self.training_mode and self.dropout_rate > 0:
            if self.dropout_rate < 1.0:  # Avoid division by zero
                scale = 1.0 / (1.0 - self.dropout_rate)
            else:
                scale = 0.0
            input_gradient = output_gradient * self.mask * scale
        else:
            input_gradient = output_gradient
            
        return input_gradient
    
    def set_training_mode(self, is_training):
        """Set layer to training or inference mode"""
        self.training_mode = is_training