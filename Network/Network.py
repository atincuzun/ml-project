import numpy as np


# 7. Simple Neural Network
class NeuralNetwork:
    """Simple neural network model"""
    
    def __init__(self):
        self.layers = []
        self.loss = None
    
    def add(self, layer):
        """Add a layer to the network"""
        self.layers.append(layer)
    
    def set_loss(self, loss):
        """Set the loss function"""
        self.loss = loss
    
    def predict(self, input_data):
        """Make predictions using the network"""
        output = input_data
        for layer in self.layers:
            output = layer.forward(output)
        return output
    
    def fit(self, X_train, y_train, epochs, batch_size, learning_rate):
        """
        Train the neural network
        
        Parameters:
        -----------
        X_train : numpy.ndarray
            Training data
        y_train : numpy.ndarray
            Target values
        epochs : int
            Number of training epochs
        batch_size : int
            Batch size for mini-batch gradient descent
        learning_rate : float
            Learning rate for gradient descent
        """
        n_samples = X_train.shape[0]
        
        for epoch in range(epochs):
            # Shuffle the training data
            permutation = np.random.permutation(n_samples)
            X_shuffled = X_train[permutation]
            y_shuffled = y_train[permutation]
            
            # Mini-batch gradient descent
            for i in range(0, n_samples, batch_size):
                # Get batch
                X_batch = X_shuffled[i:i+batch_size]
                y_batch = y_shuffled[i:i+batch_size]
                
                # Forward pass
                y_pred = self.predict(X_batch)
                
                # Compute loss for monitoring
                loss = self.loss.loss(y_batch, y_pred)
                
                # Backward pass
                grad = self.loss.gradient(y_batch, y_pred)
                for layer in reversed(self.layers):
                    grad = layer.backward(grad, learning_rate)
            
            # Calculate loss for the epoch (using all data)
            y_pred = self.predict(X_train)
            loss = self.loss.loss(y_train, y_pred)
            print(f"Epoch {epoch+1}/{epochs}, Loss: {loss:.4f}")
