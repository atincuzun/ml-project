import numpy as np
from Network.Network import NeuralNetwork
from Network.Layers.Dense import Dense
from Network.ActivationFunctions.ReLU import ReLU
from Network.ActivationFunctions.Softmax import Softmax
from Network.Losses.CategoricalCrossEntropy import CategoricalCrossEntropy

class GestureClassificationNetwork:
    """
    Neural Network specialized for gesture classification
    
    Takes gesture features and classifies them into predefined gesture categories
    """
    
    def __init__(self, input_size, num_gestures):
        """
        Initialize the gesture classification network
        
        Parameters:
        -----------
        input_size : int
            Size of input feature vector
        num_gestures : int
            Number of gesture classes to recognize
        """
        self.network = NeuralNetwork()
        
        # Architecture: Input -> Dense(128) -> ReLU -> Dense(64) -> ReLU -> Dense(num_gestures) -> Softmax
        self.network.add(Dense(input_size, 512))
        self.network.add(ReLU())
        self.network.add(Dense(512, 256))
        self.network.add(ReLU())
        self.network.add(Dense(256, 128))
        self.network.add(ReLU())
        self.network.add(Dense(128, 64))
        self.network.add(ReLU())

        self.network.add(Dense(64, num_gestures))
        self.network.add(Softmax())
        
        # Set loss function
        self.network.set_loss(CategoricalCrossEntropy())
        
        # Gesture classes (to be set during training)
        self.gesture_classes = None
        
        # Frame buffer for sequence handling
        self.frame_buffer = []
        self.buffer_size = 30  # Approximately 1 second at 30fps
        
    def train(self, X_train, y_train, epochs=50, batch_size=32, learning_rate=0.01):
        """
        Train the network on gesture data
        
        Parameters:
        -----------
        X_train : numpy.ndarray
            Training features
        y_train : numpy.ndarray
            Training labels (either one-hot encoded or class indices)
        epochs : int
            Number of training epochs
        batch_size : int
            Batch size for mini-batch gradient descent
        learning_rate : float
            Learning rate for gradient descent
        """
        # If labels are not one-hot encoded, convert them
        if len(y_train.shape) == 1:
            # Get number of classes
            num_classes = len(np.unique(y_train))
            # Convert to one-hot
            y_one_hot = np.zeros((y_train.shape[0], num_classes))
            y_one_hot[np.arange(y_train.shape[0]), y_train] = 1
            y_train = y_one_hot
        
        # Store unique gesture classes
        self.gesture_classes = np.unique(np.argmax(y_train, axis=1))
        
        # Train network
        self.network.fit(X_train, y_train, epochs, batch_size, learning_rate)
    
    def predict(self, X):
        """
        Predict gesture class for input features
        
        Parameters:
        -----------
        X : numpy.ndarray
            Input features
            
        Returns:
        --------
        int
            Predicted gesture class index
        """
        # Get raw network predictions (probabilities)
        probabilities = self.network.predict(X)
        
        # Return class with highest probability
        return np.argmax(probabilities, axis=1)
    
    def add_frame(self, frame_features):
        """
        Add frame features to buffer for sequence handling
        
        Parameters:
        -----------
        frame_features : numpy.ndarray
            Features extracted from current frame
        """
        self.frame_buffer.append(frame_features)
        
        # Keep buffer at fixed size
        if len(self.frame_buffer) > self.buffer_size:
            self.frame_buffer.pop(0)
    
    def predict_from_buffer(self):
        """
        Make prediction using current frame buffer
        
        Returns:
        --------
        int or None
            Predicted gesture class index or None if buffer not full
        """
        if len(self.frame_buffer) < self.buffer_size:
            return None
        
        # Combine features from buffer (simple approach: average them)
        # More sophisticated approaches could be used here
        combined_features = np.mean(self.frame_buffer, axis=0).reshape(1, -1)
        
        # Get prediction
        return self.predict(combined_features)[0]
    
    def save_model(self, filename):
        """
        Save model weights to file
        
        Parameters:
        -----------
        filename : str
            Path to save model
        """
        # Simple numpy save for each layer's weights
        model_data = {
            'layers': [],
            'gesture_classes': self.gesture_classes
        }
        
        for layer in self.network.layers:
            if hasattr(layer, 'weights') and hasattr(layer, 'bias'):
                model_data['layers'].append({
                    'weights': layer.weights,
                    'bias': layer.bias
                })
            else:
                model_data['layers'].append(None)
        
        np.save(filename, model_data, allow_pickle=True)
    
    def load_model(self, filename):
        """
        Load model weights from file
        
        Parameters:
        -----------
        filename : str
            Path to load model from
        """
        # Load saved model
        model_data = np.load(filename, allow_pickle=True).item()
        
        # Load gesture classes
        self.gesture_classes = model_data['gesture_classes']
        
        # Load weights for each layer
        layer_idx = 0
        for i, layer in enumerate(self.network.layers):
            if hasattr(layer, 'weights') and hasattr(layer, 'bias'):
                if model_data['layers'][i] is not None:
                    layer.weights = model_data['layers'][i]['weights']
                    layer.bias = model_data['layers'][i]['bias']
                    layer_idx += 1