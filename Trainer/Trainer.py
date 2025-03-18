import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import pickle

# Import the AnnotationGenerator and GestureClassificationNetwork
# Note: Update import paths as needed for your project structure
from annotation_generator import AnnotationGenerator
from Network.GestureClassificationNetwork import GestureClassificationNetwork

class GestureTrainer:
    """
    Class to train gesture classification models using data from AnnotationGenerator.
    Ensures only current and past frames are used for prediction.
    """
    
    def __init__(self, window_size=30, step_size=1):
        """
        Initialize the gesture trainer.
        
        Parameters:
        -----------
        window_size : int
            Number of frames to use in each training sample (temporal window)
        step_size : int
            Step size between consecutive windows when generating training data
        """
        self.window_size = window_size
        self.step_size = step_size
        self.data_generator = None
        self.model = None
        self.gesture_mapping = {}  # Maps gesture names to indices
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
    
    def set_data_generator(self, generator):
        """
        Set the data generator to use for training.
        
        Parameters:
        -----------
        generator : AnnotationGenerator
            Initialized and processed AnnotationGenerator instance
        """
        self.data_generator = generator
        return self
    
    def create_or_load_dataset(self, cache_file=None):
        """
        Create dataset from the generator or load from cache if available.
        
        Parameters:
        -----------
        cache_file : str or None
            File path to save/load processed data
            
        Returns:
        --------
        self
        """
        # Check if we should load from cache
        if cache_file and os.path.exists(cache_file):
            print(f"Loading dataset from cache: {cache_file}")
            with open(cache_file, 'rb') as f:
                data = pickle.load(f)
                self.X_train = data['X_train']
                self.y_train = data['y_train']
                self.X_test = data['X_test']
                self.y_test = data['y_test']
                self.gesture_mapping = data['gesture_mapping']
            return self
        
        # Ensure generator is set and processed
        if not self.data_generator:
            raise ValueError("Data generator not set. Use set_data_generator() first.")
        
        if not self.data_generator.processed:
            print("Data not processed. Calling prepare_training_data()...")
            self.data_generator.prepare_training_data()
        
        # Get landmark data and labels
        landmarks = self.data_generator.get_landmark_data()
        labels = self.data_generator.get_landmark_label()
        
        print(f"Creating dataset with window_size={self.window_size}, step_size={self.step_size}")
        
        # Create dataset with temporal windows
        X = []
        y = []
        
        # Create mapping from gesture names to indices
        unique_labels = sorted(set(labels))
        self.gesture_mapping = {label: idx for idx, label in enumerate(unique_labels)}
        
        print(f"Gesture mapping: {self.gesture_mapping}")
        
        # Generate windows
        for i in range(0, len(landmarks) - self.window_size + 1, self.step_size):
            # Get window of landmarks
            window = landmarks[i:i + self.window_size]
            
            # Get label for this window (use the label of the last frame in the window)
            window_label = labels[i + self.window_size - 1]
            
            # Extract features from window
            features = self._extract_features_from_window(window)
            
            if features is not None:
                X.append(features)
                y.append(self.gesture_mapping[window_label])
        
        # Convert to numpy arrays
        X = np.array(X)
        y = np.array(y)
        
        print(f"Created dataset with {len(X)} samples")
        
        # Split into train and test sets
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"Train set: {self.X_train.shape}, Test set: {self.X_test.shape}")
        
        # Cache dataset if requested
        if cache_file:
            print(f"Saving dataset to cache: {cache_file}")
            os.makedirs(os.path.dirname(cache_file) if os.path.dirname(cache_file) else '.', exist_ok=True)
            with open(cache_file, 'wb') as f:
                pickle.dump({
                    'X_train': self.X_train,
                    'y_train': self.y_train,
                    'X_test': self.X_test,
                    'y_test': self.y_test,
                    'gesture_mapping': self.gesture_mapping
                }, f)
        
        return self
    
    def _extract_features_from_window(self, window):
        """
        Extract features from a window of landmark frames.
        
        Parameters:
        -----------
        window : list
            List of landmark frames
            
        Returns:
        --------
        numpy.ndarray or None
            Extracted features or None if no valid landmarks found
        """
        # Check if we have any valid hand landmarks in the window
        has_landmarks = False
        for frame in window:
            if 'hands' in frame and frame['hands']:
                has_landmarks = True
                break
        
        if not has_landmarks:
            return None
        
        # Initialize features array
        # We'll use a simple approach: flatten all landmarks for all frames
        
        # For each frame, we have:
        # - Left hand: 21 landmarks x 3 coordinates (x,y,z)
        # - Right hand: 21 landmarks x 3 coordinates (x,y,z)
        
        # For temporal information, we'll use statistical features over the window
        
        # First, create arrays to hold landmarks
        left_hand_data = np.zeros((len(window), 21, 3))
        right_hand_data = np.zeros((len(window), 21, 3))
        hand_present = np.zeros((len(window), 2))  # [left_present, right_present]
        
        # Fill in landmark data
        for i, frame in enumerate(window):
            if 'hands' in frame and frame['hands']:
                for hand in frame['hands']:
                    hand_type = hand['hand_type']
                    
                    # Extract landmarks
                    landmarks = np.array(hand['landmarks']) if 'landmarks' in hand else []
                    
                    if len(landmarks) == 21:
                        if hand_type == 'Left':
                            left_hand_data[i] = landmarks
                            hand_present[i, 0] = 1
                        else:
                            right_hand_data[i] = landmarks
                            hand_present[i, 1] = 1
        
        # Extract statistical features
        features = []
        
        # Hand presence features
        features.append(np.mean(hand_present, axis=0))
        
        # Position features (mean, std, min, max)
        for hand_data in [left_hand_data, right_hand_data]:
            # Only use frames where hand is present
            if np.any(hand_data):
                # Mean position
                features.append(np.mean(hand_data, axis=0).flatten())
                
                # Standard deviation
                features.append(np.std(hand_data, axis=0).flatten())
                
                # Range (max - min)
                hand_max = np.max(hand_data, axis=0)
                hand_min = np.min(hand_data, axis=0)
                features.append((hand_max - hand_min).flatten())
            else:
                # Pad with zeros if hand is not present
                features.append(np.zeros(21 * 3))
                features.append(np.zeros(21 * 3))
                features.append(np.zeros(21 * 3))
        
        # Flatten all features
        return np.concatenate(features)
    
    def create_model(self, input_size=None):
        """
        Create a new gesture classification model.
        
        Parameters:
        -----------
        input_size : int or None
            Size of input feature vector. If None, calculated from training data.
            
        Returns:
        --------
        self
        """
        # Ensure we have training data
        if self.X_train is None:
            raise ValueError("Training data not created. Use create_or_load_dataset() first.")
        
        # Determine input size if not provided
        if input_size is None:
            input_size = self.X_train.shape[1]
        
        # Create model
        self.model = GestureClassificationNetwork(
            input_size=input_size,
            num_gestures=len(self.gesture_mapping)
        )
        
        return self
    
    def train_model(self, epochs=50, batch_size=32, learning_rate=0.01):
        """
        Train the gesture classification model.
        
        Parameters:
        -----------
        epochs : int
            Number of training epochs
        batch_size : int
            Batch size for mini-batch gradient descent
        learning_rate : float
            Learning rate for gradient descent
            
        Returns:
        --------
        self
        """
        # Ensure we have model and training data
        if self.model is None:
            raise ValueError("Model not created. Use create_model() first.")
        
        if self.X_train is None or self.y_train is None:
            raise ValueError("Training data not created. Use create_or_load_dataset() first.")
        
        print("Training model...")
        
        # Train model
        self.model.train(
            X_train=self.X_train,
            y_train=self.y_train,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate
        )
        
        return self
    
    def evaluate_model(self):
        """
        Evaluate the trained model on test data.
        
        Returns:
        --------
        float
            Accuracy on test set
        """
        # Ensure we have model and test data
        if self.model is None:
            raise ValueError("Model not created or trained. Use create_model() and train_model() first.")
        
        if self.X_test is None or self.y_test is None:
            raise ValueError("Test data not created. Use create_or_load_dataset() first.")
        
        # Make predictions
        y_pred = self.model.predict(self.X_test)
        
        # Calculate accuracy
        accuracy = np.mean(y_pred == self.y_test)
        
        print(f"Test accuracy: {accuracy:.4f}")
        
        # Print confusion matrix
        self._print_confusion_matrix(self.y_test, y_pred)
        
        return accuracy
    
    def _print_confusion_matrix(self, y_true, y_pred):
        """
        Print confusion matrix for model evaluation.
        
        Parameters:
        -----------
        y_true : numpy.ndarray
            True labels
        y_pred : numpy.ndarray
            Predicted labels
        """
        # Get unique labels
        labels = sorted(set(y_true))
        
        # Create confusion matrix
        cm = np.zeros((len(labels), len(labels)), dtype=int)
        for i in range(len(y_true)):
            cm[y_true[i], y_pred[i]] += 1
        
        # Print confusion matrix
        print("Confusion matrix:")
        print("    " + " ".join(f"{i:4d}" for i in range(len(labels))))
        for i, row in enumerate(cm):
            print(f"{i:2d}: " + " ".join(f"{val:4d}" for val in row))
        
        # Print per-class accuracy
        print("Per-class accuracy:")
        for i in range(len(labels)):
            true_positive = cm[i, i]
            total = np.sum(cm[i, :])
            accuracy = true_positive / total if total > 0 else 0
            
            # Get gesture name from mapping
            gesture = [k for k, v in self.gesture_mapping.items() if v == i][0]
            print(f"  {gesture}: {accuracy:.4f} ({true_positive}/{total})")
    
    def save_model(self, filename):
        """
        Save trained model to file.
        
        Parameters:
        -----------
        filename : str
            Path to save model
            
        Returns:
        --------
        self
        """
        # Ensure we have a trained model
        if self.model is None:
            raise ValueError("Model not created or trained. Use create_model() and train_model() first.")
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filename) if os.path.dirname(filename) else '.', exist_ok=True)
        
        # Save model
        self.model.save_model(filename)
        
        # Save gesture mapping
        mapping_file = os.path.splitext(filename)[0] + "_mapping.pkl"
        with open(mapping_file, 'wb') as f:
            pickle.dump(self.gesture_mapping, f)
        
        print(f"Model saved to {filename}")
        print(f"Gesture mapping saved to {mapping_file}")
        
        return self
    
    def load_model(self, filename):
        """
        Load trained model from file.
        
        Parameters:
        -----------
        filename : str
            Path to load model from
            
        Returns:
        --------
        self
        """
        # Load gesture mapping
        mapping_file = os.path.splitext(filename)[0] + "_mapping.pkl"
        if os.path.exists(mapping_file):
            with open(mapping_file, 'rb') as f:
                self.gesture_mapping = pickle.load(f)
                
            print(f"Loaded gesture mapping from {mapping_file}")
        else:
            print(f"Warning: Gesture mapping file not found: {mapping_file}")
        
        # Create model (with dummy input size, will be overwritten on load)
        self.model = GestureClassificationNetwork(
            input_size=1,
            num_gestures=len(self.gesture_mapping) if self.gesture_mapping else 1
        )
        
        # Load weights
        self.model.load_model(filename)
        
        print(f"Model loaded from {filename}")
        
        return self
    
    def predict_gesture(self, landmark_frame):
        """
        Predict gesture for a single landmark frame.
        
        Parameters:
        -----------
        landmark_frame : dict
            Landmark data for a single frame
            
        Returns:
        --------
        str
            Predicted gesture name
        """
        # Ensure we have a trained model
        if self.model is None:
            raise ValueError("Model not created or trained. Use create_model() and train_model() first.")
        
        # Add frame to model's buffer
        features = self._extract_features_from_window([landmark_frame])
        if features is not None:
            self.model.add_frame(features)
            
            # Predict from buffer
            prediction = self.model.predict_from_buffer()
            
            if prediction is not None:
                # Convert prediction index to gesture name
                idx_to_gesture = {idx: gesture for gesture, idx in self.gesture_mapping.items()}
                return idx_to_gesture.get(prediction, "unknown")
        
        return "unknown"


# Example usage
if __name__ == "__main__":
    # Create annotation generator
    generator = AnnotationGenerator()
    
    # Process data
    generator.set_video_path("rotate.mp4") \
             .set_annotation_path("rotate(1).eaf") \
             .prepare_training_data()
    
    # Create and train model
    trainer = GestureTrainer(window_size=30, step_size=5)
    
    trainer.set_data_generator(generator) \
           .create_or_load_dataset(cache_file="gesture_data.pkl") \
           .create_model() \
           .train_model(epochs=100, batch_size=32, learning_rate=0.01) \
           .evaluate_model() \
           .save_model("gesture_model.npy")
    
    # Test prediction on a single frame
    test_frame = generator.get_landmark_data()[100]  # Get a sample frame
    predicted_gesture = trainer.predict_gesture(test_frame)
    print(f"Predicted gesture: {predicted_gesture}")