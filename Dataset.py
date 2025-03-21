import numpy as np
import matplotlib.pyplot as plt
import os

class GestureDataset:
    """
    Creates sliding windows of temporal data from pose landmarks with labels.
    Uses only the specified arm and hand landmarks with no derived features.
    """
    
    def __init__(self, window_size=30, step_size=1, noise_level=0.001):
        """
        Initialize the dataset class.
        
        Parameters:
        -----------
        window_size : int
            Number of frames in each sliding window
        step_size : int
            Number of frames to advance when creating the next window
        noise_level : float
            Amount of random noise to add to training data (default: 0.001)
        """
        self.window_size = window_size
        self.step_size = step_size
        self.gesture_mapping = {}
        self.noise_level = noise_level
        
        # Define which landmarks to include (indices 11-22: shoulders, arms, hands)
        self.included_landmarks = list(range(11, 23))  # 11-22 inclusive
        
    def prepare_windows(self, landmarks, labels):
        """
        Create sliding windows from landmark data and labels.
        
        Parameters:
        -----------
        landmarks : list
            List of landmark data for each frame (containing pose landmarks)
        labels : list
            List of gesture labels for each frame
            
        Returns:
        --------
        tuple
            (X, y) where X is the feature array and y is the label array
        """
        print(f"Creating sliding windows with size={self.window_size}, step={self.step_size}")
        
        # Create gesture mapping (string labels to indices)
        unique_labels = sorted(set(labels))
        self.gesture_mapping = {label: idx for idx, label in enumerate(unique_labels)}
        
        print(f"Found {len(unique_labels)} unique gestures: {unique_labels}")
        
        # Create sliding windows
        X = []  # Features
        y = []  # Labels
        
        # Iterate through frames with sliding window
        for i in range(0, len(landmarks) - self.window_size + 1, self.step_size):
            # Get window of landmarks
            window = landmarks[i:i + self.window_size]
            
            # Get label for this window (use majority vote)
            window_labels = labels[i:i + self.window_size]
            unique, counts = np.unique(window_labels, return_counts=True)
            window_label = unique[np.argmax(counts)]  # Most common label
            label_idx = self.gesture_mapping[window_label]
            
            # Extract features from the window
            features = self._extract_features(window)
            
            if features is not None:
                X.append(features)
                y.append(label_idx)
        
        # Convert to numpy arrays
        X = np.array(X)
        y = np.array(y)
        
        # Add small random noise to training data for robustness
        X = X + np.random.normal(0, self.noise_level, X.shape)
        
        print(f"Created {len(X)} windows with shape {X.shape}")
        
        return X, y
    
    def _extract_features(self, window):
        """
        Extract only the raw landmark data for specified landmarks.
        No additional features are calculated.
        
        Parameters:
        -----------
        window : list
            List of landmark frames in the window
            
        Returns:
        --------
        numpy.ndarray or None
            Extracted features or None if no valid landmarks
        """
        # Check if we have any valid landmarks
        has_landmarks = False
        for frame in window:
            if 'landmarks' in frame and frame['landmarks']:
                has_landmarks = True
                break
        
        if not has_landmarks:
            return None
        
        # Get the number of included landmarks
        num_included = len(self.included_landmarks)
        
        # Prepare array for landmarks - only for included landmarks
        # Each landmark has x, y, z, visibility (4 values)
        pose_data = np.zeros((len(window), num_included, 4))
        
        # Presence flag for each frame
        pose_present = np.zeros(len(window))
        
        # Extract landmark data for each frame
        for i, frame in enumerate(window):
            if 'landmarks' in frame and frame['landmarks']:
                pose_present[i] = 1
                landmarks = frame['landmarks']
                
                # Only extract the included landmarks
                for idx, j in enumerate(self.included_landmarks):
                    if j < len(landmarks):  # Ensure landmark exists
                        landmark = landmarks[j]
                        pose_data[i, idx, 0] = landmark['x']
                        pose_data[i, idx, 1] = landmark['y']
                        pose_data[i, idx, 2] = landmark['z']
                        pose_data[i, idx, 3] = landmark['visibility']
        
        # Combine the features - only raw landmarks and presence flags
        features = np.concatenate([
            pose_data.flatten(),         # Raw landmark positions
            pose_present.flatten()       # Presence flags
        ])
        
        return features
    
    def get_label_mapping(self):
        """
        Get the mapping from gesture labels to indices.
        
        Returns:
        --------
        dict
            Mapping from gesture names to indices
        """
        return self.gesture_mapping
    
    def get_index_mapping(self):
        """
        Get the mapping from indices to gesture labels.
        
        Returns:
        --------
        dict
            Mapping from indices to gesture names
        """
        return {idx: label for label, idx in self.gesture_mapping.items()}


def train_test_split(X, y, test_size=0.2, random_seed=None):
    """
    Split arrays into random train and test subsets.
    
    Parameters:
    -----------
    X : numpy.ndarray
        Features array
    y : numpy.ndarray
        Target array
    test_size : float
        Proportion of the dataset to include in the test split (0-1)
    random_seed : int or None
        Seed for random number generator
    
    Returns:
    --------
    tuple
        (X_train, X_test, y_train, y_test)
    """
    # Set random seed for reproducibility
    if random_seed is not None:
        np.random.seed(random_seed)
    
    # Get number of samples
    n_samples = len(X)
    
    # Generate random indices
    indices = np.arange(n_samples)
    np.random.shuffle(indices)
    
    # Calculate split point
    test_samples = int(n_samples * test_size)
    
    # Split indices
    test_indices = indices[:test_samples]
    train_indices = indices[test_samples:]
    
    # Split data
    X_train = X[train_indices]
    X_test = X[test_indices]
    y_train = y[train_indices]
    y_test = y[test_indices]
    
    return X_train, X_test, y_train, y_test


def stratified_train_test_split(X, y, test_size=0.2, random_seed=None):
    """
    Split arrays into random train and test subsets, preserving class proportions.
    
    Parameters:
    -----------
    X : numpy.ndarray
        Features array
    y : numpy.ndarray
        Target array
    test_size : float
        Proportion of the dataset to include in the test split (0-1)
    random_seed : int or None
        Seed for random number generator
    
    Returns:
    --------
    tuple
        (X_train, X_test, y_train, y_test)
    """
    # Set random seed for reproducibility
    if random_seed is not None:
        np.random.seed(random_seed)
    
    # Find unique classes and their indices
    classes = np.unique(y)
    
    # Initialize empty arrays for train and test sets
    X_train = []
    X_test = []
    y_train = []
    y_test = []
    
    # Split each class separately to maintain class proportions
    for cls in classes:
        # Get indices for this class
        cls_indices = np.where(y == cls)[0]
        n_samples = len(cls_indices)
        
        # Shuffle indices
        np.random.shuffle(cls_indices)
        
        # Calculate split point
        test_samples = int(n_samples * test_size)
        
        # Split indices
        test_indices = cls_indices[:test_samples]
        train_indices = cls_indices[test_samples:]
        
        # Add to train and test sets
        X_train.append(X[train_indices])
        X_test.append(X[test_indices])
        y_train.append(y[train_indices])
        y_test.append(y[test_indices])
    
    # Combine arrays
    X_train = np.vstack(X_train)
    X_test = np.vstack(X_test)
    y_train = np.concatenate(y_train)
    y_test = np.concatenate(y_test)
    
    return X_train, X_test, y_train, y_test


# Data augmentation functions
def augment_spatial(landmarks, labels, augmentation_factor=1):
    """
    Apply spatial transformations to landmark data for augmentation.
    
    Parameters:
    -----------
    landmarks : list
        Original landmark data
    labels : list
        Original labels
    augmentation_factor : int
        Number of augmented versions to create
        
    Returns:
    --------
    tuple
        (augmented_landmarks, augmented_labels)
    """
    augmented_landmarks = []
    augmented_labels = []
    
    # Keep original data
    augmented_landmarks.extend(landmarks)
    augmented_labels.extend(labels)
    
    for _ in range(augmentation_factor):
        # Create transformed versions
        transformed_landmarks = []
        for frame in landmarks:
            # Create a copy of the frame
            new_frame = {key: value for key, value in frame.items()}
            
            if 'landmarks' not in frame or not frame['landmarks']:
                transformed_landmarks.append(new_frame)
                continue
            
            # Copy landmarks
            new_landmarks = []
            for landmark in frame['landmarks']:
                new_landmark = {key: value for key, value in landmark.items()}
                
                # Apply random transformations
                # 1. Scale (0.9-1.1x)
                scale = np.random.uniform(0.9, 1.1)
                # Find center point (average of all landmarks)
                center_x = np.mean([l['x'] for l in frame['landmarks']])
                center_y = np.mean([l['y'] for l in frame['landmarks']])
                # Scale relative to center
                new_landmark['x'] = center_x + scale * (new_landmark['x'] - center_x)
                new_landmark['y'] = center_y + scale * (new_landmark['y'] - center_y)
                
                # 2. Translation (±0.05)
                shift_x = np.random.uniform(-0.05, 0.05)
                shift_y = np.random.uniform(-0.05, 0.05)
                new_landmark['x'] += shift_x
                new_landmark['y'] += shift_y
                
                # 3. Small rotation (±5 degrees)
                theta = np.random.uniform(-5, 5) * np.pi / 180  # convert to radians
                x_rot = new_landmark['x'] - center_x
                y_rot = new_landmark['y'] - center_y
                new_landmark['x'] = center_x + x_rot * np.cos(theta) - y_rot * np.sin(theta)
                new_landmark['y'] = center_y + x_rot * np.sin(theta) + y_rot * np.cos(theta)
                
                new_landmarks.append(new_landmark)
            
            # Update frame with new landmarks
            new_frame['landmarks'] = new_landmarks
            transformed_landmarks.append(new_frame)
        
        # Add transformed data
        augmented_landmarks.extend(transformed_landmarks)
        augmented_labels.extend(labels)
    
    return augmented_landmarks, augmented_labels


def augment_mirroring(landmarks, labels, mirror_horizontal=True):
    """
    Create mirrored versions of gestures.
    
    Parameters:
    -----------
    landmarks : list
        Original landmark data
    labels : list
        Original labels
    mirror_horizontal : bool
        Whether to mirror horizontally (left-right)
        
    Returns:
    --------
    tuple
        (augmented_landmarks, augmented_labels)
    """
    mirrored_landmarks = []
    mirrored_labels = []
    
    # MediaPipe pose landmark indices mapping for left-right swap
    # This maps left-side landmarks to right-side landmarks and vice versa
    pose_lr_swap = {
        # Arms
        11: 12, 12: 11,  # shoulders
        13: 14, 14: 13,  # elbows
        15: 16, 16: 15,  # wrists
        17: 18, 18: 17,  # pinkies
        19: 20, 20: 19,  # indices
        21: 22, 22: 21,  # thumbs
    }
    
    # Create label mapping for mirrored gestures
    label_map = {
        "swipe_left": "swipe_right",
        "swipe_right": "swipe_left",
        "rotate_cw": "rotate_ccw",
        "rotate_ccw": "rotate_cw",
        "rotate": "rotate",  # Same label if no direction specified
        "idle": "idle"       # Idle stays the same
        # Add more mappings for other gestures
    }
    
    for i, frame in enumerate(landmarks):
        mirrored_frame = {key: value for key, value in frame.items()}
        
        if 'landmarks' in frame and frame['landmarks']:
            mirrored_landmarks_data = [None] * len(frame['landmarks'])
            
            for j, landmark in enumerate(frame['landmarks']):
                # If this is a landmark that should be swapped
                if j in pose_lr_swap:
                    target_idx = pose_lr_swap[j]
                    mirrored_landmarks_data[target_idx] = {
                        'x': 1.0 - landmark['x'],  # Mirror x-coordinate (assuming 0-1 range)
                        'y': landmark['y'],        # Keep y-coordinate the same
                        'z': -landmark['z'],       # Invert z-coordinate
                        'visibility': landmark['visibility']
                    }
                else:
                    # For landmarks that don't have left-right pairs
                    mirrored_landmarks_data[j] = {
                        'x': 1.0 - landmark['x'],  # Mirror x-coordinate
                        'y': landmark['y'],
                        'z': -landmark['z'],
                        'visibility': landmark['visibility']
                    }
            
            # Remove None values (in case landmarks array has gaps)
            mirrored_landmarks_data = [lm for lm in mirrored_landmarks_data if lm is not None]
            mirrored_frame['landmarks'] = mirrored_landmarks_data
        
        mirrored_landmarks.append(mirrored_frame)
        
        # Map the label to its mirrored version
        original_label = labels[i]
        mirrored_label = label_map.get(original_label, original_label)
        mirrored_labels.append(mirrored_label)
    
    # Return both original and mirrored data
    augmented_landmarks = landmarks + mirrored_landmarks
    augmented_labels = labels + mirrored_labels
    
    return augmented_landmarks, augmented_labels


def augment_dataset(landmarks, labels):
    """
    Apply multiple augmentation techniques to create a more robust dataset.
    
    Parameters:
    -----------
    landmarks : list
        Original landmark data
    labels : list
        Original labels
        
    Returns:
    --------
    tuple
        (augmented_landmarks, augmented_labels)
    """
    print("Applying spatial transformations...")
    landmarks_aug, labels_aug = augment_spatial(landmarks, labels, augmentation_factor=1)
    
    
    return landmarks_aug, labels_aug


# Modified version of the main function in Dataset.py

if __name__ == "__main__":
    import cv2
    import mediapipe as mp
    from annotation_generator import AnnotationGenerator
    from Network.GestureClassificationNetwork import GestureClassificationNetwork
    from Trainer.Trainer import GestureTrainer
    import matplotlib.pyplot as plt
    import seaborn as sns
    from collections import deque
    import time
    import os
    
    # Path settings
    TRAINING_DATA_DIR = "./TrainingData"
    VALIDATION_DATA_DIR = "./ValidationData"
    MODEL_SAVE_PATH = "multi_gesture_model.npy"
    
    # Set to True to skip training and just use existing model for webcam prediction
    SKIP_TRAINING = False
    
    # Confidence threshold for prediction
    CONFIDENCE_THRESHOLD = 0.6
    
    if not SKIP_TRAINING:
        # 1. Process training data
        print("Step 1: Processing videos from Training Data directory...")
        generator = AnnotationGenerator()
        generator.process_directory(TRAINING_DATA_DIR)
        training_landmarks = generator.get_landmark_data()
        training_labels = generator.get_landmark_label()
        
        print(f"Combined dataset: {len(training_landmarks)} frames")
        
        # Count label occurrences
        label_counts = {}
        for label in training_labels:
            label_counts[label] = label_counts.get(label, 0) + 1
        
        print("Label distribution:")
        for label, count in label_counts.items():
            print(f"  {label}: {count} frames ({count/len(training_labels)*100:.1f}%)")
        
        # 2. Create dataset
        print("\nStep 2: Creating dataset...")
        window_size = 10
        step_size = 1
        dataset = GestureDataset(window_size=window_size, step_size=step_size)
        
        # 3. Process data to get input size and classes
        print("\nStep 3: Processing data and initializing GestureTrainer...")
        X, y = dataset.prepare_windows(training_landmarks, training_labels)
        input_size = X.shape[1]
        num_classes = len(dataset.get_label_mapping())
        print(f"Found {num_classes} classes: {list(dataset.get_label_mapping().keys())}")
        
        # Initialize the trainer BEFORE using it 
        trainer = GestureTrainer(
            input_size=input_size,
            num_classes=num_classes,
            window_size=window_size,
            model_save_path=MODEL_SAVE_PATH
        )
        
        # 4. Now we can use the trainer for data preparation
        print("\nStep 4: Preparing data...")
        
        # Process validation data if available
        val_landmarks = None
        val_labels = None
        
        if os.path.exists(VALIDATION_DATA_DIR):
            print("Processing validation data...")
            val_generator = AnnotationGenerator()
            val_generator.process_directory(VALIDATION_DATA_DIR)
            val_landmarks = val_generator.get_landmark_data()
            val_labels = val_generator.get_landmark_label()
        
        # Prepare data with the trainer
        X_train, y_train, X_val, y_val, X_test, y_test = trainer.prepare_data(
            dataset, training_landmarks, training_labels, val_landmarks, val_labels
        )
        
        # 5. Train the model
        print("\nStep 5: Training the model...")
        epochs = 500
        batch_size = 16
        learning_rate = 0.0001
        
        trainer.train_with_validation(
            X_train, y_train, X_val, y_val,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate
        )
        
        # 6. Evaluate and save the model
        print("\nStep 6: Evaluating the model...")
        metrics = trainer.evaluate(X_test, y_test)
        
        print(f"\nStep 7: Saving the model to {MODEL_SAVE_PATH}...")
        trainer.save_model(included_landmarks=dataset.included_landmarks)
        
        # 8. Plot training history and confusion matrix
        print("\nStep 8: Plotting training history...")
        trainer.plot_training_history("training_history.png")
        trainer.plot_confusion_matrix(metrics['confusion_matrix'], "confusion_matrix.png")
        
        # Save model and mapping for webcam prediction
        idx_to_gesture = trainer.get_index_to_label()
        model = trainer.model
    else:
        # Load the model and mapping for prediction only
        print("Loading saved model for webcam prediction...")
        
        # Load gesture mapping
        mapping_path = os.path.splitext(MODEL_SAVE_PATH)[0] + "_mapping.npy"
        gesture_mapping = np.load(mapping_path, allow_pickle=True).item()
        idx_to_gesture = {idx: label for label, idx in gesture_mapping.items()}
        
        # Load parameters
        params_path = os.path.splitext(MODEL_SAVE_PATH)[0] + "_params.npy"
        if os.path.exists(params_path):
            params = np.load(params_path, allow_pickle=True).item()
            window_size = params.get('window_size', 10)
            input_size = params.get('feature_size', 0)
            included_landmarks = params.get('included_landmarks', list(range(11, 23)))
        else:
            window_size = 10
            included_landmarks = list(range(11, 23))
            
            # Rough estimate of feature size if params file is missing
            num_included = len(included_landmarks)
            pose_size = num_included * 4 * window_size  # Raw landmarks
            presence_size = window_size                 # Presence flags
            input_size = pose_size + presence_size
        
        print(f"Using window_size={window_size}, input_size={input_size}")
        
        # Load model
        num_classes = len(gesture_mapping)
        model = GestureClassificationNetwork(input_size=input_size, num_gestures=num_classes)
        model.load_model(MODEL_SAVE_PATH)
        
        print(f"Model loaded with {num_classes} gesture classes: {list(gesture_mapping.keys())}")
    
    # 9. Real-time webcam prediction (same as original code)
    print("\nStep 9: Starting real-time webcam prediction...")
    print("Press 'q' to quit.")
    
    # Initialize MediaPipe for webcam
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        smooth_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    
    # Get the list of included landmarks (either from params or default)
    included_landmarks = dataset.included_landmarks if 'dataset' in locals() else (
        params.get('included_landmarks', list(range(11, 23))) if 'params' in locals() else list(range(11, 23))
    )
    num_included = len(included_landmarks)
    
    # Arrays for feature extraction during webcam
    pose_data = np.zeros((window_size, num_included, 4))
    pose_present = np.zeros(window_size)
    buffer_idx = 0
    
    # Initialize webcam
    cap = cv2.VideoCapture(0)  # 0 for default webcam
    
    if not cap.isOpened():
        print("Error: Could not open webcam")
    else:
        # Create window
        cv2.namedWindow("Gesture Recognition", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Gesture Recognition", 1280, 720)
        
        # Variables for prediction
        current_prediction = "Waiting for data..."
        prediction_confidence = 0.0
        prev_time = time.time()
        fps = 0
        
        # For debouncing predictions (smoothing)
        last_predictions = deque(maxlen=10)  # Increased for more stability
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Calculate FPS
            current_time = time.time()
            elapsed = current_time - prev_time
            prev_time = current_time
            fps = 1.0 / elapsed if elapsed > 0 else 0
            
            # Process frame with MediaPipe
            # Flip frame horizontally for more intuitive interaction
            frame = cv2.flip(frame, 1)
            
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(rgb_frame)
            
            # Extract landmarks
            if results.pose_landmarks:
                # Draw landmarks on frame
                mp_drawing.draw_landmarks(
                    frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                    mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2)
                )
                
                # Store landmarks in buffer for prediction
                current_idx = buffer_idx % window_size
                pose_present[current_idx] = 1
                
                landmarks_list = []
                for i, landmark in enumerate(results.pose_landmarks.landmark):
                    landmarks_list.append({
                        'x': landmark.x,
                        'y': landmark.y,
                        'z': landmark.z,
                        'visibility': landmark.visibility
                    })
                
                # Only store the included landmarks
                for idx, j in enumerate(included_landmarks):
                    if j < len(landmarks_list):
                        landmark = landmarks_list[j]
                        pose_data[current_idx, idx, 0] = landmark['x']
                        pose_data[current_idx, idx, 1] = landmark['y']
                        pose_data[current_idx, idx, 2] = landmark['z']
                        pose_data[current_idx, idx, 3] = landmark['visibility']
                
                buffer_idx += 1
            
            # Make prediction when buffer is full
            if buffer_idx >= window_size:
                # Combine features - only raw landmarks and presence flags
                features = np.concatenate([
                    pose_data.flatten(),         # Raw landmark positions
                    pose_present.flatten()       # Presence flags
                ])
                
                # Reshape for model input
                features_reshaped = features.reshape(1, -1)
                
                # Make prediction
                prediction = model.predict(features_reshaped)[0]
                
                # Get gesture name
                predicted_gesture = idx_to_gesture.get(prediction, "unknown")
                
                # Add to recent predictions for smoothing
                last_predictions.append(predicted_gesture)
                
                # Get most common prediction (smoothing)
                prediction_counts = {}
                for pred in last_predictions:
                    prediction_counts[pred] = prediction_counts.get(pred, 0) + 1
                
                # Find the most common prediction
                max_count = 0
                for pred, count in prediction_counts.items():
                    if count > max_count:
                        max_count = count
                        current_prediction = pred
                
                # Calculate confidence as percentage of consistent predictions
                prediction_confidence = max_count / len(last_predictions)
                
                # Only update prediction if confidence is above threshold
                if prediction_confidence < CONFIDENCE_THRESHOLD:
                    current_prediction = "Uncertain - " + current_prediction
            
            # Display frame with prediction
            # Create a black bar at the bottom for text
            bar_height = 100
            bar = np.zeros((bar_height, frame.shape[1], 3), dtype=np.uint8)
            
            # Add text to the bar
            # Color based on confidence
            confidence_color = (
                0,
                int(255 * prediction_confidence),
                int(255 * (1 - prediction_confidence))
            )
            
            cv2.putText(bar, f"Gesture: {current_prediction}", (20, 30), 
                      cv2.FONT_HERSHEY_SIMPLEX, 1, confidence_color, 2)
            cv2.putText(bar, f"Confidence: {prediction_confidence:.2f}", (20, 70), 
                      cv2.FONT_HERSHEY_SIMPLEX, 1, confidence_color, 2)
            cv2.putText(bar, f"FPS: {fps:.1f}", (frame.shape[1] - 150, 30), 
                      cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            # Combine frame and bar
            display = np.vstack([frame, bar])
            
            # Display the result
            cv2.imshow("Gesture Recognition", display)
            
            # Break on 'q' key
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        # Clean up
        cap.release()
        cv2.destroyAllWindows()
    
    # Clean up
    if not SKIP_TRAINING:
        generator.close()
    
    print("\nGesture recognition complete!")