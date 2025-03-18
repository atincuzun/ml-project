import numpy as np
import matplotlib.pyplot as plt
import os

class GestureDataset:
    """
    Creates sliding windows of temporal data from pose landmarks with labels.
    Focused only on the 33 MediaPipe Pose landmarks for gesture recognition.
    """
    
    def __init__(self, window_size=30, step_size=1):
        """
        Initialize the dataset class.
        
        Parameters:
        -----------
        window_size : int
            Number of frames in each sliding window
        step_size : int
            Number of frames to advance when creating the next window
        """
        self.window_size = window_size
        self.step_size = step_size
        self.gesture_mapping = {}
        
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
            
            # Get label for this window (use the last frame's label)
            window_label = labels[i + self.window_size - 1]
            label_idx = self.gesture_mapping[window_label]
            
            # Extract features from the window
            features = self._extract_features(window)
            
            if features is not None:
                X.append(features)
                y.append(label_idx)
        
        # Convert to numpy arrays
        X = np.array(X)
        y = np.array(y)
        
        print(f"Created {len(X)} windows with shape {X.shape}")
        
        return X, y
    
    def _extract_features(self, window):
        """
        Extract features from a window of frames with pose landmarks.
        
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
        
        # Prepare array for landmarks
        # Pose: 33 landmarks with x,y,z,visibility (4 values per landmark)
        pose_data = np.zeros((len(window), 33, 4))
        
        # Presence flag for each frame
        pose_present = np.zeros(len(window))
        
        # Extract landmark data for each frame
        for i, frame in enumerate(window):
            if 'landmarks' in frame and frame['landmarks']:
                pose_present[i] = 1
                landmarks = frame['landmarks']
                
                for j, landmark in enumerate(landmarks):
                    if j < 33:  # Ensure we stay within array bounds
                        pose_data[i, j, 0] = landmark['x']
                        pose_data[i, j, 1] = landmark['y']
                        pose_data[i, j, 2] = landmark['z']
                        pose_data[i, j, 3] = landmark['visibility']
        
        # Method 1: Full sequence - flatten the entire temporal sequence
        # This preserves all temporal information within the window
        features = np.concatenate([
            pose_data.flatten(),      # All pose landmarks across time
            pose_present.flatten()    # Pose presence flags
        ])
        
        # Method 2: Statistical features only
        # Uncomment this section if you prefer statistical summaries instead
        """
        features = []
        
        # First add presence information
        features.append(np.mean(pose_present))
        
        # Only process frames where pose is detected
        if np.any(pose_present):
            # Calculate statistics across frames for each landmark
            # Mean position
            features.append(np.mean(pose_data, axis=0).flatten())
            
            # Standard deviation (movement variability)
            features.append(np.std(pose_data, axis=0).flatten())
            
            # Min and max values to capture range of motion
            features.append(np.min(pose_data, axis=0).flatten())
            features.append(np.max(pose_data, axis=0).flatten())
            
        else:
            # If no poses detected, add zero placeholders
            features.append(np.zeros(33 * 4))  # mean
            features.append(np.zeros(33 * 4))  # std
            features.append(np.zeros(33 * 4))  # min
            features.append(np.zeros(33 * 4))  # max
            
        features = np.concatenate(features)
        """
        
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

# Example usage with complete training pipeline and visualization
if __name__ == "__main__":
    import cv2
    import mediapipe as mp
    from annotation_generator import AnnotationGenerator
    from Network.GestureClassificationNetwork import GestureClassificationNetwork
    from sklearn.model_selection import train_test_split
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.metrics import confusion_matrix
    import os
    from collections import deque
    
    # Path settings
    VIDEO_PATH = "./TrainingData/rotate.mp4"
    ANNOTATION_PATH = "./TrainingData/rotate.eaf"
    MODEL_SAVE_PATH = "gesture_model.npy"
    
    # 1. Generate landmarks and labels
    print("Step 1: Generating landmarks and labels...")
    generator = AnnotationGenerator()
    generator.set_video_path(VIDEO_PATH) \
             .set_annotation_path(ANNOTATION_PATH) \
             .prepare_training_data()
    
    landmarks = generator.get_landmark_data()
    labels = generator.get_landmark_label()
    
    # 2. Create dataset with sliding windows
    print("\nStep 2: Creating sliding windows of temporal data...")
    window_size = 10
    step_size = 1
    dataset = GestureDataset(window_size=window_size, step_size=step_size)
    X, y = dataset.prepare_windows(landmarks, labels)
    
    # 3. Split data into training and test sets
    print("\nStep 3: Splitting data into training and test sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    
    # Convert labels to one-hot encoded format
    num_classes = len(dataset.get_label_mapping())
    y_train_one_hot = np.zeros((y_train.shape[0], num_classes))
    y_train_one_hot[np.arange(y_train.shape[0]), y_train] = 1
    
    # 4. Initialize the model
    print("\nStep 4: Initializing the GestureClassificationNetwork...")
    input_size = X_train.shape[1]  # Number of features
    model = GestureClassificationNetwork(input_size=input_size, num_gestures=num_classes)
    
    # 5. Train the model
    print("\nStep 5: Training the model...")
    epochs = 300
    batch_size = 32
    learning_rate = 0.001
    
    model.train(
        X_train=X_train,
        y_train=y_train_one_hot,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate
    )
    
    # 6. Evaluate the model
    print("\nStep 6: Evaluating the model...")
    # Make predictions - this already returns class indices, not probabilities
    y_pred_classes = model.predict(X_test)
    
    # Calculate accuracy
    accuracy = np.mean(y_pred_classes == y_test)
    print(f"Test accuracy: {accuracy:.4f}")
    
    # Calculate per-class accuracy
    class_accuracies = {}
    for i in range(num_classes):
        # Get indices where true label is class i
        indices = np.where(y_test == i)[0]
        if len(indices) > 0:
            # Calculate accuracy for this class
            class_acc = np.mean(y_pred_classes[indices] == y_test[indices])
            # Get class name
            class_name = dataset.get_index_mapping()[i]
            class_accuracies[class_name] = class_acc
    
    print("\nPer-class accuracy:")
    for class_name, acc in class_accuracies.items():
        print(f"  {class_name}: {acc:.4f}")
    
    # 7. Save the model
    print(f"\nStep 7: Saving the model to {MODEL_SAVE_PATH}...")
    model.save_model(MODEL_SAVE_PATH)
    print("Model saved successfully.")
    
    # Save the gesture mapping alongside the model
    mapping_path = os.path.splitext(MODEL_SAVE_PATH)[0] + "_mapping.npy"
    np.save(mapping_path, dataset.get_label_mapping())
    print(f"Gesture mapping saved to {mapping_path}")
    
    # 8. Plot confusion matrix for visualization
    print("\nStep 8: Generating confusion matrix...")
    cm = confusion_matrix(y_test, y_pred_classes)
    
    # Create a figure
    plt.figure(figsize=(10, 8))
    
    # Create labels for the plot
    class_names = [dataset.get_index_mapping()[i] for i in range(num_classes)]
    
    # Plot the confusion matrix
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title('Confusion Matrix')
    
    # Save the plot
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    print("Confusion matrix saved to confusion_matrix.png")
    
    # 9. Visualize predictions on the original video
    print("\nStep 9: Visualizing predictions on the original video...")
    
    # Initialize MediaPipe for visualization
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        smooth_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    
    # Create a frame buffer to match the window size used for training
    frame_buffer = deque(maxlen=window_size)
    
    # Get label to index mapping
    idx_to_gesture = dataset.get_index_mapping()
    
    # Open the video
    cap = cv2.VideoCapture(VIDEO_PATH)
    
    if not cap.isOpened():
        print(f"Error: Could not open video {VIDEO_PATH}")
    else:
        # Create window
        cv2.namedWindow("Gesture Prediction", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Gesture Prediction", 1280, 720)
        
        # Process frames
        frame_idx = 0
        current_prediction = "Waiting for frames..."
        ground_truth = "Unknown"
        
        # For calculating real-time model inference
        pose_data = np.zeros((window_size, 33, 4))
        pose_present = np.zeros(window_size)
        buffer_idx = 0
        
        print("\nPress 'q' to quit the visualization...")
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Get ground truth label for this frame if available
            if frame_idx < len(labels):
                ground_truth = labels[frame_idx]
            
            # Process frame with MediaPipe
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
                for i, landmark in enumerate(results.pose_landmarks.landmark):
                    if i < 33:  # Ensure we stay within array bounds
                        pose_data[buffer_idx % window_size, i, 0] = landmark.x
                        pose_data[buffer_idx % window_size, i, 1] = landmark.y
                        pose_data[buffer_idx % window_size, i, 2] = landmark.z
                        pose_data[buffer_idx % window_size, i, 3] = landmark.visibility
                
                pose_present[buffer_idx % window_size] = 1
                buffer_idx += 1
            
            # Make prediction when buffer is full
            if buffer_idx >= window_size:
                # Extract features
                features = np.concatenate([
                    pose_data.flatten(),
                    pose_present.flatten()
                ])
                
                # Reshape for model input
                X_pred = np.array([features])
                
                # Make prediction
                prediction = model.predict(X_pred)[0]
                
                # Convert to gesture name
                current_prediction = idx_to_gesture.get(prediction, "unknown")
            
            # Display frame with prediction and ground truth
            # Create a black bar at the bottom for text
            bar_height = 100
            bar = np.zeros((bar_height, frame.shape[1], 3), dtype=np.uint8)
            
            # Add text to the bar
            cv2.putText(bar, f"Prediction: {current_prediction}", (20, 30), 
                      cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(bar, f"Ground Truth: {ground_truth}", (20, 70), 
                      cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            cv2.putText(bar, f"Frame: {frame_idx}", (frame.shape[1] - 200, 30), 
                      cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            # Combine frame and bar
            display = np.vstack([frame, bar])
            
            # Display the result
            cv2.imshow("Gesture Prediction", display)
            
            # Increment frame counter
            frame_idx += 1
            
            # Break on 'q' key
            if cv2.waitKey(30) & 0xFF == ord('q'):
                break
        
        # Clean up
        cap.release()
        cv2.destroyAllWindows()
    
    print("\nTraining, evaluation, and visualization complete!")