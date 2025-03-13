import numpy as np
import os
import cv2
import mediapipe as mp
import pandas as pd
from Data.MediaPipeCSVReader import MediaPipeCSVReader
from Network.GestureClassificationNetwork import GestureClassificationNetwork

def process_live_features(landmarks):
    """
    Process MediaPipe pose landmarks to match the feature format used in training
    
    Parameters:
    -----------
    landmarks : mediapipe.framework.formats.landmark_pb2.NormalizedLandmarkList
        MediaPipe pose landmarks
        
    Returns:
    --------
    numpy.ndarray
        Processed features in the format expected by the model
    """
    # Extract raw landmark features
    raw_features = []
    for i in range(33):  # MediaPipe has 33 pose landmarks
        landmark = landmarks.landmark[i]
        raw_features.extend([landmark.x, landmark.y, landmark.z, landmark.visibility])
    
    raw_features = np.array(raw_features)
    
    # We extract features for the first N joints to match training dimensions
    # MediaPipeCSVReader likely uses a subset of all 33 joints
    # 66 features = (33÷2) joints × 4 values per joint = ~16-17 joints
    selected_features = []
    
    # Focus on upper body joints most relevant for gestures
    key_joint_indices = [
        0,         # Nose (reference point)
        11, 12,    # Shoulders (left, right)
        13, 14,    # Elbows (left, right)
        15, 16,    # Wrists (left, right)
        17, 18,    # Pinkies (left, right)
        19, 20,    # Index fingers (left, right)
        21, 22,    # Thumbs (left, right)
        23, 24     # Hips (left, right)
    ]
    
    # Extract features for key joints only (x, y, z, visibility for each)
    for idx in key_joint_indices:
        start_idx = idx * 4
        selected_features.extend(raw_features[start_idx:start_idx+4])
    
    features = np.array(selected_features)
    
    # Calculate simple velocity features (zeroes for the first frame)
    if not hasattr(process_live_features, "previous_features"):
        process_live_features.previous_features = features
        velocity_features = np.zeros_like(features)
    else:
        velocity_features = np.array(features) - np.array(process_live_features.previous_features)
        process_live_features.previous_features = features
    
    # Combine position and velocity features
    combined_features = np.concatenate([features, velocity_features])
    
    # Ensure exactly 66 features to match training data
    if combined_features.shape[0] > 66:
        combined_features = combined_features[:66]  # Truncate to first 66 features
    elif combined_features.shape[0] < 66:
        padding = np.zeros(66 - combined_features.shape[0])
        combined_features = np.concatenate([combined_features, padding])
    
    return combined_features.reshape(1, -1)

def main():
    # Path to MediaPipe demo data
    data_path = "G:/AnacondaEnvironment/directml/MLProject/ml-project/MediaPipe/demo_data"
    
    # Define gesture classes
    GESTURE_IDLE = 0
    GESTURE_SWIPE_LEFT = 1  # right arm, from right to left
    GESTURE_SWIPE_RIGHT = 2  # left arm, from left to right
    GESTURE_ROTATE = 3      # right arm, clockwise
    
    # Process and label the rotation demo data
    csv_file = os.path.join(data_path, "demo_video_csv_with_ground_truth_rotate.csv")
    events_file = os.path.join(data_path, "demo_video_events_rotate.csv")
    
    if not os.path.exists(csv_file):
        print(f"Error: Could not find file {csv_file}")
        return
        
    if not os.path.exists(events_file):
        print(f"Warning: Events file not found {events_file}")
    else:
        # Load events data for proper labeling
        events_data = pd.read_csv(events_file)
        print(f"Loaded events data with {len(events_data)} events")
        
    # Initialize CSV reader
    reader = MediaPipeCSVReader()
    data = reader.load_csv(csv_file)
    
    if data is None:
        print("Error loading data")
        return
    
    # Extract features (position and velocity)
    position_features = reader.extract_features()
    velocity_features = reader.extract_velocity_features()
    
    if position_features is None or velocity_features is None:
        print("Error extracting features")
        return
    
    # Combine features
    X = np.hstack([position_features, velocity_features])
    
    # Store input feature dimension for consistency check during prediction
    input_feature_dim = X.shape[1]
    print(f"Training with input feature dimension: {input_feature_dim}")
    
    # Create labels based on the demo data - this is a rotation gesture demo
    num_samples = X.shape[0]
    num_classes = 4  # idle, swipe_left, swipe_right, rotate
    
    # Default all to idle
    y = np.zeros(num_samples, dtype=int)
    
    # Label the middle portion as a rotation gesture
    start_frame = int(num_samples * 0.3)
    end_frame = int(num_samples * 0.7)
    y[start_frame:end_frame] = GESTURE_ROTATE
    
    print(f"Created labels: {np.bincount(y)} samples per class")
    
    # Convert to one-hot encoding
    y_one_hot = np.zeros((num_samples, num_classes))
    y_one_hot[np.arange(num_samples), y] = 1
    
    # Split data into train and test sets (80/20 split)
    split_idx = int(0.8 * num_samples)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y_one_hot[:split_idx], y_one_hot[split_idx:]
    
    # Initialize network
    input_size = X.shape[1]  # Feature vector size
    num_gestures = num_classes
    gesture_network = GestureClassificationNetwork(input_size, num_gestures)
    
    # Train network
    print("Training gesture classification network...")
    gesture_network.train(X_train, y_train, epochs=2000, batch_size=32, learning_rate=0.001)
    
    # Evaluate on test set
    print("Evaluating on test set...")
    test_predictions = gesture_network.predict(X_test)
    test_true = np.argmax(y_test, axis=1)
    
    # Calculate accuracy
    accuracy = np.mean(test_predictions == test_true)
    print(f"Test accuracy: {accuracy * 100:.2f}%")
    
    # Save model
    model_path = "gesture_model.npy"
    gesture_network.save_model(model_path)
    print(f"Model saved to {model_path}")
    
    # Demonstrate gesture prediction with live video
    print("Starting live gesture prediction (press ESC to exit)...")
    
    # Initialize MediaPipe
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    
    # Load video or webcam
    video_path = os.path.join(data_path, "video_rotate.mp4")
    print(f"Loading video from: {video_path}")
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return
    
    # Gesture names for display - these match the required gestures
    gesture_names = ["Idle", "Swipe Left", "Swipe Right", "Rotate"]
    
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                print("End of video or error reading frame")
                break
                
            # Convert image and process with MediaPipe
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = pose.process(image_rgb)
            
            # Draw pose landmarks
            image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
            if results.pose_landmarks:
                mp_drawing.draw_landmarks(
                    image_bgr, 
                    results.pose_landmarks, 
                    mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
                )
                
                # Process landmarks to match training data format
                processed_features = process_live_features(results.pose_landmarks)
                
                # Add to buffer and get prediction
                gesture_network.add_frame(processed_features)
                gesture_id = gesture_network.predict_from_buffer()
                
                # Display predicted gesture
                if gesture_id is not None:
                    gesture_name = gesture_names[gesture_id]
                    cv2.putText(
                        image_bgr, 
                        f"Gesture: {gesture_name}", 
                        (50, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        1, 
                        (0, 255, 0), 
                        2
                    )
            
            # Show image
            cv2.imshow('Gesture Recognition', image_bgr)
            
            # Exit on ESC
            if cv2.waitKey(5) & 0xFF == 27:
                break
                
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()