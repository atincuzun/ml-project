import numpy as np
import os
import cv2
import mediapipe as mp
import pandas as pd
from Data.MediaPipeCSVReader import MediaPipeCSVReader
from Network.GestureClassificationNetwork import GestureClassificationNetwork
import time

swipeRight = 1
pinchSpread = 1
swipeUpDown = 1
bothHandsUp = 1
pointHands = 1

def process_live_features(landmarks):
    """
    Process MediaPipe pose landmarks to match the feature format used in training
    """
    # Load normalization parameters
    if not hasattr(process_live_features, "normalization_params"):
        try:
            params = np.load("feature_normalization.npy", allow_pickle=True).item()
            process_live_features.normalization_params = params
            print("Loaded normalization parameters successfully")
        except:
            # Default to no normalization if file not found
            process_live_features.normalization_params = None
            print("Warning: No normalization parameters found")
    
    # Extract raw landmark features
    raw_features = []
    for i in range(33):  # MediaPipe has 33 pose landmarks
        landmark = landmarks.landmark[i]
        raw_features.extend([landmark.x, landmark.y, landmark.z, landmark.visibility])
    
    raw_features = np.array(raw_features)
    features = np.array(raw_features)
    
    # Calculate velocity features
    if not hasattr(process_live_features, "previous_features"):
        process_live_features.previous_features = features
        velocity_features = np.zeros_like(features)
    else:
        velocity_features = np.array(features) - np.array(process_live_features.previous_features)
        process_live_features.previous_features = features
    
    # Combine position and velocity features
    combined_features = np.concatenate([features, velocity_features])
    
    # Ensure exactly 165 features
    if combined_features.shape[0] > 165:
        combined_features = combined_features[:165]
    elif combined_features.shape[0] < 165:
        padding = np.zeros(165 - combined_features.shape[0])
        combined_features = np.concatenate([combined_features, padding])
    
    # Apply normalization if available
    if process_live_features.normalization_params:
        mean = process_live_features.normalization_params["mean"]
        std = process_live_features.normalization_params["std"]
        combined_features = (combined_features - mean) / std
    
    return combined_features.reshape(1, -1)

def main():
    # Path to MediaPipe demo data
    data_path = "G:/AnacondaEnvironment/directml/MLProject/ml-project/MediaPipe/demo_data"
    
    # Define gesture classes
    GESTURE_IDLE = 0
    GESTURE_ROTATE = 1
    
    # Process and label the rotation demo data
    csv_file = os.path.join(data_path, "demo_video_csv_with_ground_truth_rotate.csv")
    annotation_file = os.path.join(data_path, "annotation_rotate.txt")
    
    if not os.path.exists(csv_file):
        print(f"Error: Could not find file {csv_file}")
        return
        
    # Initialize CSV reader
    reader = MediaPipeCSVReader()
    data = reader.load_csv(csv_file)
    
    if data is None:
        print("Error loading data")
        return
    
    # Extract features - get all x, y, z, visibility for each landmark
    position_features = []
    for _, row in data.iterrows():
        frame_features = []
        # Extract all 33 landmarks with x, y, z, visibility
        for i in range(33):
            prefix = f"landmark_{i}_"
            # Get x, y, z, visibility for each landmark
            x = row.get(f"{prefix}x", 0)
            y = row.get(f"{prefix}y", 0)
            z = row.get(f"{prefix}z", 0)
            visibility = row.get(f"{prefix}visibility", 0)
            frame_features.extend([x, y, z, visibility])
        position_features.append(frame_features)
    position_features = np.array(position_features)

    # Calculate velocity features
    velocity_features = np.zeros_like(position_features)
    for i in range(1, position_features.shape[0]):
        velocity_features[i] = position_features[i] - position_features[i-1]

    # Combine position and velocity features
    X = np.hstack([position_features, velocity_features])

    # Ensure we have exactly 165 dimensions
    if X.shape[1] > 165:
        X = X[:, :165]  # Truncate to 165 features
    elif X.shape[1] < 165:
        padding = np.zeros((X.shape[0], 165 - X.shape[1]))
        X = np.hstack([X, padding])
    
    # Normalize features - IMPORTANT
    feature_mean = np.mean(X, axis=0)
    feature_std = np.std(X, axis=0)
    feature_std[feature_std == 0] = 1  # Avoid division by zero
    X = (X - feature_mean) / feature_std

    # Save normalization parameters for live prediction
    np.save("feature_normalization.npy", {"mean": feature_mean, "std": feature_std})
    print("Saved normalization parameters")
    
    if position_features is None or velocity_features is None:
        print("Error extracting features")
        return

    # Get timestamps (assuming they're the index of the DataFrame)
    timestamps = data.index.tolist()
    
    # Load annotation file for precise rotation windows
    rotation_windows = []
    if os.path.exists(annotation_file):
        print("Using annotation file for labeling")
        with open(annotation_file, 'r') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 7 and parts[6] == 'rotate':
                    # Convert time from seconds to milliseconds
                    start_time = float(parts[2]) * 1000
                    end_time = float(parts[4]) * 1000
                    rotation_windows.append((start_time, end_time))
                    print(f"Rotation window: {start_time}ms to {end_time}ms")
    
    # If no windows from the annotation file, use hardcoded ones
    if not rotation_windows:
        rotation_windows = [
            (790, 2090),    # 00:00:00.790 to 00:00:02.090
            (5110, 6470),   # 00:00:05.110 to 00:00:06.470 
            (9500, 11010),  # 00:00:09.500 to 00:00:11.010
            (13700, 14930), # 00:00:13.700 to 00:00:14.930
            (16870, 18160), # 00:00:16.870 to 00:00:18.160
            (20340, 21780)  # 00:00:20.340 to 00:00:21.780
        ]
    
    # Create labels based on rotation windows
    num_samples = X.shape[0]
    y = np.zeros(num_samples, dtype=int)
    
    # Mark each frame as idle or rotate based on windows
    for i, timestamp in enumerate(timestamps):
        in_rotation = False
        for start_ms, end_ms in rotation_windows:
            if start_ms <= timestamp <= end_ms:
                in_rotation = True
                break
        
        if in_rotation:
            y[i] = GESTURE_ROTATE
    
    # Print class distribution
    rotate_count = np.sum(y == GESTURE_ROTATE)
    idle_count = np.sum(y == GESTURE_IDLE)
    print(f"Class distribution: IDLE={idle_count}, ROTATE={rotate_count}")
    print(f"Percentage of rotation frames: {rotate_count/num_samples:.2%}")
    
    # Create frame ranges for each rotation window
    frame_windows = []
    for start_ms, end_ms in rotation_windows:
        # Find frames that fall within this window
        start_frame = None
        end_frame = None
        
        for i, ts in enumerate(timestamps):
            if start_frame is None and ts >= start_ms:
                start_frame = i
            if end_frame is None and ts >= end_ms:
                end_frame = i
                break
        
        if start_frame is not None and end_frame is not None:
            frame_windows.append((start_frame, end_frame))
            print(f"Rotation window frames: {start_frame} to {end_frame}")
    
    # Split rotation windows for train/test (use 2/3 for training, 1/3 for testing)
    num_windows = len(frame_windows)
    num_train_windows = int(num_windows * 0.67)  # Use 67% of windows for training
    
    # Shuffle windows to get random train/test split
    np.random.seed(42)  # For reproducibility
    window_indices = np.random.permutation(num_windows)
    
    train_window_indices = window_indices[:num_train_windows]
    test_window_indices = window_indices[num_train_windows:]
    
    print(f"Using windows {train_window_indices} for training")
    print(f"Using windows {test_window_indices} for testing")
    
    # Create train/test masks
    train_mask = np.ones(num_samples, dtype=bool)
    test_mask = np.zeros(num_samples, dtype=bool)
    
    # Mark test frames based on test windows
    for idx in test_window_indices:
        start_frame, end_frame = frame_windows[idx]
        train_mask[start_frame:end_frame] = False
        test_mask[start_frame:end_frame] = True
        
    # Also include some idle frames in test set
    idle_frames = np.where(y == GESTURE_IDLE)[0]
    num_idle_test = len(idle_frames) // 3  # Use 1/3 of idle frames for testing
    
    # Randomly select idle frames for testing
    idle_test_indices = np.random.choice(idle_frames, num_idle_test, replace=False)
    
    for idx in idle_test_indices:
        train_mask[idx] = False
        test_mask[idx] = True
    
    # Create train/test sets
    X_train = X[train_mask]
    y_train = y[train_mask]
    X_test = X[test_mask]
    y_test = y[test_mask]
    
    # Print split statistics
    print(f"Training set: {len(X_train)} samples")
    print(f"  IDLE: {np.sum(y_train == GESTURE_IDLE)}, ROTATE: {np.sum(y_train == GESTURE_ROTATE)}")
    
    print(f"Testing set: {len(X_test)} samples")
    print(f"  IDLE: {np.sum(y_test == GESTURE_IDLE)}, ROTATE: {np.sum(y_test == GESTURE_ROTATE)}")
    
    # Convert to one-hot encoding
    num_classes = 2
    y_train_one_hot = np.zeros((len(y_train), num_classes))
    y_train_one_hot[np.arange(len(y_train)), y_train] = 1
    
    y_test_one_hot = np.zeros((len(y_test), num_classes))
    y_test_one_hot[np.arange(len(y_test)), y_test] = 1
    
    # Calculate class weights to handle imbalance
    rotate_weight = len(y_train) / (2 * np.sum(y_train == GESTURE_ROTATE))
    idle_weight = len(y_train) / (2 * np.sum(y_train == GESTURE_IDLE))
    print(f"Class weights - IDLE: {idle_weight:.2f}, ROTATE: {rotate_weight:.2f}")
    
    # Initialize network with 165 features
    input_size = 165
    print('Input feature size:', input_size)
    gesture_network = GestureClassificationNetwork(input_size, num_classes)
    
    # Train the network with better parameters
    print("Training gesture classification network...")
    gesture_network.train(X_train, y_train_one_hot, epochs=1000, batch_size=32, learning_rate=0.001)
    
    # Evaluate on training data
    print("Evaluating on training data...")
    train_predictions = gesture_network.predict(X_train)
    train_accuracy = np.mean(train_predictions == y_train)
    print(f"Training accuracy: {train_accuracy * 100:.2f}%")
    
    # Calculate class-specific accuracy for training
    train_rotate_correct = np.sum((train_predictions == GESTURE_ROTATE) & (y_train == GESTURE_ROTATE))
    train_rotate_total = np.sum(y_train == GESTURE_ROTATE)
    train_rotate_accuracy = train_rotate_correct / train_rotate_total if train_rotate_total > 0 else 0
    
    train_idle_correct = np.sum((train_predictions == GESTURE_IDLE) & (y_train == GESTURE_IDLE))
    train_idle_total = np.sum(y_train == GESTURE_IDLE)
    train_idle_accuracy = train_idle_correct / train_idle_total if train_idle_total > 0 else 0
    
    print(f"Training rotation accuracy: {train_rotate_accuracy * 100:.2f}%")
    print(f"Training idle accuracy: {train_idle_accuracy * 100:.2f}%")
    
    # Evaluate on test data
    print("Evaluating on test data...")
    test_predictions = gesture_network.predict(X_test)
    test_accuracy = np.mean(test_predictions == y_test)
    print(f"Test accuracy: {test_accuracy * 100:.2f}%")
    
    # Calculate class-specific accuracy for testing
    test_rotate_correct = np.sum((test_predictions == GESTURE_ROTATE) & (y_test == GESTURE_ROTATE))
    test_rotate_total = np.sum(y_test == GESTURE_ROTATE)
    test_rotate_accuracy = test_rotate_correct / test_rotate_total if test_rotate_total > 0 else 0
    
    test_idle_correct = np.sum((test_predictions == GESTURE_IDLE) & (y_test == GESTURE_IDLE))
    test_idle_total = np.sum(y_test == GESTURE_IDLE)
    test_idle_accuracy = test_idle_correct / test_idle_total if test_idle_total > 0 else 0
    
    print(f"Test rotation accuracy: {test_rotate_accuracy * 100:.2f}%")
    print(f"Test idle accuracy: {test_idle_accuracy * 100:.2f}%")
    
    # Save model
    model_path = "rotation_model.npy"
    gesture_network.save_model(model_path)
    print(f"Model saved to {model_path}")
    
    # Run live demo with the trained model
    # Load video
    video_path = os.path.join(data_path, "video_rotate.mp4")
    print(f"Loading video from: {video_path}")
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return
    
    # Gesture names
    gesture_names = ["Idle", "Rotate"]
    
    # Parameters for detection
    rotation_threshold = 0.002  # Medium-low sensitivity
    buffer_size = 20  # Larger buffer for better stability
    
    # Initialize MediaPipe
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    
    # Frame counter and fps tracking
    frame_count = 0
    fps = 30  # Assumed fps
    
    # For tracking predictions
    prediction_buffer = []
    predictions_log = []
    
    # Enable debug mode
    debug_mode = True
    
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                print("End of video or error reading frame")
                break
            
            frame_count += 1
            current_time = frame_count / fps * 1000  # Current time in milliseconds
            
            # Process with MediaPipe
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = pose.process(image_rgb)
            
            # Draw landmarks
            image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
            
            # Check if in a rotation window (ground truth)
            in_rotation_window = False
            for start_ms, end_ms in rotation_windows:
                if start_ms <= current_time <= end_ms:
                    in_rotation_window = True
                    break
            
            ground_truth = GESTURE_ROTATE if in_rotation_window else GESTURE_IDLE
            
            if results.pose_landmarks:
                mp_drawing.draw_landmarks(
                    image_bgr, 
                    results.pose_landmarks, 
                    mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
                )
                
                # Process landmarks for prediction
                processed_features = process_live_features(results.pose_landmarks)
                
                # Make prediction
                direct_prediction = gesture_network.predict(processed_features)[0]
                
                # Add to buffer
                prediction_buffer.append(direct_prediction)
                if len(prediction_buffer) > buffer_size:
                    prediction_buffer.pop(0)
                
                # Calculate rotation confidence
                rotation_ratio = prediction_buffer.count(GESTURE_ROTATE) / len(prediction_buffer) if prediction_buffer else 0
                
                # Final prediction
                final_prediction = GESTURE_ROTATE if rotation_ratio >= rotation_threshold else GESTURE_IDLE
                
                # Debug info
                if debug_mode and frame_count % 15 == 0:
                    print(f"Frame {frame_count}: Raw prediction={direct_prediction}, Confidence={rotation_ratio:.3f}")
                    
                    # Get raw probabilities for debugging
                    raw_probs = gesture_network.network.predict(processed_features)
                    print(f"  Probabilities: IDLE={raw_probs[0][0]:.3f}, ROTATE={raw_probs[0][1]:.3f}")
                    print(f"  Ground truth: {gesture_names[ground_truth]}")
                
                # Log prediction
                predictions_log.append({
                    'frame': frame_count,
                    'time_ms': current_time,
                    'direct_prediction': direct_prediction,
                    'final_prediction': final_prediction,
                    'ground_truth': ground_truth,
                    'rotation_ratio': rotation_ratio
                })
                
                # Display prediction
                gesture_name = gesture_names[final_prediction]
                color = (0, 255, 0) if final_prediction == GESTURE_ROTATE else (255, 255, 255)
                
                # Draw indicator circle
                circle_color = (0, 255, 0) if final_prediction == GESTURE_ROTATE else (0, 0, 255)
                cv2.circle(image_bgr, (400, 50), 30, circle_color, -1)
                
                # Draw ground truth indicator
                truth_color = (0, 255, 0) if ground_truth == GESTURE_ROTATE else (0, 0, 255)
                cv2.circle(image_bgr, (450, 50), 30, truth_color, -1)
                cv2.putText(image_bgr, "GT", (445, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                # Show gesture
                cv2.putText(
                    image_bgr, 
                    f"Gesture: {gesture_name}", 
                    (50, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    1, 
                    color, 
                    2
                )
                
                # Show confidence
                cv2.putText(
                    image_bgr,
                    f"Confidence: {rotation_ratio:.2f}",
                    (50, 90),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 255),
                    2
                )
                
                # Show frame prediction
                cv2.putText(
                    image_bgr,
                    f"Frame: {gesture_names[direct_prediction]}", 
                    (50, 130),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 255),
                    2
                )
                
                # Show time
                cv2.putText(
                    image_bgr,
                    f"Time: {current_time/1000:.2f}s, Frame: {frame_count}",
                    (50, 170),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 0, 255),
                    2
                )
                
                # Show ground truth
                cv2.putText(
                    image_bgr,
                    f"Ground Truth: {gesture_names[ground_truth]}",
                    (50, 210),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    truth_color,
                    2
                )
            
            # Show image
            cv2.imshow('Rotation Detection', image_bgr)
            
            # Exit on ESC
            if cv2.waitKey(5) & 0xFF == 27:
                break
    
    # Print prediction statistics
    correct = 0
    total = len(predictions_log)
    rotate_detected = 0
    rotate_ground_truth = 0
    
    for pred in predictions_log:
        if pred['final_prediction'] == pred['ground_truth']:
            correct += 1
        if pred['final_prediction'] == GESTURE_ROTATE:
            rotate_detected += 1
        if pred['ground_truth'] == GESTURE_ROTATE:
            rotate_ground_truth += 1
    
    print(f"\nPrediction Statistics:")
    print(f"Overall accuracy: {correct/total:.2%}")
    print(f"Detected rotation in {rotate_detected}/{total} frames ({rotate_detected/total:.2%})")
    print(f"Ground truth had rotation in {rotate_ground_truth}/{total} frames ({rotate_ground_truth/total:.2%})")
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()