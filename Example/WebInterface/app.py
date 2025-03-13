from flask import Flask, render_template, send_from_directory, jsonify, request
from flask_socketio import SocketIO, emit
import threading
import time
import cv2
import mediapipe as mp
import numpy as np
import os
import sys
import json
import base64
import pandas as pd

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Network.GestureClassificationNetwork import GestureClassificationNetwork
from Data.MediaPipeCSVReader import MediaPipeCSVReader

app = Flask(__name__)
app.config['SECRET_KEY'] = 'gesture-control-secret!'
socketio = SocketIO(app, cors_allowed_origins="*")

# Global variables
detection_active = False
current_gesture = "idle"
last_event_time = 0
event_cooldown = 1.0  # seconds between events to prevent flooding
training_in_progress = False
training_progress = 0
training_log = []

# Gesture detection settings
GESTURE_IDLE = 0
GESTURE_ROTATE = 1
GESTURE_SWIPE_LEFT = 2
GESTURE_SWIPE_RIGHT = 3
gesture_names = ["idle", "rotate", "swipe_left", "swipe_right"]
rotation_threshold = 0.3
buffer_size = 7
prediction_buffer = []

# Load the trained model or create a new one to be trained
model_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "gesture_model.npy")
gesture_network = None  # Will be loaded on demand

# Routes for serving existing files
@app.route('/')
def index():
    # Use send_from_directory instead of render_template for index.html
    # This is because index.html is in the WebInterface directory, not templates
    return send_from_directory('.', 'index.html')

@app.route('/<path:path>')
def send_static(path):
    return send_from_directory('.', path)

@app.route('/presentation')
def presentation():
    return render_template('presentation.html')

@app.route('/status')
def status():
    return jsonify({
        'detection_active': detection_active,
        'current_gesture': current_gesture,
        'training_in_progress': training_in_progress,
        'training_progress': training_progress,
        'model_loaded': gesture_network is not None
    })

# Route to get training logs
@app.route('/training_log')
def get_training_log():
    return jsonify(training_log)

# New route to get current gesture via HTTP
@app.route('/get_current_gesture', methods=['GET'])
def get_current_gesture():
    global current_gesture
    return jsonify({'gesture': current_gesture})

# API endpoint to start training with parameters
@app.route('/start_training', methods=['POST'])
def start_training():
    global training_in_progress
    
    if training_in_progress:
        return jsonify({'status': 'error', 'message': 'Training already in progress'})
    
    # Get training parameters from request
    try:
        params = request.json
        epochs = int(params.get('epochs', 500))
        batch_size = int(params.get('batch_size', 16))
        learning_rate = float(params.get('learning_rate', 0.0005))
        buffer_size_param = int(params.get('buffer_size', 7))
        
        # Start training in separate thread
        training_thread = threading.Thread(
            target=train_network, 
            args=(epochs, batch_size, learning_rate, buffer_size_param)
        )
        training_thread.start()
        
        return jsonify({
            'status': 'success', 
            'message': 'Training started', 
            'params': {
                'epochs': epochs, 
                'batch_size': batch_size, 
                'learning_rate': learning_rate
            }
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

# WebSocket for real-time communication
@socketio.on('connect')
def handle_connect():
    print('Client connected')

@socketio.on('start_detection')
def handle_start_detection():
    global detection_active, gesture_network, buffer_size
    
    if not detection_active:
        # Load model if it hasn't been loaded yet
        if gesture_network is None:
            try:
                # Initialize network with same parameters as in test.py
                input_size = 66  # The feature vector size
                num_classes = len(gesture_names)  # idle, rotate, swipe_left, swipe_right
                gesture_network = GestureClassificationNetwork(input_size, num_classes)
                
                if os.path.exists(model_path):
                    gesture_network.load_model(model_path)
                    print(f"Model loaded from {model_path}")
                else:
                    print("No model found, please train first")
                    return {'status': 'error', 'message': 'No model found, please train first'}
                
                # Set buffer size
                gesture_network.buffer_size = buffer_size
                
                detection_active = True
                threading.Thread(target=gesture_detection_loop).start()
                return {'status': 'started'}
            except Exception as e:
                print(f"Error loading model: {e}")
                return {'status': 'error', 'message': str(e)}
        else:
            detection_active = True
            threading.Thread(target=gesture_detection_loop).start()
            return {'status': 'started'}
    
    return {'status': 'already_running'}

@socketio.on('stop_detection')
def handle_stop_detection():
    global detection_active
    detection_active = False
    return {'status': 'stopped'}

# Function to train the network
def train_network(epochs=500, batch_size=16, learning_rate=0.0005, buffer_size_param=7):
    global gesture_network, training_in_progress, training_progress, training_log, buffer_size
    
    if training_in_progress:
        return False
    
    training_in_progress = True
    training_progress = 0
    training_log = []
    buffer_size = buffer_size_param
    
    try:
        # Path to MediaPipe demo data - same as in test.py
        data_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "MediaPipe/demo_data")
        
        # Define gesture classes
        GESTURE_IDLE = 0
        GESTURE_ROTATE = 1
        
        # Process and label the rotation demo data
        csv_file = os.path.join(data_path, "demo_video_csv_with_ground_truth_rotate.csv")
        annotation_file = os.path.join(data_path, "annotation_rotate.txt")
        
        log_and_emit("Starting training process")
        log_and_emit(f"Using CSV file: {csv_file}")
        
        if not os.path.exists(csv_file):
            log_and_emit(f"Error: Could not find file {csv_file}")
            training_in_progress = False
            return
            
        # Initialize CSV reader
        reader = MediaPipeCSVReader()
        data = reader.load_csv(csv_file)
        
        if data is None:
            log_and_emit("Error loading data")
            training_in_progress = False
            return
        
        # Extract features
        log_and_emit("Extracting position features...")
        position_features = reader.extract_features()
        
        log_and_emit("Extracting velocity features...")
        velocity_features = reader.extract_velocity_features()
        
        if position_features is None or velocity_features is None:
            log_and_emit("Error extracting features")
            training_in_progress = False
            return
        
        # Combine features
        X = np.hstack([position_features, velocity_features])
        input_feature_dim = X.shape[1]
        log_and_emit(f"Feature dimension: {input_feature_dim}")
        
        # Get timestamps
        timestamps = data.index.tolist()
        
        # Load annotation file for precise rotation windows
        rotation_windows = []
        if os.path.exists(annotation_file):
            log_and_emit("Using annotation file for labeling")
            with open(annotation_file, 'r') as f:
                for line in f:
                    parts = line.strip().split('\t')
                    if len(parts) >= 7 and parts[6] == 'rotate':
                        # Convert time from seconds to milliseconds
                        start_time = float(parts[2]) * 1000
                        end_time = float(parts[4]) * 1000
                        rotation_windows.append((start_time, end_time))
                        log_and_emit(f"Rotation window: {start_time}ms to {end_time}ms")
        
        # If no windows from the annotation file, use hardcoded ones
        if not rotation_windows:
            log_and_emit("Using hardcoded rotation windows")
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
        log_and_emit(f"Class distribution: IDLE={idle_count}, ROTATE={rotate_count}")
        log_and_emit(f"Percentage of rotation frames: {rotate_count/num_samples:.2%}")
        
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
                log_and_emit(f"Rotation window frames: {start_frame} to {end_frame}")
        
        # Split rotation windows for train/test (use 2/3 for training, 1/3 for testing)
        num_windows = len(frame_windows)
        num_train_windows = int(num_windows * 0.67)  # Use 67% of windows for training
        
        # Shuffle windows to get random train/test split
        np.random.seed(42)  # For reproducibility
        window_indices = np.random.permutation(num_windows)
        
        train_window_indices = window_indices[:num_train_windows]
        test_window_indices = window_indices[num_train_windows:]
        
        log_and_emit(f"Using windows {train_window_indices} for training")
        log_and_emit(f"Using windows {test_window_indices} for testing")
        
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
        log_and_emit(f"Training set: {len(X_train)} samples")
        log_and_emit(f"  IDLE: {np.sum(y_train == GESTURE_IDLE)}, ROTATE: {np.sum(y_train == GESTURE_ROTATE)}")
        
        log_and_emit(f"Testing set: {len(X_test)} samples")
        log_and_emit(f"  IDLE: {np.sum(y_test == GESTURE_IDLE)}, ROTATE: {np.sum(y_test == GESTURE_ROTATE)}")
        
        # Convert to one-hot encoding
        num_classes = len(gesture_names)  # idle, rotate, swipe_left, swipe_right
        y_train_one_hot = np.zeros((len(y_train), num_classes))
        y_train_one_hot[np.arange(len(y_train)), y_train] = 1
        
        y_test_one_hot = np.zeros((len(y_test), num_classes))
        y_test_one_hot[np.arange(len(y_test)), y_test] = 1
        
        # Initialize network
        input_size = X.shape[1]
        global gesture_network
        gesture_network = GestureClassificationNetwork(input_size, num_classes)
        
        # Train the network
        log_and_emit(f"Training network with params: epochs={epochs}, batch_size={batch_size}, learning_rate={learning_rate}")
        
        # Custom training loop to update progress
        total_epochs = epochs
        epoch_size = len(X_train)
        batch_count = (epoch_size + batch_size - 1) // batch_size  # ceil division
        
        # Initialize layers
        gesture_network.network.initialize(X_train, y_train_one_hot)
        
        # For each epoch
        for epoch in range(epochs):
            # Shuffle data
            indices = np.random.permutation(epoch_size)
            X_train_shuffled = X_train[indices]
            y_train_shuffled = y_train_one_hot[indices]
            
            epoch_loss = 0
            
            # For each batch
            for batch_start in range(0, epoch_size, batch_size):
                batch_end = min(batch_start + batch_size, epoch_size)
                batch_X = X_train_shuffled[batch_start:batch_end]
                batch_y = y_train_shuffled[batch_start:batch_end]
                
                # Forward pass
                output = gesture_network.network.forward(batch_X)
                
                # Calculate loss
                batch_loss = gesture_network.network.loss.calculate(output, batch_y)
                epoch_loss += batch_loss
                
                # Backward pass
                gradients = gesture_network.network.loss.backward(output, batch_y)
                gesture_network.network.backward(gradients)
                
                # Update weights
                gesture_network.network.update_params(learning_rate)
                
            # Calculate average loss for this epoch
            epoch_loss /= batch_count
            
            # Update progress
            progress_pct = (epoch + 1) / total_epochs * 100
            training_progress = int(progress_pct)
            
            # Log every 10% or when finished
            if (epoch + 1) % max(1, total_epochs // 10) == 0 or epoch == total_epochs - 1:
                log_and_emit(f"Epoch {epoch + 1}/{total_epochs}, Loss: {epoch_loss:.4f}, Progress: {training_progress}%")
                socketio.emit('training_update', {'progress': training_progress, 'epoch': epoch + 1, 'loss': float(epoch_loss)})
        
        # Evaluate on training data
        log_and_emit("Evaluating on training data...")
        train_predictions = gesture_network.predict(X_train)
        train_accuracy = np.mean(train_predictions == y_train)
        log_and_emit(f"Training accuracy: {train_accuracy * 100:.2f}%")
        
        # Calculate class-specific accuracy for training
        train_rotate_correct = np.sum((train_predictions == GESTURE_ROTATE) & (y_train == GESTURE_ROTATE))
        train_rotate_total = np.sum(y_train == GESTURE_ROTATE)
        train_rotate_accuracy = train_rotate_correct / train_rotate_total if train_rotate_total > 0 else 0
        
        train_idle_correct = np.sum((train_predictions == GESTURE_IDLE) & (y_train == GESTURE_IDLE))
        train_idle_total = np.sum(y_train == GESTURE_IDLE)
        train_idle_accuracy = train_idle_correct / train_idle_total if train_idle_total > 0 else 0
        
        log_and_emit(f"Training rotation accuracy: {train_rotate_accuracy * 100:.2f}%")
        log_and_emit(f"Training idle accuracy: {train_idle_accuracy * 100:.2f}%")
        
        # Evaluate on test data
        log_and_emit("Evaluating on test data...")
        test_predictions = gesture_network.predict(X_test)
        test_accuracy = np.mean(test_predictions == y_test)
        log_and_emit(f"Test accuracy: {test_accuracy * 100:.2f}%")
        
        # Calculate class-specific accuracy for testing
        test_rotate_correct = np.sum((test_predictions == GESTURE_ROTATE) & (y_test == GESTURE_ROTATE))
        test_rotate_total = np.sum(y_test == GESTURE_ROTATE)
        test_rotate_accuracy = test_rotate_correct / test_rotate_total if test_rotate_total > 0 else 0
        
        test_idle_correct = np.sum((test_predictions == GESTURE_IDLE) & (y_test == GESTURE_IDLE))
        test_idle_total = np.sum(y_test == GESTURE_IDLE)
        test_idle_accuracy = test_idle_correct / test_idle_total if test_idle_total > 0 else 0
        
        log_and_emit(f"Test rotation accuracy: {test_rotate_accuracy * 100:.2f}%")
        log_and_emit(f"Test idle accuracy: {test_idle_accuracy * 100:.2f}%")
        
        # Save model
        gesture_network.save_model(model_path)
        log_and_emit(f"Model saved to {model_path}")
        
        # Update buffer size from parameter
        buffer_size = buffer_size_param
        
        # Finish
        log_and_emit("Training complete!")
        socketio.emit('training_complete', {
            'train_accuracy': float(train_accuracy * 100), 
            'test_accuracy': float(test_accuracy * 100)
        })
        
        training_in_progress = False
        training_progress = 100
        
    except Exception as e:
        log_and_emit(f"Error during training: {str(e)}")
        training_in_progress = False

def log_and_emit(message):
    """Log message to console and add to training log"""
    global training_log
    print(message)
    training_log.append({'time': time.time(), 'message': message})
    socketio.emit('training_log', {'message': message})

# Function to process features for gesture recognition
def process_live_features(landmarks):
    """
    Process MediaPipe pose landmarks to match the feature format used in training
    """
    # Extract raw landmark features
    raw_features = []
    for i in range(33):  # MediaPipe has 33 pose landmarks
        landmark = landmarks.landmark[i]
        raw_features.extend([landmark.x, landmark.y, landmark.z, landmark.visibility])
    
    raw_features = np.array(raw_features)
    
    # Extract features for key joints
    selected_features = []
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
    
    for idx in key_joint_indices:
        start_idx = idx * 4
        selected_features.extend(raw_features[start_idx:start_idx+4])
    
    features = np.array(selected_features)
    
    # Calculate velocity features
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
        combined_features = combined_features[:66]
    elif combined_features.shape[0] < 66:
        padding = np.zeros(66 - combined_features.shape[0])
        combined_features = np.concatenate([combined_features, padding])
    
    return combined_features.reshape(1, -1)

def emit_gesture_event(gesture_name):
    """Emit a gesture event with cooldown to prevent flooding"""
    global last_event_time
    
    current_time = time.time()
    if current_time - last_event_time >= event_cooldown:
        last_event_time = current_time
        socketio.emit('gesture', {'gesture': gesture_name})
        print(f"Emitted gesture: {gesture_name}")

def gesture_detection_loop():
    """Main loop for gesture detection"""
    global detection_active, current_gesture, prediction_buffer, buffer_size
    
    # Initialize MediaPipe
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    
    # Initialize webcam
    cap = cv2.VideoCapture(0)  # Use default webcam (change index if needed)
    
    if not cap.isOpened():
        print("Error: Could not open webcam")
        detection_active = False
        return
    
    try:
        with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
            while detection_active:
                # Get frame from webcam
                success, frame = cap.read()
                if not success:
                    print("Error reading from webcam")
                    break
                
                # Process with MediaPipe
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = pose.process(frame_rgb)
                
                # Create a copy for drawing
                output_frame = frame.copy()
                
                if results.pose_landmarks:
                    # Draw landmarks
                    mp_drawing.draw_landmarks(
                        output_frame, 
                        results.pose_landmarks, 
                        mp_pose.POSE_CONNECTIONS,
                        landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
                    )
                    
                    # Process landmarks for prediction
                    processed_features = process_live_features(results.pose_landmarks)
                    
                    # Make prediction using the network's predict function
                    direct_prediction = gesture_network.predict(processed_features)[0]
                    
                    # Add to buffer for smoothing
                    prediction_buffer.append(direct_prediction)
                    if len(prediction_buffer) > buffer_size:
                        prediction_buffer.pop(0)
                    
                    # Count occurrences of each gesture in the buffer
                    gesture_counts = [prediction_buffer.count(i) for i in range(len(gesture_names))]
                    
                    # Calculate rotation confidence
                    rotation_ratio = gesture_counts[GESTURE_ROTATE] / len(prediction_buffer) if prediction_buffer else 0
                    
                    # Final prediction
                    final_prediction = GESTURE_ROTATE if rotation_ratio >= rotation_threshold else GESTURE_IDLE
                    
                    # Update current gesture
                    new_gesture = gesture_names[final_prediction]
                    
                    # Draw indicator circle
                    circle_color = (0, 255, 0) if final_prediction == GESTURE_ROTATE else (0, 0, 255)
                    cv2.circle(output_frame, (30, 30), 20, circle_color, -1)
                    
                    # Show gesture name
                    cv2.putText(
                        output_frame, 
                        f"Gesture: {gesture_names[final_prediction]}", 
                        (70, 35), 
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        0.7, 
                        (255, 255, 255), 
                        2
                    )
                    
                    # Show confidence
                    cv2.putText(
                        output_frame,
                        f"Confidence: {rotation_ratio:.2f}",
                        (70, 70),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 255),
                        2
                    )
                    
                    # Only emit if gesture changed
                    if new_gesture != current_gesture:
                        current_gesture = new_gesture
                        socketio.emit('gesture_update', {'gesture': current_gesture})
                        
                        # Also emit the specific control event
                        emit_gesture_event(current_gesture)
                
                # Convert to JPEG
                ret, buffer = cv2.imencode('.jpg', output_frame)
                frame_bytes = base64.b64encode(buffer.tobytes()).decode('utf-8')
                
                # Send frame to client
                socketio.emit('video_feed', {'image': f'data:image/jpeg;base64,{frame_bytes}'})
                
                # Don't hog the CPU
                time.sleep(0.03)  # ~30fps
    finally:
        cap.release()
        detection_active = False
        print("Gesture detection stopped")
        socketio.emit('detection_stopped')

# Create templates directory if it doesn't exist
os.makedirs('templates', exist_ok=True)

if __name__ == '__main__':
    socketio.run(app, debug=True, host='0.0.0.0')