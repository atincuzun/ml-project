import os

# Flask app configuration
SECRET_KEY = os.urandom(24)
DEBUG = True

# File upload settings
UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'uploads')
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv'}
MAX_CONTENT_LENGTH = 500 * 1024 * 1024  # 500MB max upload size

# MediaPipe settings
MEDIAPIPE_MIN_DETECTION_CONFIDENCE = 0.5
MEDIAPIPE_MIN_TRACKING_CONFIDENCE = 0.5

# Gesture detection settings
GESTURE_DETECTION_INTERVAL = 500  # milliseconds
GESTURE_LOG_STEP_SIZE = 1       # Update gesture log every N frames

# Gesture recognition sliding window settings
# Must match the window size used during training
SLIDING_WINDOW_STEP_SIZE = 1  # Process every n frames
SLIDING_WINDOW_SIZE = 10  # Match the window size used during training
EXCLUDED_LANDMARKS = None  # List of landmark names to exclude, or None

# Model paths (these should be absolute paths in production)
GESTURE_MODEL_PATH = "gesture_model.npy"
GESTURE_MAPPING_PATH = "gesture_model_mapping.npy"  # Should match actual extension

# Gesture post-processing settings
POSTPROCESSING_REGISTERING_THRESHOLD_SIZE = 10  # Size of buffer for registering mode
POSTPROCESSING_REGISTERING_THRESHOLD_LIMIT = 8  # Minimum occurrences to register a gesture
POSTPROCESSING_REGISTERED_THRESHOLD_SIZE = 10  # Size of buffer for registered mode
POSTPROCESSING_REGISTERED_THRESHOLD_LIMIT = 6  # Maximum occurrences to remain in registered mode