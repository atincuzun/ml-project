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