import cv2
import mediapipe as mp
import numpy as np
import time
import yaml
import os


class MediapipeProcessor:
	def __init__(self):
		self.mp_drawing = mp.solutions.drawing_utils
		self.mp_drawing_styles = mp.solutions.drawing_styles
		self.mp_pose = mp.solutions.pose
		
		# Initialize the pose detector
		self.pose = self.mp_pose.Pose(
			min_detection_confidence=0.5,
			min_tracking_confidence=0.5
		)
		
		# Load keypoint mapping
		script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
		mapping_path = os.path.join(script_dir, 'keypoint_mapping.yml')
		if os.path.exists(mapping_path):
			with open(mapping_path, "r") as yaml_file:
				self.mappings = yaml.safe_load(yaml_file)
				self.keypoint_names = self.mappings["face"]
				self.keypoint_names += self.mappings["body"]
		else:
			# Default keypoint names if the file doesn't exist
			self.keypoint_names = []
			print(f"Warning: Keypoint mapping file not found at {mapping_path}")
		
		# Video processing variables
		self.video_path = None
		self.video_cap = None
		self.is_logging = False
		self.last_pose_data = None
		self.last_gesture = "idle"
		
		# Gesture detection variables
		self.last_gesture_time = 0
		self.gesture_cooldown = 1000  # 1 second cooldown between gestures
		
		# Previous frames for gesture detection
		self.frame_history = []
		self.max_history = 10
	
	def set_video_path(self, video_path):
		"""Set the video file to process"""
		self.video_path = video_path
		if self.video_cap is not None:
			self.video_cap.release()
		self.video_cap = cv2.VideoCapture(video_path)
	
	def process_frame(self, frame):
		"""Process a single frame with MediaPipe and detect gestures"""
		# Convert the image to RGB for MediaPipe
		image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
		image_rgb.flags.writeable = False
		
		# Process the image and detect pose
		results = self.pose.process(image_rgb)
		
		# Convert back to BGR for display
		image_rgb.flags.writeable = True
		image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
		
		# Create a blank image for pose visualization
		pose_frame = np.zeros(image_bgr.shape, dtype=np.uint8)
		
		# Draw pose landmarks if detected
		if results.pose_landmarks:
			self.mp_drawing.draw_landmarks(
				image_bgr,
				results.pose_landmarks,
				self.mp_pose.POSE_CONNECTIONS,
				landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style()
			)
			
			self.mp_drawing.draw_landmarks(
				pose_frame,
				results.pose_landmarks,
				self.mp_pose.POSE_CONNECTIONS,
				landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style()
			)
			
			# Store the pose data
			self.last_pose_data = results.pose_landmarks
			
			# Add to frame history for gesture detection
			self.add_to_frame_history(results.pose_landmarks)
			
			# Detect gestures
			gesture = self.detect_gesture()
		else:
			gesture = "idle"
		
		return image_bgr, pose_frame, gesture
	
	def get_next_video_frame(self):
		"""Get and process the next frame from the video file"""
		if self.video_cap is None or not self.video_cap.isOpened():
			return None, None, "idle", False
		
		success, frame = self.video_cap.read()
		if not success:
			return None, None, "idle", False
		
		processed_frame, pose_frame, gesture = self.process_frame(frame)
		return processed_frame, pose_frame, gesture, True
	
	def get_current_video_timestamp(self):
		"""Get the current timestamp of the video in milliseconds"""
		if self.video_cap is None:
			return 0
		return int(self.video_cap.get(cv2.CAP_PROP_POS_MSEC))
	
	def toggle_logging(self):
		"""Toggle data logging state"""
		self.is_logging = not self.is_logging
		return self.is_logging
	
	def add_to_frame_history(self, landmarks):
		"""Add landmarks to frame history for gesture detection"""
		self.frame_history.append(landmarks)
		if len(self.frame_history) > self.max_history:
			self.frame_history.pop(0)
	
	def detect_gesture(self):
		"""Detect gestures based on pose landmarks"""
		# Check if enough time has passed since the last gesture
		current_time = time.time() * 1000  # Current time in milliseconds
		if current_time - self.last_gesture_time < self.gesture_cooldown:
			return self.last_gesture
		
		# Check if we have enough frames for gesture detection
		if len(self.frame_history) < 3:
			return "idle"
		
		# THIS IS WHERE YOU WOULD CALL YOUR AI MODEL
		# -------------------------------------------
		# 1. Convert MediaPipe landmarks to format expected by your model
		keypoints_data = []
		for landmark in self.last_pose_data.landmark:
			keypoints_data.append({
				'x': landmark.x,
				'y': landmark.y,
				'z': landmark.z,
				'visibility': landmark.visibility
			})
		
		# 2. Call your AI model with the prepared data
		# gesture = your_ai_model.predict(keypoints_data)
		
		# 3. For now, we use the simple detection logic as a placeholder
		# The actual integration would replace this with your model's output
		
		# ... existing simple detection logic ...
		
		return self.last_gesture