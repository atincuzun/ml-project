import cv2
import mediapipe as mp
import numpy as np
import time
import yaml
import os
from utils.gesture_recognition import GestureRecognitionModel, GesturePostProcessor
from utils import data_handler


class MediapipeProcessor:
	def __init__(self):
		self.mp_drawing = mp.solutions.drawing_utils
		self.mp_drawing_styles = mp.solutions.drawing_styles
		self.mp_pose = mp.solutions.pose
		
		# Load app configuration
		from config import (
			MEDIAPIPE_MIN_DETECTION_CONFIDENCE,
			MEDIAPIPE_MIN_TRACKING_CONFIDENCE,
			SLIDING_WINDOW_STEP_SIZE,
			SLIDING_WINDOW_SIZE,
			EXCLUDED_LANDMARKS,
			POSTPROCESSING_REGISTERING_THRESHOLD_SIZE,
			POSTPROCESSING_REGISTERING_THRESHOLD_LIMIT,
			POSTPROCESSING_REGISTERED_THRESHOLD_SIZE,
			POSTPROCESSING_REGISTERED_THRESHOLD_LIMIT,
		)
		
		# Initialize the pose detector with appropriate parameters
		self.pose = self.mp_pose.Pose(
			min_detection_confidence=MEDIAPIPE_MIN_DETECTION_CONFIDENCE,
			min_tracking_confidence=MEDIAPIPE_MIN_TRACKING_CONFIDENCE,
			model_complexity=1,  # 0=Lite, 1=Full, 2=Heavy
			smooth_landmarks=True
		)
		
		# Gesture recognition parameters
		self.sliding_window_step_size = SLIDING_WINDOW_STEP_SIZE
		self.sliding_window_size = SLIDING_WINDOW_SIZE
		self.excluded_landmarks = EXCLUDED_LANDMARKS
		
		# Initialize gesture recognition model and post-processor
		self.gesture_model = GestureRecognitionModel()
		self.gesture_processor = GesturePostProcessor(
			registering_threshold_size=POSTPROCESSING_REGISTERING_THRESHOLD_SIZE,
			registering_threshold_limit=POSTPROCESSING_REGISTERING_THRESHOLD_LIMIT,
			registered_threshold_size=POSTPROCESSING_REGISTERED_THRESHOLD_SIZE,
			registered_threshold_limit=POSTPROCESSING_REGISTERED_THRESHOLD_LIMIT
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
		self.frame_count = 0  # Counter for sliding window step size
		
		# Sliding window for gesture detection
		self.sliding_window = []
		
		# Logging variables
		self.log_data = {
			"timestamp": [],
			"events": [],
			"gesture": [],
			"mode": [],
			"mode_percentage": []
		}
	
	def set_video_path(self, video_path):
		"""Set the video file to process with improved error handling"""
		try:
			# Clean up existing video capture if any
			if self.video_cap is not None:
				self.video_cap.release()
				self.video_cap = None
			
			# Open the video file - try with different backends if available
			self.video_path = video_path
			
			# Try different backends for better compatibility
			backends = [None]  # Default backend
			if os.name == 'nt':  # Windows
				backends.extend([cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_FFMPEG])
			else:  # Linux/Mac
				backends.append(cv2.CAP_FFMPEG)
			
			# Try each backend until one works
			for backend in backends:
				if backend is None:
					self.video_cap = cv2.VideoCapture(video_path)
				else:
					try:
						self.video_cap = cv2.VideoCapture(video_path, backend)
					except Exception:
						continue
				
				if self.video_cap is not None and self.video_cap.isOpened():
					print(f"Successfully opened video with backend: {backend}")
					break
			
			if self.video_cap is None or not self.video_cap.isOpened():
				raise Exception(f"Failed to open video: {video_path}")
			
			# Get video properties
			self.video_fps = self.video_cap.get(cv2.CAP_PROP_FPS)
			if self.video_fps <= 0 or self.video_fps > 120:  # Sanity check
				self.video_fps = 25.0  # Default to standard frame rate
				print(f"Invalid FPS detected, using default: {self.video_fps}")
			
			self.video_frame_count = int(self.video_cap.get(cv2.CAP_PROP_FRAME_COUNT))
			self.video_width = int(self.video_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
			self.video_height = int(self.video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
			
			# Calculate frame interval in milliseconds
			self.frame_interval = 1000.0 / self.video_fps
			self.last_frame_time = time.time() * 1000
			
			print(f"Loaded video: {video_path}")
			print(
				f"FPS: {self.video_fps}, Frames: {self.video_frame_count}, Resolution: {self.video_width}x{self.video_height}")
			print(f"Frame interval: {self.frame_interval:.2f} ms")
			
			# Reset for processing
			self.current_frame = 0
			self.frame_count = 0
			self.sliding_window = []
			self.gesture_processor.reset()
			self.log_data = {
				"timestamp": [],
				"events": [],
				"gesture": [],
				"mode": [],
				"mode_percentage": []
			}
			
			return True
		except Exception as e:
			print(f"Error loading video: {e}")
			import traceback
			traceback.print_exc()
			return False
	
	def process_frame(self, frame):
		"""Process a single frame with MediaPipe and detect gestures"""
		if frame is None:
			return None, None, "idle"
		
		try:
			# Get image dimensions to fix MediaPipe warning
			frame_height, frame_width = frame.shape[:2]
			
			# Convert the image to RGB for MediaPipe
			image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
			image_rgb.flags.writeable = False  # Performance optimization
			
			# Process the image and detect pose
			results = self.pose.process(image_rgb)
			
			# Make image writable again
			image_rgb.flags.writeable = True
			
			# Create a copy of the original frame for drawing
			processed_frame = frame.copy()
			
			# Create a blank image for pose visualization
			pose_frame = np.zeros(frame.shape, dtype=np.uint8)
			
			gesture = "idle"
			event = "idle"
			mode = "Registering"
			mode_percentage = 0.0
			
			# Draw pose landmarks if detected
			if results.pose_landmarks:
				# Draw landmarks on the frame - ONLY the landmarks
				self.mp_drawing.draw_landmarks(
					processed_frame,
					results.pose_landmarks,
					self.mp_pose.POSE_CONNECTIONS,
					landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style()
				)
				
				# Draw on pose frame too
				self.mp_drawing.draw_landmarks(
					pose_frame,
					results.pose_landmarks,
					self.mp_pose.POSE_CONNECTIONS,
					landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style()
				)
				
				# Store the pose data using the _convert_landmarks method
				landmarks_data = self._convert_landmarks(results.pose_landmarks)
				
				# Process for gesture recognition with sliding window
				self.frame_count += 1
				
				# Add to sliding window
				self.sliding_window.append(landmarks_data)
				
				# Keep sliding window to specified size
				if len(self.sliding_window) > self.sliding_window_size:
					self.sliding_window.pop(0)
				
				# Process for gesture recognition every step_size frames
				if self.frame_count % self.sliding_window_step_size == 0 and len(
						self.sliding_window) == self.sliding_window_size:
					# Get the recognized gesture from the model
					recognized_gesture = self.gesture_model.predict(self.sliding_window, self.excluded_landmarks)
					
					# Post-process the gesture
					event, mode, mode_percentage = self.gesture_processor.process(recognized_gesture)
					
					# Update current timestamp
					timestamp = self.current_frame if hasattr(self, 'current_frame') else self.frame_count
					
					# Check if we should update the gesture log based on GESTURE_LOG_STEP_SIZE
					# or if this is a non-idle event
					should_log = (
							             hasattr(self, 'gesture_log_step_size') and
							             timestamp % self.gesture_log_step_size == 0
					             ) or event != "idle"
					
					if should_log:
						# Add to the gesture log data
						self.log_data["timestamp"].append(timestamp)
						self.log_data["events"].append(event)
						self.log_data["gesture"].append(recognized_gesture)
						self.log_data["mode"].append(mode)
						self.log_data["mode_percentage"].append(mode_percentage)
					
					# Update the gesture if it's not idle
					if event != "idle":
						gesture = event
						self.last_gesture = gesture
						self.last_gesture_time = time.time() * 1000
			
			return processed_frame, pose_frame, gesture
		
		except Exception as e:
			print(f"Error processing frame: {e}")
			import traceback
			traceback.print_exc()
			# Return original frame on error with minimal error indicator
			try:
				error_frame = frame.copy()
				# Simple error indicator, no extensive text
				cv2.rectangle(error_frame, (0, 0), (20, 20), (0, 0, 255), -1)
				return error_frame, np.zeros(frame.shape, dtype=np.uint8), "idle"
			except:
				return None, None, "idle"
	
	def _convert_landmarks(self, pose_landmarks):
		"""
		Convert MediaPipe landmarks to the format expected by our gesture recognition model.
		Returns the landmarks in a format compatible with the Dataset._extract_features method.
		"""
		# Check if we have landmarks
		if pose_landmarks is None:
			return None
		
		# Create a frame data structure with the pose landmarks
		# No need to extract hand landmarks specifically, as the model was trained on pose landmarks
		frame_data = {
			'pose_landmarks': pose_landmarks  # Pass the entire pose_landmarks object
		}
		
		return frame_data
	
	def get_next_video_frame(self):
		"""Get and process the next frame with improved error handling"""
		if self.video_cap is None or not self.video_cap.isOpened():
			return None, None, "idle", False
		
		try:
			# Enforce frame rate timing
			current_time = time.time() * 1000  # Current time in milliseconds
			if hasattr(self, 'last_frame_time') and hasattr(self, 'frame_interval'):
				elapsed = current_time - self.last_frame_time
				if elapsed < self.frame_interval:
					# Not time for next frame yet
					time_to_wait = max(0, (self.frame_interval - elapsed) / 1000.0)
					if time_to_wait > 0:
						time.sleep(time_to_wait)
			
			# Update last frame time
			self.last_frame_time = time.time() * 1000
			
			# Read the next frame with timeout/retry
			max_retries = 3
			for attempt in range(max_retries):
				success, frame = self.video_cap.read()
				if success and frame is not None:
					break
				elif attempt < max_retries - 1:
					print(f"Frame read failed, retrying ({attempt + 1}/{max_retries})...")
					time.sleep(0.1)  # Short delay before retry
			
			if not success or frame is None:
				print("Failed to read frame after all retries")
				return None, None, "idle", False
			
			# Track current frame for progress reporting
			if hasattr(self, 'current_frame'):
				self.current_frame += 1
			
			# Process the frame with MediaPipe
			try:
				processed_frame, pose_frame, gesture = self.process_frame(frame)
			except Exception as e:
				print(f"Error processing frame: {e}")
				# Return original frame on processing error
				gesture = "idle"
				processed_frame = frame.copy()
				pose_frame = np.zeros(frame.shape, dtype=np.uint8)
			
			# Track last detected gesture
			if gesture != "idle":
				self.last_gesture = gesture
				self.last_gesture_time = time.time() * 1000
			
			# Add progress indicator to the frame if available
			if processed_frame is not None and hasattr(self, 'current_frame') and hasattr(self,
			                                                                              'video_frame_count') and self.video_frame_count > 0:
				progress = self.current_frame / self.video_frame_count
				
				# Draw progress bar at the bottom of the frame
				bar_height = 5  # 5 pixels height
				progress_width = int(processed_frame.shape[1] * progress)
				
				try:
					processed_frame = cv2.rectangle(
						processed_frame,
						(0, processed_frame.shape[0] - bar_height),
						(progress_width, processed_frame.shape[0]),
						(0, 255, 0),
						-1
					)
					
					# Add frame count text
					text = f"Frame: {self.current_frame}/{self.video_frame_count}"
					cv2.putText(
						processed_frame,
						text,
						(10, 30),
						cv2.FONT_HERSHEY_SIMPLEX,
						0.7,
						(0, 255, 0),
						2
					)
				except Exception as e:
					print(f"Error adding progress indicator: {e}")
			
			return processed_frame, pose_frame, gesture, True
		
		except Exception as e:
			print(f"Error in get_next_video_frame: {e}")
			import traceback
			traceback.print_exc()
			return None, None, "idle", False
	
	def get_current_video_timestamp(self):
		"""Get the current timestamp of the video in milliseconds"""
		if self.video_cap is None:
			return 0
		return int(self.video_cap.get(cv2.CAP_PROP_POS_MSEC))
	
	def toggle_logging(self):
		"""Toggle data logging state and prepare CSV data"""
		self.is_logging = not self.is_logging
		
		# If stopping logging, prepare CSV data
		if not self.is_logging:
			# Get the log data as CSV string
			# This should now be handled by the data_handler
			return self.is_logging
		else:
			# If starting logging, clear previous data
			if hasattr(self, 'log_data'):
				self.log_data = {
					"timestamp": [],
					"events": [],
					"gesture": [],
					"mode": [],
					"mode_percentage": []
				}
			return self.is_logging
	
	def get_csv_data(self):
		"""Get the log data as CSV string"""
		if hasattr(self, 'csv_data'):
			return self.csv_data
		return ""
	
	def detect_gesture(self):
		"""Legacy method for gesture detection, now uses the sliding window approach"""
		# This is kept for backward compatibility
		return self.last_gesture