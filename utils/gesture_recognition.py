import numpy as np
from typing import List, Dict, Union, Optional, Tuple
import os
import pickle

# Import the neural network
try:
	from Network.GestureClassificationNetwork import GestureClassificationNetwork
	
	print("Successfully imported GestureClassificationNetwork")
except ImportError as e:
	print(f"Warning: Could not import GestureClassificationNetwork. Error: {e}")
	GestureClassificationNetwork = None


class GestureRecognitionModel:
	"""
	AI model for gesture recognition using pre-trained neural network.

	This class loads the pre-trained neural network model for gesture recognition
	and handles the preprocessing and postprocessing of input data.
	"""
	
	def __init__(self, threshold: float = 0.6, model_path: str = None, mapping_path: str = None):
		"""
		Initialize the gesture recognition model.

		Args:
			threshold: Confidence threshold for gesture detection
			model_path: Path to the pre-trained model
			mapping_path: Path to the gesture mapping file
		"""
		# Import config for model paths
		try:
			from config import GESTURE_MODEL_PATH, GESTURE_MAPPING_PATH
			if model_path is None:
				model_path = GESTURE_MODEL_PATH
			if mapping_path is None:
				mapping_path = GESTURE_MAPPING_PATH
			print(f"Using model path from config: {model_path}")
		except (ImportError, AttributeError):
			print("Warning: Could not import model paths from config")
			if model_path is None:
				model_path = "gesture_model.npy"  # Default fallback
		
		self.threshold = threshold
		self.gestures = ["idle"]  # Default gesture list
		self.model = None
		self.model_loaded = False
		self.frame_buffer = []
		self.buffer_size = 10  # Size of sliding window
		self.gesture_mapping = {}
		
		# Try to load gesture mapping
		if mapping_path is None:
			mapping_path = os.path.splitext(model_path)[0] + "_mapping.npy"
		
		if os.path.exists(mapping_path):
			try:
				self.gesture_mapping = np.load(mapping_path, allow_pickle=True).item()
				print(f"Loaded gesture mapping from {mapping_path}: {self.gesture_mapping}")
				
				# Update gestures list from mapping
				if self.gesture_mapping:
					# Create reverse mapping from index to gesture name
					idx_to_gesture = {idx: gesture for gesture, idx in self.gesture_mapping.items()}
					self.gestures = [idx_to_gesture.get(i, "idle") for i in range(len(self.gesture_mapping))]
					print(f"Using gestures from mapping: {self.gestures}")
			except Exception as e:
				print(f"Error loading gesture mapping: {e}")
				import traceback
				traceback.print_exc()
		
		# Try to load the pre-trained model
		try:
			if GestureClassificationNetwork is not None:
				print(f"Trying to load model from {model_path}")
				
				# Check if model file exists
				if os.path.exists(model_path):
					print(f"Model file exists at {model_path}")
					
					# Expected input size: 33 landmarks × 4 values × window_size + window_size presence flags
					input_size = (33 * 4 * self.buffer_size) + self.buffer_size
					print(f"Using input size: {input_size}")
					
					# Initialize the neural network with proper number of gesture classes
					num_gestures = len(self.gestures)
					self.model = GestureClassificationNetwork(input_size, num_gestures)
					print(f"Created network instance with {num_gestures} gesture classes")
					
					# Load pre-trained weights
					try:
						self.model.load_model(model_path)
						self.model_loaded = True
						print(f"Successfully loaded gesture recognition model from {model_path}")
					except Exception as model_load_error:
						print(f"Error loading model weights: {model_load_error}")
						import traceback
						traceback.print_exc()
				else:
					print(f"Model file {model_path} not found. Model will not work.")
					print("Keyboard gesture simulation will be available.")
			else:
				print("GestureClassificationNetwork module not available. Model will not work.")
				print("Keyboard gesture simulation will be available.")
		except Exception as e:
			print(f"Error initializing gesture recognition model: {e}")
			import traceback
			traceback.print_exc()
	
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
		# Check if window is empty
		if not window or len(window) == 0:
			return None
		
		# Prepare array for landmarks
		# Pose: 33 landmarks with x,y,z,visibility (4 values per landmark)
		pose_data = np.zeros((len(window), 33, 4))
		
		# Presence flag for each frame
		pose_present = np.ones(len(window))  # Default to 1 as requested
		
		# Extract landmark data for each frame
		valid_landmarks = False
		for i, frame in enumerate(window):
			# Check different possible formats of input data
			if hasattr(frame, 'landmark'):
				# MediaPipe pose_landmarks format
				landmarks = frame.landmark
				if landmarks and len(landmarks) > 0:
					valid_landmarks = True
					for j, landmark in enumerate(landmarks):
						if j < 33:  # Ensure we stay within array bounds
							pose_data[i, j, 0] = landmark.x
							pose_data[i, j, 1] = landmark.y
							pose_data[i, j, 2] = landmark.z
							pose_data[i, j, 3] = landmark.visibility
			
			elif isinstance(frame, dict) and 'pose_landmarks' in frame and frame['pose_landmarks']:
				# Format where pose_landmarks is stored in a dict
				landmarks = frame['pose_landmarks'].landmark
				if landmarks and len(landmarks) > 0:
					valid_landmarks = True
					for j, landmark in enumerate(landmarks):
						if j < 33:  # Ensure we stay within array bounds
							pose_data[i, j, 0] = landmark.x
							pose_data[i, j, 1] = landmark.y
							pose_data[i, j, 2] = landmark.z
							pose_data[i, j, 3] = landmark.visibility
			
			elif isinstance(frame, list) and len(frame) > 0 and all(isinstance(lm, dict) for lm in frame):
				# List of landmark dictionaries
				landmarks = frame
				if landmarks and len(landmarks) > 0:
					valid_landmarks = True
					for j, landmark in enumerate(landmarks):
						if j < 33 and 'x' in landmark and 'y' in landmark and 'z' in landmark:
							pose_data[i, j, 0] = landmark['x']
							pose_data[i, j, 1] = landmark['y']
							pose_data[i, j, 2] = landmark['z']
							pose_data[i, j, 3] = landmark.get('visibility', 1.0)
		
		if not valid_landmarks:
			return None
		
		# Flatten everything to match the exact feature extraction in Dataset.py
		features = np.concatenate([
			pose_data.flatten(),  # All pose landmarks across time (33*4*window_size)
			pose_present.flatten()  # Pose presence flags (window_size)
		])
		
		return features
	
	def predict(self, landmarks_window, excluded_landmarks=None):
		"""
		Predict the gesture from the given window of landmarks.

		Args:
			landmarks_window: List of frames, each containing MediaPipe pose landmarks
			excluded_landmarks: List of landmark names to exclude (unused)

		Returns:
			Predicted gesture as string
		"""
		# If model not loaded, return idle
		if not self.model_loaded or self.model is None:
			print("Model not loaded, returning idle")
			return "idle"
		
		try:
			# If landmarks window is empty, return idle
			if not landmarks_window:
				print("Empty landmarks window, returning idle")
				return "idle"
			
			# Update frame buffer with new frames
			self.frame_buffer.extend(landmarks_window)
			
			# Keep buffer at correct size
			if len(self.frame_buffer) > self.buffer_size:
				self.frame_buffer = self.frame_buffer[-self.buffer_size:]
			
			# If buffer not full yet, return idle
			if len(self.frame_buffer) < self.buffer_size:
				print(f"Buffer not full yet ({len(self.frame_buffer)}/{self.buffer_size}), returning idle")
				return "idle"
			
			# Extract features using the same method as during training in Dataset.py
			features = self._extract_features(self.frame_buffer)
			
			if features is None:
				print("No valid features extracted, returning idle")
				return "idle"
			
			# Print shape for debugging
			print(f"Feature shape: {features.shape}")
			
			# Reshape for model input
			features = features.reshape(1, -1)
			
			# Predict using the model
			if self.model is not None:
				try:
					# Get the predicted class from the model
					predicted_class = self.model.predict(features)[0]
					
					# Map class index to gesture name
					if 0 <= predicted_class < len(self.gestures):
						gesture = self.gestures[predicted_class]
						print(f"Predicted gesture: {gesture} (class {predicted_class})")
						return gesture
					else:
						print(f"Invalid class index: {predicted_class}, returning idle")
				except Exception as pred_error:
					print(f"Error during model prediction: {pred_error}")
					import traceback
					traceback.print_exc()
			
			# Default to idle if something goes wrong
			return "idle"
		
		except Exception as e:
			print(f"Error in predict function: {e}")
			import traceback
			traceback.print_exc()
			return "idle"


class GesturePostProcessor:
	"""
	Post-processor for gesture recognition with two modes:
	- Registering: Looking for a consistent gesture to register
	- Registered: Monitoring if a registered gesture is still present
	"""
	
	def __init__(self,
	             registering_threshold_size: int = 10,
	             registering_threshold_limit: int = 8,
	             registered_threshold_size: int = 10,
	             registered_threshold_limit: int = 6):
		"""
		Initialize the post-processor.

		Args:
			registering_threshold_size: Size of buffer for registering mode
			registering_threshold_limit: Minimum occurrences to register a gesture
			registered_threshold_size: Size of buffer for registered mode
			registered_threshold_limit: Maximum occurrences to remain in registered mode
		"""
		self.registering_threshold_size = registering_threshold_size
		self.registering_threshold_limit = registering_threshold_limit
		self.registered_threshold_size = registered_threshold_size
		self.registered_threshold_limit = registered_threshold_limit
		
		self.recognized_gestures = []  # Buffer of recognized gestures
		self.mode = "Registering"  # Current mode (Registering or Registered)
		self.registered_gesture = None  # Currently registered gesture
	
	def process(self, gesture: str) -> tuple:
		"""
		Process a recognized gesture through the post-processing system.

		Args:
			gesture: The recognized gesture from the model

		Returns:
			Tuple of (event, mode, mode_percentage)
			- event: The event to register ("idle" or a gesture name)
			- mode: Current mode ("Registering" or "Registered")
			- mode_percentage: Percentage indicating recognition strength
		"""
		
		# Add the gesture to the buffer
		self.recognized_gestures.append(gesture)
		
		# Keep buffer size limited
		if self.mode == "Registering":
			max_size = self.registering_threshold_size
		else:  # Registered mode
			max_size = self.registered_threshold_size
		
		# Trim buffer if necessary
		while len(self.recognized_gestures) > max_size:
			self.recognized_gestures.pop(0)
		
		# Process based on current mode
		if self.mode == "Registering":
			return self._process_registering_mode()
		else:  # Registered mode
			return self._process_registered_mode()
	
	def _process_registering_mode(self) -> tuple:
		"""Process gestures in Registering mode."""
		# Count occurrences of each gesture in the buffer
		gesture_counts = {}
		for g in self.recognized_gestures:
			if g not in gesture_counts:
				gesture_counts[g] = 0
			gesture_counts[g] += 1
		
		# Find the most frequent gesture
		max_gesture = "idle"
		max_count = 0
		for gesture, count in gesture_counts.items():
			if gesture != "idle" and count > max_count:
				max_gesture = gesture
				max_count = count
		
		# Calculate percentage
		buffer_size = len(self.recognized_gestures)
		percentage = (max_count / buffer_size) * 100 if buffer_size > 0 else 0
		
		# Check if we should register this gesture
		event = "idle"
		if max_count >= self.registering_threshold_limit:
			# Register the gesture and switch to Registered mode
			self.registered_gesture = max_gesture
			self.mode = "Registered"
			event = max_gesture
		
		return event, "Registering", percentage
	
	def _process_registered_mode(self) -> tuple:
		"""Process gestures in Registered mode."""
		if not self.registered_gesture:
			# Something went wrong, reset to Registering mode
			self.mode = "Registering"
			return "idle", "Registering", 0.0
		
		# Count occurrences of the registered gesture
		registered_count = self.recognized_gestures.count(self.registered_gesture)
		
		# Calculate percentage
		buffer_size = len(self.recognized_gestures)
		percentage = (registered_count / buffer_size) * 100 if buffer_size > 0 else 0
		
		# Check if we should stay in Registered mode
		if registered_count <= self.registered_threshold_limit:
			# Switch back to Registering mode
			self.mode = "Registering"
			self.registered_gesture = None
		
		return "idle", "Registered", percentage
	
	def reset(self):
		"""Reset the post-processor to initial state."""
		self.recognized_gestures = []
		self.mode = "Registering"
		self.registered_gesture = None