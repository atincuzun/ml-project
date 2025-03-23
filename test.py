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
		
input = [
	("idle", 11),
	("g1", 11),
	("idle", 11),
	("g2", 110),
	("idle", 11),
	("g1", 11),
	("g2", 11),
	("idle", 11)
]
postprocessor = GesturePostProcessor()
events = []
for gesture, amount in input:
	for i in range(amount):
		events.append((gesture, postprocessor.process(gesture)))

import pprint
pprint.pprint(events)