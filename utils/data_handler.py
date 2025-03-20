import pandas as pd
import os
import yaml


class CSVDataHandler:
	def __init__(self):
		self.frame_list = []
		self.timestamps = []
		self.gestures = []
		self.events = []
		self.modes = []
		self.mode_percentages = []
		
		# Load keypoint mapping
		script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
		mapping_path = os.path.join(script_dir, 'keypoint_mapping.yml')
		
		if os.path.exists(mapping_path):
			self.column_names = self.load_keypoint_mapping_from_file(mapping_path)
		else:
			# Create default column names if mapping file doesn't exist
			self.column_names = []
			print(f"Warning: Keypoint mapping file not found at {mapping_path}")
	
	def load_keypoint_mapping_from_file(self, file):
		"""Load keypoint mapping from YAML file"""
		with open(file, "r") as yaml_file:
			mappings = yaml.safe_load(yaml_file)
			keypoint_names = mappings["face"]
			keypoint_names += mappings["body"]
		
		return ["%s_%s" % (joint_name, jdn) for joint_name in keypoint_names
		        for jdn in ["x", "y", "z", "confidence"]]
	
	def add_frame(self, pose_data, timestamp, gesture="idle", event="idle", mode="Registering", mode_percentage=0.0):
		"""
		Add a frame of pose data with timestamp and detected gesture information

		Args:
			pose_data: MediaPipe pose landmark data
			timestamp: Frame timestamp
			gesture: Recognized gesture
			event: Event that was registered (from switching modes)
			mode: Current processing mode (Registering or Registered)
			mode_percentage: Percentage of recognition confidence
		"""
		if pose_data is None:
			return
		
		frame = []
		for i in range(33):  # MediaPipe poses have 33 landmarks
			frame.append(pose_data.landmark[i].x)
			frame.append(pose_data.landmark[i].y)
			frame.append(pose_data.landmark[i].z)
			frame.append(pose_data.landmark[i].visibility)
		
		self.frame_list.append(frame)
		self.timestamps.append(timestamp)
		self.gestures.append(gesture)
		self.events.append(event)
		self.modes.append(mode)
		self.mode_percentages.append(mode_percentage)
	
	def clear_data(self):
		"""Clear all collected data"""
		self.frame_list = []
		self.timestamps = []
		self.gestures = []
		self.events = []
		self.modes = []
		self.mode_percentages = []
	
	def get_csv_data(self):
		"""Get the data in CSV format"""
		if not self.frame_list:
			return "timestamp,events,gesture,mode,mode_percentage\n"
		
		# Create DataFrame with pose data
		frames_df = pd.DataFrame(self.frame_list, columns=self.column_names, index=self.timestamps)
		frames_df.index.name = "timestamp"
		
		# Add additional columns
		frames_df['events'] = self.events
		frames_df['gesture'] = self.gestures
		frames_df['mode'] = self.modes
		frames_df['mode_percentage'] = self.mode_percentages
		
		# Convert to CSV string
		csv_data = frames_df.to_csv()
		return csv_data
	
	def save_to_csv(self, output_path):
		"""Save data to CSV file"""
		try:
			# Create directory if it doesn't exist
			os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
			
			# Create DataFrame
			frames_df = pd.DataFrame(self.frame_list, columns=self.column_names, index=self.timestamps)
			frames_df.index.name = "timestamp"
			
			# Add additional columns
			frames_df['events'] = self.events
			frames_df['gesture'] = self.gestures
			frames_df['mode'] = self.modes
			frames_df['mode_percentage'] = self.mode_percentages
			
			# Round to 5 decimal places and save
			frames_df.round(5).to_csv(output_path)
			return True
		
		except Exception as e:
			print(f"Error saving CSV: {e}")
			return False