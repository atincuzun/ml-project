import pandas as pd
import os
import yaml


class CSVDataHandler:
	def __init__(self):
		self.frame_list = []
		self.timestamps = []
		self.gestures = []
		
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
	
	def add_frame(self, pose_data, timestamp, gesture="idle"):
		"""Add a frame of pose data with timestamp and detected gesture"""
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
	
	def clear_data(self):
		"""Clear all collected data"""
		self.frame_list = []
		self.timestamps = []
		self.gestures = []
	
	def get_csv_data(self):
		"""Get the data in CSV format"""
		if not self.frame_list:
			return ""
		
		# Create DataFrame with pose data
		frames_df = pd.DataFrame(self.frame_list, columns=self.column_names, index=self.timestamps)
		frames_df.index.name = "timestamp"
		
		# Add gesture column
		frames_df['events'] = self.gestures
		
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
			
			# Add gesture column
			frames_df['events'] = self.gestures
			
			# Round to 5 decimal places and save
			frames_df.round(5).to_csv(output_path)
			return True
		
		except Exception as e:
			print(f"Error saving CSV: {e}")
			return False