#!/usr/bin/env python3
"""
Gesture Log Processor

This script processes CSV files containing MediaPipe landmarks to generate gesture logs.
It uses the same gesture recognition model as the main application.

Usage:
    python gesture_log_processor.py --input input_file.csv --output output_dir

The script will output two files:
1. gesture_log.csv - Full gesture log with all columns
2. performance_results.csv - Only events column for performance evaluation
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
import yaml
import time
import mediapipe as mp

# Add parent directory to path to import from utils
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
sys.path.append(parent_dir)

try:
	from utils.gesture_recognition import GestureRecognitionModel, GesturePostProcessor
except ImportError:
	print("Error: Could not import from utils. Make sure utils directory exists.")
	print(f"Current working directory: {os.getcwd()}")
	print(f"Script directory: {script_dir}")
	print(f"Parent directory: {parent_dir}")
	sys.exit(1)


def load_keypoint_mapping(mapping_file):
	"""Load keypoint mapping from YAML file."""
	try:
		with open(mapping_file, "r") as yaml_file:
			mappings = yaml.safe_load(yaml_file)
			keypoint_names = mappings["face"]
			keypoint_names += mappings["body"]
		
		# Generate column names for landmarks
		columns = []
		for joint_name in keypoint_names:
			for component in ["x", "y", "z", "confidence"]:
				columns.append(f"{joint_name}_{component}")
		
		return columns
	except Exception as e:
		print(f"Error loading keypoint mapping: {e}")
		sys.exit(1)


def preprocess_csv(input_file, output_dir, exclude_landmarks=None):
	"""
	Preprocess the CSV file by filtering out excluded landmarks.

	Args:
		input_file: Path to input CSV file
		output_dir: Directory to save augmented file
		exclude_landmarks: List of landmark names to exclude

	Returns:
		Path to augmented CSV file
	"""
	print(f"Preprocessing {input_file}...")
	
	# Create output directory if it doesn't exist
	os.makedirs(output_dir, exist_ok=True)
	
	# Output file path
	augmented_path = os.path.join(output_dir, "augmented_data.csv")
	
	try:
		# Load the CSV file
		df = pd.read_csv(input_file)
		print(f"Loaded {len(df)} rows from {input_file}")
		
		# If no exclusion is needed, just save a copy
		if not exclude_landmarks:
			df.to_csv(augmented_path, index=False)
			print(f"No landmarks to exclude, saved copy to {augmented_path}")
			return augmented_path
		
		# Load keypoint mapping
		mapping_file = os.path.join(parent_dir, "keypoint_mapping.yml")
		if not os.path.exists(mapping_file):
			print(f"Error: Keypoint mapping file not found at {mapping_file}")
			sys.exit(1)
		
		columns = load_keypoint_mapping(mapping_file)
		
		# Filter columns to exclude specified landmarks
		if exclude_landmarks:
			filtered_columns = []
			for col in df.columns:
				# Check if this column belongs to an excluded landmark
				if not any(col.startswith(f"{landmark}_") for landmark in exclude_landmarks):
					filtered_columns.append(col)
			
			# Create new DataFrame with filtered columns
			filtered_df = df[filtered_columns]
			filtered_df.to_csv(augmented_path, index=False)
			print(f"Saved augmented data with {len(filtered_columns)} columns to {augmented_path}")
			
			return augmented_path
		else:
			# No filtering needed
			df.to_csv(augmented_path, index=False)
			return augmented_path
	
	except Exception as e:
		print(f"Error preprocessing CSV: {e}")
		sys.exit(1)


def extract_landmarks_from_row(row, column_names):
	"""
	Extract landmark data from a row in the DataFrame.

	Args:
		row: Row from DataFrame
		column_names: List of column names for landmarks

	Returns:
		List of landmarks in format expected by the gesture recognition model
	"""
	# Create list of landmarks
	landmarks = []
	
	# Process landmark columns in groups of 4 (x, y, z, confidence)
	landmark_index = 0
	for i in range(0, len(column_names), 4):
		if i + 3 < len(column_names):
			try:
				# Get column names for this landmark
				x_col = column_names[i]
				y_col = column_names[i + 1]
				z_col = column_names[i + 2]
				conf_col = column_names[i + 3]
				
				# Check if columns exist in the row
				if all(col in row.index for col in [x_col, y_col, z_col, conf_col]):
					landmark = {
						'x': float(row[x_col]),
						'y': float(row[y_col]),
						'z': float(row[z_col]),
						'visibility': float(row[conf_col])
					}
					landmarks.append(landmark)
					landmark_index += 1
			except Exception as e:
				print(f"Error extracting landmark {landmark_index}: {e}")
				# Add placeholder with zeros
				landmarks.append({
					'x': 0.0,
					'y': 0.0,
					'z': 0.0,
					'visibility': 0.0
				})
				landmark_index += 1
	
	return landmarks


def process_csv(input_file, output_dir, window_size=10, step_size=1):
	"""
	Process the CSV file to detect gestures and generate logs.

	Args:
		input_file: Path to input CSV file
		output_dir: Directory to save output files
		window_size: Size of sliding window for gesture detection
		step_size: Step size for sliding window

	Returns:
		Tuple of paths to gesture_log.csv and performance_results.csv
	"""
	print(f"Processing {input_file} with window size {window_size} and step size {step_size}...")
	
	# Ensure output directory exists
	os.makedirs(output_dir, exist_ok=True)
	
	# Output file paths
	gesture_log_path = os.path.join(output_dir, "gesture_log.csv")
	performance_results_path = os.path.join(output_dir, "performance_results.csv")
	
	try:
		# Get keypoint column names
		mapping_file = os.path.join(parent_dir, "keypoint_mapping.yml")
		if not os.path.exists(mapping_file):
			print(f"Error: Keypoint mapping file not found at {mapping_file}")
			sys.exit(1)
		
		column_names = load_keypoint_mapping(mapping_file)
		
		# Load the CSV file
		df = pd.read_csv(input_file)
		print(f"Loaded {len(df)} rows from {input_file}")
		
		# Make sure we have a timestamp column
		if 'timestamp' not in df.columns:
			print("No timestamp column found, using row index as timestamp")
			df['timestamp'] = df.index
		
		# Initialize gesture recognition model from model file
		model_path = os.path.join(parent_dir, "gesture_model.npy")
		gesture_model = GestureRecognitionModel(model_path=model_path)
		
		# Initialize post-processor
		gesture_processor = GesturePostProcessor(
			registering_threshold_size=10,
			registering_threshold_limit=8,
			registered_threshold_size=10,
			registered_threshold_limit=6
		)
		
		# Initialize result DataFrame
		result_df = pd.DataFrame({
			'timestamp': df['timestamp'],
			'events': 'idle',
			'gesture': 'idle',
			'mode': 'Registering',
			'mode_percentage': 0.0
		})
		
		# Process the data using sliding window
		sliding_window = []
		total_frames = len(df)
		start_time = time.time()
		
		for i, (_, row) in enumerate(df.iterrows()):
			# Extract landmarks
			landmarks = extract_landmarks_from_row(row, column_names)
			
			# Add to sliding window
			sliding_window.append(landmarks)
			
			# Keep sliding window at specified size
			if len(sliding_window) > window_size:
				sliding_window.pop(0)
			
			# Process window when full and at step intervals
			if len(sliding_window) == window_size and i % step_size == 0:
				# Make prediction
				recognized_gesture = gesture_model.predict(sliding_window)
				
				# Process the gesture
				event, mode, mode_percentage = gesture_processor.process(recognized_gesture)
				
				# Store results
				result_df.at[i, 'events'] = event
				result_df.at[i, 'gesture'] = recognized_gesture
				result_df.at[i, 'mode'] = mode
				result_df.at[i, 'mode_percentage'] = mode_percentage
			
			# Show progress
			if i % 100 == 0 or i == total_frames - 1:
				progress = (i + 1) / total_frames * 100
				elapsed = time.time() - start_time
				frames_per_sec = (i + 1) / elapsed if elapsed > 0 else 0
				eta = (total_frames - i - 1) / frames_per_sec if frames_per_sec > 0 else 0
				
				sys.stdout.write(f"\rProgress: {progress:.1f}% ({i + 1}/{total_frames}) | "
				                 f"Speed: {frames_per_sec:.1f} frames/s | "
				                 f"ETA: {eta:.1f}s")
				sys.stdout.flush()
		
		print("\nProcessing complete!")
		
		# Save gesture log (complete results)
		result_df.to_csv(gesture_log_path, index=False)
		print(f"Saved complete gesture log to {gesture_log_path}")
		
		# Save performance results (events column only)
		performance_df = pd.DataFrame({
			'timestamp': df['timestamp'],
			'events': result_df['events']
		})
		performance_df.to_csv(performance_results_path, index=False)
		print(f"Saved performance results to {performance_results_path}")
		
		return gesture_log_path, performance_results_path
	
	except Exception as e:
		print(f"Error processing CSV: {e}")
		import traceback
		traceback.print_exc()
		sys.exit(1)


def main():
	"""Main function to parse arguments and run the script."""
	parser = argparse.ArgumentParser(
		description="Process CSV files with MediaPipe landmark data to generate gesture logs.")
	parser.add_argument("--input", required=True, help="Path to input CSV file with landmark data")
	parser.add_argument("--output", default="./output", help="Directory to save output files")
	parser.add_argument("--window", type=int, default=10, help="Size of sliding window for gesture detection")
	parser.add_argument("--step", type=int, default=1, help="Step size for sliding window")
	parser.add_argument("--exclude", nargs='+', help="List of landmarks to exclude (e.g. nose left_eye)")
	
	args = parser.parse_args()
	
	if not os.path.exists(args.input):
		print(f"Error: Input file {args.input} not found")
		sys.exit(1)
	
	print(f"Gesture Log Processor")
	print(f"====================")
	print(f"Input file: {args.input}")
	print(f"Output directory: {args.output}")
	print(f"Window size: {args.window}")
	print(f"Step size: {args.step}")
	print(f"Excluded landmarks: {args.exclude or 'None'}")
	print("")
	
	# First preprocess the data to filter out excluded landmarks
	augmented_file = preprocess_csv(args.input, args.output, args.exclude)
	
	# Then process the augmented file to generate gesture logs
	gesture_log, performance_results = process_csv(
		augmented_file,
		args.output,
		window_size=args.window,
		step_size=args.step
	)
	
	print("\nProcessing completed successfully!")
	print(f"Gesture log: {gesture_log}")
	print(f"Performance results: {performance_results}")


if __name__ == "__main__":
	main()