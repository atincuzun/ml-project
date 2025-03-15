#!/usr/bin/env python3
"""
Data Processing Utility for Gesture Recognition

This script processes exported annotation data to prepare it for model training:
1. Converts CSV annotation files to appropriate training format
2. Calculates velocity features
3. Normalizes landmark data
4. Splits data into training and testing sets
5. Saves processed data in numpy format

Usage:
python data_processor.py input_csv output_dir [--split 0.8] [--smooth 3]
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Define gesture mapping
GESTURE_MAP = {
	"idle": 0,
	"swipe_left": 1,
	"swipe_right": 2,
	"rotate_cw": 3,
	"rotate_ccw": 4
}


def load_annotation_data(csv_path):
	"""Load and validate annotation data from CSV"""
	try:
		data = pd.read_csv(csv_path)
		
		# Check required columns
		required_cols = ["frame", "gesture"]
		if not all(col in data.columns for col in required_cols):
			print(f"Error: CSV file must contain columns: {', '.join(required_cols)}")
			return None
		
		# Check if there are any landmarks
		landmark_cols = [col for col in data.columns if col.startswith("landmark_")]
		if not landmark_cols:
			print("Error: No landmark data found in CSV")
			return None
		
		print(f"Loaded {len(data)} frames with {len(landmark_cols)} landmark features")
		return data
	
	except Exception as e:
		print(f"Error loading CSV file: {e}")
		return None


def preprocess_data(data, smoothing_window=0):
	"""Preprocess the annotation data"""
	# Sort by frame number
	data = data.sort_values("frame")
	
	# Extract landmark columns
	landmark_cols = [col for col in data.columns if col.startswith("landmark_")]
	landmark_data = data[landmark_cols].values
	
	# Apply smoothing if requested
	if smoothing_window > 0:
		# Simple moving average
		kernel = np.ones(smoothing_window) / smoothing_window
		# Apply smoothing to each landmark dimension
		smoothed_data = np.zeros_like(landmark_data)
		for i in range(landmark_data.shape[1]):
			smoothed_data[:, i] = np.convolve(landmark_data[:, i], kernel, mode='same')
		landmark_data = smoothed_data
	
	# Extract gesture labels
	labels = data["gesture"].map(GESTURE_MAP).values
	
	return landmark_data, labels


def calculate_velocity_features(landmark_data):
	"""Calculate velocity features from landmark positions"""
	# Velocity is difference between consecutive frames
	velocity = np.zeros_like(landmark_data)
	velocity[1:] = landmark_data[1:] - landmark_data[:-1]
	
	# First frame has zero velocity
	velocity[0] = 0
	
	return velocity


def normalize_data(position_features, velocity_features):
	"""Normalize position and velocity features"""
	# Combine for consistent scaling
	combined = np.hstack([position_features, velocity_features])
	
	# Apply standard scaling
	scaler = StandardScaler()
	normalized = scaler.fit_transform(combined)
	
	# Split back into position and velocity
	n_features = position_features.shape[1]
	normalized_pos = normalized[:, :n_features]
	normalized_vel = normalized[:, n_features:]
	
	return normalized_pos, normalized_vel, scaler


def split_train_test(features, labels, test_size=0.2, random_state=42):
	"""Split data into training and testing sets"""
	return train_test_split(
		features, labels,
		test_size=test_size,
		random_state=random_state,
		stratify=labels  # Ensure proportional representation of gestures
	)


def save_processed_data(output_dir, X_train, X_test, y_train, y_test):
	"""Save processed data to output directory"""
	os.makedirs(output_dir, exist_ok=True)
	
	# Save as numpy files
	np.save(os.path.join(output_dir, "X_train.npy"), X_train)
	np.save(os.path.join(output_dir, "X_test.npy"), X_test)
	np.save(os.path.join(output_dir, "y_train.npy"), y_train)
	np.save(os.path.join(output_dir, "y_test.npy"), y_test)
	
	print(f"Data saved to {output_dir}")


def visualize_data(output_dir, position_features, velocity_features, labels):
	"""Generate visualization of the data"""
	os.makedirs(output_dir, exist_ok=True)
	
	# Create class distribution plot
	plt.figure(figsize=(10, 6))
	gesture_names = {v: k for k, v in GESTURE_MAP.items()}
	counts = np.bincount(labels)
	plt.bar(range(len(counts)), counts)
	plt.xticks(range(len(counts)), [gesture_names.get(i, f"Unknown-{i}") for i in range(len(counts))])
	plt.title("Gesture Class Distribution")
	plt.ylabel("Number of Samples")
	plt.savefig(os.path.join(output_dir, "class_distribution.png"))
	
	# Create feature visualization
	plt.figure(figsize=(12, 8))
	
	# Plot a sample of position features (first 10 dimensions)
	plt.subplot(2, 1, 1)
	plt.imshow(position_features[:100, :10].T, aspect='auto', cmap='viridis')
	plt.colorbar()
	plt.title("Position Features (First 10 Dimensions)")
	plt.xlabel("Frame")
	plt.ylabel("Feature Dimension")
	
	# Plot a sample of velocity features
	plt.subplot(2, 1, 2)
	plt.imshow(velocity_features[:100, :10].T, aspect='auto', cmap='viridis')
	plt.colorbar()
	plt.title("Velocity Features (First 10 Dimensions)")
	plt.xlabel("Frame")
	plt.ylabel("Feature Dimension")
	
	plt.tight_layout()
	plt.savefig(os.path.join(output_dir, "feature_visualization.png"))
	
	# Create an example sequence plot
	plt.figure(figsize=(14, 8))
	
	# Choose a few representative landmarks
	landmark_indices = [0, 4, 8, 12]  # For example, landmarks for key body parts
	
	for i, idx in enumerate(landmark_indices):
		plt.subplot(len(landmark_indices), 1, i + 1)
		plt.plot(position_features[:100, idx])
		plt.ylabel(f"Landmark {idx // 4}-{idx % 4}")
		
		# Mark the gestures with different colors
		for gesture_id in set(labels[:100]):
			if gesture_id == 0:  # Skip idle
				continue
			
			gesture_frames = np.where(labels[:100] == gesture_id)[0]
			if len(gesture_frames) > 0:
				plt.scatter(gesture_frames, position_features[gesture_frames, idx],
				            label=gesture_names.get(gesture_id, f"Unknown-{gesture_id}"))
	
	plt.legend()
	plt.tight_layout()
	plt.savefig(os.path.join(output_dir, "landmark_sequence.png"))
	
	plt.close('all')
	print(f"Visualizations saved to {output_dir}")


def write_metadata(output_dir, data_shape, num_samples, split_ratio):
	"""Write metadata about the processed dataset"""
	with open(os.path.join(output_dir, "metadata.txt"), "w") as f:
		f.write(f"Total samples: {num_samples}\n")
		f.write(f"Feature shape: {data_shape}\n")
		f.write(f"Train/test split: {split_ratio}\n")
		f.write(f"Gesture mapping: {GESTURE_MAP}\n")
		f.write("\nClass distribution:\n")
		
		# Count samples per class
		for gesture_name, gesture_id in GESTURE_MAP.items():
			count = np.sum(labels == gesture_id)
			f.write(f"  {gesture_name}: {count} samples\n")


def main():
	parser = argparse.ArgumentParser(description="Process gesture annotation data for model training")
	parser.add_argument("input_csv", help="Input CSV file with annotation data")
	parser.add_argument("output_dir", help="Output directory for processed data")
	parser.add_argument("--split", type=float, default=0.8, help="Training set proportion (default: 0.8)")
	parser.add_argument("--smooth", type=int, default=0, help="Smoothing window size (default: 0, no smoothing)")
	parser.add_argument("--visualize", action="store_true", help="Generate data visualizations")
	
	args = parser.parse_args()
	
	# Load data
	data = load_annotation_data(args.input_csv)
	if data is None:
		return 1
	
	# Preprocess data
	position_features, labels = preprocess_data(data, args.smooth)
	
	# Calculate velocity features
	velocity_features = calculate_velocity_features(position_features)
	
	# Normalize data
	norm_pos, norm_vel, _ = normalize_data(position_features, velocity_features)
	
	# Combine features
	X = np.hstack([norm_pos, norm_vel])
	
	# Split data
	X_train, X_test, y_train, y_test = split_train_test(X, labels, test_size=1.0 - args.split)
	
	# Save processed data
	save_processed_data(args.output_dir, X_train, X_test, y_train, y_test)
	
	# Generate visualizations if requested
	if args.visualize:
		visualize_data(args.output_dir, position_features, velocity_features, labels)
	
	# Write metadata
	write_metadata(args.output_dir, X.shape, len(labels), args.split)
	
	print(f"Processing complete! Training data shape: {X_train.shape}, Test data shape: {X_test.shape}")
	return 0


if __name__ == "__main__":
	sys.exit(main())
	