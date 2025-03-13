import pandas as pd
import numpy as np
import os


class MediaPipeCSVReader:
    """
    Reader for MediaPipe pose landmark data saved in CSV format.
    Provides methods to load, process, and prepare data for neural network training.
    """
    
    def __init__(self, keypoint_names=None):
        """
        Initialize CSV reader with optional keypoint mapping
        
        Parameters:
        -----------
        keypoint_names : list or None
            List of keypoint names to use. If None, all keypoints will be loaded.
        """
        self.keypoint_names = keypoint_names
        self.data = None
        self.column_mapping = None
        
    def load_csv(self, filepath):
        """
        Load CSV file containing MediaPipe landmarks
        
        Parameters:
        -----------
        filepath : str
            Path to the CSV file
            
        Returns:
        --------
        pandas.DataFrame
            Loaded data
        """
        try:
            # Load CSV file with timestamp as index
            self.data = pd.read_csv(filepath, index_col="timestamp")
            print(f"Loaded {len(self.data)} frames from {filepath}")
            return self.data
        except Exception as e:
            print(f"Error loading CSV file: {e}")
            return None
            
    def get_column_indices(self, joint_names=None):
        """
        Get column indices for specified joints
        
        Parameters:
        -----------
        joint_names : list or None
            List of joint names to get indices for. If None, all joints are returned.
            
        Returns:
        --------
        dict
            Mapping of joint names to column indices
        """
        if self.data is None:
            print("No data loaded. Call load_csv() first.")
            return None
            
        # If no specific joints requested, return all
        if joint_names is None:
            joint_names = self.keypoint_names
            
        column_indices = {}
        
        # For each joint, find its x, y, z, confidence columns
        for joint in joint_names:
            joint_cols = {}
            for coord in ["x", "y", "z", "confidence"]:
                col_name = f"{joint}_{coord}"
                if col_name in self.data.columns:
                    joint_cols[coord] = self.data.columns.get_loc(col_name)
                else:
                    print(f"Warning: Column {col_name} not found in data")
            
            column_indices[joint] = joint_cols
            
        return column_indices
        
    def extract_features(self, joint_names=None, coords=["x", "y", "z"]):
        """
        Extract features for specified joints and coordinates
        
        Parameters:
        -----------
        joint_names : list or None
            List of joint names to extract. If None, uses pre-defined list of important joints.
        coords : list
            List of coordinates to extract (default: ["x", "y", "z"])
            
        Returns:
        --------
        numpy.ndarray
            Array of shape (n_samples, n_features) with extracted features
        """
        if self.data is None:
            print("No data loaded. Call load_csv() first.")
            return None
            
        # Default to important joints for gesture recognition if not specified
        if joint_names is None:
            joint_names = [
                "left_wrist", "right_wrist",
                "left_elbow", "right_elbow",
                "left_shoulder", "right_shoulder",
                "left_index", "right_index",
                "left_thumb", "right_thumb",
                "nose"  # Reference point
            ]
        
        # Get column indices for the requested joints
        column_indices = self.get_column_indices(joint_names)
        
        # Extract features
        features = []
        for _, row in self.data.iterrows():
            frame_features = []
            
            for joint in joint_names:
                for coord in coords:
                    if coord in column_indices[joint]:
                        idx = column_indices[joint][coord]
                        frame_features.append(row.iloc[idx])
                        
            features.append(frame_features)
            
        return np.array(features)
    
    def load_multiple_csv_with_labels(self, csv_dir, label_mapping):
        """
        Load multiple CSV files and assign labels based on filename or directory
        
        Parameters:
        -----------
        csv_dir : str
            Directory containing CSV files
        label_mapping : dict
            Mapping of filename patterns to class labels
            
        Returns:
        --------
        tuple
            (features, labels) arrays for training
        """
        all_features = []
        all_labels = []
        
        for file in os.listdir(csv_dir):
            if file.endswith(".csv"):
                filepath = os.path.join(csv_dir, file)
                
                # Determine label based on filename
                label = None
                for pattern, class_label in label_mapping.items():
                    if pattern in file:
                        label = class_label
                        break
                
                if label is None:
                    print(f"Skipping {file}: No matching label pattern")
                    continue
                    
                # Load and extract features
                self.load_csv(filepath)
                features = self.extract_features()
                
                if features is not None and len(features) > 0:
                    all_features.append(features)
                    # Assign same label to all frames in this file
                    all_labels.extend([label] * len(features))
        
        if len(all_features) > 0:
            return np.vstack(all_features), np.array(all_labels)
        else:
            return np.array([]), np.array([])
    
    def extract_sequence_features(self, sequence_length=10, stride=5):
        """
        Extract sequence features for temporal modeling
        
        Parameters:
        -----------
        sequence_length : int
            Number of frames to include in each sequence
        stride : int
            Number of frames to skip between sequences
            
        Returns:
        --------
        numpy.ndarray
            Array of shape (n_sequences, sequence_length, n_features)
        """
        features = self.extract_features()
        
        if features is None or len(features) < sequence_length:
            print(f"Not enough frames for sequences (need {sequence_length}, have {len(features) if features is not None else 0})")
            return None
            
        sequences = []
        
        for i in range(0, len(features) - sequence_length + 1, stride):
            sequence = features[i:i+sequence_length]
            sequences.append(sequence)
            
        return np.array(sequences)
    
    def extract_velocity_features(self, joint_names=None):
        """
        Calculate velocity features (change between consecutive frames)
        
        Parameters:
        -----------
        joint_names : list or None
            List of joint names to extract velocity for
            
        Returns:
        --------
        numpy.ndarray
            Array of velocity features
        """
        features = self.extract_features(joint_names)
        
        if features is None or len(features) < 2:
            print("Not enough frames to calculate velocity")
            return None
            
        # Calculate differences between consecutive frames
        velocity = np.diff(features, axis=0)
        
        # Padding to keep the same number of frames (first velocity is zero)
        velocity = np.vstack([np.zeros((1, velocity.shape[1])), velocity])
        
        return velocity
    
    def combine_features(self, include_position=True, include_velocity=True):
        """
        Combine different feature types
        
        Parameters:
        -----------
        include_position : bool
            Whether to include position features
        include_velocity : bool
            Whether to include velocity features
            
        Returns:
        --------
        numpy.ndarray
            Combined features
        """
        features_list = []
        
        if include_position:
            position = self.extract_features()
            if position is not None:
                features_list.append(position)
        
        if include_velocity:
            velocity = self.extract_velocity_features()
            if velocity is not None:
                features_list.append(velocity)
        
        if not features_list:
            return None
            
        # Combine all feature types horizontally
        return np.hstack(features_list)
