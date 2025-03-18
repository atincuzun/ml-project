import cv2
import numpy as np
import mediapipe as mp
import os
import xml.etree.ElementTree as ET
from tqdm import tqdm

class AnnotationGenerator:
    """
    Generates landmark annotations from videos, extracts only body pose landmarks
    (33 landmarks) and associates them with gesture labels.
    """
    
    def __init__(self, video_path=None, annotation_path=None):
        """Initialize the annotation generator."""
        self.video_path = video_path
        self.annotation_path = annotation_path
        self.output_path = None
        self.cap = None
        self.fps = 0
        self.total_frames = 0
        self.annotations = []
        self.frame_labels = []
        self.frame_landmarks = []
        self.processed = False
        
        # Initialize MediaPipe Pose only
        self.mp_pose = mp.solutions.pose
        
        # Initialize pose model
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,  # 0, 1, or 2 (higher = more accurate but slower)
            smooth_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
    
    def set_video_path(self, video_path):
        """Set the path to the video file and initialize video capture."""
        self.video_path = video_path
        
        try:
            self.cap = cv2.VideoCapture(video_path)
            if not self.cap.isOpened():
                print(f"Error: Could not open video {video_path}")
                return self
                
            self.fps = self.cap.get(cv2.CAP_PROP_FPS)
            self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            print(f"Video: {video_path}")
            print(f"  - FPS: {self.fps}")
            print(f"  - Total frames: {self.total_frames}")
        except Exception as e:
            print(f"Error setting video path: {e}")
        
        return self
    
    def set_annotation_path(self, annotation_path):
        """Set the path to the annotation file."""
        self.annotation_path = annotation_path
        return self
    
    def set_output_path(self, output_path):
        """Set the path for the output file."""
        self.output_path = output_path
        return self
    
    def assign_labels(self):
        """Assign labels to all frames based on annotations."""
        if not self.annotations:
            self.extract_labels(self.annotation_path)
        
        if not self.cap:
            print("Video not loaded. Please set video path first.")
            return self
            
        # Initialize all frames with "idle" label
        self.frame_labels = ["idle"] * self.total_frames
        
        # Assign gesture labels based on timestamps
        for frame_idx in range(self.total_frames):
            timestamp = frame_idx / self.fps
            
            # Check if this timestamp falls within any annotation
            for annotation in self.annotations:
                if annotation['start_time'] <= timestamp <= annotation['end_time']:
                    self.frame_labels[frame_idx] = annotation['label']
                    break
        
        # Count occurrences of each label
        label_counts = {}
        for label in self.frame_labels:
            label_counts[label] = label_counts.get(label, 0) + 1
        
        print(f"Frame labels assigned: {label_counts}")
        return self
    
    def extract_landmarks(self, video_file=None):
        """
        Extract only pose landmarks (33) from each frame of the video.
        
        Parameters:
        -----------
        video_file : str or None
            Path to video file (optional, uses self.video_path if None)
            
        Returns:
        --------
        self
        """
        if video_file:
            self.set_video_path(video_file)
            
        if not self.cap:
            print("Video not loaded. Please set video path first.")
            return self
            
        # Reset video to beginning
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        
        # Process each frame
        self.frame_landmarks = []
        
        for frame_idx in tqdm(range(self.total_frames), desc="Extracting landmarks"):
            ret, frame = self.cap.read()
            if not ret:
                break
            
            # Convert to RGB (MediaPipe requires RGB)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process with MediaPipe Pose
            pose_results = self.pose.process(rgb_frame)
            
            # Store landmarks for this frame
            frame_data = {
                'frame_idx': frame_idx,
                'timestamp': frame_idx/self.fps,
                'landmarks': []  # Will store all 33 pose landmarks
            }
            
            # Extract pose landmarks (33 landmarks)
            if pose_results.pose_landmarks:
                for i, landmark in enumerate(pose_results.pose_landmarks.landmark):
                    frame_data['landmarks'].append({
                        'x': landmark.x,
                        'y': landmark.y,
                        'z': landmark.z,
                        'visibility': landmark.visibility
                    })
            
            self.frame_landmarks.append(frame_data)
        
        # Count frames with valid landmarks
        valid_frames = sum(1 for frame in self.frame_landmarks if frame['landmarks'])
        
        print(f"Extracted landmarks from {len(self.frame_landmarks)} frames")
        print(f"Frames with valid pose landmarks: {valid_frames}/{len(self.frame_landmarks)} ({valid_frames/len(self.frame_landmarks)*100:.1f}%)")
        
        return self
    
    def extract_labels(self, annotation_file=None):
        """Extract labels from the annotation file."""
        if annotation_file:
            self.annotation_path = annotation_file
            
        if not self.annotation_path:
            print("Annotation path not set. Please set annotation path first.")
            return self
            
        print(f"Reading annotation file: {self.annotation_path}")
        self.annotations = []
        
        try:
            if self.annotation_path.endswith('.eaf'):
                # Parse EAF file
                tree = ET.parse(self.annotation_path)
                root = tree.getroot()
                
                # Get time slots
                time_slots = {}
                for time_slot in root.findall(".//TIME_SLOT"):
                    time_slot_id = time_slot.get('TIME_SLOT_ID')
                    time_value = int(time_slot.get('TIME_VALUE')) / 1000.0  # convert to seconds
                    time_slots[time_slot_id] = time_value
                
                # Get annotations
                for annotation in root.findall(".//ALIGNABLE_ANNOTATION"):
                    start_slot = annotation.get('TIME_SLOT_REF1')
                    end_slot = annotation.get('TIME_SLOT_REF2')
                    
                    start_time = time_slots[start_slot]
                    end_time = time_slots[end_slot]
                    
                    # Get annotation value (gesture label)
                    annotation_value = annotation.find(".//ANNOTATION_VALUE").text
                    
                    self.annotations.append({
                        'start_time': start_time,
                        'end_time': end_time,
                        'label': annotation_value
                    })
                    
            elif self.annotation_path.endswith('.txt'):
                # Parse TXT file
                with open(self.annotation_path, 'r') as f:
                    lines = f.readlines()
                    
                for line in lines:
                    parts = line.strip().split('\t')
                    if len(parts) >= 8:  # Ensure we have enough columns
                        self.annotations.append({
                            'start_time': float(parts[2]),
                            'end_time': float(parts[4]),
                            'label': parts[7]
                        })
        except Exception as e:
            print(f"Error extracting labels: {e}")
            
        print(f"Found {len(self.annotations)} gesture annotations")
        return self
    
    def prepare_training_data(self):
        """Prepare training data by combining landmarks and labels."""
        # Extract landmarks if not already done
        if not self.frame_landmarks:
            self.extract_landmarks()
            
        # Extract and assign labels if not already done
        if not self.frame_labels:
            self.assign_labels()
            
        print("Preparing training data...")
        
        # Make sure we have the same number of frames for landmarks and labels
        min_frames = min(len(self.frame_landmarks), len(self.frame_labels))
        
        if min_frames < len(self.frame_landmarks):
            self.frame_landmarks = self.frame_landmarks[:min_frames]
            
        if min_frames < len(self.frame_labels):
            self.frame_labels = self.frame_labels[:min_frames]
            
        # Add labels to landmark data
        for i in range(min_frames):
            self.frame_landmarks[i]['label'] = self.frame_labels[i]
            
        self.processed = True
        print(f"Training data prepared with {min_frames} frames")
        
        return self
    
    def get_landmark_data(self):
        """Get the landmark data after calling prepare_training_data."""
        if not self.processed:
            print("Data not processed. Call prepare_training_data() first.")
            return None
            
        return self.frame_landmarks
    
    def get_landmark_label(self):
        """Get the landmark labels after calling prepare_training_data."""
        if not self.processed:
            print("Data not processed. Call prepare_training_data() first.")
            return None
            
        return self.frame_labels
    
    def save_data(self, output_path=None):
        """Save the processed data to a file."""
        if output_path:
            self.output_path = output_path
            
        if not self.output_path:
            print("Output path not set. Please set output path first.")
            return self
            
        if not self.processed:
            print("Data not processed. Call prepare_training_data() first.")
            return self
            
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(self.output_path) if os.path.dirname(self.output_path) else '.', exist_ok=True)
        
        # Save the data
        data = {
            'landmarks': self.frame_landmarks,
            'labels': self.frame_labels
        }
        
        np.save(self.output_path, data)
        print(f"Data saved to {self.output_path}")
        
        return self
    
    def close(self):
        """Release resources."""
        if self.cap and self.cap.isOpened():
            self.cap.release()

