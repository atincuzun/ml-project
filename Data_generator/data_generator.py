#!/usr/bin/env python3
"""
Data Generator Tool for Gesture Recognition

This tool allows users to:
1. Load videos
2. Play/pause videos with MediaPipe overlay
3. Step frame by frame (forward/backward)
4. Adjust playback speed
5. Annotate frames with gesture labels
6. Select multiple frames at once for batch annotation
7. Export annotated data for AI training
"""

import os
import sys
import time
import csv
import json
import numpy as np
import cv2
import mediapipe as mp
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from tkinter.scrolledtext import ScrolledText
from PIL import Image, ImageTk

# Set up MediaPipe Pose
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles


class VideoAnnotationApp:
	def __init__(self, root):
		self.root = root
		self.root.title("Gesture Data Generator")
		self.root.geometry("1280x720")
		self.root.minsize(1024, 700)
		
		# Set app state variables
		self.video_path = None
		self.cap = None
		self.total_frames = 0
		self.current_frame = 0
		self.frame_rate = 30
		self.playing = False
		self.playback_speed = 1.0
		self.last_frame_time = 0
		self.selected_frames = set()
		self.frame_annotations = {}  # {frame_number: gesture_label}
		self.pose_data = {}  # {frame_number: landmarks_data}
		
		# Gesture types
		self.gestures = {
			0: "idle",
			1: "swipe_left",
			2: "swipe_right",
			3: "rotate_cw",
			4: "rotate_ccw"
		}
		
		# Set up MediaPipe processor
		self.pose = mp_pose.Pose(
			min_detection_confidence=0.5,
			min_tracking_confidence=0.5,
			model_complexity=1
		)
		
		# Create UI elements
		self.create_menu()
		self.create_main_frame()
		self.create_status_bar()
		
		# Start with disabled controls
		self.toggle_controls(False)
	
	def create_menu(self):
		"""Create application menu bar"""
		menubar = tk.Menu(self.root)
		
		# File menu
		file_menu = tk.Menu(menubar, tearoff=0)
		file_menu.add_command(label="Open Video", command=self.open_video)
		file_menu.add_command(label="Export Annotations", command=self.export_annotations)
		file_menu.add_separator()
		file_menu.add_command(label="Exit", command=self.root.quit)
		menubar.add_cascade(label="File", menu=file_menu)
		
		# Edit menu
		edit_menu = tk.Menu(menubar, tearoff=0)
		edit_menu.add_command(label="Clear All Annotations", command=self.clear_annotations)
		menubar.add_cascade(label="Edit", menu=edit_menu)
		
		# Help menu
		help_menu = tk.Menu(menubar, tearoff=0)
		help_menu.add_command(label="Controls", command=self.show_controls)
		help_menu.add_command(label="About", command=self.show_about)
		menubar.add_cascade(label="Help", menu=help_menu)
		
		self.root.config(menu=menubar)
	
	def create_main_frame(self):
		"""Create the main application frame"""
		# Main container
		main_container = ttk.Frame(self.root)
		main_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
		
		# Top panel: Video display
		self.video_panel = ttk.Frame(main_container)
		self.video_panel.pack(fill=tk.BOTH, expand=True)
		
		# Video canvas
		self.canvas = tk.Canvas(self.video_panel, bg="black")
		self.canvas.pack(fill=tk.BOTH, expand=True)
		
		# Middle panel: Controls
		controls_panel = ttk.Frame(main_container)
		controls_panel.pack(fill=tk.X, pady=10)
		
		# Playback controls
		playback_frame = ttk.Frame(controls_panel)
		playback_frame.pack(fill=tk.X)
		
		# Play/Pause button
		self.play_button = ttk.Button(playback_frame, text="▶ Play", command=self.toggle_play)
		self.play_button.pack(side=tk.LEFT, padx=5)
		
		# Frame navigation
		self.prev_frame_button = ttk.Button(playback_frame, text="◀ Previous", command=self.prev_frame)
		self.prev_frame_button.pack(side=tk.LEFT, padx=5)
		
		self.next_frame_button = ttk.Button(playback_frame, text="Next ▶", command=self.next_frame)
		self.next_frame_button.pack(side=tk.LEFT, padx=5)
		
		# Frame position indicator
		self.frame_position = ttk.Label(playback_frame, text="Frame: 0 / 0")
		self.frame_position.pack(side=tk.LEFT, padx=20)
		
		# Playback speed control
		speed_label = ttk.Label(playback_frame, text="Speed:")
		speed_label.pack(side=tk.LEFT, padx=5)
		
		self.speed_var = tk.StringVar(value="1.0x")
		speed_values = ["0.25x", "0.5x", "1.0x", "1.5x", "2.0x"]
		self.speed_menu = ttk.Combobox(playback_frame, textvariable=self.speed_var, values=speed_values, width=5,
		                               state="readonly")
		self.speed_menu.pack(side=tk.LEFT, padx=5)
		self.speed_menu.bind("<<ComboboxSelected>>", self.change_speed)
		
		# Frame slider
		slider_frame = ttk.Frame(controls_panel)
		slider_frame.pack(fill=tk.X, pady=5)
		
		self.frame_slider = ttk.Scale(slider_frame, from_=0, to=100, orient=tk.HORIZONTAL, command=self.slider_changed)
		self.frame_slider.pack(fill=tk.X, padx=5)
		
		# Bottom panel: Annotation controls
		annotation_panel = ttk.LabelFrame(main_container, text="Annotation")
		annotation_panel.pack(fill=tk.X, pady=5)
		
		# Gesture selection buttons
		gestures_frame = ttk.Frame(annotation_panel)
		gestures_frame.pack(fill=tk.X, padx=5, pady=5)
		
		self.gesture_var = tk.IntVar(value=0)
		
		for gesture_id, gesture_name in self.gestures.items():
			gesture_radio = ttk.Radiobutton(
				gestures_frame,
				text=gesture_name.replace('_', ' ').title(),
				variable=self.gesture_var,
				value=gesture_id
			)
			gesture_radio.pack(side=tk.LEFT, padx=10)
		
		# Annotation buttons
		annotation_buttons_frame = ttk.Frame(annotation_panel)
		annotation_buttons_frame.pack(fill=tk.X, padx=5, pady=5)
		
		self.annotate_frame_button = ttk.Button(
			annotation_buttons_frame,
			text="Annotate Current Frame",
			command=self.annotate_current_frame
		)
		self.annotate_frame_button.pack(side=tk.LEFT, padx=5)
		
		self.annotate_multiple_button = ttk.Button(
			annotation_buttons_frame,
			text="Annotate Selected Frames",
			command=self.annotate_selected_frames
		)
		self.annotate_multiple_button.pack(side=tk.LEFT, padx=5)
		
		self.clear_selection_button = ttk.Button(
			annotation_buttons_frame,
			text="Clear Selection",
			command=self.clear_selection
		)
		self.clear_selection_button.pack(side=tk.LEFT, padx=5)
		
		# Right panel: Annotation list
		self.annotations_panel = ttk.LabelFrame(main_container, text="Annotations List")
		self.annotations_panel.pack(fill=tk.BOTH, expand=True, pady=5)
		
		# Annotations display
		self.annotations_display = ScrolledText(self.annotations_panel, height=6)
		self.annotations_display.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
		self.annotations_display.config(state=tk.DISABLED)
	
	def create_status_bar(self):
		"""Create status bar at the bottom of the window"""
		self.status_bar = ttk.Label(self.root, text="Ready. Open a video file to begin.", relief=tk.SUNKEN, anchor=tk.W)
		self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)
	
	def toggle_controls(self, enabled=True):
		"""Enable or disable UI controls based on video availability"""
		state = "normal" if enabled else "disabled"
		
		self.play_button.config(state=state)
		self.prev_frame_button.config(state=state)
		self.next_frame_button.config(state=state)
		self.frame_slider.config(state=state)
		self.speed_menu.config(state="readonly" if enabled else "disabled")
		self.annotate_frame_button.config(state=state)
		self.annotate_multiple_button.config(state=state)
		self.clear_selection_button.config(state=state)
	
	def open_video(self):
		"""Open a video file for annotation"""
		# Stop any current video playback
		self.playing = False
		
		# Open file dialog
		video_path = filedialog.askopenfilename(
			title="Select Video File",
			filetypes=[
				("Video files", "*.mp4 *.avi *.mov *.mkv"),
				("All files", "*.*")
			]
		)
		
		if not video_path:
			return
		
		# Release any existing video capture
		if self.cap is not None:
			self.cap.release()
		
		# Open new video
		self.cap = cv2.VideoCapture(video_path)
		
		if not self.cap.isOpened():
			messagebox.showerror("Error", "Could not open video file")
			self.status_bar.config(text=f"Error: Could not open video file")
			return
		
		# Get video properties
		self.video_path = video_path
		self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
		self.frame_rate = self.cap.get(cv2.CAP_PROP_FPS)
		self.current_frame = 0
		
		# Update UI
		self.frame_slider.config(to=self.total_frames - 1)
		self.frame_position.config(text=f"Frame: {self.current_frame} / {self.total_frames - 1}")
		self.status_bar.config(
			text=f"Loaded: {os.path.basename(video_path)} | {self.total_frames} frames | {self.frame_rate:.2f} FPS")
		
		# Reset annotations
		self.frame_annotations = {}
		self.pose_data = {}
		self.selected_frames = set()
		self.update_annotations_display()
		
		# Enable controls
		self.toggle_controls(True)
		
		# Show first frame
		self.read_frame(0)
	
	def read_frame(self, frame_num):
		"""Read a specific frame from the video"""
		if self.cap is None:
			return
		
		# Validate frame number
		if frame_num < 0:
			frame_num = 0
		elif frame_num >= self.total_frames:
			frame_num = self.total_frames - 1
		
		# Set position and read frame
		self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
		ret, frame = self.cap.read()
		
		if not ret:
			self.status_bar.config(text=f"Error reading frame {frame_num}")
			return
		
		# Process with MediaPipe
		processed_frame, landmarks = self.process_frame_with_mediapipe(frame)
		
		# Store pose data if landmarks were detected
		if landmarks:
			self.pose_data[frame_num] = self.landmarks_to_array(landmarks)
		
		# Update current frame
		self.current_frame = frame_num
		
		# Highlight if this frame is selected
		if frame_num in self.selected_frames:
			cv2.rectangle(processed_frame, (0, 0), (processed_frame.shape[1], processed_frame.shape[0]), (0, 255, 0),
			              10)
		
		# Add annotation label if it exists
		if frame_num in self.frame_annotations:
			gesture = self.frame_annotations[frame_num]
			cv2.putText(processed_frame, gesture, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
		
		# Update display
		self.display_frame(processed_frame)
		
		# Update UI
		self.frame_position.config(text=f"Frame: {frame_num} / {self.total_frames - 1}")
		self.frame_slider.set(frame_num)
	
	def process_frame_with_mediapipe(self, frame):
		"""Process a frame with MediaPipe Pose"""
		# Convert to RGB for MediaPipe
		frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
		frame_rgb.flags.writeable = False
		
		# Process with MediaPipe
		results = self.pose.process(frame_rgb)
		
		# Convert back to BGR for display
		frame_rgb.flags.writeable = True
		frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
		
		# Draw landmarks if detected
		if results.pose_landmarks:
			mp_drawing.draw_landmarks(
				frame_bgr,
				results.pose_landmarks,
				mp_pose.POSE_CONNECTIONS,
				landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
			)
		
		return frame_bgr, results.pose_landmarks
	
	def landmarks_to_array(self, landmarks):
		"""Convert MediaPipe landmarks to array format for storage"""
		data = []
		for landmark in landmarks.landmark:
			data.extend([landmark.x, landmark.y, landmark.z, landmark.visibility])
		return data
	
	def display_frame(self, frame):
		"""Display the processed frame on the canvas"""
		# Resize frame to fit canvas while maintaining aspect ratio
		canvas_width = self.canvas.winfo_width()
		canvas_height = self.canvas.winfo_height()
		
		if canvas_width <= 1 or canvas_height <= 1:
			# Canvas not properly initialized yet, use default size
			canvas_width = 800
			canvas_height = 600
		
		# Calculate scaling factor
		frame_height, frame_width = frame.shape[:2]
		width_ratio = canvas_width / frame_width
		height_ratio = canvas_height / frame_height
		scale_factor = min(width_ratio, height_ratio)
		
		# Resize the frame
		if scale_factor < 1:
			new_width = int(frame_width * scale_factor)
			new_height = int(frame_height * scale_factor)
			frame = cv2.resize(frame, (new_width, new_height))
		
		# Convert to PhotoImage
		image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
		image = Image.fromarray(image)
		photo = ImageTk.PhotoImage(image=image)
		
		# Update canvas
		self.canvas.config(width=photo.width(), height=photo.height())
		self.canvas.create_image(0, 0, image=photo, anchor=tk.NW)
		self.canvas.image = photo  # Keep a reference
	
	def toggle_play(self):
		"""Toggle video playback"""
		self.playing = not self.playing
		
		if self.playing:
			self.play_button.config(text="⏸ Pause")
			self.last_frame_time = time.time()
			self.update_video()
		else:
			self.play_button.config(text="▶ Play")
	
	def update_video(self):
		"""Update video frame during playback"""
		if not self.playing:
			return
		
		current_time = time.time()
		elapsed = current_time - self.last_frame_time
		
		# Calculate time per frame based on playback speed
		time_per_frame = 1.0 / (self.frame_rate * self.playback_speed)
		
		if elapsed >= time_per_frame:
			# Move to next frame
			next_frame = self.current_frame + 1
			
			if next_frame >= self.total_frames:
				# End of video, stop playback
				self.playing = False
				self.play_button.config(text="▶ Play")
				return
			
			self.read_frame(next_frame)
			self.last_frame_time = current_time
		
		# Schedule next update
		self.root.after(10, self.update_video)
	
	def prev_frame(self):
		"""Move to previous frame"""
		self.read_frame(self.current_frame - 1)
	
	def next_frame(self):
		"""Move to next frame"""
		self.read_frame(self.current_frame + 1)
	
	def slider_changed(self, value):
		"""Handle frame slider change"""
		# Only update if the value has actually changed to avoid recursion
		new_frame = int(float(value))
		if new_frame != self.current_frame:
			self.read_frame(new_frame)
	
	def change_speed(self, event=None):
		"""Change video playback speed"""
		speed_text = self.speed_var.get()
		self.playback_speed = float(speed_text.replace('x', ''))
	
	def annotate_current_frame(self):
		"""Annotate the current frame with selected gesture"""
		gesture_id = self.gesture_var.get()
		gesture_name = self.gestures[gesture_id]
		
		self.frame_annotations[self.current_frame] = gesture_name
		self.status_bar.config(text=f"Frame {self.current_frame} annotated as '{gesture_name}'")
		
		# Update the display and annotations list
		self.read_frame(self.current_frame)
		self.update_annotations_display()
	
	def toggle_frame_selection(self, event):
		"""Toggle selection of the current frame"""
		if self.current_frame in self.selected_frames:
			self.selected_frames.remove(self.current_frame)
		else:
			self.selected_frames.add(self.current_frame)
		
		# Update the display
		self.read_frame(self.current_frame)
		self.status_bar.config(text=f"{len(self.selected_frames)} frames selected")
	
	def annotate_selected_frames(self):
		"""Annotate all selected frames with the current gesture"""
		if not self.selected_frames:
			messagebox.showinfo("Info", "No frames selected")
			return
		
		gesture_id = self.gesture_var.get()
		gesture_name = self.gestures[gesture_id]
		
		# Apply annotation to all selected frames
		for frame_num in self.selected_frames:
			self.frame_annotations[frame_num] = gesture_name
		
		# Update the display
		self.read_frame(self.current_frame)
		self.update_annotations_display()
		
		self.status_bar.config(text=f"{len(self.selected_frames)} frames annotated as '{gesture_name}'")
	
	def clear_selection(self):
		"""Clear the selected frames"""
		self.selected_frames = set()
		self.read_frame(self.current_frame)
		self.status_bar.config(text="Selection cleared")
	
	def clear_annotations(self):
		"""Clear all annotations"""
		if not self.frame_annotations:
			return
		
		if messagebox.askyesno("Confirm", "Are you sure you want to clear all annotations?"):
			self.frame_annotations = {}
			self.update_annotations_display()
			self.read_frame(self.current_frame)
			self.status_bar.config(text="All annotations cleared")
	
	def update_annotations_display(self):
		"""Update the annotations display"""
		self.annotations_display.config(state=tk.NORMAL)
		self.annotations_display.delete(1.0, tk.END)
		
		if not self.frame_annotations:
			self.annotations_display.insert(tk.END, "No annotations yet.")
		else:
			# Sort by frame number
			sorted_annotations = sorted(self.frame_annotations.items())
			
			for frame, gesture in sorted_annotations:
				self.annotations_display.insert(tk.END, f"Frame {frame}: {gesture}\n")
		
		self.annotations_display.config(state=tk.DISABLED)
	
	def export_annotations(self):
		"""Export annotations to CSV file"""
		if not self.frame_annotations:
			messagebox.showinfo("Info", "No annotations to export")
			return
		
		# Ask for export file
		export_path = filedialog.asksaveasfilename(
			title="Export Annotations",
			defaultextension=".csv",
			filetypes=[
				("CSV files", "*.csv"),
				("All files", "*.*")
			]
		)
		
		if not export_path:
			return
		
		try:
			with open(export_path, 'w', newline='') as f:
				writer = csv.writer(f)
				
				# Write header
				header = ["frame", "gesture"]
				
				# Add landmark headers
				for i in range(33):  # MediaPipe has 33 pose landmarks
					for coord in ["x", "y", "z", "visibility"]:
						header.append(f"landmark_{i}_{coord}")
				
				writer.writerow(header)
				
				# Write data rows
				for frame_num, gesture in sorted(self.frame_annotations.items()):
					row = [frame_num, gesture]
					
					# Add landmark data if available
					if frame_num in self.pose_data:
						row.extend(self.pose_data[frame_num])
					else:
						# Fill with zeros if no landmark data
						row.extend([0] * (33 * 4))
					
					writer.writerow(row)
			
			self.status_bar.config(text=f"Annotations exported to {export_path}")
			messagebox.showinfo("Success", f"Annotations exported to {export_path}")
		
		except Exception as e:
			messagebox.showerror("Error", f"Failed to export annotations: {e}")
	
	def show_controls(self):
		"""Show keyboard shortcuts and controls"""
		controls = """
        Keyboard Controls:

        Space      - Play/Pause
        Left Arrow - Previous Frame
        Right Arrow - Next Frame
        Up Arrow   - Increase Playback Speed
        Down Arrow - Decrease Playback Speed
        S         - Select/Deselect Current Frame
        A         - Annotate Current Frame
        C         - Clear Selection
        """
		
		messagebox.showinfo("Controls", controls)
	
	def show_about(self):
		"""Show about information"""
		about_text = """
        Gesture Data Generator

        A tool for annotating videos with gesture labels for AI training.

        1. Open a video file
        2. Navigate and annotate frames
        3. Export annotations as CSV
        """
		
		messagebox.showinfo("About", about_text)
	
	def handle_keyboard(self, event):
		"""Handle keyboard shortcuts"""
		if self.cap is None:
			return
		
		if event.keysym == "space":
			self.toggle_play()
		elif event.keysym == "Left":
			self.prev_frame()
		elif event.keysym == "Right":
			self.next_frame()
		elif event.keysym == "Up":
			# Increase speed
			speeds = ["0.25x", "0.5x", "1.0x", "1.5x", "2.0x"]
			current_idx = speeds.index(self.speed_var.get())
			if current_idx < len(speeds) - 1:
				self.speed_var.set(speeds[current_idx + 1])
				self.change_speed()
		elif event.keysym == "Down":
			# Decrease speed
			speeds = ["0.25x", "0.5x", "1.0x", "1.5x", "2.0x"]
			current_idx = speeds.index(self.speed_var.get())
			if current_idx > 0:
				self.speed_var.set(speeds[current_idx - 1])
				self.change_speed()
		elif event.keysym.lower() == "s":
			self.toggle_frame_selection(None)
		elif event.keysym.lower() == "a":
			self.annotate_current_frame()
		elif event.keysym.lower() == "c":
			self.clear_selection()


def main():
	root = tk.Tk()
	app = VideoAnnotationApp(root)
	
	# Bind keyboard shortcuts
	root.bind("<Key>", app.handle_keyboard)
	
	# Bind canvas click to toggle frame selection
	app.canvas.bind("<Button-1>", app.toggle_frame_selection)
	
	root.mainloop()


if __name__ == "__main__":
	main()