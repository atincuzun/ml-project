# Gesture Data Generator

A standalone tool for creating training data for gesture recognition by annotating videos with MediaPipe pose landmarks.

## Features

- Load and play videos with MediaPipe pose detection overlay
- Frame-by-frame navigation (forward/backward)
- Adjustable playback speed (0.25x to 2.0x)
- Select and annotate single or multiple frames 
- Assign gesture labels (idle, swipe_left, swipe_right, rotate_cw, rotate_ccw)
- Export annotations with landmark data to CSV for AI training

## Requirements

- Python 3.6+
- OpenCV
- MediaPipe
- NumPy
- Pillow (PIL)
- Tkinter (usually comes with Python)

## Installation

1. Create a new Python environment (recommended):
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. Install required packages:
   ```
   pip install -r requirements.txt
   ```

## Usage

1. Run the application:
   ```
   python data_generator.py
   ```

2. Open a video file through the File menu or keyboard shortcut

3. Navigate through the video using:
   - Play/Pause button or Space key
   - Previous/Next frame buttons or Left/Right arrow keys
   - Frame slider
   - Adjust playback speed with the dropdown or Up/Down arrow keys

4. Annotate frames:
   - Select a gesture type from the radio buttons
   - Click "Annotate Current Frame" or press 'A' key to annotate the current frame
   - Select multiple frames by clicking on them (or press 'S' key) and use "Annotate Selected Frames"
   - Clear selection with "Clear Selection" button or 'C' key

5. Export your annotations from the File menu to create a CSV file for training

## Keyboard Shortcuts

- **Space**: Play/Pause video
- **Left Arrow**: Previous frame
- **Right Arrow**: Next frame
- **Up Arrow**: Increase playback speed
- **Down Arrow**: Decrease playback speed
- **S**: Select/Deselect current frame
- **A**: Annotate current frame with selected gesture
- **C**: Clear frame selection

## Output Format

The exported CSV file contains:
- Frame number
- Gesture label
- 132 columns of MediaPipe pose landmark data (33 landmarks × 4 values: x, y, z, visibility)

This format is compatible with the machine learning pipeline used in the main project.

## License

MIT License