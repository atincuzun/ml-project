# Gesture Control Application

A Flask-based web application for gesture recognition and control using MediaPipe. This application provides webcam and video input for gesture detection, a Tetris game controllable by gestures, and a presentation browser.

## Features

- **Webcam Input**: Real-time pose detection and gesture recognition
- **Video Input**: Process video files for gesture recognition and performance evaluation
- **Tetris Game**: Playable Tetris game controlled via keyboard or gestures
- **Presentation Browser**: View and open RevealJS presentations

## Supported Gestures

- **Swipe Left**: Move left in Tetris or go to previous slide in presentations
- **Swipe Right**: Move right in Tetris or go to next slide in presentations
- **Rotate Clockwise**: Rotate piece clockwise in Tetris or enter overview mode in presentations
- **Rotate Counter-clockwise**: Rotate piece counter-clockwise in Tetris or exit overview mode in presentations

## Installation

1. Clone the repository:
   ```
   git clone <repository-url>
   cd gesture-control-app
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Create necessary directories:
   ```
   mkdir -p uploads
   mkdir -p static/img
   mkdir -p static/presentations
   ```

## Usage

1. Run the application:
   ```
   python app.py
   ```

2. The application will automatically open in your default web browser at `http://127.0.0.1:5000/`

3. Navigate through the different sections:
   - **Webcam Input**: Use your webcam for real-time gesture recognition
   - **Video Input**: Upload and process video files
   - **Tetris**: Play Tetris using keyboard or gestures
   - **Presentation**: Browse and open RevealJS presentations

## Adding Presentations

To add RevealJS presentations:

1. Create a new directory for each presentation in `static/presentations/`
2. Add the RevealJS presentation files to the directory
3. The presentation will automatically appear in the Presentation browser

## Project Structure

```
gesture-control-app/
├── static/
│   ├── css/           # CSS stylesheets
│   ├── js/            # JavaScript files
│   ├── img/           # Images and icons
│   └── presentations/ # RevealJS presentations
│
├── templates/         # HTML templates
│
├── utils/             # Utility modules
│   ├── __init__.py
│   ├── mediapipe_processor.py
│   └── data_handler.py
│
├── uploads/           # Directory for uploaded videos
├── app.py             # Main Flask application
├── config.py          # Configuration settings
├── keypoint_mapping.yml # MediaPipe keypoint definitions
└── requirements.txt   # Dependencies
```

## Requirements

- Python 3.6+
- OpenCV
- MediaPipe
- NumPy
- Flask
- Pandas
- PyYAML

## Performance Evaluation

The application can generate CSV files containing gesture detection results for performance evaluation. When processing videos, the results are saved as `performance_results.csv` with timestamps and detected gestures.

## Customization

- Modify `keypoint_mapping.yml` to adjust the MediaPipe keypoints used for gesture detection
- Update gesture detection logic in `utils/mediapipe_processor.py`
- Customize the appearance by modifying the CSS files in `static/css/`

## License

[MIT License](LICENSE)