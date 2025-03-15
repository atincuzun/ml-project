from flask import Flask, render_template, request, Response, jsonify, send_from_directory
import os
import json
import webbrowser
from threading import Timer
import cv2
import time
import numpy as np
from utils.mediapipe_processor import MediapipeProcessor
from utils.data_handler import CSVDataHandler
import mediapipe as mp
import yaml

# Configure logging to reduce noise
import logging


# Setup MediaPipe
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose
pose_detector = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)



# Create a global pose detector
pose_detector = mp_pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
    model_complexity=1,  # 0=Lite, 1=Full, 2=Heavy
    smooth_landmarks=True
)


logging.getLogger('werkzeug').setLevel(logging.WARNING)

app = Flask(__name__)
app.config.from_pyfile('config.py')

# Global variables
processor = MediapipeProcessor()
data_handler = CSVDataHandler()


# Open browser automatically when the app starts
def open_browser():
    webbrowser.open_new('http://127.0.0.1:5000/')


@app.route('/')
def index():
    return render_template('webcam.html', active_page='webcam')


@app.route('/webcam')
def webcam():
    return render_template('webcam.html', active_page='webcam')


@app.route('/video')
def video():
    return render_template('video.html', active_page='video')


@app.route('/tetris')
def tetris():
    return render_template('tetris.html', active_page='tetris')


@app.route('/presentation')
def presentation():
    # Get list of available presentations
    presentations_dir = os.path.join(app.static_folder, 'presentations')
    presentations = []
    
    if os.path.exists(presentations_dir):
        # Add directories (which may contain index.html)
        for item in os.listdir(presentations_dir):
            if os.path.isdir(os.path.join(presentations_dir, item)):
                presentations.append({'name': item, 'type': 'directory'})
        
        # Add HTML files in the root of presentations directory
        for item in os.listdir(presentations_dir):
            if item.lower().endswith('.html') and os.path.isfile(os.path.join(presentations_dir, item)):
                presentations.append({'name': item, 'type': 'file'})
    
    return render_template('presentation.html', active_page='presentation', presentations=presentations)


@app.route('/favicon.ico')
def favicon():
    """Serve the favicon"""
    return send_from_directory(app.static_folder, 'favicon.ico')


@app.route('/get_cameras')
def get_cameras():
    """Get a list of available camera devices with improved detection"""
    cameras = []
    
    # Always include camera 0 as default
    cameras.append({
        'id': 0, 
        'name': 'Default Camera'
    })
    
    # Only check first few indices to avoid hanging
    for index in range(1, 8):  # Only check cameras 1-2
        try:
            # Try DirectShow on Windows (more reliable)
            if os.name == 'nt':
                cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
            else:
                cap = cv2.VideoCapture(index)
            
            if cap.isOpened():
                cameras.append({
                    'id': index,
                    'name': f'Camera {index}'
                })
            
            cap.release()
        except Exception as e:
            print(f"Error checking camera {index}: {e}")
    
    return jsonify(cameras)



@app.route('/video_feed')
def video_feed():
    """Video streaming route for webcam"""
    camera_id = request.args.get('camera_id', default=0, type=int)
    
    def generate():
        try:
            cap = cv2.VideoCapture(camera_id)
            
            # Check if camera opened successfully
            if not cap.isOpened():
                # Create dummy frames with error message
                frame_height, frame_width = 480, 640
                dummy_frame = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)
                cv2.putText(dummy_frame, "Camera not available", (50, frame_height // 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                
                # Encode and send the dummy frames
                ret, buffer_original = cv2.imencode('.jpg', dummy_frame)
                ret, buffer_pose = cv2.imencode('.jpg', dummy_frame)
                
                # Combine frames with a delimiter for frontend to split
                combined_data = buffer_original.tobytes() + b"FRAME_DELIMITER" + buffer_pose.tobytes() + b"FRAME_DELIMITER"
                
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + combined_data + b'\r\n')
                return
            
            while cap.isOpened():
                success, frame = cap.read()
                if not success:
                    break
                
                # Process frame with MediaPipe
                processed_frame, pose_frame, gesture = processor.process_frame(frame)
                
                # If logging is enabled, record the data
                if processor.is_logging:
                    timestamp = int(time.time() * 1000)  # Current time in milliseconds
                    data_handler.add_frame(processor.last_pose_data, timestamp, gesture)
                
                # Encode the frames
                ret, buffer_original = cv2.imencode('.jpg', processed_frame)
                ret, buffer_pose = cv2.imencode('.jpg', pose_frame)
                
                # Combine frames with a delimiter for frontend to split
                combined_data = buffer_original.tobytes() + b"FRAME_DELIMITER" + buffer_pose.tobytes() + b"FRAME_DELIMITER"
                
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + combined_data + b'\r\n')
        except Exception as e:
            print(f"Error in video feed: {e}")
        finally:
            if 'cap' in locals() and cap is not None:
                cap.release()
    
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/process_video', methods=['POST'])
def process_video():
    """Process uploaded video file"""
    if 'video' not in request.files:
        return jsonify({'error': 'No video file uploaded'}), 400
    
    video_file = request.files['video']
    video_path = os.path.join(app.config['UPLOAD_FOLDER'], video_file.filename)
    video_file.save(video_path)
    
    # Set the video path for processing
    processor.set_video_path(video_path)
    
    return jsonify({'success': True, 'video_path': video_path})


@app.route('/video_process_feed')
def video_process_feed():
    """Stream processed video frames"""
    
    def generate():
        success = True
        
        while success:
            processed_frame, pose_frame, gesture, success = processor.get_next_video_frame()
            
            if not success:
                break
            
            # If we're supposed to be logging, add the frame to our data
            if processor.is_logging:
                timestamp = processor.get_current_video_timestamp()
                data_handler.add_frame(processor.last_pose_data, timestamp, gesture)
            
            # Encode the frames
            ret, buffer_original = cv2.imencode('.jpg', processed_frame)
            ret, buffer_pose = cv2.imencode('.jpg', pose_frame)
            
            # Combine frames with a delimiter for frontend to split
            combined_data = buffer_original.tobytes() + b"FRAME_DELIMITER" + buffer_pose.tobytes() + b"FRAME_DELIMITER"
            
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + combined_data + b'\r\n')
    
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/gesture_stream')
def gesture_stream():
    """Stream of detected gestures using Server-Sent Events (SSE)"""
    
    def generate():
        last_gesture = "idle"
        
        while True:
            current_gesture = processor.last_gesture
            
            # Only send when gesture changes
            if current_gesture != last_gesture:
                data = json.dumps({'gesture': current_gesture})
                yield f"data: {data}\n\n"
                last_gesture = current_gesture
            
            # Still need a small delay to prevent CPU overuse
            time.sleep(0.1)
    
    return Response(generate(), mimetype='text/event-stream')


@app.route('/toggle_logging', methods=['POST'])
def toggle_logging():
    """Toggle data logging state"""
    processor.toggle_logging()
    
    if not processor.is_logging:
        # Logging was turned off, return the collected data
        csv_data = data_handler.get_csv_data()
        return jsonify({
            'logging': False,
            'csv_data': csv_data
        })
    else:
        # Logging was turned on, clear previous data
        data_handler.clear_data()
        return jsonify({'logging': True})


@app.route('/save_csv', methods=['POST'])
def save_csv():
    """Save CSV data to a file"""
    data = request.json
    file_path = data.get('file_path', 'performance_results.csv')
    
    success = data_handler.save_to_csv(file_path)
    
    if success:
        return jsonify({'success': True, 'file_path': file_path})
    else:
        return jsonify({'success': False, 'error': 'Failed to save CSV file'})


@app.route('/open_presentation/<name>')
def open_presentation(name):
    """Return the path to the presentation to open in a new tab"""
    presentations_dir = os.path.join(app.static_folder, 'presentations')
    controller_path = '/static/js/presentation-controller.js'
    
    # Check if it's a file directly in the presentations directory
    if os.path.isfile(os.path.join(presentations_dir, name)):
        presentation_path = f'/static/presentations/{name}'
    else:
        presentation_path = f'/static/presentations/{name}'
    
    return jsonify({
        'path': presentation_path,
        'controller': controller_path
    })

@app.route('/simple')
def simple():
    return render_template('simple_camera.html')
    
@app.route('/simple_camera_feed')
def simple_camera_feed():
    """Video stream optimized for display quality"""
    camera_id = request.args.get('camera_id', default=0, type=int)
    # Allow requesting specific resolutions via query parameters
    requested_width = request.args.get('width', default=0, type=int)
    requested_height = request.args.get('height', default=0, type=int)
    
    def generate():
        cap = None
        try:
            print(f"Attempting to open camera {camera_id}")
            
            # Try multiple backends in order of preference
            backends = [None]  # Default backend first
            
            if os.name == 'nt':  # Windows-specific backends
                backends.extend([cv2.CAP_DSHOW, cv2.CAP_MSMF])
            
            # Try each backend until one works
            for backend in backends:
                if cap is not None:
                    cap.release()
                
                if backend is None:
                    cap = cv2.VideoCapture(camera_id)
                    print("Using default backend")
                else:
                    cap = cv2.VideoCapture(camera_id, backend)
                    if backend == cv2.CAP_DSHOW:
                        print("Using DirectShow backend")
                    elif backend == cv2.CAP_MSMF:
                        print("Using Media Foundation backend")
                
                if cap is not None and cap.isOpened():
                    # Get original dimensions before trying to change them
                    orig_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    orig_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    print(f"Original camera resolution: {orig_width}x{orig_height}")
                    
                    # Try to set requested resolution if provided
                    if requested_width > 0 and requested_height > 0:
                        cap.set(cv2.CAP_PROP_FRAME_WIDTH, requested_width)
                        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, requested_height)
                    else:
                        # Otherwise try common resolutions in descending order
                        resolutions = [
                            (1920, 1080),  # Full HD
                            (1280, 720),   # HD
                            (864, 480),    # 480p (16:9)
                            (640, 480)     # VGA
                        ]
                        
                        for width, height in resolutions:
                            cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
                            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
                            actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                            
                            # If we successfully set a resolution, break
                            if actual_width == width and actual_height == height:
                                print(f"Successfully set resolution to {width}x{height}")
                                break
                    
                    # Verify final resolution
                    final_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    final_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    print(f"Final camera resolution: {final_width}x{final_height}")
                    
                    break
            
            if cap is None or not cap.isOpened():
                print(f"Failed to open camera {camera_id}")
                yield (b'--frame\r\n'
                      b'Content-Type: text/plain\r\n\r\n'
                      b'Camera open failed\r\n')
                return
                
            while True:
                success, frame = cap.read()
                if not success:
                    print("Failed to read frame")
                    break
                
                # Process the frame with MediaPipe Pose
                try:
                    # Convert to RGB for MediaPipe
                    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    
                    # Process the image and get landmarks
                    results = pose_detector.process(image_rgb)
                    
                    # If landmarks were detected, draw them on the frame
                    if results.pose_landmarks:
                        # Draw the pose landmarks on the original frame
                        mp_drawing.draw_landmarks(
                            frame,
                            results.pose_landmarks,
                            mp_pose.POSE_CONNECTIONS,
                            landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
                        )
                except Exception as e:
                    print(f"Error processing pose: {e}")
                    # Continue with the original frame if pose detection fails
                
                # Use highest quality JPEG encoding
                encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 95]
                _, buffer = cv2.imencode('.jpg', frame, encode_param)
                frame_bytes = buffer.tobytes()
                
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                       
        except Exception as e:
            print(f"Camera error: {str(e)}")
            import traceback
            traceback.print_exc()
            
        finally:
            if cap is not None and cap.isOpened():
                cap.release()
                print(f"Camera {camera_id} released")

    return Response(generate(), 
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/get_camera_info')
def get_camera_info():
    """Get native camera resolution and aspect ratio"""
    camera_id = request.args.get('camera_id', default=0, type=int)
    
    try:
        # Try multiple backends
        backends = [None]
        if os.name == 'nt':
            backends.extend([cv2.CAP_DSHOW, cv2.CAP_MSMF])
            
        for backend in backends:
            if backend is None:
                cap = cv2.VideoCapture(camera_id)
            else:
                cap = cv2.VideoCapture(camera_id, backend)
                
            if cap is not None and cap.isOpened():
                break
        
        if not cap.isOpened():
            return jsonify({
                'success': False,
                'message': f"Failed to open camera {camera_id}"
            })
        
        # Get native resolution
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Capture one frame to verify resolution
        ret, frame = cap.read()
        if ret:
            actual_height, actual_width = frame.shape[:2]
            if actual_width > 0 and actual_height > 0:
                width = actual_width
                height = actual_height
        
        # Calculate aspect ratio
        aspect_ratio = width / height if height > 0 else 16/9
        
        cap.release()
        
        return jsonify({
            'success': True,
            'width': width,
            'height': height,
            'aspectRatio': aspect_ratio
        })
    
    except Exception as e:
        print(f"Error getting camera info: {str(e)}")
        return jsonify({
            'success': False,
            'message': str(e)
        })
    
if __name__ == '__main__':
    # Open browser tab after a small delay
    Timer(1.0, open_browser).start()
    
    # Create upload folder if it doesn't exist
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
    
    app.run(debug=True)