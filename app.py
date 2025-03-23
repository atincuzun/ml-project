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
    import webbrowser
    import time
    import platform
    
    print("\n* Waiting for Flask server to start...")
    time.sleep(1.0)  # Wait longer for server to fully initialize
    
    url = 'http://127.0.0.1:5000/'
    print(f"* Opening browser to {url}")
    
    try:
        # Different approaches for different platforms
        system = platform.system()
        if system == 'Windows':
            # On Windows, try to use the default browser
            webbrowser.get('windows-default').open(url)
        elif system == 'Darwin':  # macOS
            # On macOS, try to use the default browser
            webbrowser.get('safari').open(url)
        else:
            # On Linux and other platforms, use the default
            webbrowser.open(url)
        
        print("* Browser should be opening now")
    except Exception as e:
        print(f"* Error opening browser: {e}")
        print(f"* Please manually navigate to {url}")


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
    """Process uploaded video file with improved error handling"""
    try:
        # Check if this is a request to get just the first frame
        if request.is_json:
            data = request.json
            video_path = data.get('video_path')
            get_first_frame_only = data.get('get_first_frame_only', False)
            
            if not video_path or not os.path.exists(video_path):
                return jsonify({'error': 'Invalid video path'}), 400
        else:
            # Normal file upload
            if 'video' not in request.files:
                return jsonify({'error': 'No video file uploaded'}), 400
            
            video_file = request.files['video']
            if video_file.filename == '':
                return jsonify({'error': 'No video file selected'}), 400
            
            # Ensure the filename is safe
            from werkzeug.utils import secure_filename
            filename = secure_filename(video_file.filename)
            video_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            
            # Save the video file
            try:
                video_file.save(video_path)
            except Exception as e:
                return jsonify({'error': f'Failed to save video: {str(e)}'}), 500
            
            get_first_frame_only = False
        
        # Check if video file exists and is readable
        if not os.path.isfile(video_path):
            return jsonify({'error': 'Video file not found or not accessible'}), 404
        
        # Extract the first frame for preview regardless of mode
        first_frame = None
        try:
            cap = cv2.VideoCapture(video_path)
            
            # Try different backends if the default fails
            if not cap.isOpened():
                backends = []
                if os.name == 'nt':  # Windows
                    backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_FFMPEG]
                else:
                    backends = [cv2.CAP_FFMPEG]
                
                for backend in backends:
                    if cap is not None:
                        cap.release()
                    try:
                        cap = cv2.VideoCapture(video_path, backend)
                        if cap.isOpened():
                            break
                    except Exception:
                        continue
            
            if not cap.isOpened():
                raise Exception("Failed to open video with any backend")
            
            # Read first frame
            success, frame = cap.read()
            if success:
                # Create a thumbnail version
                height, width = frame.shape[:2]
                max_size = 800
                if height > max_size or width > max_size:
                    scale = max_size / max(height, width)
                    new_height = int(height * scale)
                    new_width = int(width * scale)
                    frame = cv2.resize(frame, (new_width, new_height))
                
                # Encode the frame to JPEG
                ret, buffer = cv2.imencode('.jpg', frame)
                if ret:
                    # Convert to base64 for sending in JSON
                    import base64
                    first_frame = base64.b64encode(buffer).decode('utf-8')
            
            # Get video info
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            cap.release()
            
            # If we just need the first frame, return it now
            if get_first_frame_only:
                return jsonify({
                    'success': True,
                    'first_frame': first_frame
                })
        
        except Exception as e:
            print(f"Error extracting first frame: {e}")
            import traceback
            traceback.print_exc()
        
        # Set the video path for processing if we're not just getting the first frame
        if not get_first_frame_only:
            # Initialize the processor with this video
            processor.set_video_path(video_path)
        
        # Return success response with video path and first frame
        return jsonify({
            'success': True,
            'video_path': video_path,
            'first_frame': first_frame,
            'message': 'Video uploaded and ready for processing'
        })
    
    except Exception as e:
        print(f"Unexpected error in process_video: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/video_process_feed')
def video_process_feed():
    """Stream processed video frames without mode display"""
    
    def generate():
        try:
            # Create a new video capture object directly
            video_path = processor.video_path
            if not video_path or not os.path.exists(video_path):
                print("Video path not set or file doesn't exist")
                return
            
            # Open the video file
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                print(f"Failed to open video: {video_path}")
                return
            
            # Get video properties
            fps = cap.get(cv2.CAP_PROP_FPS)
            if fps <= 0 or fps > 120:
                fps = 25.0  # Default if invalid
            
            frame_interval = 1000.0 / fps
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            # Process frames with timing control
            current_frame = 0
            last_frame_time = time.time()
            
            # Create MediaPipe Pose instance for video processing
            mp_pose = mp.solutions.pose
            pose = mp_pose.Pose(
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
                model_complexity=1,
                static_image_mode=False  # Important for video
            )
            
            print(f"Processing video: {video_path}")
            print(f"FPS: {fps}, Frames: {frame_count}")
            
            while True:
                # Read frame
                success, frame = cap.read()
                if not success:
                    print(f"End of video after {current_frame} frames")
                    break
                
                current_frame += 1
                
                # Process frame with MediaPipe
                try:
                    # Create a copy for drawing
                    processed_frame = frame.copy()
                    
                    # CRITICAL FIX: Set proper image processing for MediaPipe
                    # Convert to RGB (MediaPipe requires RGB)
                    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    
                    # Set image as non-writeable to save memory
                    image_rgb.flags.writeable = False
                    
                    # Get dimensions for fixing the MediaPipe error
                    height, width, _ = image_rgb.shape
                    
                    # Process the image
                    results = pose.process(image_rgb)
                    
                    # Set image back to writeable before drawing
                    image_rgb.flags.writeable = True
                    
                    # Draw landmarks if detected
                    if results.pose_landmarks:
                        mp.solutions.drawing_utils.draw_landmarks(
                            processed_frame,
                            results.pose_landmarks,
                            mp_pose.POSE_CONNECTIONS,
                            landmark_drawing_spec=mp.solutions.drawing_styles.get_default_pose_landmarks_style()
                        )
                        
                        # Store pose data for gesture detection
                        processor.last_pose_data = results.pose_landmarks
                        
                        # Process this frame through processor's sliding window
                        landmarks_data = []
                        for i, landmark in enumerate(results.pose_landmarks.landmark):
                            landmarks_data.append({
                                'x': landmark.x,
                                'y': landmark.y,
                                'z': landmark.z,
                                'visibility': landmark.visibility
                            })
                        
                        # Add to sliding window
                        if hasattr(processor, 'sliding_window'):
                            processor.sliding_window.append(landmarks_data)
                            
                            # Keep sliding window to specified size
                            if len(processor.sliding_window) > processor.sliding_window_size:
                                processor.sliding_window.pop(0)
                            
                            # Process for gesture recognition every step_size frames
                            processor.frame_count += 1
                            if processor.frame_count % processor.sliding_window_step_size == 0:
                                # Process sliding window through gesture recognition
                                if hasattr(processor, 'gesture_model') and hasattr(processor, 'gesture_processor'):
                                    recognized_gesture = processor.gesture_model.predict(
                                        processor.sliding_window, processor.excluded_landmarks)
                                    
                                    # Post-process the gesture
                                    event, mode, mode_percentage = processor.gesture_processor.process(
                                        recognized_gesture)
                                    
                                    # Log data for later retrieval
                                    timestamp = current_frame
                                    processor.log_data["timestamp"].append(timestamp)
                                    processor.log_data["events"].append(event)
                                    processor.log_data["gesture"].append(recognized_gesture)
                                    processor.log_data["mode"].append(mode)
                                    processor.log_data["mode_percentage"].append(mode_percentage)
                    
                    # Add frame count text - only show frame numbers, not mode
                    cv2.putText(
                        processed_frame,
                        f"Frame: {current_frame}/{frame_count}",
                        (20, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 0),
                        2
                    )
                    
                    # Add progress bar at the bottom of the frame
                    if frame_count > 0:
                        progress = current_frame / frame_count
                        bar_height = 10
                        progress_width = int(processed_frame.shape[1] * progress)
                        
                        cv2.rectangle(
                            processed_frame,
                            (0, processed_frame.shape[0] - bar_height),
                            (progress_width, processed_frame.shape[0]),
                            (0, 255, 0),
                            -1
                        )
                
                except Exception as e:
                    print(f"Error processing frame: {e}")
                    import traceback
                    traceback.print_exc()
                    processed_frame = frame  # Use original frame on error
                    
                    # Add error message to frame
                    cv2.putText(
                        processed_frame,
                        f"Error: {str(e)[:50]}",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 0, 255),
                        2
                    )
                
                # Timing control to maintain original FPS
                now = time.time()
                elapsed = (now - last_frame_time) * 1000  # ms
                
                if elapsed < frame_interval:
                    # Sleep to maintain timing
                    time.sleep((frame_interval - elapsed) / 1000.0)
                
                last_frame_time = time.time()
                
                # Encode and send frame
                try:
                    # Use JPEG encoding for browser compatibility
                    ret, buffer = cv2.imencode('.jpg', processed_frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                    if not ret:
                        print("Failed to encode frame")
                        continue
                    
                    # Send frame
                    frame_data = buffer.tobytes()
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + frame_data + b'\r\n')
                except Exception as e:
                    print(f"Error sending frame: {e}")
            
            # Return a signal to JavaScript that video has ended
            completion_frame = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(
                completion_frame,
                "Video Processing Complete",
                (50, 240),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 255),
                2
            )
            
            # Set processor state to indicate completion
            processor.current_frame = processor.video_frame_count  # Mark as finished
            
            # Encode and send the completion notification frame
            ret, buffer = cv2.imencode('.jpg', completion_frame)
            if ret:
                frame_data = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n'
                       b'X-Video-End: true\r\n\r\n' + frame_data + b'\r\n')
        
        except Exception as e:
            print(f"Video streaming error: {e}")
            import traceback
            traceback.print_exc()
            
            # Create an error frame to display to the user
            error_frame = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(
                error_frame,
                "Error processing video",
                (50, 240),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 255),
                2
            )
            cv2.putText(
                error_frame,
                str(e)[:50],
                (50, 280),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )
            
            # Encode and send the error frame
            ret, buffer = cv2.imencode('.jpg', error_frame)
            if ret:
                frame_data = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_data + b'\r\n')
        
        finally:
            # Clean up
            if 'cap' in locals() and cap is not None:
                cap.release()
                print("Video released")
            
            if 'pose' in locals():
                pose.close()
    
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/gesture_stream')
def gesture_stream():
    """Stream of detected gestures using Server-Sent Events (SSE) with enhanced data"""
    
    def generate():
        last_gesture = "idle"
        last_mode = "Registering"
        last_percentage = 0
        last_event = "idle"
        
        while True:
            # Get current gesture and its information
            current_gesture = processor.last_gesture
            current_event = "idle"  # Default: not an event
            
            # Get additional info if available
            current_mode = processor.gesture_processor.mode if hasattr(processor,
                                                                       'gesture_processor') else "Registering"
            current_percentage = 0
            
            # Check if this should be registered as an event
            if current_mode == "Registered" and hasattr(processor, 'gesture_processor'):
                # In Registered mode, the latest gesture has become an event
                current_event = processor.gesture_processor.registered_gesture
            
            if hasattr(processor, 'gesture_processor'):
                # Calculate mode percentage
                if current_mode == "Registering":
                    # Count most frequent non-idle gesture
                    gesture_counts = {}
                    for g in processor.gesture_processor.recognized_gestures:
                        if g != "idle" and g not in gesture_counts:
                            gesture_counts[g] = 0
                        if g != "idle":
                            gesture_counts[g] = gesture_counts.get(g, 0) + 1
                    
                    # Find max count
                    max_count = 0
                    for g, count in gesture_counts.items():
                        max_count = max(max_count, count)
                    
                    # Calculate percentage
                    buffer_size = len(processor.gesture_processor.recognized_gestures)
                    current_percentage = (max_count / buffer_size) * 100 if buffer_size > 0 else 0
                else:  # Registered mode
                    if processor.gesture_processor.registered_gesture:
                        # Count occurrences of registered gesture
                        registered_count = processor.gesture_processor.recognized_gestures.count(
                            processor.gesture_processor.registered_gesture)
                        
                        # Calculate percentage
                        buffer_size = len(processor.gesture_processor.recognized_gestures)
                        current_percentage = (registered_count / buffer_size) * 100 if buffer_size > 0 else 0
            
            # Only send when anything changes
            if (current_gesture != last_gesture or
                    current_mode != last_mode or
                    current_event != last_event or
                    abs(current_percentage - last_percentage) > 1.0):
                
                data = json.dumps({
                    'gesture': current_gesture,
                    'event': current_event,
                    'mode': current_mode,
                    'percentage': current_percentage
                })
                yield f"data: {data}\n\n"
                
                last_gesture = current_gesture
                last_mode = current_mode
                last_percentage = current_percentage
                last_event = current_event
                
                # Print debug info on event changes
                if current_event != "idle" and current_event != last_event:
                    print(f"* Gesture '{current_gesture}' registered as event: '{current_event}'")
            
            # Still need a small delay to prevent CPU overuse
            time.sleep(0.1)
    
    return Response(generate(), mimetype='text/event-stream')


@app.route('/get_gesture_log')
def get_gesture_log():
    """Get current gesture log data as HTML or JSON"""
    format_type = request.args.get('format', 'html')
    
    if format_type == 'json':
        # Return as JSON for JavaScript processing
        log_data = []
        
        if hasattr(processor, 'log_data') and processor.log_data["timestamp"]:
            for i in range(len(processor.log_data["timestamp"])):
                log_data.append({
                    'timestamp': processor.log_data["timestamp"][i],
                    'event': processor.log_data["events"][i],
                    'gesture': processor.log_data["gesture"][i],
                    'mode': processor.log_data["mode"][i],
                    'mode_percentage': processor.log_data["mode_percentage"][i]
                })
        
        return jsonify(log_data)
    else:
        # Return as pre-rendered HTML
        html = """
        <table class="gesture-log-table" style="width:100%; border-collapse: collapse;">
            <thead>
                <tr style="background-color: #f2f2f2;">
                    <th style="border:1px solid #ddd; padding:8px; text-align:left;">Frame</th>
                    <th style="border:1px solid #ddd; padding:8px; text-align:left;">Event</th>
                    <th style="border:1px solid #ddd; padding:8px; text-align:left;">Gesture</th>
                    <th style="border:1px solid #ddd; padding:8px; text-align:left;">Mode</th>
                    <th style="border:1px solid #ddd; padding:8px; text-align:left;">Confidence</th>
                </tr>
            </thead>
            <tbody>
        """
        
        if hasattr(processor, 'log_data') and processor.log_data["timestamp"]:
            for i in range(len(processor.log_data["timestamp"])):
                # Highlight non-idle events
                row_style = "background-color: #ffffcc;" if processor.log_data["events"][i] != "idle" else ""
                
                html += f"""
                <tr style="{row_style}">
                    <td style="border:1px solid #ddd; padding:8px;">{processor.log_data["timestamp"][i]}</td>
                    <td style="border:1px solid #ddd; padding:8px;">{processor.log_data["events"][i]}</td>
                    <td style="border:1px solid #ddd; padding:8px;">{processor.log_data["gesture"][i]}</td>
                    <td style="border:1px solid #ddd; padding:8px;">{processor.log_data["mode"][i]}</td>
                    <td style="border:1px solid #ddd; padding:8px;">{processor.log_data["mode_percentage"][i]:.1f}%</td>
                </tr>
                """
        else:
            html += """
            <tr>
                <td colspan="5" style="border:1px solid #ddd; padding:12px; text-align:center;">No gesture data recorded yet.</td>
            </tr>
            """
        
        html += """
            </tbody>
        </table>
        """
        
        return html


@app.route('/toggle_logging', methods=['POST'])
def toggle_logging():
    """Toggle data logging state"""
    is_logging = processor.toggle_logging()
    
    if not is_logging:
        # Logging was turned off, return the collected data
        # Format the CSV data with the specified columns
        request_path = request.path
        
        # Different column names for webcam and video
        if '/webcam' in request.headers.get('Referer', ''):
            csv_data = "time_elapsed,events,gesture,mode,mode_percentage\n"
        else:
            # Default to video format
            csv_data = "frame,events,gesture,mode,mode_percentage\n"
        
        # Add sample data
        if hasattr(processor, 'log_data') and processor.log_data["timestamp"]:
            for i in range(len(processor.log_data["timestamp"])):
                csv_data += f"{processor.log_data['timestamp'][i]},"
                csv_data += f"{processor.log_data['events'][i]},"
                csv_data += f"{processor.log_data['gesture'][i]},"
                csv_data += f"{processor.log_data['mode'][i]},"
                csv_data += f"{processor.log_data['mode_percentage'][i]:.1f}\n"
        
        return jsonify({
            'logging': False,
            'csv_data': csv_data
        })
    else:
        # Logging was turned on
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


@app.route('/check_video_status')
def check_video_status():
    """Check if video processing has completed"""
    # Check if the video is still playing
    if processor.video_cap is None or not processor.video_cap.isOpened():
        return jsonify({
            'completed': True,
            'message': 'Video processing complete'
        })
    
    # Check if we've reached the end of the video
    if hasattr(processor, 'current_frame') and hasattr(processor, 'video_frame_count'):
        if processor.current_frame >= processor.video_frame_count:
            return jsonify({
                'completed': True,
                'message': 'Video processing complete'
            })
    
    # Still processing
    return jsonify({
        'completed': False,
        'message': 'Video still processing'
    })


# Add this route to app.py
@app.route('/simulate_gesture', methods=['POST'])
def simulate_gesture():
    """
    Endpoint to simulate a gesture via keyboard input or API call.
    Simulated gestures undergo the same post-processing as real ones.
    """
    try:
        data = request.json
        gesture = data.get('gesture', 'idle')
        
        print(f"\n* Simulating gesture: {gesture}")
        
        # Only allow simulation if model isn't loaded
        if not hasattr(processor, 'gesture_model') or not processor.gesture_model.model_loaded:
            # First ensure we have a gesture processor
            if not hasattr(processor, 'gesture_processor'):
                return jsonify({
                    'success': False,
                    'message': 'Gesture processor not available',
                    'event': 'idle'
                }), 500
            
            # Add to the processor's last gesture info (for display purposes only)
            processor.last_gesture = gesture
            processor.last_gesture_time = time.time() * 1000
            
            # Process through post-processor to determine if this should become an event
            event, mode, mode_percentage = processor.gesture_processor.process(gesture)
            print(f"* Post-processed: gesture={gesture}, event={event}, mode={mode}, percentage={mode_percentage:.1f}%")
            
            # Add to log data
            timestamp = int(time.time() * 1000)
            processor.log_data["timestamp"].append(timestamp)
            processor.log_data["events"].append(event)  # This is key - only events trigger actions
            processor.log_data["gesture"].append(gesture)
            processor.log_data["mode"].append(mode)
            processor.log_data["mode_percentage"].append(mode_percentage)
            
            # Return both the simulated gesture and the post-processed event
            return jsonify({
                'success': True,
                'message': f'Simulated gesture: {gesture}',
                'model_active': False,
                'gesture': gesture,  # The raw simulated gesture
                'event': event,  # The post-processed event (might be 'idle')
                'mode': mode,
                'mode_percentage': mode_percentage
            })
        else:
            print("* Cannot simulate gesture when model is active")
            return jsonify({
                'success': False,
                'message': 'Cannot simulate gesture when model is active',
                'model_active': True,
                'event': 'idle'
            })
    
    except Exception as e:
        print(f"* Error simulating gesture: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'message': f'Error simulating gesture: {str(e)}',
            'event': 'idle'
        }), 500


@app.route('/check_model_status')
def check_model_status():
    """Check if the gesture recognition model is loaded"""
    model_loaded = False
    
    try:
        if hasattr(processor, 'gesture_model') and processor.gesture_model:
            model_loaded = processor.gesture_model.model_loaded
        
        return jsonify({
            'model_loaded': model_loaded,
            'gestures': processor.gesture_model.gestures if model_loaded else []
        })
    except Exception as e:
        return jsonify({
            'model_loaded': False,
            'error': str(e)
        })


if __name__ == '__main__':
    # Create upload folder if it doesn't exist
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
    
    open_browser()
    
    # Print clear startup message
    print("\n* Starting Flask development server...")
    print("* Debug mode: ENABLED")
    print("* Server will be available at: http://127.0.0.1:5000/")
    
    # Run the app with debug explicitly set
    app.run(host='127.0.0.1', port=5000, debug=True)