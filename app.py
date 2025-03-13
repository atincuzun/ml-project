from flask import Flask, render_template, request, Response, jsonify
import os
import json
import webbrowser
from threading import Timer
import cv2
import time
from utils.mediapipe_processor import MediapipeProcessor
from utils.data_handler import CSVDataHandler

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


@app.route('/get_cameras')
def get_cameras():
	"""Get a list of available camera devices"""
	cameras = []
	index = 0
	
	while True:
		cap = cv2.VideoCapture(index)
		if not cap.isOpened():
			break
		
		cameras.append({
			'id': index,
			'name': f'Camera {index}'
		})
		cap.release()
		index += 1
	
	return jsonify(cameras)


@app.route('/video_feed')
def video_feed():
	"""Video streaming route for webcam"""
	camera_id = request.args.get('camera_id', default=0, type=int)
	
	def generate():
		cap = cv2.VideoCapture(camera_id)
		
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
			
			# Add gesture info
			gesture_data = json.dumps({"gesture": gesture}).encode('utf-8')
			combined_data += gesture_data
			
			yield (b'--frame\r\n'
			       b'Content-Type: image/jpeg\r\n\r\n' + combined_data + b'\r\n')
		
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
			
			# Add gesture info
			gesture_data = json.dumps({"gesture": gesture}).encode('utf-8')
			combined_data += gesture_data
			
			yield (b'--frame\r\n'
			       b'Content-Type: image/jpeg\r\n\r\n' + combined_data + b'\r\n')
	
	return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')


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
	presentation_path = f'/static/presentations/{name}/index.html'
	return jsonify({'path': presentation_path})


@app.route('/get_gesture')
def get_gesture():
	"""Get the current detected gesture for Tetris"""
	return jsonify({'gesture': processor.last_gesture})


if __name__ == '__main__':
	# Open browser tab after a small delay
	Timer(1.0, open_browser).start()
	
	# Create upload folder if it doesn't exist
	if not os.path.exists(app.config['UPLOAD_FOLDER']):
		os.makedirs(app.config['UPLOAD_FOLDER'])
	
	app.run(debug=True)