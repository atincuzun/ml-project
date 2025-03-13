document.addEventListener('DOMContentLoaded', function() {
    // DOM Elements
    const selectCameraBtn = document.getElementById('selectCameraBtn');
    const startLoggingBtn = document.getElementById('startLoggingBtn');
    const saveLogBtn = document.getElementById('saveLogBtn');
    const cameraSelect = document.getElementById('cameraSelect');
    const cameraImage = document.getElementById('cameraImage');
    const poseImage = document.getElementById('poseImage');
    const gestureIcon = document.getElementById('gestureIcon');
    const gestureText = document.getElementById('gestureText');
    const logOutput = document.getElementById('logOutput');
    const saveModal = document.getElementById('saveModal');
    const confirmSaveBtn = document.getElementById('confirmSaveBtn');
    const cancelSaveBtn = document.getElementById('cancelSaveBtn');
    const fileName = document.getElementById('fileName');
    const filePath = document.getElementById('filePath');

    // State variables
    let selectedCamera = 0;
    let isStreaming = false;
    let isLogging = false;
    let logStartTime = 0;
    let logData = '';

    // Load available cameras
    fetchCameras();

    // Event listeners
    selectCameraBtn.addEventListener('click', () => {
        cameraSelect.style.display = cameraSelect.style.display === 'inline-block' ? 'none' : 'inline-block';
    });

    cameraSelect.addEventListener('change', () => {
        selectedCamera = cameraSelect.value;
        if (selectedCamera) {
            stopVideoStream();
            startVideoStream();
            cameraSelect.style.display = 'none';
        }
    });

    startLoggingBtn.addEventListener('click', () => {
        toggleLogging();
    });

    saveLogBtn.addEventListener('click', () => {
        saveModal.style.display = 'flex';
    });

    confirmSaveBtn.addEventListener('click', () => {
        saveCSVLog();
    });

    cancelSaveBtn.addEventListener('click', () => {
        saveModal.style.display = 'none';
    });

    // Functions
    async function fetchCameras() {
        try {
            const response = await fetch('/get_cameras');
            const cameras = await response.json();

            cameraSelect.innerHTML = '';
            if (cameras.length === 0) {
                cameraSelect.innerHTML = '<option value="">No cameras found</option>';
            } else {
                cameras.forEach(camera => {
                    const option = document.createElement('option');
                    option.value = camera.id;
                    option.textContent = camera.name;
                    cameraSelect.appendChild(option);
                });

                // Select first camera by default
                selectedCamera = cameras[0].id;
                cameraSelect.value = selectedCamera;

                // Start stream with default camera
                startVideoStream();
            }
        } catch (error) {
            console.error('Error fetching cameras:', error);
            cameraSelect.innerHTML = '<option value="">Error loading cameras</option>';
        }
    }

    function startVideoStream() {
        if (isStreaming) return;

        const videoUrl = `/video_feed?camera_id=${selectedCamera}`;

        // Set up the event source for video feed
        const eventSource = new EventSource(videoUrl);

        // Handle streaming events
        eventSource.onmessage = function(event) {
            // Split the combined data (original image, pose image, gesture data)
            const parts = event.data.split('FRAME_DELIMITER');

            if (parts.length >= 3) {
                // Handle original image
                const originalImageBlob = new Blob([parts[0]], { type: 'image/jpeg' });
                cameraImage.src = URL.createObjectURL(originalImageBlob);

                // Handle pose image
                const poseImageBlob = new Blob([parts[1]], { type: 'image/jpeg' });
                poseImage.src = URL.createObjectURL(poseImageBlob);

                // Handle gesture data
                try {
                    const gestureData = JSON.parse(parts[2]);
                    updateGestureDisplay(gestureData.gesture);

                    // If logging is active, log the gesture
                    if (isLogging && gestureData.gesture !== 'idle') {
                        logGesture(gestureData.gesture);
                    }
                } catch (e) {
                    console.error('Error parsing gesture data:', e);
                }
            }
        };

        eventSource.onerror = function(error) {
            console.error('Error with video stream:', error);
            stopVideoStream();
        };

        isStreaming = true;
    }

    function stopVideoStream() {
        if (!isStreaming) return;

        // Close event source
        if (window.eventSource) {
            window.eventSource.close();
        }

        isStreaming = false;
    }

    function updateGestureDisplay(gesture) {
        if (gesture === 'idle') {
            gestureIcon.style.backgroundColor = '#f4f4f4';
            gestureText.textContent = 'No gesture';
        } else {
            gestureIcon.style.backgroundColor = '#3498db';
            gestureText.textContent = formatGestureName(gesture);

            // Flash effect
            setTimeout(() => {
                gestureIcon.style.backgroundColor = '#f4f4f4';
                gestureText.textContent = 'No gesture';
            }, 1000);
        }
    }

    function formatGestureName(gesture) {
        // Convert snake_case to Title Case
        return gesture.split('_')
            .map(word => word.charAt(0).toUpperCase() + word.slice(1))
            .join(' ');
    }

    async function toggleLogging() {
        try {
            const response = await fetch('/toggle_logging', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                }
            });

            const data = await response.json();
            isLogging = data.logging;

            if (isLogging) {
                // Logging started
                startLoggingBtn.textContent = 'Stop Logging';
                saveLogBtn.disabled = true;
                logStartTime = Date.now();
                logOutput.value = 'timestamp,events\n';
            } else {
                // Logging stopped
                startLoggingBtn.textContent = 'Start Logging';
                saveLogBtn.disabled = false;
                logData = data.csv_data;
                logOutput.value = logData;
            }
        } catch (error) {
            console.error('Error toggling logging:', error);
        }
    }

    function logGesture(gesture) {
        const timeElapsed = Date.now() - logStartTime;
        const newLogEntry = `${timeElapsed},${gesture}\n`;
        logOutput.value += newLogEntry;

        // Scroll to bottom of log
        logOutput.scrollTop = logOutput.scrollHeight;
    }

    async function saveCSVLog() {
        try {
            const fullPath = `${filePath.value}${fileName.value}`;

            const response = await fetch('/save_csv', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    file_path: fullPath
                })
            });

            const data = await response.json();

            if (data.success) {
                alert(`Log saved successfully to ${data.file_path}`);
            } else {
                alert(`Error saving log: ${data.error}`);
            }

            // Close modal
            saveModal.style.display = 'none';
        } catch (error) {
            console.error('Error saving CSV:', error);
            alert('Error saving CSV file');
        }
    }

    // Clean up resources when page unloads
    window.addEventListener('beforeunload', () => {
        stopVideoStream();
    });
});