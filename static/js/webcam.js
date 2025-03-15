document.addEventListener('DOMContentLoaded', function() {
    // DOM Elements
    const selectCameraBtn = document.getElementById('selectCameraBtn');
    const startLoggingBtn = document.getElementById('startLoggingBtn');
    const saveLogBtn = document.getElementById('saveLogBtn');
    const cameraSelect = document.getElementById('cameraSelect');
    const cameraImage = document.getElementById('cameraImage');
    const gestureIcon = document.getElementById('gestureIcon');
    const gestureText = document.getElementById('gestureText');
    const logOutput = document.getElementById('logOutput');
    const saveModal = document.getElementById('saveModal');
    const confirmSaveBtn = document.getElementById('confirmSaveBtn');
    const cancelSaveBtn = document.getElementById('cancelSaveBtn');
    const fileName = document.getElementById('fileName');
    const filePath = document.getElementById('filePath');
    const keyboardInfoElement = document.getElementById('keyboardInfo');

    // State variables
    let selectedCamera = 0;
    let isStreaming = false;
    let isLogging = false;
    let logStartTime = 0;
    let logData = '';
    let lastGestureTime = 0;
    const GESTURE_COOLDOWN = 1000; // 1 second cooldown between gesture simulations

    // Gesture icons
    const gestureIcons = {
        'swipe_left': '/static/img/swipe_left.png',
        'swipe_right': '/static/img/swipe_right.png',
        'rotate_cw': '/static/img/rotate_cw.png',
        'rotate_ccw': '/static/img/rotate_ccw.png'
    };

    // Load available cameras
    fetchCameras();

    // Connect to gesture stream
    connectToGestureStream();

    // Connect to the Server-Sent Events stream for gestures
    function connectToGestureStream() {
        const eventSource = new EventSource('/gesture_stream');

        eventSource.onmessage = function(event) {
            try {
                const data = JSON.parse(event.data);
                if (data.gesture !== 'idle') {
                    updateGestureDisplay(data.gesture);

                    // If logging is active, log the gesture
                    if (isLogging) {
                        logGesture(data.gesture);
                    }
                }
            } catch (error) {
                console.error('Error parsing gesture data:', error);
            }
        };

        eventSource.onerror = function(error) {
            console.error('EventSource error:', error);
            setTimeout(() => {
                console.log('Reconnecting to gesture stream...');
                connectToGestureStream();
            }, 3000);
        };

        // Store the eventSource to close it when needed
        window.gestureEventSource = eventSource;
    }

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

    // Add keyboard event listener for gesture simulation
    document.addEventListener('keydown', (e) => {
        if (e.key === 'ArrowLeft' || e.key === 'ArrowRight' ||
            e.key === 'ArrowUp' || e.key === 'ArrowDown') {
            // Prevent default behavior for arrow keys
            e.preventDefault();

            // Check cooldown to prevent gesture spamming
            const currentTime = Date.now();
            if (currentTime - lastGestureTime < GESTURE_COOLDOWN) {
                return;
            }

            lastGestureTime = currentTime;

            let gesture = '';
            switch (e.key) {
                case 'ArrowLeft':
                    gesture = 'swipe_left';
                    break;
                case 'ArrowRight':
                    gesture = 'swipe_right';
                    break;
                case 'ArrowUp':
                    gesture = 'rotate_cw';
                    break;
                case 'ArrowDown':
                    gesture = 'rotate_ccw';
                    break;
            }

            if (gesture) {
                updateGestureDisplay(gesture);
                if (isLogging) {
                    logGesture(gesture);
                }
            }
        }
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
        cameraImage.src = videoUrl;
        isStreaming = true;
    }

    function stopVideoStream() {
        if (!isStreaming) return;
        isStreaming = false;
    }

    function updateGestureDisplay(gesture) {
        if (gesture === 'idle' || !gestureIcons[gesture]) {
            // Set default appearance for idle or unknown gestures
            gestureIcon.innerHTML = '<span id="gestureText">No gesture</span>';
            gestureIcon.style.backgroundColor = '#f4f4f4';
        } else {
            // Use gesture icon images
            gestureIcon.innerHTML = `
                <img src="${gestureIcons[gesture]}" alt="${formatGestureName(gesture)}" style="width: 80px; height: 80px;">
                <span id="gestureText">${formatGestureName(gesture)}</span>
            `;
            gestureIcon.style.backgroundColor = '#3498db';

            // Flash effect
            setTimeout(() => {
                gestureIcon.innerHTML = '<span id="gestureText">No gesture</span>';
                gestureIcon.style.backgroundColor = '#f4f4f4';
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
        if (window.gestureEventSource) {
            window.gestureEventSource.close();
        }
    });
});