document.addEventListener('DOMContentLoaded', function() {
    // DOM Elements
    const selectCameraBtn = document.getElementById('selectCameraBtn');
    const startLoggingBtn = document.getElementById('startLoggingBtn');
    const saveLogBtn = document.getElementById('saveLogBtn');
    const cameraSelect = document.getElementById('cameraSelect');
    const cameraImage = document.getElementById('cameraImage');
    const gestureIcon = document.getElementById('gestureIcon');
    const gestureText = document.getElementById('gestureText');
    const saveModal = document.getElementById('saveModal');
    const confirmSaveBtn = document.getElementById('confirmSaveBtn');
    const cancelSaveBtn = document.getElementById('cancelSaveBtn');

    // State variables
    let selectedCamera = 0;
    let isStreaming = false;
    let isLogging = false;
    let logStartTime = 0;
    let logData = '';
    let lastGestureTime = 0;
    const GESTURE_COOLDOWN = 1000; // 1 second cooldown between gesture simulations
    let currentTimestamp = 0;

    // Gesture history tracking
    const gestureHistory = [];
    const MAX_HISTORY_ITEMS = 5;
    let gestureDisplayTimeout = null;
    const GESTURE_DISPLAY_DURATION = 3000; // 3 seconds

    // Gesture icons
    const gestureIcons = {
        'swipe_left': '/static/img/swipe_left.png',
        'swipe_right': '/static/img/swipe_right.png',
        'rotate_cw': '/static/img/rotate_cw.png',
        'rotate_ccw': '/static/img/rotate_ccw.png',
        'hand_up': '/static/img/hand_up.png',
        'hand_down': '/static/img/hand_down.png'
    };

    // Create gesture overlay element
    createGestureOverlay();

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

                // Update current timestamp
                currentTimestamp = Date.now() - logStartTime;

                if (data.gesture !== 'idle') {
                    // Update the gesture display
                    updateGestureDisplay(data.gesture);

                    // Add to gesture history
                    addToGestureHistory(data.gesture, currentTimestamp);

                    // Update the gesture history display
                    updateGestureHistoryDisplay();
                }

                // Always update the current timestamp display
                updateTimestampDisplay(currentTimestamp);
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
            e.key === 'ArrowUp' || e.key === 'ArrowDown' ||
            e.key === 'u' || e.key === 'd') {
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
                case 'u':
                    gesture = 'hand_up';
                    break;
                case 'd':
                    gesture = 'hand_down';
                    break;
            }

            if (gesture) {
                updateGestureDisplay(gesture);
                addToGestureHistory(gesture, currentTimestamp);
                updateGestureHistoryDisplay();
            }
        }
    });

    // Functions
    function createGestureOverlay() {
        // Create overlay container
        const overlayContainer = document.createElement('div');
        overlayContainer.id = 'gestureOverlay';
        overlayContainer.className = 'gesture-overlay';

        // Create current timestamp display
        const timestampDisplay = document.createElement('div');
        timestampDisplay.id = 'currentTimestamp';
        timestampDisplay.className = 'current-timestamp';
        timestampDisplay.textContent = 'Time: 0 ms';

        // Create gesture history container
        const historyContainer = document.createElement('div');
        historyContainer.id = 'gestureHistory';
        historyContainer.className = 'gesture-history';
        historyContainer.innerHTML = '<h4>Recent Gestures</h4><ul></ul>';

        // Add elements to overlay
        overlayContainer.appendChild(timestampDisplay);
        overlayContainer.appendChild(historyContainer);

        // Add overlay to the page
        const container = document.querySelector('.container');
        container.appendChild(overlayContainer);

        // Add styles for overlay
        const style = document.createElement('style');
        style.textContent = `
            .gesture-overlay {
                position: fixed;
                top: 20px;
                right: 20px;
                background-color: rgba(0, 0, 0, 0.7);
                color: white;
                padding: 15px;
                border-radius: 8px;
                width: 250px;
                z-index: 1000;
            }
            
            .current-timestamp {
                font-size: 16px;
                font-weight: bold;
                margin-bottom: 10px;
                text-align: center;
            }
            
            .gesture-history h4 {
                margin: 0 0 10px 0;
                text-align: center;
                border-bottom: 1px solid rgba(255, 255, 255, 0.3);
                padding-bottom: 5px;
            }
            
            .gesture-history ul {
                list-style: none;
                padding: 0;
                margin: 0;
            }
            
            .gesture-history li {
                display: flex;
                justify-content: space-between;
                margin-bottom: 5px;
                padding: 5px;
                background-color: rgba(255, 255, 255, 0.1);
                border-radius: 4px;
            }
            
            .gesture-history li:last-child {
                margin-bottom: 0;
            }
            
            .gesture-history .gesture-name {
                font-weight: bold;
            }
            
            .gesture-history .gesture-time {
                opacity: 0.8;
            }
            
            .gesture-icon {
                transition: all 0.3s ease-in-out;
            }
            
            .gesture-icon.active {
                transform: scale(1.1);
                box-shadow: 0 0 20px rgba(52, 152, 219, 0.7);
            }
        `;
        document.head.appendChild(style);
    }

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

        // Initialize log start time when streaming begins
        logStartTime = Date.now();
    }

    function stopVideoStream() {
        if (!isStreaming) return;
        isStreaming = false;
    }

    function updateGestureDisplay(gesture) {
        // Clear any existing timeout
        if (gestureDisplayTimeout) {
            clearTimeout(gestureDisplayTimeout);
        }

        if (gesture === 'idle' || !gestureIcons[gesture]) {
            // Set default appearance for idle or unknown gestures
            gestureIcon.innerHTML = '<span id="gestureText">No gesture</span>';
            gestureIcon.style.backgroundColor = '#f4f4f4';
            gestureIcon.classList.remove('active');
        } else {
            // Use gesture icon images
            gestureIcon.innerHTML = `
                <img src="${gestureIcons[gesture]}" alt="${formatGestureName(gesture)}" style="width: 80px; height: 80px;">
                <span id="gestureText">${formatGestureName(gesture)}</span>
            `;
            gestureIcon.style.backgroundColor = '#3498db';
            gestureIcon.classList.add('active');

            // Set timeout to revert after 3 seconds
            gestureDisplayTimeout = setTimeout(() => {
                gestureIcon.innerHTML = '<span id="gestureText">No gesture</span>';
                gestureIcon.style.backgroundColor = '#f4f4f4';
                gestureIcon.classList.remove('active');
            }, GESTURE_DISPLAY_DURATION);
        }
    }

    function formatGestureName(gesture) {
        // Convert snake_case to Title Case
        return gesture.split('_')
            .map(word => word.charAt(0).toUpperCase() + word.slice(1))
            .join(' ');
    }

    function addToGestureHistory(gesture, timestamp) {
        // Add gesture to history
        gestureHistory.unshift({
            gesture: gesture,
            timestamp: timestamp
        });

        // Keep only the most recent items
        while (gestureHistory.length > MAX_HISTORY_ITEMS) {
            gestureHistory.pop();
        }
    }

    function updateGestureHistoryDisplay() {
        const historyList = document.querySelector('#gestureHistory ul');
        if (!historyList) return;

        // Clear the list
        historyList.innerHTML = '';

        // Add each history item
        gestureHistory.forEach(item => {
            const listItem = document.createElement('li');

            const gestureName = document.createElement('span');
            gestureName.className = 'gesture-name';
            gestureName.textContent = formatGestureName(item.gesture);

            const gestureTime = document.createElement('span');
            gestureTime.className = 'gesture-time';
            gestureTime.textContent = `${item.timestamp} ms`;

            listItem.appendChild(gestureName);
            listItem.appendChild(gestureTime);

            historyList.appendChild(listItem);
        });
    }

    function updateTimestampDisplay(timestamp) {
        const timestampDisplay = document.getElementById('currentTimestamp');
        if (timestampDisplay) {
            timestampDisplay.textContent = `Time: ${timestamp} ms`;
        }
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

                // Reset gesture history
                gestureHistory.length = 0;
                updateGestureHistoryDisplay();
            } else {
                // Logging stopped
                startLoggingBtn.textContent = 'Start Logging';
                saveLogBtn.disabled = false;
                logData = data.csv_data;
            }
        } catch (error) {
            console.error('Error toggling logging:', error);
        }
    }

    async function saveCSVLog() {
        try {
            const fullPath = `${document.getElementById('filePath').value}${document.getElementById('fileName').value}`;

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