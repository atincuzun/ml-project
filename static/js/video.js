document.addEventListener('DOMContentLoaded', function() {
    // DOM Elements
    const selectVideoBtn = document.getElementById('selectVideoBtn');
    const startVideoBtn = document.getElementById('startVideoBtn');
    const saveLogBtn = document.getElementById('saveLogBtn');
    const videoFileInput = document.getElementById('videoFileInput');
    const selectedVideo = document.getElementById('selectedVideo');
    const videoImage = document.getElementById('videoImage');
    const gestureIcon = document.getElementById('gestureIcon');
    const gestureText = document.getElementById('gestureText');
    const logOutput = document.getElementById('logOutput');
    const progressModal = document.getElementById('progressModal');
    const progressBar = document.getElementById('progressBar');
    const progressText = document.getElementById('progressText');
    const saveModal = document.getElementById('saveModal');
    const confirmSaveBtn = document.getElementById('confirmSaveBtn');
    const cancelSaveBtn = document.getElementById('cancelSaveBtn');

    // State variables
    let selectedVideoFile = null;
    let isProcessing = false;
    let isVideoPlaying = false;
    let videoEventSource = null;

    // Gesture icons
    const gestureIcons = {
        'swipe_left': '/static/img/swipe_left.png',
        'swipe_right': '/static/img/swipe_right.png',
        'rotate_cw': '/static/img/rotate_cw.png',
        'rotate_ccw': '/static/img/rotate_ccw.png'
    };

    // Event listeners
    selectVideoBtn.addEventListener('click', () => {
        videoFileInput.click();
    });

    videoFileInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) {
            selectedVideoFile = e.target.files[0];
            selectedVideo.textContent = selectedVideoFile.name;

            // Automatically process the video when selected
            processVideo();
        }
    });

    startVideoBtn.addEventListener('click', () => {
        if (isVideoPlaying) {
            stopVideo();
            startVideoBtn.textContent = 'Start Video';
        } else {
            startVideoPlayback();
            startVideoBtn.textContent = 'Stop Video';
        }
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
    async function processVideo() {
        if (!selectedVideoFile || isProcessing) return;

        isProcessing = true;

        // Show progress modal
        progressModal.style.display = 'flex';
        progressBar.style.width = '0%';
        progressText.textContent = 'Processing video...';

        // Create FormData to send the file
        const formData = new FormData();
        formData.append('video', selectedVideoFile);

        try {
            // Upload video for processing
            const response = await fetch('/process_video', {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                throw new Error('Failed to process video');
            }

            const data = await response.json();

            // Update progress
            progressBar.style.width = '100%';
            progressText.textContent = 'Processing complete!';

            // Enable start button
            startVideoBtn.disabled = false;

            // Update progress modal
            setTimeout(() => {
                progressModal.style.display = 'none';
                isProcessing = false;
            }, 1000);

        } catch (error) {
            console.error('Error processing video:', error);
            progressModal.style.display = 'none';
            alert('Error processing video: ' + error.message);
            isProcessing = false;
        }
    }

    function startVideoPlayback() {
        // Clear previous log
        logOutput.value = 'timestamp,events\n';

        // Enable logging on server
        fetch('/toggle_logging', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            }
        });

        // Set up EventSource for video stream
        videoEventSource = new EventSource('/video_process_feed');

        // Connect to gesture stream for the video processor
        const gestureEventSource = new EventSource('/gesture_stream');

        gestureEventSource.onmessage = function(event) {
            try {
                const data = JSON.parse(event.data);
                if (data.gesture !== 'idle') {
                    updateGestureDisplay(data.gesture);
                    logGesture(data.gesture);
                }
            } catch (error) {
                console.error('Error parsing gesture data:', error);
            }
        };

        videoEventSource.onmessage = function(event) {
            // Split the combined data (original image, pose image)
            const parts = event.data.split('FRAME_DELIMITER');

            if (parts.length >= 3) {
                // Handle original image (which should now contain MediaPipe landmarks)
                const imageBlob = new Blob([parts[0]], { type: 'image/jpeg' });
                videoImage.src = URL.createObjectURL(imageBlob);
            }
        };

        videoEventSource.onerror = function(error) {
            console.log('Video stream ended or error occurred');
            stopVideo();
            videoCompleted();
            if (gestureEventSource) {
                gestureEventSource.close();
            }
        };

        // Store the gestureEventSource to close it when needed
        window.gestureEventSource = gestureEventSource;

        isVideoPlaying = true;
        saveLogBtn.disabled = true;
    }

    function stopVideo() {
        if (videoEventSource) {
            videoEventSource.close();
            videoEventSource = null;
        }

        if (window.gestureEventSource) {
            window.gestureEventSource.close();
        }

        isVideoPlaying = false;

        // Disable logging on server and get the log data
        fetch('/toggle_logging', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            }
        })
        .then(response => response.json())
        .then(data => {
            if (!data.logging) {
                logOutput.value = data.csv_data;
                saveLogBtn.disabled = false;
            }
        });
    }

    function videoCompleted() {
        startVideoBtn.textContent = 'Start Video';
        saveLogBtn.disabled = false;

        // Alert the user
        alert('Video processing complete. You can now save the log.');
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

    function logGesture(gesture) {
        const timestamp = new Date().getTime();
        const newLogEntry = `${timestamp},${gesture}\n`;
        logOutput.value += newLogEntry;

        // Scroll to bottom of log
        logOutput.scrollTop = logOutput.scrollHeight;
    }

    async function saveCSVLog() {
        try {
            const response = await fetch('/save_csv', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    file_path: 'performance_results.csv'
                })
            });

            const data = await response.json();

            if (data.success) {
                alert(`Log saved successfully as performance_results.csv`);
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
        if (isVideoPlaying) {
            stopVideo();
        }
    });
});