document.addEventListener('DOMContentLoaded', function() {
    // DOM Elements
    const selectVideoBtn = document.getElementById('selectVideoBtn');
    const startVideoBtn = document.getElementById('startVideoBtn');
    const saveLogBtn = document.getElementById('saveLogBtn');
    const videoFileInput = document.getElementById('videoFileInput');
    const selectedVideo = document.getElementById('selectedVideo');
    const videoFeed = document.getElementById('videoFeed');
    const gestureIcon = document.getElementById('gestureIcon');
    const gestureText = document.getElementById('gestureText');
    const progressModal = document.getElementById('progressModal');
    const progressBar = document.getElementById('progressBar');
    const progressText = document.getElementById('progressText');
    const saveModal = document.getElementById('saveModal');
    const confirmSaveBtn = document.getElementById('confirmSaveBtn');
    const cancelSaveBtn = document.getElementById('cancelSaveBtn');
    const loadingOverlay = document.getElementById('loadingOverlay');
    const loadingText = document.getElementById('loadingText');

    // State variables
    let selectedVideoFile = null;
    let isProcessing = false;
    let isVideoPlaying = false;
    let uploadedVideoPath = null;
    let gestureEventSource = null;
    let videoStreamImg = null;
    let videoEndCheckInterval = null;
    let currentFrame = 0;

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

    // Event listeners
    selectVideoBtn.addEventListener('click', () => {
        videoFileInput.click();
    });

    videoFileInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) {
            selectedVideoFile = e.target.files[0];
            selectedVideo.textContent = selectedVideoFile.name;

            // Upload the video automatically
            uploadVideo();
        }
    });

    startVideoBtn.addEventListener('click', () => {
        if (isVideoPlaying) {
            stopVideo();
            startVideoBtn.textContent = 'Start Video';
        } else {
            if (uploadedVideoPath) {
                startVideoPlayback();
                startVideoBtn.textContent = 'Stop Video';
            } else {
                alert('Please select and upload a video first');
            }
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
    function createGestureOverlay() {
        // Create overlay container
        const overlayContainer = document.createElement('div');
        overlayContainer.id = 'gestureOverlay';
        overlayContainer.className = 'gesture-overlay';

        // Create current frame display
        const frameDisplay = document.createElement('div');
        frameDisplay.id = 'currentFrame';
        frameDisplay.className = 'current-frame';
        frameDisplay.textContent = 'Frame: 0 (0.0 sec)';

        // Create gesture history container
        const historyContainer = document.createElement('div');
        historyContainer.id = 'gestureHistory';
        historyContainer.className = 'gesture-history';
        historyContainer.innerHTML = '<h4>Recent Gestures</h4><ul></ul>';

        // Add elements to overlay
        overlayContainer.appendChild(frameDisplay);
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
            
            .current-frame {
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
            
            .gesture-history .gesture-frame {
                opacity: 0.8;
            }
            
            .gesture-icon {
                transition: all 0.3s ease-in-out;
            }
            
            .gesture-icon.active {
                transform: scale(1.1);
                box-shadow: 0 0 20px rgba(52, 152, 219, 0.7);
            }
            
            .log-container {
                display: none; /* Hide the log textarea */
            }
        `;
        document.head.appendChild(style);
    }

    async function uploadVideo() {
        if (!selectedVideoFile || isProcessing) return;

        isProcessing = true;

        // Show progress modal
        progressModal.style.display = 'flex';
        progressBar.style.width = '0%';
        progressText.textContent = 'Uploading video...';

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
                throw new Error('Failed to upload video');
            }

            const data = await response.json();
            uploadedVideoPath = data.video_path;

            // Update progress
            progressBar.style.width = '100%';
            progressText.textContent = 'Video uploaded successfully!';

            // Enable start button
            startVideoBtn.disabled = false;

            // Display first frame if available
            if (data.first_frame) {
                // Create img element if it doesn't exist
                if (!videoStreamImg) {
                    videoStreamImg = document.createElement('img');
                    videoStreamImg.alt = "Video feed";
                    videoStreamImg.style.width = "100%";
                    videoStreamImg.style.height = "100%";
                    videoStreamImg.style.objectFit = "contain";

                    // Clear videoFeed contents
                    while (videoFeed.firstChild) {
                        videoFeed.removeChild(videoFeed.firstChild);
                    }

                    videoFeed.appendChild(videoStreamImg);
                }

                videoStreamImg.src = 'data:image/jpeg;base64,' + data.first_frame;
            }

            // Close progress modal after a delay
            setTimeout(() => {
                progressModal.style.display = 'none';
                isProcessing = false;
            }, 1000);

        } catch (error) {
            console.error('Error uploading video:', error);
            progressModal.style.display = 'none';
            alert('Error uploading video: ' + error.message);
            isProcessing = false;
        }
    }

    function startVideoPlayback() {
        if (!uploadedVideoPath) {
            alert('No video has been uploaded. Please select a video first.');
            return;
        }

        // Reset frame counter and gesture history
        currentFrame = 0;
        gestureHistory.length = 0;
        updateGestureHistoryDisplay();
        updateFrameDisplay(0);

        // Enable logging on server
        fetch('/toggle_logging', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            }
        });

        // Show loading until first frame appears
        loadingOverlay.style.display = 'flex';
        loadingText.textContent = 'Starting video...';

        // Connect to gesture stream
        connectToGestureStream();

        // Create a completely new image element for each playback
        if (videoStreamImg) {
            videoStreamImg.remove();
        }

        videoStreamImg = document.createElement('img');
        videoStreamImg.alt = "Video feed";
        videoStreamImg.style.width = "100%";
        videoStreamImg.style.height = "100%";
        videoStreamImg.style.objectFit = "contain";

        // Clear existing content
        while (videoFeed.firstChild) {
            videoFeed.removeChild(videoFeed.firstChild);
        }

        // Set up error handling before setting src
        videoStreamImg.onerror = function(e) {
            console.error('Error loading video stream:', e);
            loadingOverlay.style.display = 'none';
            alert('Error playing video. Please try again.');
            stopVideo();
            startVideoBtn.textContent = 'Start Video';
        };

        // Set up load handler to check for video completion
        videoStreamImg.onload = function() {
            // Hide loading overlay after first frame loads
            loadingOverlay.style.display = 'none';

            // Check for video end signal
            if (this.src.includes('X-Video-End')) {
                videoCompleted();
                return;
            }
        };

        // Add timestamp and unique identifier to prevent caching
        const timestamp = new Date().getTime();
        const uniqueId = Math.random().toString(36).substring(2, 15);
        videoStreamImg.src = `/video_process_feed?t=${timestamp}&id=${uniqueId}`;

        // Add to DOM
        videoFeed.appendChild(videoStreamImg);

        isVideoPlaying = true;
        saveLogBtn.disabled = true;

        // Set up video end detection
        setupVideoEndDetection();
    }

    function connectToGestureStream() {
        // Close existing connection if any
        if (gestureEventSource) {
            gestureEventSource.close();
        }

        // Connect to gesture stream for the video processor
        gestureEventSource = new EventSource('/gesture_stream');

        gestureEventSource.onmessage = function(event) {
            try {
                const data = JSON.parse(event.data);

                // Update current frame counter - estimate based on events received
                currentFrame++;
                updateFrameDisplay(currentFrame);

                // CHANGE: Only process and display registered events
                // Previously we were showing all gestures, now only show events
                if (data.event && data.event !== 'idle') {
                    // Update gesture display with the event
                    updateGestureDisplay(data.event);

                    // Add to gesture history with frame converted to seconds
                    // Assume default 30fps if we don't have specific info
                    const estimatedSeconds = (currentFrame / 30).toFixed(1);
                    addToGestureHistory(data.event, estimatedSeconds);

                    // Update the gesture history display
                    updateGestureHistoryDisplay();
                }
            } catch (error) {
                console.error('Error parsing gesture data:', error);
            }
        };

        gestureEventSource.onerror = function(error) {
            console.error('EventSource error:', error);

            // Try to reconnect after a short delay
            setTimeout(() => {
                if (isVideoPlaying) {
                    console.log('Attempting to reconnect to gesture stream...');
                    connectToGestureStream();
                }
            }, 3000);
        };
    }

    function setupVideoEndDetection() {
        // Clear any existing interval
        if (videoEndCheckInterval) {
            clearInterval(videoEndCheckInterval);
        }

        // Check for video completion every 2 seconds
        videoEndCheckInterval = setInterval(() => {
            if (!isVideoPlaying) {
                clearInterval(videoEndCheckInterval);
                return;
            }

            // Check if video has ended (no new frames for a while)
            fetch('/check_video_status')
                .then(response => response.json())
                .then(data => {
                    if (data.completed) {
                        console.log("Video completed, updating UI");
                        videoCompleted();
                    }
                })
                .catch(error => {
                    console.error("Error checking video status:", error);
                });
        }, 2000);
    }

    function videoCompleted() {
        console.log("Video processing complete");

        // Stop the video and update UI
        isVideoPlaying = false;

        // Clear video end check interval
        if (videoEndCheckInterval) {
            clearInterval(videoEndCheckInterval);
            videoEndCheckInterval = null;
        }

        // Remove event listeners and stop stream
        if (videoStreamImg) {
            // Remove the error handler to prevent alerts
            videoStreamImg.onerror = null;

            // Set src to empty to stop the stream
            videoStreamImg.src = '';
        }

        // Close gesture event source
        if (gestureEventSource) {
            gestureEventSource.close();
            gestureEventSource = null;
        }

        // Disable logging and get log data
        fetch('/toggle_logging', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            }
        })
        .then(response => response.json())
        .then(data => {
            if (!data.logging) {
                saveLogBtn.disabled = false;
            }
        })
        .catch(error => {
            console.error('Error toggling logging:', error);
            saveLogBtn.disabled = false;
        });

        // Show first frame or placeholder
        if (uploadedVideoPath) {
            fetch('/process_video', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    video_path: uploadedVideoPath,
                    get_first_frame_only: true
                })
            })
            .then(response => response.json())
            .then(data => {
                if (data.first_frame && videoStreamImg) {
                    videoStreamImg.src = 'data:image/jpeg;base64,' + data.first_frame;
                }
            })
            .catch(error => {
                console.error('Error getting first frame:', error);
                if (videoStreamImg) {
                    videoStreamImg.src = '/static/img/video-placeholder.jpg';
                }
            });
        } else {
            if (videoStreamImg) {
                videoStreamImg.src = '/static/img/video-placeholder.jpg';
            }
        }

        // Update UI buttons
        startVideoBtn.textContent = 'Start Video';
        saveLogBtn.disabled = false;
    }

    function stopVideo() {
        isVideoPlaying = false;

        // Clear video end check interval
        if (videoEndCheckInterval) {
            clearInterval(videoEndCheckInterval);
            videoEndCheckInterval = null;
        }

        // Remove event listeners and stop stream
        if (videoStreamImg) {
            // Remove the error handler to prevent alerts
            videoStreamImg.onerror = null;

            // Set src to empty to stop the stream
            videoStreamImg.src = '';
        }

        // Close gesture event source
        if (gestureEventSource) {
            gestureEventSource.close();
            gestureEventSource = null;
        }

        // Disable logging and get log data
        fetch('/toggle_logging', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            }
        })
        .then(response => response.json())
        .then(data => {
            if (!data.logging) {
                saveLogBtn.disabled = false;
            }
        })
        .catch(error => {
            console.error('Error toggling logging:', error);
            saveLogBtn.disabled = false;
        });

        // Show first frame or placeholder
        if (uploadedVideoPath) {
            fetch('/process_video', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    video_path: uploadedVideoPath,
                    get_first_frame_only: true
                })
            })
            .then(response => response.json())
            .then(data => {
                if (data.first_frame && videoStreamImg) {
                    videoStreamImg.src = 'data:image/jpeg;base64,' + data.first_frame;
                }
            })
            .catch(error => {
                console.error('Error getting first frame:', error);
                if (videoStreamImg) {
                    videoStreamImg.src = '/static/img/video-placeholder.jpg';
                }
            });
        } else {
            if (videoStreamImg) {
                videoStreamImg.src = '/static/img/video-placeholder.jpg';
            }
        }
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

    function addToGestureHistory(gesture, seconds) {
        // Add gesture to history
        gestureHistory.unshift({
            gesture: gesture,
            seconds: seconds
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
            // CHANGE: Display time in seconds
            gestureTime.textContent = `${item.seconds} sec`;

            listItem.appendChild(gestureName);
            listItem.appendChild(gestureTime);

            historyList.appendChild(listItem);
        });
    }

    function updateFrameDisplay(frame) {
        const frameDisplay = document.getElementById('currentFrame');
        if (frameDisplay) {
            frameDisplay.textContent = `Frame: ${frame}`;
        }
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

        if (gestureEventSource) {
            gestureEventSource.close();
        }

        if (videoEndCheckInterval) {
            clearInterval(videoEndCheckInterval);
        }
    });
});