document.addEventListener('DOMContentLoaded', function() {
    // DOM Elements
    const openButtons = document.querySelectorAll('.open-btn');
    const webcamFeed = document.getElementById('webcamFeed');
    const gestureIcon = document.getElementById('gestureIcon');
    const gestureText = document.getElementById('gestureText');

    // Start webcam feed for gesture detection
    initWebcam();

    // Poll for gestures
    const gestureInterval = setInterval(pollGestures, 200);

    // Add click event handlers to open presentation buttons
    openButtons.forEach(button => {
        button.addEventListener('click', function() {
            const presentationName = this.getAttribute('data-presentation');
            const presentationType = this.getAttribute('data-type');
            openPresentation(presentationName, presentationType);
        });
    });

    // Function to initialize webcam
    function initWebcam() {
        webcamFeed.src = '/video_feed?camera_id=0';
    }

    // Function to poll for gestures
    async function pollGestures() {
        try {
            const response = await fetch('/get_gesture');
            const data = await response.json();

            if (data.gesture !== 'idle') {
                updateGestureDisplay(data.gesture);
            }
        } catch (error) {
            console.error('Error polling gestures:', error);
        }
    }

    // Function to update gesture display
    function updateGestureDisplay(gesture) {
        if (gesture === 'idle' || !gesture) {
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

    // Function to format gesture name
    function formatGestureName(gesture) {
        return gesture.split('_')
            .map(word => word.charAt(0).toUpperCase() + word.slice(1))
            .join(' ');
    }

    // Function to open a presentation in a new tab
    async function openPresentation(presentationName, presentationType) {
        try {
            const response = await fetch(`/open_presentation/${presentationName}`);
            const data = await response.json();

            if (data.path) {
                // Determine the correct path based on presentation type
                let fullPath = data.path;

                // If it's a directory, append index.html if not already included
                if (presentationType === 'directory' && !fullPath.endsWith('index.html')) {
                    fullPath = `${fullPath}/index.html`;
                }

                // Open the presentation in a new tab
                const presentationWindow = window.open(fullPath, '_blank');

                // Inject the presentation controller script
                if (presentationWindow && data.controller) {
                    presentationWindow.addEventListener('load', function() {
                        const script = presentationWindow.document.createElement('script');
                        script.src = data.controller;
                        presentationWindow.document.body.appendChild(script);
                    });
                }
            } else {
                alert('Error opening presentation');
            }
        } catch (error) {
            console.error('Error opening presentation:', error);
            alert('Error opening presentation');
        }
    }

    // Set up gesture communication with presentation window
    function setupGestureCommunication(presentationWindow) {
        // Modify the gesture polling to also send gestures to presentation window
        const enhancedGestureInterval = setInterval(async function() {
            try {
                const response = await fetch('/get_gesture');
                const data = await response.json();

                if (data.gesture !== 'idle') {
                    updateGestureDisplay(data.gesture);

                    // Send the gesture to the presentation window
                    presentationWindow.postMessage({
                        type: 'gesture',
                        gesture: data.gesture
                    }, '*');
                }
            } catch (error) {
                console.error('Error polling gestures:', error);
            }
        }, 200);

        // Clear interval when presentation window closes
        const checkWindowInterval = setInterval(function() {
            if (presentationWindow.closed) {
                clearInterval(enhancedGestureInterval);
                clearInterval(checkWindowInterval);
            }
        }, 1000);
    }

    // Clean up on page unload
    window.addEventListener('beforeunload', () => {
        clearInterval(gestureInterval);
    });
});document.addEventListener('DOMContentLoaded', function() {
    // Find all open presentation buttons
    const openButtons = document.querySelectorAll('.open-btn');

    // Add click event handlers to open presentation buttons
    openButtons.forEach(button => {
        button.addEventListener('click', function() {
            const presentationName = this.getAttribute('data-presentation');
            const presentationType = this.getAttribute('data-type');
            openPresentation(presentationName, presentationType);
        });
    });

    // Function to open a presentation in a new tab
    async function openPresentation(presentationName, presentationType) {
        try {
            const response = await fetch(`/open_presentation/${presentationName}`);
            const data = await response.json();

            if (data.path) {
                // Determine the correct path based on presentation type
                let fullPath = data.path;

                // If it's a directory, append index.html if not already included
                if (presentationType === 'directory' && !fullPath.endsWith('index.html')) {
                    fullPath = `${fullPath}/index.html`;
                }

                // Open the presentation in a new tab
                window.open(fullPath, '_blank');
            } else {
                alert('Error opening presentation');
            }
        } catch (error) {
            console.error('Error opening presentation:', error);
            alert('Error opening presentation');
        }
    }
});