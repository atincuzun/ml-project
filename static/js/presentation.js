document.addEventListener('DOMContentLoaded', function() {
    // DOM Elements
    const openButtons = document.querySelectorAll('.open-btn');
    const webcamFeed = document.getElementById('webcamFeed');
    const gestureIcon = document.getElementById('gestureIcon');
    const gestureText = document.getElementById('gestureText');

    // Start webcam feed for gesture detection
    initWebcam();

    // Connect to gesture stream
    connectToGestureStream();

    // Function to initialize webcam
    function initWebcam() {
        webcamFeed.src = '/video_feed?camera_id=0';
    }

    // Connect to the Server-Sent Events stream for gestures
    function connectToGestureStream() {
        const eventSource = new EventSource('/gesture_stream');

        eventSource.onmessage = function(event) {
            try {
                const data = JSON.parse(event.data);
                if (data.gesture !== 'idle') {
                    updateGestureDisplay(data.gesture);
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

    // Add click event handlers to open presentation buttons
    openButtons.forEach(button => {
        button.addEventListener('click', function() {
            const presentationName = this.getAttribute('data-presentation');
            const presentationType = this.getAttribute('data-type');
            openPresentation(presentationName, presentationType);
        });
    });

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

    // Clean up on page unload
    window.addEventListener('beforeunload', () => {
        if (window.gestureEventSource) {
            window.gestureEventSource.close();
        }
    });
});