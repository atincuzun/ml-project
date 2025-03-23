document.addEventListener('DOMContentLoaded', function() {
    // DOM Elements
    const openButtons = document.querySelectorAll('.open-btn');
    const webcamFeed = document.getElementById('webcamFeed');
    const gestureIcon = document.getElementById('gestureIcon');
    const gestureText = document.getElementById('gestureText');

    // References to presentation windows we've opened
    const openPresentations = [];

    // Gesture icons
    const gestureIcons = {
        'swipe_left': '/static/img/swipe_left.png',
        'swipe_right': '/static/img/swipe_right.png',
        'rotate_cw': '/static/img/rotate_cw.png',
        'rotate_ccw': '/static/img/rotate_ccw.png',
        'hand_up': '/static/img/hand_up.png',
        'hand_down': '/static/img/hand_down.png'
    };

    // Gesture display timeout
    let gestureDisplayTimeout = null;
    const GESTURE_DISPLAY_DURATION = 3000; // 3 seconds

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

                // Only forward *registered events* to presentations
                // In SSE data, events are signaled by mode changing to "Registered"
                if (data.mode === "Registered") {
                    console.log("Forwarding registered event to presentations:", data.gesture);

                    // Update the gesture display
                    updateGestureDisplay(data.gesture);

                    // Forward the event to open presentations
                    forwardEventToOpenPresentations(data.gesture);
                } else if (data.gesture !== 'idle') {
                    // For raw gestures, just update the display with visual feedback
                    // but don't forward to presentations
                    updateGestureDisplay(data.gesture, false);
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

    // Function to forward events to open presentations
    function forwardEventToOpenPresentations(event) {
        // Remove closed windows from our array
        for (let i = openPresentations.length - 1; i >= 0; i--) {
            if (openPresentations[i].closed) {
                openPresentations.splice(i, 1);
            }
        }

        // Forward event to remaining windows
        openPresentations.forEach(win => {
            if (!win.closed) {
                try {
                    win.postMessage({
                        type: 'gesture',
                        gesture: event // Technically an event now
                    }, '*');
                } catch (e) {
                    console.error('Error sending event to presentation window:', e);
                }
            }
        });

        // Log only if we have open presentations
        if (openPresentations.length > 0) {
            console.log('Forwarded event to', openPresentations.length, 'presentations:', event);
        }
    }

    // Function to update gesture display
    function updateGestureDisplay(gesture, isEvent = true) {
        if (gesture === 'idle' || !gestureIcons[gesture]) {
            // Set default appearance for idle or unknown gestures
            gestureIcon.innerHTML = '<span id="gestureText">No gesture</span>';
            gestureIcon.style.backgroundColor = '#f4f4f4';
        } else {
            // Use gesture icon images
            gestureIcon.innerHTML = `
                <img src="${gestureIcons[gesture]}" alt="${formatGestureName(gesture)}" style="width: 80px; height: 80px;">
                <span id="gestureText">${formatGestureName(gesture)}${isEvent ? ' (Event)' : ''}</span>
            `;

            // Strong effect for events, subdued for raw gestures
            if (isEvent) {
                gestureIcon.style.backgroundColor = '#3498db';
            } else {
                gestureIcon.style.backgroundColor = '#a4c9e8'; // Lighter blue
                gestureIcon.style.opacity = 0.7;
            }

            // Flash effect after a delay
            setTimeout(() => {
                gestureIcon.innerHTML = '<span id="gestureText">No gesture</span>';
                gestureIcon.style.backgroundColor = '#f4f4f4';
                gestureIcon.style.opacity = 1.0;
            }, 1000);
        }
    }

    // Function to simulate a gesture via API with event handling
    function simulateGesture(gesture) {
        fetch('/simulate_gesture', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ gesture: gesture })
        })
        .then(response => response.json())
        .then(data => {
            console.log('Gesture simulation response:', data);

            // Show the gesture visually
            updateGestureDisplay(data.gesture, false);

            // If it became an event, actually handle it
            if (data.event && data.event !== 'idle') {
                console.log('Simulation produced event:', data.event);

                // Update display with event styling
                updateGestureDisplay(data.event, true);

                // Forward the event to presentations
                forwardEventToOpenPresentations(data.event);
            }
        })
        .catch(error => {
            console.error('Error simulating gesture:', error);
        });
    }

    // Function to format gesture name
    function formatGestureName(gesture) {
        return gesture.split('_')
            .map(word => word.charAt(0).toUpperCase() + word.slice(1))
            .join(' ');
    }

    // Function to forward gestures to open presentations
    function forwardGestureToOpenPresentations(gesture) {
        // Remove closed windows from our array
        for (let i = openPresentations.length - 1; i >= 0; i--) {
            if (openPresentations[i].closed) {
                openPresentations.splice(i, 1);
            }
        }

        // Forward gesture to remaining windows
        openPresentations.forEach(win => {
            if (!win.closed) {
                try {
                    win.postMessage({
                        type: 'gesture',
                        gesture: gesture
                    }, '*');
                } catch (e) {
                    console.error('Error sending gesture to presentation window:', e);
                }
            }
        });

        // Log only if we have open presentations
        if (openPresentations.length > 0) {
            console.log('Forwarded gesture to', openPresentations.length, 'presentations:', gesture);
        }
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

                // Add the window to our list of open presentations
                if (presentationWindow) {
                    openPresentations.push(presentationWindow);

                    // After window is loaded, inject our controller
                    presentationWindow.addEventListener('load', function() {
                        // Wait a moment for RevealJS to initialize if present
                        setTimeout(() => {
                            try {
                                // Inject the controller script
                                const script = presentationWindow.document.createElement('script');
                                script.src = data.controller;
                                presentationWindow.document.body.appendChild(script);

                                console.log("Injected presentation controller");
                            } catch (err) {
                                console.error("Error injecting controller:", err);
                            }
                        }, 1000);
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

    // Update the gesture guide
    updateGestureGuide();

    // Function to update gesture guide with new gestures
    function updateGestureGuide() {
        const gestureGuide = document.querySelector('.gesture-guide');
        if (gestureGuide) {
            gestureGuide.innerHTML = `
                <li><strong>Swipe Left:</strong> Previous slide</li>
                <li><strong>Swipe Right:</strong> Next slide</li>
                <li><strong>Rotate Clockwise:</strong> Rotate elements clockwise</li>
                <li><strong>Rotate Counter-clockwise:</strong> Rotate elements counter-clockwise</li>
                <li><strong>Hand Up:</strong> Navigate to slide above</li>
                <li><strong>Hand Down:</strong> Navigate to slide below</li>
            `;
        }
    }

    // Clean up on page unload
    window.addEventListener('beforeunload', () => {
        if (window.gestureEventSource) {
            window.gestureEventSource.close();
        }
    });
});