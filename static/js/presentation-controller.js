/**
 * Enhanced Presentation controller for RevealJS.
 * This script helps control presentations using gestures detected by the main application.
 * Supports additional gestures like hand_up and hand_down.
 */
(function() {
    // Store rotation angles for each element
    const rotationAngles = {};

    // Notify the parent window that the presentation is ready
    window.addEventListener('load', function() {
        if (window.opener) {
            window.opener.postMessage('presentation_ready', '*');
        }
        console.log("Presentation controller loaded and ready for gestures");
    });

    // Listen for gesture messages from the parent window
    window.addEventListener('message', function(event) {
        if (event.data && event.data.type === 'gesture') {
            handleGesture(event.data.gesture);
        }
    });

    // Handle different gestures
    function handleGesture(gesture) {
        console.log("Received gesture:", gesture);

        // Check if Reveal is available (RevealJS)
        if (typeof Reveal !== 'undefined') {
            switch (gesture) {
                case 'swipe_left':
                    console.log("Navigating to previous slide");
                    Reveal.left();
                    break;
                case 'swipe_right':
                    console.log("Navigating to next slide");
                    Reveal.right();
                    break;
                case 'rotate_cw':
                    console.log("Rotating elements clockwise");
                    rotateElements(90);
                    break;
                case 'rotate_ccw':
                    console.log("Rotating elements counter-clockwise");
                    rotateElements(-90);
                    break;
                case 'hand_up':
                    console.log("Navigating to slide above");
                    Reveal.up();
                    break;
                case 'hand_down':
                    console.log("Navigating to slide below");
                    Reveal.down();
                    break;
            }
        } else {
            // For other presentation types, simulate key presses
            console.log("RevealJS not detected, simulating keyboard events for gesture:", gesture);
            const keyEvent = new KeyboardEvent('keydown', {
                bubbles: true,
                cancelable: true,
                key: getKeyForGesture(gesture)
            });
            document.dispatchEvent(keyEvent);
        }
    }

    // Map gestures to keys for non-RevealJS presentations
    function getKeyForGesture(gesture) {
        switch (gesture) {
            case 'swipe_left':
                return 'ArrowLeft'; // Previous slide
            case 'swipe_right':
                return 'ArrowRight'; // Next slide
            case 'rotate_cw':
                return 'r'; // Rotate (custom key)
            case 'rotate_ccw':
                return 'l'; // Rotate (custom key)
            case 'hand_up':
                return 'ArrowUp'; // Up
            case 'hand_down':
                return 'ArrowDown'; // Down
            default:
                return '';
        }
    }

    // Function to rotate elements with class 'rotatable'
    function rotateElements(degrees) {
        try {
            // First try to use the helper method if available
            if (typeof rotateRotatables === 'function') {
                console.log("Using rotateRotatables helper function");
                const currentSlide = Reveal.getCurrentSlide();
                rotateRotatables(currentSlide);
                return;
            }

            // Fallback to our own implementation
            console.log("Using fallback rotation implementation");
            const currentSlide = Reveal.getCurrentSlide();
            const rotatables = currentSlide.querySelectorAll('.rotatable');

            if (rotatables.length === 0) {
                console.log("No rotatable elements found in current slide");
                return;
            }

            rotatables.forEach(function(elem) {
                // Generate ID if needed
                if (!elem.id) {
                    elem.id = 'rotatable-' + Math.random().toString(36).substr(2, 9);
                }

                // Get current rotation
                if (!rotationAngles[elem.id]) {
                    rotationAngles[elem.id] = 0;
                }

                // Update rotation
                rotationAngles[elem.id] += degrees;

                // Apply rotation
                elem.style.transform = `rotate(${rotationAngles[elem.id]}deg)`;
                elem.style.transition = 'transform 0.3s ease-in-out';

                console.log(`Rotated element ${elem.id} to ${rotationAngles[elem.id]} degrees`);
            });
        } catch (error) {
            console.error("Error rotating elements:", error);
        }
    }

    // Add indicator to show that gesture control is active
    function addGestureIndicator() {
        if (document.getElementById('gesture-indicator')) {
            return; // Already exists
        }

        const indicator = document.createElement('div');
        indicator.id = 'gesture-indicator';
        indicator.style.position = 'fixed';
        indicator.style.bottom = '10px';
        indicator.style.right = '10px';
        indicator.style.backgroundColor = 'rgba(52, 152, 219, 0.7)';
        indicator.style.color = 'white';
        indicator.style.padding = '5px 10px';
        indicator.style.borderRadius = '3px';
        indicator.style.fontSize = '12px';
        indicator.style.zIndex = '9999';
        indicator.style.boxShadow = '0 2px 5px rgba(0,0,0,0.3)';
        indicator.textContent = 'Gesture Control Active';

        document.body.appendChild(indicator);
        console.log("Added gesture control indicator");
    }

    // Wait for document to be ready before adding the indicator
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', addGestureIndicator);
    } else {
        addGestureIndicator();
    }
})();