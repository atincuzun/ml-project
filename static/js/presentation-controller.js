/**
 * Presentation controller for RevealJS.
 * This script helps control presentations using gestures detected by the main application.
 */
(function() {
    // Notify the parent window that the presentation is ready
    window.addEventListener('load', function() {
        if (window.opener) {
            window.opener.postMessage('presentation_ready', '*');
        }
    });

    // Listen for gesture messages from the parent window
    window.addEventListener('message', function(event) {
        if (event.data && event.data.type === 'gesture') {
            handleGesture(event.data.gesture);
        }
    });

    // Handle different gestures
    function handleGesture(gesture) {
        // Check if Reveal is available (RevealJS)
        if (typeof Reveal !== 'undefined') {
            switch (gesture) {
                case 'swipe_left':
                    Reveal.next();
                    break;
                case 'swipe_right':
                    Reveal.prev();
                    break;
                case 'rotate_cw':
                    Reveal.toggleOverview(true);
                    break;
                case 'rotate_ccw':
                    Reveal.toggleOverview(false);
                    break;
            }
        } else {
            // For other presentation types, simulate key presses
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
                return 'ArrowRight'; // Next slide
            case 'swipe_right':
                return 'ArrowLeft'; // Previous slide
            case 'rotate_cw':
                return 'o'; // Common key for overview
            case 'rotate_ccw':
                return 'Escape'; // Common key to escape overview
            default:
                return '';
        }
    }

    // Add indicator to show that gesture control is active
    function addGestureIndicator() {
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
        indicator.textContent = 'Gesture Control Active';

        document.body.appendChild(indicator);
    }

    // Wait for document to be ready before adding the indicator
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', addGestureIndicator);
    } else {
        addGestureIndicator();
    }
})();