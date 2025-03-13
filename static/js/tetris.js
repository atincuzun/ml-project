document.addEventListener('DOMContentLoaded', function() {
    // DOM Elements
    const tetrisCanvas = document.getElementById('tetris');
    const nextPieceCanvas = document.getElementById('nextPiece');
    const scoreElement = document.getElementById('score');
    const linesElement = document.getElementById('lines');
    const levelElement = document.getElementById('level');
    const gestureText = document.getElementById('gestureText');
    const gestureIcon = document.getElementById('gestureIcon');
    const startGameBtn = document.getElementById('startGameBtn');
    const pauseGameBtn = document.getElementById('pauseGameBtn');
    const restartGameBtn = document.getElementById('restartGameBtn');
    const gameOverModal = document.getElementById('gameOverModal');
    const finalScoreElement = document.getElementById('finalScore');
    const finalLinesElement = document.getElementById('finalLines');
    const newGameBtn = document.getElementById('newGameBtn');
    const webcamFeed = document.getElementById('webcamFeed');

    // Canvas contexts
    const ctx = tetrisCanvas.getContext('2d');
    const nextCtx = nextPieceCanvas.getContext('2d');

    // Game constants
    const ROWS = 20;
    const COLS = 10;
    const BLOCK_SIZE = 20;
    const COLORS = [
        null,
        '#FF0D72', // I
        '#0DC2FF', // J
        '#0DFF72', // L
        '#F538FF', // O
        '#FF8E0D', // S
        '#FFE138', // T
        '#3877FF'  // Z
    ];

    // Tetromino shapes
    const SHAPES = [
        null,
        [[0,0,0,0], [1,1,1,1], [0,0,0,0], [0,0,0,0]], // I
        [[2,0,0], [2,2,2], [0,0,0]],                  // J
        [[0,0,3], [3,3,3], [0,0,0]],                  // L
        [[0,4,4], [0,4,4], [0,0,0]],                  // O
        [[0,5,5], [5,5,0], [0,0,0]],                  // S
        [[0,6,0], [6,6,6], [0,0,0]],                  // T
        [[7,7,0], [0,7,7], [0,0,0]]                   // Z
    ];

    // Game state variables
    let board = createBoard();
    let score = 0;
    let lines = 0;
    let level = 1;
    let dropCounter = 0;
    let dropInterval = 1000; // milliseconds
    let lastTime = 0;
    let gameActive = false;
    let gamePaused = false;
    let player = {
        pos: {x: 0, y: 0},
        shape: null,
        next: null
    };

    // Initialize the webcam stream
    startWebcamFeed();

    // Event listeners
    startGameBtn.addEventListener('click', startGame);
    pauseGameBtn.addEventListener('click', togglePause);
    restartGameBtn.addEventListener('click', restartGame);
    newGameBtn.addEventListener('click', () => {
        gameOverModal.style.display = 'none';
        startGame();
    });

    document.addEventListener('keydown', handleKeyPress);

    // Poll for gestures
    const gestureInterval = setInterval(pollGestures, 200);

    // Functions
    function createBoard() {
        return Array.from(Array(ROWS), () => Array(COLS).fill(0));
    }

    function draw() {
        // Clear canvas
        ctx.fillStyle = '#000';
        ctx.fillRect(0, 0, tetrisCanvas.width, tetrisCanvas.height);

        // Draw board
        drawMatrix(board, {x: 0, y: 0});

        // Draw current piece
        if (player.shape) {
            drawMatrix(player.shape, player.pos);
        }
    }

    function drawMatrix(matrix, offset) {
        matrix.forEach((row, y) => {
            row.forEach((value, x) => {
                if (value !== 0) {
                    ctx.fillStyle = COLORS[value];
                    ctx.fillRect(
                        (x + offset.x) * BLOCK_SIZE,
                        (y + offset.y) * BLOCK_SIZE,
                        BLOCK_SIZE,
                        BLOCK_SIZE
                    );

                    // Draw border
                    ctx.strokeStyle = '#000';
                    ctx.lineWidth = 1;
                    ctx.strokeRect(
                        (x + offset.x) * BLOCK_SIZE,
                        (y + offset.y) * BLOCK_SIZE,
                        BLOCK_SIZE,
                        BLOCK_SIZE
                    );
                }
            });
        });
    }

    function drawNextPiece() {
        // Clear next piece canvas
        nextCtx.fillStyle = '#000';
        nextCtx.fillRect(0, 0, nextPieceCanvas.width, nextPieceCanvas.height);

        if (player.next) {
            // Center the piece in the next canvas
            const offset = {
                x: (nextPieceCanvas.width / BLOCK_SIZE - player.next[0].length) / 2,
                y: (nextPieceCanvas.height / BLOCK_SIZE - player.next.length) / 2
            };

            player.next.forEach((row, y) => {
                row.forEach((value, x) => {
                    if (value !== 0) {
                        nextCtx.fillStyle = COLORS[value];
                        nextCtx.fillRect(
                            (x + offset.x) * BLOCK_SIZE,
                            (y + offset.y) * BLOCK_SIZE,
                            BLOCK_SIZE,
                            BLOCK_SIZE
                        );

                        // Draw border
                        nextCtx.strokeStyle = '#000';
                        nextCtx.lineWidth = 1;
                        nextCtx.strokeRect(
                            (x + offset.x) * BLOCK_SIZE,
                            (y + offset.y) * BLOCK_SIZE,
                            BLOCK_SIZE,
                            BLOCK_SIZE
                        );
                    }
                });
            });
        }
    }

    function createPiece() {
        const pieceType = Math.floor(Math.random() * 7) + 1;
        return SHAPES[pieceType];
    }

    function resetPlayer() {
        player.shape = player.next || createPiece();
        player.next = createPiece();
        player.pos.y = 0;
        player.pos.x = Math.floor(COLS / 2) - Math.floor(player.shape[0].length / 2);

        // Check if game over
        if (collide(board, player)) {
            gameOver();
        }

        drawNextPiece();
    }

    function collide(board, player) {
        const [shape, pos] = [player.shape, player.pos];

        for (let y = 0; y < shape.length; y++) {
            for (let x = 0; x < shape[y].length; x++) {
                if (shape[y][x] !== 0 &&
                    (board[y + pos.y] === undefined ||
                    board[y + pos.y][x + pos.x] === undefined ||
                    board[y + pos.y][x + pos.x] !== 0)) {
                    return true;
                }
            }
        }

        return false;
    }

    function merge(board, player) {
        player.shape.forEach((row, y) => {
            row.forEach((value, x) => {
                if (value !== 0) {
                    board[y + player.pos.y][x + player.pos.x] = value;
                }
            });
        });
    }

    function rotate(matrix) {
        // Transpose
        for (let y = 0; y < matrix.length; y++) {
            for (let x = 0; x < y; x++) {
                [matrix[x][y], matrix[y][x]] = [matrix[y][x], matrix[x][y]];
            }
        }

        // Reverse each row
        matrix.forEach(row => row.reverse());

        return matrix;
    }

    function playerRotate(dir) {
        // Make a deep copy of the shape to avoid modifying the original
        const originalShape = JSON.parse(JSON.stringify(player.shape));
        const originalPos = {...player.pos};

        // Apply rotation
        if (dir === 1) { // Clockwise
            rotate(player.shape);
        } else { // Counter-clockwise
            for (let i = 0; i < 3; i++) {
                rotate(player.shape);
            }
        }

        // Check if rotation causes collision
        let offset = 1;
        while (collide(board, player)) {
            player.pos.x += offset;
            offset = -(offset + (offset > 0 ? 1 : -1));

            // If we've tried moving too far, revert rotation
            if (offset > player.shape[0].length) {
                player.shape = originalShape;
                player.pos = originalPos;
                return;
            }
        }

        // Flash gesture indicator
        updateGestureDisplay(dir === 1 ? 'rotate_cw' : 'rotate_ccw');
    }

    function playerDrop() {
        player.pos.y++;

        if (collide(board, player)) {
            player.pos.y--;
            merge(board, player);
            resetPlayer();
            sweepLines();
            updateScore();
        }

        dropCounter = 0;
    }

    function playerMove(dir) {
        player.pos.x += dir;

        if (collide(board, player)) {
            player.pos.x -= dir;
            return false;
        }

        // Flash gesture indicator
        updateGestureDisplay(dir === 1 ? 'swipe_right' : 'swipe_left');
        return true;
    }

    function sweepLines() {
        let linesCleared = 0;

        outer: for (let y = board.length - 1; y >= 0; y--) {
            for (let x = 0; x < board[y].length; x++) {
                if (board[y][x] === 0) {
                    continue outer;
                }
            }

            // Remove the line
            const row = board.splice(y, 1)[0].fill(0);
            board.unshift(row);
            y++;
            linesCleared++;
        }

        if (linesCleared > 0) {
            lines += linesCleared;

            // Update level every 10 lines
            level = Math.floor(lines / 10) + 1;

            // Update drop interval based on level
            dropInterval = 1000 - (level - 1) * 50;
            if (dropInterval < 100) dropInterval = 100; // Cap at 100ms
        }
    }

    function updateScore() {
        score += 10;
        scoreElement.textContent = score;
        linesElement.textContent = lines;
        levelElement.textContent = level;
    }

    function gameOver() {
        gameActive = false;
        finalScoreElement.textContent = score;
        finalLinesElement.textContent = lines;
        gameOverModal.style.display = 'flex';
    }

    function update(time = 0) {
        if (!gameActive || gamePaused) return;

        const deltaTime = time - lastTime;
        lastTime = time;

        dropCounter += deltaTime;
        if (dropCounter > dropInterval) {
            playerDrop();
        }

        draw();
        requestAnimationFrame(update);
    }

    function startGame() {
        // Reset game state
        board = createBoard();
        score = 0;
        lines = 0;
        level = 1;
        dropInterval = 1000;

        // Update display
        scoreElement.textContent = score;
        linesElement.textContent = lines;
        levelElement.textContent = level;

        // Initialize player
        player.next = createPiece();
        resetPlayer();

        // Start game loop
        gameActive = true;
        gamePaused = false;
        lastTime = 0;
        update();

        // Update button states
        startGameBtn.disabled = true;
        pauseGameBtn.disabled = false;
    }

    function togglePause() {
        gamePaused = !gamePaused;

        if (!gamePaused) {
            pauseGameBtn.textContent = 'Pause';
            lastTime = 0;
            update();
        } else {
            pauseGameBtn.textContent = 'Resume';
        }
    }

    function restartGame() {
        if (gameActive) {
            gameActive = false;

            // Reset and start a new game
            setTimeout(startGame, 0);
        }
    }

    function handleKeyPress(e) {
        if (!gameActive || gamePaused) return;

        // Prevent default behavior for arrow keys and space
        if ([32, 37, 38, 39, 40].includes(e.keyCode)) {
            e.preventDefault();
        }

        switch (e.keyCode) {
            case 37: // Left arrow
                playerMove(-1);
                break;
            case 39: // Right arrow
                playerMove(1);
                break;
            case 40: // Down arrow
                playerDrop();
                break;
            case 38: // Up arrow (rotate clockwise)
                playerRotate(1);
                break;
            case 90: // Z key (rotate counter-clockwise)
                playerRotate(-1);
                break;
            case 32: // Space (hard drop)
                while (!collide(board, player)) {
                    player.pos.y++;
                }
                player.pos.y--;
                playerDrop();
                break;
        }
    }

    function updateGestureDisplay(gesture) {
        // Update the gesture indicator
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

    function formatGestureName(gesture) {
        // Convert snake_case to Title Case
        return gesture.split('_')
            .map(word => word.charAt(0).toUpperCase() + word.slice(1))
            .join(' ');
    }

    function startWebcamFeed() {
        // Set up webcam for gesture recognition
        const videoUrl = '/video_feed?camera_id=0';

        // Create an img to display webcam feed
        webcamFeed.src = videoUrl;
    }

    async function pollGestures() {
        if (!gameActive || gamePaused) return;

        try {
            const response = await fetch('/get_gesture');
            const data = await response.json();

            // Apply gesture actions
            if (data.gesture !== 'idle') {
                handleGesture(data.gesture);
            }
        } catch (error) {
            console.error('Error polling gestures:', error);
        }
    }

    function handleGesture(gesture) {
        if (!gameActive || gamePaused) return;

        switch (gesture) {
            case 'swipe_left':
                playerMove(-1);
                break;
            case 'swipe_right':
                playerMove(1);
                break;
            case 'rotate_cw':
                playerRotate(1);
                break;
            case 'rotate_ccw':
                playerRotate(-1);
                break;
        }
    }

    // Initialize game display
    draw();
    drawNextPiece();

    // Clean up when page unloads
    window.addEventListener('beforeunload', () => {
        clearInterval(gestureInterval);
    });
});