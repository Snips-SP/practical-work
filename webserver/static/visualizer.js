document.addEventListener('DOMContentLoaded', function () {
    const audio = document.getElementById('audio-player');
    const visualizer = document.getElementById('ascii-visualizer');

    if (!audio || !visualizer) {
        console.error('Audio player or visualizer element not found.');
        return;
    }

    const audioContext = new (window.AudioContext || window.webkitAudioContext)();
    const analyser = audioContext.createAnalyser();
    // Results in 64 frequency bins
    analyser.fftSize = 128;

    const source = audioContext.createMediaElementSource(audio);
    source.connect(analyser);
    analyser.connect(audioContext.destination);

    const dataArray = new Uint8Array(analyser.frequencyBinCount);
    // Max height of bars in characters
    const maxHeight = 10;
    // Will hold the calculated width of a single character
    let charWidth = 0;

    // Dynamically calculates the width of a single monospace character by rendering a temporary element and measuring it.
    function calculateCharWidth() {
        const tempSpan = document.createElement('span');
        // Use styles to make it invisible and not affect layout
        tempSpan.style.visibility = 'hidden';
        tempSpan.style.position = 'absolute';
        // A sample character
        tempSpan.textContent = '#';

        visualizer.appendChild(tempSpan);
        // Use measured width, with a fallback
        charWidth = tempSpan.offsetWidth || 7;
        visualizer.removeChild(tempSpan);
    }

    //Calculates the maximum number of characters that can fit in the visualizer.
    function calculateMaxBars() {
        // A safe default if width isn't calculated yet
        if (charWidth === 0) return 80;
        // Use the visualizer's width
        return Math.floor(visualizer.clientWidth / charWidth);
    }

    function getAsciiBars(dataArray) {
        const totalBars = calculateMaxBars();
        if (totalBars <= 0) return '';

        const leftBarCount = Math.floor(totalBars / 2);
        const bars = Array(leftBarCount).fill(0);

        // Map the frequency data to the number of bars available for the left side
        for (let i = 0; i < leftBarCount; i++) {
            const percent = i / leftBarCount;
            const dataIndex = Math.floor(percent * dataArray.length);
            const value = dataArray[dataIndex] / 255; // Normalize to 0-1
            bars[i] = Math.round(value * maxHeight);
        }

        let asciiOutput = '';
        for (let row = maxHeight; row >= 0; row--) {
            let leftPart = '';
            for (let col = 0; col < leftBarCount; col++) {
                leftPart += bars[col] > row ? '#' : ' ';
            }

            // Create the right part by reversing the left, ensuring it fits the total width
            let rightPart = leftPart.split('').reverse().join('');
            // If total is odd, the middle bar is duplicated
            if (totalBars % 2 !== 0) {
                rightPart = rightPart.slice(1);
            }

            asciiOutput += leftPart + rightPart + '\n';
        }

        return asciiOutput;
    }

    function updateVisualizer() {
        analyser.getByteFrequencyData(dataArray);
        visualizer.textContent = getAsciiBars(dataArray);
        requestAnimationFrame(updateVisualizer);
    }

    window.addEventListener('resize', () => {
        // Recalculate dimensions when the window size changes
        calculateCharWidth();
        // Redraw immediately to prevent lag
        analyser.getByteFrequencyData(dataArray);
        visualizer.textContent = getAsciiBars(dataArray);
    });

    audio.onplay = () => {
        // The AudioContext must be resumed by a user gesture
        if (audioContext.state === 'suspended') {
            audioContext.resume();
        }
        updateVisualizer();
    };

    // Perform initial calculations once the DOM is ready
    calculateCharWidth();
});