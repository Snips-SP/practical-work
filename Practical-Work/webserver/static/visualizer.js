document.addEventListener("DOMContentLoaded", function () {
    const audio = document.getElementById("audio-player");
    const visualizer = document.getElementById("ascii-visualizer");

    const audioContext = new (window.AudioContext || window.webkitAudioContext)();
    const analyser = audioContext.createAnalyser();
    analyser.fftSize = 128;

    const source = audioContext.createMediaElementSource(audio);
    source.connect(analyser);
    analyser.connect(audioContext.destination);

    let dataArray = new Uint8Array(analyser.frequencyBinCount);
    let maxHeight = 10; // Max height of bars

    function calculateMaxBars() {
        return Math.floor(window.innerWidth / 7.1); // Smaller divisor = more bars
    }

    function getAsciiBars(dataArray) {
        let maxBars = calculateMaxBars();
        let halfBars = Math.floor(maxBars / 2); // Half for left, half for right
        let bars = Array(halfBars).fill(0);

        // Distribute frequency data evenly
        for (let i = 0; i < halfBars; i++) {
            const value = dataArray[Math.floor(i * (dataArray.length / halfBars))] / 255;
            bars[i] = Math.round(value * maxHeight);
        }

        let asciiOutput = "";
        for (let row = maxHeight; row >= 0; row--) {
            let leftPart = "";
            let rightPart = "";

            for (let col = 0; col < halfBars; col++) {
                leftPart += bars[col] > row ? "#" : " ";
            }

            rightPart = leftPart.split("").reverse().join(""); // Mirroring the left side

            asciiOutput += leftPart + rightPart + "\n"; // No gap in between!
        }

        return asciiOutput;
    }

    function updateVisualizer() {
        analyser.getByteFrequencyData(dataArray);
        visualizer.textContent = getAsciiBars(dataArray);
        requestAnimationFrame(updateVisualizer);
    }

    window.addEventListener("resize", () => {
        visualizer.textContent = getAsciiBars(dataArray);
    });

    audio.onplay = () => {
        audioContext.resume();
        updateVisualizer();
    };
});
