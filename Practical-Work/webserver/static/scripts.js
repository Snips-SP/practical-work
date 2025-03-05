document.addEventListener('DOMContentLoaded', function () {
    // Get the audio element and the buttons
    const audioPlayer = document.getElementById('audio-player');
    const playButton = document.getElementById('play-button');
    const generateButton = document.getElementById('generate-button')
    const volumeSlider = document.getElementById('volume-slider')
    const volumeDisplay = document.getElementById('volume-display');
    const skipButton = document.querySelector('.skip');

    audioPlayer.volume = volumeSlider.value;
    audioPlayer.loop = true;

    function getAsciiBar(value) {
        const totalBars = 10; // Number of segments in the bar
        const filledBars = Math.round(value * totalBars); // Calculate filled segments
        const emptyBars = totalBars - filledBars;

        return `|${'='.repeat(filledBars)}${'-'.repeat(emptyBars)}|`;
    }

    function updateVolumeBar() {
        volumeDisplay.textContent = getAsciiBar(volumeSlider.value);
        audioPlayer.volume = volumeSlider.value
    }

    volumeSlider.addEventListener('input', updateVolumeBar);

    // Initialize display
    updateVolumeBar();

    generateButton.addEventListener('click', function () {
        const chord_progression_value = document.getElementById('chord_progression').value;

        fetch('/generate-music', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({
                chord_progression: chord_progression_value,
            }),
            credentials: 'include' 
        })
            .then(response => response.json())
            .then(data => {
                if (data.audio_url) {
                    document.getElementById('audio-player').src = data.audio_url;
                    document.getElementById('audio-player').play();
                }
            })
            .catch(error => console.error('Error:', error));
    });

    playButton.addEventListener('click', function () {
        if (audioPlayer.paused) {
            audioPlayer.play();
            playButton.textContent = 'II'; // Change button text to Pause
        } else {
            audioPlayer.pause();
            playButton.textContent = '>'; // Change button text to Play
        }
    });

    skipButton.addEventListener('click', function () {
        audioPlayer.currentTime += 10; // Skip forward 10 seconds
    });
});
