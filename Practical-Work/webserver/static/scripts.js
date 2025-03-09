document.addEventListener('DOMContentLoaded', function () {
    // Get the audio element and the buttons
    const audioPlayer = document.getElementById('audio-player');
    const playButton = document.getElementById('play-button');
    const generateButton = document.getElementById('generate-button')
    const volumeSlider = document.getElementById('volume-slider')
    const volumeDisplay = document.getElementById('volume-display');
    const loadingSpinner = document.getElementById('loading-spinner');
    const dropdownMenu = document.getElementById('dropdown-menu');
    const dropdownButton = document.getElementById('dropdown-button');
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

    // Play selected from textbox
    function playSong(song) {
        fetch(`/play-song?song=${encodeURIComponent(song)}`)
            .then(response => response.json())
            .then(data => {
                if (data.audio_url) {
                    // Play selected song
                    audioPlayer.src = data.audio_url;
                    audioPlayer.play();
                    // Update the text displaying the current song name
                    document.getElementById('textbox').textContent = song;
                }
            })
            .catch(error => console.error('Error playing song:', error));
    }

    function fetchSongList() {
    fetch('/get-songs')
        .then(response => response.json())
        .then(data => {
            dropdownMenu.innerHTML = ''; // Clear previous entries
            data.songs.forEach(song => {
                let listItem = document.createElement('li');
                listItem.textContent = song;
                listItem.addEventListener('click', function () {
                    playSong(song);
                });
                dropdownMenu.appendChild(listItem);
            });
            dropdownMenu.style.display = 'block'; // Show dropdown
        })
        .catch(error => console.error('Error fetching songs:', error));
    }

    // Toggle dropdown on click
    dropdownButton.addEventListener('click', function () {
        if (dropdownMenu.style.display === 'block') {
            dropdownMenu.style.display = 'none';
        } else {
            fetchSongList();
        }
    });

    // Hide dropdown when clicking outside
    document.addEventListener('click', function (event) {
        if (!dropdownButton.contains(event.target) && !dropdownMenu.contains(event.target)) {
            dropdownMenu.style.display = 'none';
        }
    });

    generateButton.addEventListener('click', function () {
        const chord_progression_value = document.getElementById('chord_progression').value;

        // Show loading spinner
        loadingSpinner.style.display = 'block';

        let spinnerFrames = ['[|]', '[/]', '[-]', '[\\]'];
        let frameIndex = 0;
        let spinnerInterval = setInterval(() => {
            loadingSpinner.textContent = spinnerFrames[frameIndex];
            frameIndex = (frameIndex + 1) % spinnerFrames.length;
        }, 200); // Change frame every 200ms

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
                    // Play newly generated song
                    audioPlayer.src = data.audio_url;
                    audioPlayer.play();
                    // Update the text displaying the current song name
                    document.getElementById('textbox').textContent = data.song_name;
                }
            })
            .catch(error => console.error('Error:', error))
            .finally(() => {
            // Stop and hide spinner
            clearInterval(spinnerInterval);
            loadingSpinner.style.display = 'none';
        });
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
