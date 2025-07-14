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
    const modelDropdownButton = document.getElementById('model-dropdown-button')
    const modelDropdownMenu = document.getElementById('model-dropdown-menu')
    const skipButton = document.getElementById('skip')
    const currentTimeDisplay = document.getElementById('currentTime');
    const totalTimeDisplay = document.getElementById('totalTime');

    let selectedModel = null;

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
            // Show dropdown
            dropdownMenu.style.display = 'block';
        })
        .catch(error => console.error('Error fetching songs:', error));
    }

    // Fetch and populate models (runs on load)
    function fetchModelList() {
        fetch('/get-models')
            .then(response => response.json())
            .then(data => {
                modelDropdownMenu.innerHTML = '';
                if (data.models.length === 0) {
                    modelDropdownButton.textContent = 'No models found in run directory';
                    return;
                }
                data.models.forEach((model, index) => {
                    let listItem = document.createElement('li');
                    listItem.textContent = model['name'];

                    listItem.addEventListener('click', function () {
                        selectModel(model);
                        modelDropdownMenu.style.display = 'none';
                    });
                    modelDropdownMenu.appendChild(listItem);
                });

                // Select the first model by default
                selectModel(data.models[0]);
                // Show dropdown
                modelDropdownMenu.style.display = 'block';
            })
            .catch(error => console.error('Error fetching models:', error));
    }

    // Handle model selection
    function selectModel(model) {
        selectedModel = model;
        modelDropdownButton.textContent = `${model['name']}`;
    }

    // Toggle dropdown on click
    dropdownButton.addEventListener('click', function () {
        if (dropdownMenu.style.display === 'block') {
            dropdownMenu.style.display = 'none';
        } else {
            fetchSongList();
        }
    });

    // Toggle models dropdown
    modelDropdownButton.addEventListener('click', function () {
        if (modelDropdownMenu.style.display === 'block') {
            modelDropdownMenu.style.display = 'none';
        } else {
            fetchModelList()
        }
    });

    // Hide dropdowns when clicking outside
    document.addEventListener('click', function (event) {
        if (!dropdownButton.contains(event.target) && !dropdownMenu.contains(event.target)) {
            dropdownMenu.style.display = 'none';
        }
        if (!modelDropdownButton.contains(event.target) && !modelDropdownMenu.contains(event.target)) {
            modelDropdownMenu.style.display = 'none';
        }
    });

    generateButton.addEventListener('click', function () {
        const chord_progression_value = document.getElementById('chord_progression').value;
        const bpmValue = document.getElementById('bpm').value.trim();
        const bpm = Number(bpmValue);
        const model_path = selectedModel['path']

        if (!Number.isInteger(bpm)) {
          alert('Please enter a valid whole number.');
        }
        else if (bpm < 40 || bpm > 200) {
          alert('Please enter a BPM between 40 and 200.');
        }
        else{
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
                    bpm: bpmValue,
                    model_path: model_path
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
                    if (data.error){
                        alert(`An error occurred: ${data.error}`)
                    }
                })
                .catch(error => alert('An error occurred'))
                .finally(() => {
                // Stop and hide spinner
                clearInterval(spinnerInterval);
                loadingSpinner.style.display = 'none';
            });
        }
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
    
    function formatTime(seconds) {
        const min = Math.floor(seconds / 60);
        const sec = Math.floor(seconds % 60);
        return `${min}:${sec < 10 ? '0' : ''}${sec}`;
    }

    // Update duration when metadata is loaded
    audioPlayer.addEventListener('loadedmetadata', () => {
        totalTimeDisplay.textContent = formatTime(audioPlayer.duration);
    });
    
    // Update current time as the song plays
    audioPlayer.addEventListener('timeupdate', () => {
        currentTimeDisplay.textContent = formatTime(audioPlayer.currentTime);
    });

    // Fetch model list on website load
    fetchModelList();
});
