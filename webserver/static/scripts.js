document.addEventListener('DOMContentLoaded', function () {
    // --- Get all DOM elements ---
    const audioPlayer = document.getElementById('audio-player');
    const playButton = document.getElementById('play-button');
    const generateButton = document.getElementById('generate-button');
    const skipButton = document.getElementById('skip-button');
    const volumeSlider = document.getElementById('volume-slider');
    const loadingSpinner = document.getElementById('loading-spinner');
    const textbox = document.getElementById('textbox');
    const currentTimeDisplay = document.getElementById('currentTime');
    const totalTimeDisplay = document.getElementById('totalTime');
    const bpmSlider = document.getElementById('bpm-slider');
    const bpmDisplay = document.getElementById('bpm-display');

    // New song name input
    const songNameInput = document.getElementById('song-name-input');

    // Dropdown elements
    const dropdownMenu = document.getElementById('dropdown-menu');
    const dropdownButton = document.getElementById('dropdown-button');
    const modelDropdownButton = document.getElementById('model-dropdown-button');
    const modelDropdownMenu = document.getElementById('model-dropdown-menu');

    // Advanced options elements
    const moreOptionsBtn = document.getElementById('more-options-btn');
    const advancedOptions = document.getElementById('advanced-options');
    const temperatureInput = document.getElementById('temperature-input');
    const topKInput = document.getElementById('top-k-input');
    const topPInput = document.getElementById('top-p-input');

    let selectedModel = null;
    let allModels = []; // Store all available models
    audioPlayer.loop = true;

    // --- Player Controls & UI ---
    playButton.addEventListener('click', function () {
        if (audioPlayer.paused) {
            audioPlayer.play();
        } else {
            audioPlayer.pause();
        }
    });

    audioPlayer.addEventListener('play', () => playButton.textContent = '❚❚');
    audioPlayer.addEventListener('pause', () => playButton.textContent = '▶');
    skipButton.addEventListener('click', () => audioPlayer.currentTime += 10);
    volumeSlider.addEventListener('input', () => audioPlayer.volume = volumeSlider.value);
    bpmSlider.addEventListener('input', () => bpmDisplay.textContent = bpmSlider.value);

    const formatTime = (seconds) => {
        const min = Math.floor(seconds / 60);
        const sec = Math.floor(seconds % 60);
        return `${min}:${sec < 10 ? '0' : ''}${sec}`;
    };

    audioPlayer.addEventListener('loadedmetadata', () => totalTimeDisplay.textContent = formatTime(audioPlayer.duration));
    audioPlayer.addEventListener('timeupdate', () => currentTimeDisplay.textContent = formatTime(audioPlayer.currentTime));

    moreOptionsBtn.addEventListener('click', () => {
        advancedOptions.classList.toggle('hidden');
        moreOptionsBtn.textContent = advancedOptions.classList.contains('hidden') ? 'More Options ▼' : 'Fewer Options ▲';
    });

    // --- Dropdown Logic & State Loading ---
    function populateDropdown(menu, items, clickHandler) {
        menu.innerHTML = '';
        items.forEach(item => {
            let listItem = document.createElement('li');
            listItem.textContent = item.name;
            listItem.className = 'px-4 py-2 hover:bg-slate-700 cursor-pointer text-sm';
            listItem.addEventListener('click', () => clickHandler(item));
            menu.appendChild(listItem);
        });
    }

    function selectModel(model) {
        selectedModel = model;
        console.log('Selected model:', model);
        modelDropdownButton.textContent = model.name;
        modelDropdownMenu.classList.add('hidden');
    }

    // --- Function to load state from metadata ---
    function loadStateFromMetadata(metadata) {
        // Update Song Name
        songNameInput.value = metadata.name || '';

        // Update BPM
        bpmSlider.value = metadata.bpm;
        // Trigger event to update display
        bpmSlider.dispatchEvent(new Event('input'));

        // Update Advanced Options
        temperatureInput.value = metadata.temperature || 1.0;
        topKInput.value = metadata.top_k || 8;
        topPInput.value = metadata.top_p || 0.9;

        // Find and select the correct model
        const modelToSelect = allModels.find(m => m.path === metadata.model_path);
        if (modelToSelect) {
            selectModel(modelToSelect);
        }

        // Dispatch a custom event to update the chord boxes
        document.dispatchEvent(new CustomEvent('loadChordState', { detail: metadata }));
    }

    // --- playSong function to fetch and use metadata ---
    function playSong(song) {
        fetch(`/play-song?song=${encodeURIComponent(song)}`)
            .then(response => response.json())
            .then(data => {
                if (data.audio_url && data.metadata_url) {
                    // Load the audio
                    audioPlayer.src = data.audio_url;
                    audioPlayer.play();
                    textbox.textContent = song;
                    dropdownMenu.classList.add('hidden');

                    // Fetch and apply the metadata
                    fetch(data.metadata_url)
                        .then(res => res.json())
                        .then(metadata => {
                            loadStateFromMetadata(metadata);
                        })
                        .catch(err => console.error("Failed to load metadata:", err));
                }
            })
            .catch(error => console.error('Error playing song:', error));
    }

    function fetchModelList() {
        fetch('/get-models')
            .then(response => response.json())
            .then(data => {
                // Store models globally
                allModels = data.models;
                if (allModels.length > 0) {
                    populateDropdown(modelDropdownMenu, allModels, selectModel);
                    // Select first model by default
                    selectModel(allModels[0]);
                } else {
                    modelDropdownButton.textContent = 'No models found';
                }
            })
            .catch(error => console.error('Error fetching models:', error));
    }

    function fetchSongList() {
        fetch('/get-songs')
            .then(response => response.json())
            .then(data => {
                if (data.songs.length > 0) {
                    populateDropdown(dropdownMenu, data.songs, playSong);
                }
            })
            .catch(error => console.error('Error fetching songs:', error));
    }

    modelDropdownButton.addEventListener('click', () => modelDropdownMenu.classList.toggle('hidden'));
    dropdownButton.addEventListener('click', () => {
        fetchSongList();
        dropdownMenu.classList.toggle('hidden');
    });

    document.addEventListener('click', function (event) {
        if (!modelDropdownButton.contains(event.target) && !modelDropdownMenu.contains(event.target)) {
            modelDropdownMenu.classList.add('hidden');
        }
        if (!dropdownButton.contains(event.target) && !dropdownMenu.contains(event.target)) {
            dropdownMenu.classList.add('hidden');
        }
    });

    generateButton.addEventListener('click', function () {
        if (!selectedModel) {
            alert('Please wait for models to load or select a model.');
            return;
        }

        // Collect chord progression data
        const chords = [];
        const timings = [];
        const chordBoxes = document.querySelectorAll('.chord-box');

        chordBoxes.forEach(box => {
            const chord = box.querySelector('.chord-input').value.trim();
            const duration = box.dataset.duration;
            // Only add the chord if the input is not empty
            if (chord) {
                chords.push(chord);
                timings.push(duration);
            }
        });

        if (chords.length === 0) {
            alert('Please enter at least one chord.');
            return;
        }

        // Collect all other parameters
        const generationData = {
            song_name: songNameInput.value.trim(), // Add song name
            chords: chords,
            timings: timings,
            bpm: bpmSlider.value,
            model_path: selectedModel.path,
            temperature: temperatureInput.value,
            top_k: topKInput.value,
            top_p: topPInput.value,
        };

        // Show spinner and send request
        loadingSpinner.classList.remove('hidden');
        let spinnerFrames = ['[|]', '[/]', '[-]', '[\\]'];
        let frameIndex = 0;
        const spinnerInterval = setInterval(() => {
            loadingSpinner.textContent = spinnerFrames[frameIndex];
            frameIndex = (frameIndex + 1) % spinnerFrames.length;
        }, 200);

        fetch('/generate-music', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify(generationData),
        })
        .then(response => response.json())
        .then(data => {
            if (data.audio_url) {
                audioPlayer.src = data.audio_url;
                audioPlayer.play();
                textbox.textContent = data.song_name;
            }
            if (data.error){
                alert(`An error occurred: ${data.error}`);
            }
        })
        .catch(error => {
            console.error('Generation failed:', error);
            alert('A network or server error occurred during generation.');
        })
        .finally(() => {
            clearInterval(spinnerInterval);
            loadingSpinner.classList.add('hidden');
        });
    });

    // --- Initial Load ---
    fetchModelList();
});