document.addEventListener('DOMContentLoaded', () => {
    const chordsContainer = document.getElementById('chords-container');
    const addChordBtn = document.getElementById('add-chord-btn');
    const initialChords = ['Am', 'G', 'C', 'F'];

    // --- Data for Autocomplete & Validation ---
    const rootNotes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'];
    const chordQualities = [
        '', 'm', '7', 'm7', 'M7', 'm7-5', 'dim', 'sus4', '7sus4', 'aug',
        'm6', '7(9)', 'm7(9)', 'add9', '6', 'mM7', '7-5', '7#5'
    ];
    let activeInput = null;

    const isValidChord = (chord) => {
        const rawValue = chord.trim();
        if (rawValue === '') return true;
        const valueUpperCase = rawValue.toUpperCase();
        let matchedRoot = '';
        for (const note of rootNotes) {
            if (valueUpperCase.startsWith(note) && note.length > matchedRoot.length) {
                matchedRoot = note;
            }
        }
        if (!matchedRoot) return false;
        const qualityPart = rawValue.substring(matchedRoot.length);
        return chordQualities.includes(qualityPart);
    };

    const autocompleteList = document.createElement('ul');
    autocompleteList.className = 'absolute bg-slate-700 border border-slate-500 rounded-md shadow-lg z-50 max-h-48 overflow-y-auto hidden';
    document.body.appendChild(autocompleteList);

    const showAutocomplete = (suggestions) => {
        autocompleteList.innerHTML = '';
        if (suggestions.length === 0 || !activeInput) {
            autocompleteList.classList.add('hidden');
            return;
        }
        suggestions.forEach(suggestion => {
            const item = document.createElement('li');
            item.className = 'px-4 py-2 text-left text-sm cursor-pointer hover:bg-sky-500';
            item.textContent = suggestion;
            autocompleteList.appendChild(item);
        });
        const inputRect = activeInput.getBoundingClientRect();
        autocompleteList.style.left = `${inputRect.left}px`;
        autocompleteList.style.top = `${inputRect.bottom + window.scrollY + 4}px`;
        autocompleteList.style.width = `${inputRect.width}px`;
        autocompleteList.classList.remove('hidden');
    };

    const hideAutocomplete = () => autocompleteList.classList.add('hidden');

    const createChordBoxElement = (chordValue = '', duration = 16) => {
        const box = document.createElement('div');
        box.className = 'chord-box relative group bg-slate-800 border border-slate-600 rounded-lg p-4 h-32 flex flex-col justify-between transition-all duration-300';
        box.dataset.duration = duration;
        box.style.width = `${(8 * duration) / 16}rem`;

        const input = document.createElement('input');
        input.type = 'text';
        input.className = 'chord-input bg-transparent text-white text-3xl font-bold text-center w-full focus:outline-none';
        input.value = chordValue;
        input.placeholder = '...';

        const durationContainer = document.createElement('div');
        durationContainer.className = 'flex items-center justify-center gap-2 text-xs text-slate-400 mt-1 opacity-0 group-hover:opacity-100 transition-opacity';

        const shrinkBtn = document.createElement('button');
        shrinkBtn.textContent = '−';
        shrinkBtn.className = 'shrink-btn bg-slate-700 rounded-full w-5 h-5 flex items-center justify-center hover:bg-slate-600';

        const durationDisplay = document.createElement('span');
        durationDisplay.className = 'duration-display font-mono w-12 text-center';
        durationDisplay.textContent = `${duration}`;

        const extendBtn = document.createElement('button');
        extendBtn.textContent = '+';
        extendBtn.className = 'extend-btn bg-slate-700 rounded-full w-5 h-5 flex items-center justify-center hover:bg-slate-600';

        durationContainer.appendChild(shrinkBtn);
        durationContainer.appendChild(durationDisplay);
        durationContainer.appendChild(extendBtn);

        box.appendChild(input);
        box.appendChild(durationContainer);
        addInteractionButtons(box);
        return box;
    };

    const addInteractionButtons = (box) => {
        const removeBtn = document.createElement('button');
        removeBtn.innerHTML = '×';
        removeBtn.className = 'remove-btn absolute -top-2 -right-2 bg-red-500 text-white rounded-full w-6 h-6 flex items-center justify-center text-sm font-bold opacity-0 group-hover:opacity-100 transition-opacity hover:bg-red-600';
        removeBtn.onclick = () => { if (chordsContainer.children.length > 1) { box.remove(); } };
        box.appendChild(removeBtn);

        const updateDuration = (newDuration) => {
            box.dataset.duration = newDuration;
            box.querySelector('.duration-display').textContent = `${newDuration}n`;
            box.style.width = `${(8 * newDuration) / 16}rem`;
            updateUI();
        };

        box.querySelector('.extend-btn').onclick = () => {
            const newDuration = parseInt(box.dataset.duration, 10) + 16;
            updateDuration(newDuration);
        };

        box.querySelector('.shrink-btn').onclick = () => {
            const currentDuration = parseInt(box.dataset.duration, 10);
            if (currentDuration > 16) {
                updateDuration(currentDuration - 16);
            }
        };
    };

    const updateUI = () => {
        const boxes = chordsContainer.querySelectorAll('.chord-box');
        boxes.forEach((box) => {
            const duration = parseInt(box.dataset.duration, 10);
            const shrinkBtn = box.querySelector('.shrink-btn');
            shrinkBtn.disabled = duration <= 16;
            shrinkBtn.style.opacity = duration <= 16 ? '0.3' : '1';
        });
    };

    addChordBtn.addEventListener('click', () => {
        const newBox = createChordBoxElement();
        chordsContainer.appendChild(newBox);
        updateUI();
    });

    chordsContainer.addEventListener('focusin', (e) => {
        if (e.target.classList.contains('chord-input')) {
            activeInput = e.target;
            handleInput(e.target);
        }
    });

    chordsContainer.addEventListener('focusout', (e) => {
        if (e.target.classList.contains('chord-input')) {
            const input = e.target;
            if (isValidChord(input.value)) {
                input.classList.remove('text-red-500');
            } else {
                input.classList.add('text-red-500');
            }
            setTimeout(() => {
                if (document.activeElement !== activeInput) {
                    hideAutocomplete();
                    activeInput = null;
                }
            }, 100);
        }
    });

    chordsContainer.addEventListener('input', (e) => {
        if (e.target.classList.contains('chord-input')) {
            e.target.classList.remove('text-red-500');
            handleInput(e.target);
        }
    });

    const handleInput = (inputElement) => {
        const value = inputElement.value.trim().toUpperCase();
        let suggestions = [];
        let matchedRoot = '';
        for (const note of rootNotes) {
            if (value.startsWith(note) && note.length > matchedRoot.length) {
                matchedRoot = note;
            }
        }
        if (matchedRoot) {
            const qualityPart = value.substring(matchedRoot.length);
            suggestions = chordQualities
                .filter(q => q.toUpperCase().startsWith(qualityPart))
                .map(q => matchedRoot + q);
        } else {
            suggestions = rootNotes.filter(note => note.startsWith(value));
        }
        showAutocomplete(suggestions);
    }

    autocompleteList.addEventListener('mousedown', (e) => {
        if (e.target.tagName === 'LI' && activeInput) {
            activeInput.value = e.target.textContent;
            activeInput.classList.remove('text-red-500');
            hideAutocomplete();
        }
    });

    // --- Event listener to load a song's chord data ---
    document.addEventListener('loadChordState', (e) => {
        const { chords, timings } = e.detail;
        if (!chords || !timings) return;

        chordsContainer.innerHTML = ''; // Clear existing chords
        chords.forEach((chord, index) => {
            const timing = timings[index];
            const newBox = createChordBoxElement(chord, timing);
            chordsContainer.appendChild(newBox);
        });
        updateUI();
    });

    // --- Initial Load ---
    initialChords.forEach(chord => {
        const newBox = createChordBoxElement(chord);
        chordsContainer.appendChild(newBox);
    });
    updateUI();
});