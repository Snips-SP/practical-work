from transformers import GPT2Config, GPT2LMHeadModel
from pydub.silence import split_on_silence
from pydub import AudioSegment
from pydub.utils import which
import torch
from typing import Dict, List
import subprocess
import re
import os


class EncodingConfig:
    """
    EncodingConfig holds the token layout and helper mappings for notes, drums and microtiming tokens.
    Call EncodingConfig.initialize() once at startup to populate maps.
    """

    # Public config
    tokens: List[int] = []
    padding_token: int = None
    vocab_size: int = None

    # Pianoroll config
    # Every quarter note is split into 24 steps in each pianoroll
    pianoroll_resolution: int = 24
    # We mainly encode only sixteenth notes, except for drums where we encode the whole pianoroll resolution
    encoding_resolution: int = 4

    # Instrument config
    # The order of tracks in which they appear in the dataset
    midi_tracks: List[str] = ['Drums', 'Piano', 'Guitar', 'Bass', 'Strings']
    # The order we wish to encode the notes in (As string and as indices of midi_tracks)
    encoding_order: List[str] = ['Bass', 'Piano', 'Guitar', 'Strings', 'Drums']
    # [3, 1, 2, 4, 0]
    trc_idx: List[int] = None

    programs = {
        'Drums': 0,
        'Piano': 0,
        'Guitar': 32,
        'Bass': 35,
        'Strings': 49
    }

    # General pitch packing / offset settings
    note_size: int = 84  # how many pitch slots per melodic instrument
    note_offset: int = 24  # midi pitch corresponding to index 0 inside each melodic instrument block

    # Special tokens (filled during initialize)
    time_note: int = None
    end_note: int = None

    # Drum pitches used in your dataset (sorted)
    drum_pitches: List[int] = [27, 28, 32, 33, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 49, 50,
                               51, 53, 54, 55, 56, 57, 59, 60, 61, 62, 63, 64, 65, 67, 68, 69, 70, 73, 75,
                               76, 77, 80, 81, 82, 83, 85, 87]

    # Microtiming definitions: (delta in steps)
    microtimings: List[int] = [-2, -1, 1, 2, 3]

    # Mappings filled at initialize
    instrument_bases: Dict[str, int] = {}
    drum_pitch_to_token: Dict[int, int] = {}
    drum_token_to_pitch: Dict[int, int] = {}
    microtiming_delta_to_token: Dict[int, int] = {}
    microtiming_token_to_delta: Dict[int, int] = {}

    @classmethod
    def initialize(cls):
        if cls.tokens:
            return  # already initialized

        cls.tokens = []
        cls.trc_idx = [cls.midi_tracks.index(track) for track in cls.encoding_order]
        current = 0

        # Add melodic instrument blocks in the chosen order: Bass, Piano, Guitar, Strings
        melodic_tracks = [t for t in cls.encoding_order if t != 'Drums']
        for t in melodic_tracks:
            cls.instrument_bases[t] = current
            # just reserve the integer range; tokens will be 0..(vocab-1) anyway
            current += cls.note_size

        # Drums: custom per-pitch mapping (one token per pitch from drum_pitches_sorted)
        cls.instrument_bases['Drums'] = current
        start_drum = current
        for i, midi_pitch in enumerate(cls.drum_pitches):
            token_id = start_drum + i
            cls.drum_pitch_to_token[midi_pitch] = token_id
            cls.drum_token_to_pitch[token_id] = midi_pitch
        current = start_drum + len(cls.drum_pitches)

        # Microtiming tokens
        cls.instrument_bases['Microtimings'] = current
        for delta in cls.microtimings:
            token_id = current
            cls.microtiming_delta_to_token[delta] = token_id
            cls.microtiming_token_to_delta[token_id] = delta
            current += 1

        cls.instrument_bases['Special'] = current
        # Special tokens: time_note, end_note, padding
        cls.time_note = current
        current += 1
        cls.end_note = current
        current += 1
        cls.padding_token = current
        current += 1

        # Final vocabulary size and token list
        cls.vocab_size = current
        cls.tokens = list(range(cls.vocab_size))

    @staticmethod
    def reorder_current(cur_seq):
        """
        Reorder the sequence in one timestep so that melodic tokens always come first and are ordered accendingly.
        Assumes instruments are encoded in blocks of size EncodingConfig.note_size.
        """
        # Stort melodic tokens in an accenting order
        melodic_notes = [c for c in cur_seq if 0 <= c < EncodingConfig.instrument_bases['Drums']]
        # Dont sort drum tokens, though, since microtiming tokens and special tokens depend on order
        other_notes = [c for c in cur_seq if EncodingConfig.instrument_bases['Drums'] <= c < EncodingConfig.vocab_size]

        # Sort each group to keep deterministic ordering
        return sorted(melodic_notes) + other_notes


EncodingConfig.initialize()


def mid_to_mp3(mid_file: str, sf2_file: str, output_file: str = 'output.mp3'):
    """Converts a MIDI file to an MP3 file using a SoundFont.

        This function synthesizes the MIDI file into a temporary WAV file
        using FluidSynth and the provided SF2 SoundFont. It then converts the
        WAV file to MP3, automatically trimming any leading or trailing silence
        that FluidSynth might have added.

        :param str mid_file: The path to the input MIDI file.
        :param str sf2_file: The path to the SoundFont (.sf2) file.
        :param str output_file: The path to save the resulting MP3 file,
                            defaults to 'output.mp3'.
        :raises AssertionError: If the `mid_file` or `sf2_file` is not found.
        :raises FileNotFoundError: If ffmpeg is not found in the system path.
        :returns: None. The function saves the output directly to a file.
        :rtype: None
    """
    assert os.path.isfile(mid_file), 'Mid file not found'
    assert os.path.isfile(sf2_file), 'sf2 file not found'

    # Finds ffmpeg in the system path
    ffmpeg_path = which('ffmpeg')
    if ffmpeg_path is None:
        raise FileNotFoundError('ffmpeg not found in system path')
    AudioSegment.converter = ffmpeg_path

    # Create temporary wav file
    tmp_wav = 'tmp.wav'
    subprocess.run(['fluidsynth', '-qni', sf2_file, mid_file, '-F', tmp_wav])

    # Convert wav to mp3
    audio = AudioSegment.from_wav(tmp_wav)

    # TODO:
    # FluidSynth created long periods of silence after the notes of the mid file have finished playing
    # I cannot figure out why this is happening but I will just remove it during the conversion to mp3

    # Parameters for silence detection
    # Silence threshold in dBFS
    silence_thresh = audio.dBFS - 14
    # Minimum silence duration in milliseconds
    min_silence_len = 500

    # Split the audio based on silence
    segments = split_on_silence(audio, min_silence_len=min_silence_len, silence_thresh=silence_thresh)

    # Join all non-silent segments
    trimmed_audio = segments[0]  # Start with the first segment

    for segment in segments[1:]:
        trimmed_audio += segment  # Concatenate the segments back together

    trimmed_audio.export(output_file, format='mp3')

    # Delete tmp wav file
    os.remove(tmp_wav)


def chord2tokens(chord: str) -> list:
    """Encodes a chord into a list of tokens.

        :param str chord: The chord to encode, (e.g., 'C', 'Am', 'G7').
        :return: A list of integer tokens representing the chord. The first token
                 is the root note for the bass, and subsequent tokens are for
                 the piano.
        :rtype: list
    """
    notes = ['E', 'F', 'G', 'A', 'B', 'C', 'D']
    if chord[0] in notes:
        base = notes.index(chord[0])
    else:
        raise ValueError(f'Invalid chord root note: {chord[0]}')
    basenote = [4, 5, 7, 9, 11, 12, 14][base]  # Bass
    chordtype = chord[1:]
    if len(chord) > 1 and chord[1] == '#':
        basenote += 1
        chordtype = chord[2:]
    offset = basenote + EncodingConfig.note_size * 2 + 24  # Piano notes
    if len(chordtype) == 0:
        return [basenote, offset, offset + 4, offset + 7]
    elif chordtype == 'm':
        return [basenote, offset, offset + 3, offset + 7]
    elif chordtype == '7':
        return [basenote, offset, offset + 4, offset + 7, offset + 10]
    elif chordtype == 'm7':
        return [basenote, offset, offset + 3, offset + 7, offset + 10]
    elif chordtype == 'M7':
        return [basenote, offset, offset + 4, offset + 7, offset + 11]
    elif chordtype == 'm7-5':
        return [basenote, offset, offset + 3, offset + 6, offset + 10]
    elif chordtype == 'dim':
        return [basenote, offset, offset + 3, offset + 6, offset + 9]
    elif chordtype == 'sus4':
        return [basenote, offset, offset + 5, offset + 7]
    elif chordtype == '7sus4':
        return [basenote, offset, offset + 5, offset + 7, offset + 10]
    elif chordtype == 'aug':
        return [basenote, offset, offset + 4, offset + 8]
    elif chordtype == 'm6':
        return [basenote, offset, offset + 3, offset + 7, offset + 9]
    elif chordtype == '7(9)':
        return [basenote, offset, offset + 4, offset + 7, offset + 10, offset + 14]
    elif chordtype == 'm7(9)':
        return [basenote, offset, offset + 3, offset + 7, offset + 10, offset + 14]
    elif chordtype == 'add9':
        return [basenote, offset, offset + 4, offset + 7, offset + 14]
    elif chordtype == '6':
        return [basenote, offset, offset + 4, offset + 7, offset + 9]
    elif chordtype == 'mM7':
        return [basenote, offset, offset + 3, offset + 7, offset + 11]
    elif chordtype == '7-5':
        return [basenote, offset, offset + 4, offset + 6, offset + 10]
    elif chordtype == '7#5':
        return [basenote, offset, offset + 4, offset + 8, offset + 10]
    else:
        return [basenote]


def load_latest_checkpoint(directory: str, name: str = 'checkpoint_', device: str = 'cpu', optimizer_class=None,
                           **optimizer_kwargs):
    """Load the latest checkpoint from a directory based on epoch number.

        This function searches for checkpoint files matching the pattern '{name}{epoch}.ph'
        and loads the one with the highest epoch number. It reconstructs the GPT2 model
        from the saved config, creates an optimizer with the specified class and parameters,
        and loads all saved states.

        :param str directory: Directory path containing checkpoint files.
        :param str name: Base name of checkpoint files (e.g., 'checkpoint_epoch_' for 'checkpoint_epoch_5.ph').
        :param str device: The device to load the model and optimizer to ('cpu', 'cuda' or 'cpu').
        :param optimizer_class: Optimizer class to instantiate (e.g., torch.optim.AdamW, torch.optim.Adam). If None, no optimizer is created.
        :type optimizer_class: torch.optim.Optimizer or None
        :param optimizer_kwargs: Keyword arguments passed to the optimizer constructor (e.g., lr=0.001, weight_decay=0.01, betas=(0.9, 0.999)).
        :returns: A tuple containing (model, optimizer, start_epoch, global_step) if checkpoint found, where model is GPT2LMHeadModel with loaded weights, optimizer is the optimizer instance with loaded state (or None if optimizer_class is None), start_epoch is the next epoch number to start training from, and global_step is the global step counter for continuous logging. Returns None if no valid checkpoint is found.
        :rtype: tuple or None
        :raises FileNotFoundError: If checkpoint exists but doesn't contain valid config.
        :raises ValueError: If checkpoint file is corrupted or missing required keys.
    """

    # Validate inputs
    if not os.path.exists(directory):
        print(f'Warning: Directory {directory} does not exist')
        return None

    if not os.path.isdir(directory):
        print(f'Warning: {directory} is not a directory')
        return None

    # Define regex pattern to extract epoch number from filename
    # Pattern matches: {name}{digits}.pth
    pattern = re.compile(rf'{re.escape(name)}(\d+)\.ph?$')

    latest_epoch = -1
    latest_file = None

    try:
        # Search for checkpoint files and find the one with highest epoch number
        for filename in os.listdir(directory):
            match = pattern.match(filename)
            if match:
                epoch = int(match.group(1))
                if epoch > latest_epoch:
                    latest_epoch = epoch
                    latest_file = filename

        # If no checkpoint found
        if latest_file is None:
            raise FileNotFoundError(f'No checkpoint files matching pattern {name}*.pth found in {directory}')

        # Load the latest checkpoint
        checkpoint_path = os.path.join(directory, latest_file)
        print(f'Loading checkpoint: {checkpoint_path}')

        try:
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
        except Exception as e:
            raise ValueError(f'Failed to load checkpoint file {checkpoint_path}: {e}')

        # Validate checkpoint structure
        required_keys = ['model_state_dict', 'epoch', 'global_step', 'config']
        missing_keys = [key for key in required_keys if key not in checkpoint]
        if missing_keys:
            raise ValueError(f'Checkpoint missing required keys: {missing_keys}')

        # Reconstruct from config
        try:
            config = GPT2Config(**checkpoint['config'])
        except Exception as e:
            raise ValueError(f'Failed to create GPT2Config from saved config: {e}')

        # Create model from config
        try:
            model = GPT2LMHeadModel(config)
            model.to(device)
        except Exception as e:
            raise ValueError(f'Failed to create GPT2LMHeadModel from config: {e}')

        # Load model weights
        try:
            model.load_state_dict(checkpoint['model_state_dict'])
        except Exception as e:
            raise ValueError(f'Failed to load model weights: {e}')

        # Create optimizer if class is provided
        optimizer = None
        if optimizer_class is not None:
            try:
                optimizer = optimizer_class(model.parameters(), **optimizer_kwargs)
            except Exception as e:
                raise ValueError(f'Failed to create optimizer: {e}')

            # Load optimizer state if available
            if 'optimizer_state_dict' in checkpoint:
                try:
                    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                except Exception as e:
                    raise ValueError(f'Failed to load optimizer state dict optimizer: {e}')

            # Move optimizer state to the correct device
            for state in optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to(device)

        # Extract training progress info
        start_epoch = checkpoint.get('epoch', 0) + 1
        global_step = checkpoint.get('global_step', 0)

        print(f'Successfully loaded checkpoint from epoch {checkpoint["epoch"]}')

        return model, optimizer, start_epoch, global_step

    except Exception as e:
        print(f'Error loading checkpoint: {e}')
        raise


def get_device(preferred: str = None) -> str:
    """Determines the best available PyTorch device for computation.

        Selects an appropriate device for PyTorch operations, prioritizing the user's
        preference if available, otherwise falling back to automatic detection in
        order of preference: XPU (Intel GPU) > CUDA (NVIDIA GPU) > CPU.

        :param str preferred: Preferred device type ('xpu', 'cuda', or 'cpu'), defaults to None.
        :returns: The selected device identifier string.
        :rtype: str
    """
    # Normalize and check if a preferred device was given
    preferred = preferred.lower() if preferred else None

    # Check if the preferred device is available
    if preferred == 'xpu' and hasattr(torch, 'xpu') and torch.xpu.is_available():
        return 'xpu'
    elif preferred == 'cuda' and torch.cuda.is_available():
        return 'cuda'
    elif preferred == 'cpu':
        return 'cpu'

    # Fallback to automatic selection
    if hasattr(torch, 'xpu') and torch.xpu.is_available():
        return 'xpu'
    elif torch.cuda.is_available():
        return 'cuda'
    else:
        return 'cpu'


def get_next_run_folder(name, base_dir='runs'):
    """Generates the next available run folder path with sequential numbering.

        Scans the base directory for existing folders matching the pattern '{name}_{index}'
        and returns a path for the next sequential run folder. Useful for organizing
        experiment outputs without overwriting previous runs.

        :param str name: Base name for the run folder (e.g., 'experiment', 'training').
        :param str base_dir: Directory to search for existing runs, defaults to 'runs'.
        :returns: Path to the next available run folder (e.g., 'runs/experiment_3').
        :rtype: str
    """
    # List all folders in the base directory
    existing_folders = os.listdir(base_dir)

    # Regex to capture 'run_<index>' format
    run_pattern = re.compile(fr'^{name}_(\d+)$')

    # Find the highest index
    max_index = 0
    for folder in existing_folders:
        match = run_pattern.match(folder)
        if match:
            # Extract the index from folder name
            index = int(match.group(1))
            max_index = max(max_index, index)

    # Increase the index by 1 for the next run
    new_run_name = f'{name}_{max_index + 1}'
    new_run_path = os.path.join(base_dir, new_run_name)

    return new_run_path
