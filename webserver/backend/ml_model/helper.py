import json
from backend.ml_model.encoding import EncodingConfig
from transformers import Phi3Config, Phi3ForCausalLM
from pydub import AudioSegment
from pydub.utils import which
import torch
import tempfile
import uuid
import subprocess
import re
import os


def mid_to_mp3(mid_file: str, sf2_file: str, output_file: str = 'output.mp3'):
    """Converts a MIDI file to an MP3 file using a SoundFont.

        This function synthesizes the MIDI file into a temporary WAV file
        using FluidSynth and the provided SF2 SoundFont. It then converts the
        WAV file to MP3, automatically trimming any leading or trailing silence.

        :param str mid_file: The path to the input MIDI file.
        :param str sf2_file: The path to the SoundFont (.sf2) file.
        :param str output_file: The path to save the resulting MP3 file,
                            defaults to 'output.mp3'.
        :raises AssertionError: If the `mid_file` or `sf2_file` is not found.
        :raises FileNotFoundError: If `ffmpeg` or `fluidsynth` is not found in the system PATH.
        :raises subprocess.CalledProcessError: If the `fluidsynth` command fails.
        :returns: None. The function saves the output directly to a file.
        :rtype: None
    """
    assert os.path.isfile(mid_file), f'MIDI file not found: {mid_file}'
    assert os.path.isfile(sf2_file), f'SoundFont file not found: {sf2_file}'

    if which('ffmpeg') is None:
        raise FileNotFoundError('ffmpeg not found. Please install it and ensure it is in your system PATH.')

    if which('fluidsynth') is None:
        raise FileNotFoundError('fluidsynth not found. Please install it and ensure it is in your system PATH.')

    # Manually create a unique temporary file path
    tmp_wav_path = os.path.join(tempfile.gettempdir(), f"{uuid.uuid4()}.wav")

    try:
        # Synthesize MIDI to the temporary WAV file
        subprocess.run(
            ['fluidsynth', '-qni', '-F', tmp_wav_path, sf2_file, mid_file],
            check=True
        )

        audio = AudioSegment.from_wav(tmp_wav_path)

        # Silence trimming logic remains the same
        silence_thresh = audio.dBFS - 30
        start_trim = 0
        for i, chunk in enumerate(audio):
            if chunk.dBFS > silence_thresh:
                start_trim = i
                break

        end_trim = len(audio)
        for i, chunk in enumerate(audio.reverse()):
            if chunk.dBFS > silence_thresh:
                end_trim = len(audio) - i
                break

        trimmed_audio = audio[start_trim:end_trim]

        trimmed_audio.export(output_file, format='mp3')

    finally:
        if os.path.exists(tmp_wav_path):
            os.remove(tmp_wav_path)


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


def load_latest_checkpoint(
        directory: str, name: str = 'checkpoint_epoch_', device: str = 'cpu',
        optimizer_class=None, learning_rate_scheduler_class=None, model_only=False
):
    """Load the latest checkpoint from a directory based on epoch number.

        This function searches for checkpoint files matching the pattern '{name}{epoch}.ph'
        and loads the one with the highest epoch number. It reconstructs the Phi-2 model
        from the saved config, creates an optimizer with the specified class and parameters,
        and loads all saved states.

        :param str directory: Directory path containing checkpoint files.
        :param str name: Base name of checkpoint files (e.g., 'checkpoint_epoch_' for 'checkpoint_epoch_5.ph').
        :param str device: The device to load the model and optimizer to ('cpu', 'cuda' or 'cpu').
        :param optimizer_class: Optimizer class to instantiate (e.g., torch.optim.AdamW, torch.optim.Adam). If None, no optimizer is created.
        :type optimizer_class: torch.optim.Optimizer or None
        :param optimizer_kwargs: Keyword arguments passed to the optimizer constructor (e.g., lr=0.001, weight_decay=0.01, betas=(0.9, 0.999)).
        :returns: A tuple containing (model, optimizer, start_epoch, global_step) if checkpoint found,
         where model is Phi-2LMHeadModel with loaded weights, optimizer is the optimizer instance with loaded state (or None if optimizer_class is None),
          start_epoch is the next epoch number to start training from, and global_step is the global step counter for continuous logging.
          Returns None if no valid checkpoint is found.
        :rtype: tuple or None
        :raises FileNotFoundError: If checkpoint exists but doesn't contain valid config.
        :raises ValueError: If checkpoint file is corrupted or missing required keys.
    """

    # Validate inputs
    if not os.path.isdir(directory):
        raise NotADirectoryError(f'Not a directory: {directory}')

    # Define regex pattern to extract epoch number from filename
    # Pattern matches: {name}{digits}.pth
    pattern = re.compile(rf'{re.escape(name)}(\d+)\.ph?$')

    latest_epoch = -1
    latest_file = None
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
    required_keys = ['model_state_dict', 'config']
    missing_keys = [key for key in required_keys if key not in checkpoint]
    if missing_keys:
        raise ValueError(f'Checkpoint missing required keys: {missing_keys}')

    # =========================
    # = Create and load model =
    # =========================
    try:
        config = Phi3Config(**checkpoint['config'])
        config._attn_implementation = 'eager'
    except Exception as e:
        raise ValueError(f'Failed to create Phi-2Config from saved config: {e}')

    # Create model from config
    try:
        model = Phi3ForCausalLM(config)
        model.to(device, dtype=checkpoint['model_dtype'])
    except Exception as e:
        raise ValueError(f'Failed to create PhiForCausalLM from config: {e}')

    # Load model weights
    try:
        model.load_state_dict(checkpoint['model_state_dict'])
    except Exception as e:
        raise ValueError(f'Failed to load model weights: {e}')

    if model_only:
        return model

    # =============================
    # = Create and load optimizer =
    # =============================
    optimizer = None
    optimizer_kwargs = None
    if optimizer_class is not None and 'optimizer_state_dict' in checkpoint and checkpoint['optimizer_kwargs'] is not None:
        try:
            optimizer_kwargs = checkpoint['optimizer_kwargs']
            optimizer = optimizer_class(model.parameters(), **optimizer_kwargs)
        except Exception as e:
            raise ValueError(f'Failed to create optimizer: {e}')

        # Load optimizer state if available
        try:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        except Exception as e:
            raise ValueError(f'Failed to load optimizer state dict optimizer: {e}')

        # Move optimizer state to the correct device
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)

    # ================================
    # = Create and load lr scheduler =
    # ================================
    learning_rate_scheduler = None
    learning_rate_scheduler_kwargs = None
    if learning_rate_scheduler_class is not None and 'lr_scheduler_state_dict' in checkpoint and checkpoint['lr_scheduler_kwargs'] is not None:
        try:
            learning_rate_scheduler_kwargs = checkpoint['lr_scheduler_kwargs']
            learning_rate_scheduler = learning_rate_scheduler_class(optimizer=optimizer,  **learning_rate_scheduler_kwargs)
        except Exception as e:
            raise ValueError(f'Failed to create learning rate scheduler: {e}')

        try:
            learning_rate_scheduler.load_state_dict(checkpoint['lr_scheduler_state_dict'])
        except Exception as e:
            raise ValueError(f'Failed to load learning rate scheduler state dict: {e}')

    # ===================================
    # = Load train valid and test split =
    # ===================================
    if os.path.isfile(os.path.join(directory, 'train_valid_test_split.json')):
        with open(os.path.join(directory, 'train_valid_test_split.json'), 'r') as f:
            train_valid_test_split = json.load(f)

        train_files = train_valid_test_split['train']
        valid_files = train_valid_test_split['valid']
        test_files = train_valid_test_split['test']
    else:
        train_files = None
        valid_files = None
        test_files = None
        print(f'Warning: "train_valid_test_split.json" file not found in {directory}.')

    # ==================================
    # = Extract training progress info =
    # ==================================
    if 'training_loss_per_epoch' in checkpoint:
        training_loss_per_epoch = checkpoint['training_loss_per_epoch']
    else:
        training_loss_per_epoch = None
        print(f'Warning: "training_loss_per_epoch" key not found in checkpoint.')

    if 'validation_loss_per_epoch' in checkpoint:
        validation_loss_per_epoch = checkpoint['validation_loss_per_epoch']
    else:
        validation_loss_per_epoch = None
        print(f'Warning: "validation_loss_per_epoch" key not found in checkpoint.')

    if 'patience' in checkpoint:
        patience_dict = checkpoint['patience']
    else:
        patience_dict = None
        print(f'Warning: "patience_tuple" key not found in checkpoint.')

    if 'epoch' in checkpoint:
        start_epoch = checkpoint['epoch'] + 1
    else:
        start_epoch = None
        print(f'Warning: "epoch" key not found in checkpoint.')

    if 'global_step' in checkpoint:
        global_step = checkpoint['global_step']
    else:
        global_step = None
        print(f'Warning: "global_step" key not found in checkpoint.')

    return model, training_loss_per_epoch, validation_loss_per_epoch, patience_dict, start_epoch, global_step, optimizer, optimizer_kwargs, learning_rate_scheduler, learning_rate_scheduler_kwargs, train_files, valid_files, test_files


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