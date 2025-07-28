from backend.ml_model.helper import chord2tokens
from backend.ml_model.train import EncodingConfig, get_latest_checkpoint
import numpy as np
import torch
from transformers import GPT2LMHeadModel, GPT2Config
import pypianoroll
from tqdm import trange
import os
from collections import deque


def sliding_window_generate(model, context, max_tokens=1024, window_size=1024, step_size=256):
    """Generates a sequence of tokens using a sliding window approach.

    This is useful for generating sequences longer than the model's maximum
    context window. The function generates `step_size` new tokens in each
    iteration, using the last part of the previously generated sequence as the
    new context.

    :param model: The pre-trained autoregressive model (e.g., GPT2LMHeadModel).
    :param list context: A list of initial token IDs to seed the generation.
    :param int max_tokens: The target number of new tokens to generate.
    :param int window_size: The context window size of the model.
    :param int step_size: The number of new tokens to generate in each step.
    :returns: A NumPy array containing the generated token IDs.
    :rtype: numpy.ndarray
    """
    device = model.device
    context_tokens = torch.tensor(context, device=device, dtype=torch.int64).unsqueeze(0)
    generated_token_chunks = []
    new_tokens_count = 0

    with torch.no_grad():
        while new_tokens_count <= max_tokens:
            # The current context is the last (window_size - step_size) tokens
            current_context = context_tokens[:, -(window_size - step_size):]

            output = model.generate(
                current_context,
                max_length=current_context.shape[1] + step_size,
                temperature=0.7,
                top_k=0,
                top_p=0.9,
                do_sample=True,
                use_cache=True
            )

            # Grab only the new tokens
            new_tokens = output[:, current_context.shape[1]:]

            # Append the new tokens to our list and update the full context
            generated_token_chunks.append(new_tokens)
            context_tokens = torch.cat([context_tokens, new_tokens], dim=1)
            new_tokens_count += new_tokens.shape[1]

    # Concatenate all generated chunks at once
    all_generated_tokens = torch.cat(generated_token_chunks, dim=1)

    # Return the generated tokens, trimmed to the exact max_tokens length
    return all_generated_tokens[:, :max_tokens].cpu().squeeze(0).numpy()


def generate_from_chords(chords: list, timings: list, tempo: int,  model_dir: str, output: str = 'output.mid'):
    """Generates a multitrack MIDI file from a sequence of chords.

    This function is the heart of the music generation process. It loads a pre-trained
    autoregressive model, encodes the input chord progression into tokens, and then
    iteratively generates notes one at a time. The generated notes are assembled into
    a piano roll, which is finally converted and saved as a MIDI file.

    :param list chords: A list of chord strings (e.g., ['C', 'G', 'Am', 'F']).
    :param list timings: A list of integers, where each integer specifies the
                        duration of the corresponding chord in 1/16th notes.
    :param int tempo: The tempo of the generated MIDI file in beats per minute (BPM).
    :param str model_dir: The directory path containing the saved model state
                          dictionary and configuration file.
    :param str output: The file path where the output MIDI file will be saved.
                       Defaults to 'output.mid'.
    :raises FileNotFoundError: If `model_dir` does not exist or if the required
                               model and config files are not found within it.
    :raises IndexError: If the number of chords does not match the number of
                        timing durations, leading to an attempt to access a
                        non-existent chord.
    :returns: The file path of the generated MIDI file.
    :rtype: str
    """
    # Use appropriate xpu, cude or cpu device
    device = ('xpu' if torch.xpu.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu')

    # Check if the run folder exists
    if not os.path.isdir(model_dir):
        raise FileNotFoundError('Could not find run directory to load model state dict and config file from')

    # Check if chords and timings are of equal length
    if len(chords) != len(timings):
        raise ValueError('The number of chords must be equal to the number of timings')

    state_dict_file_name = 'gpt_model_state_dict_epoch_'
    model_path = get_latest_checkpoint(model_dir, state_dict_file_name)
    config_path = os.path.join(model_dir, f'config.json')

    if model_path is None:
        raise FileNotFoundError('No state dictionary not found in folder.')

    if not os.path.exists(config_path):
        raise FileNotFoundError('No config file found in folder.')

    print(f'Loading model from: {model_dir}')

    # Load config
    config = GPT2Config.from_json_file(config_path)
    # Create model from loaded configuration
    model = GPT2LMHeadModel(config)
    # Load model weights
    model.load_state_dict(torch.load(model_path, weights_only=True, map_location=device))
    # Move to device
    model.to(device)
    model.eval()

    # Encode chords into tokens
    chord_tokens_queue = deque([chord2tokens(c) for c in chords])
    timings_queue = deque(timings)

    # Use a fixed-size deque for the context window to prevent unbounded memory growth
    # The size should be based on the model's window size.
    context_window_size = model.config.n_positions  # e.g., 1024 for GPT-2
    context_sequence = deque(maxlen=context_window_size)

    # Create an empty pianoroll array
    total_steps = sum(timings)
    pianoroll = np.zeros((len(EncodingConfig.tracks), total_steps, 128))

    # Initialize context with special token and the first chord
    context_sequence.append(EncodingConfig.end_note)
    context_sequence.extend(chord_tokens_queue.popleft())

    current_chord_duration = timings_queue.popleft()
    time_step_in_chord = 0
    generated_sequence_cache = deque()

    # TODO: Get rid of this assumption
    # We always generate 1024 tokens from a chord which we give the network in the first time step. (The conditioned chord)
    # We assume that the network will generate 1024 tokens which harmonize with the first chord and
    # that it does not just switch to a different chord on its own.

    with torch.no_grad(), trange(total_steps) as progress_bar:
        for pos in progress_bar:
            # Check if we need to generate more tokens
            if not generated_sequence_cache:
                new_tokens = sliding_window_generate(model, list(context_sequence), max_tokens=1024)
                generated_sequence_cache.extend(new_tokens)

            # Process tokens until we find a time event
            while True:
                if not generated_sequence_cache:
                    # Regenerate if cache runs out mid-step (rare but possible)
                    new_tokens = sliding_window_generate(model, list(context_sequence), max_tokens=1024)
                    generated_sequence_cache.extend(new_tokens)

                note = generated_sequence_cache.popleft()
                context_sequence.append(note)

                if note == EncodingConfig.time_note:
                    time_step_in_chord += 1
                    break  # Move to the next time step (pos)
                elif note < EncodingConfig.time_note:
                    # Place note in the pianoroll at the current position
                    track_index = EncodingConfig.trc_idx[note // EncodingConfig.note_size]
                    midi_note = note % EncodingConfig.note_size + EncodingConfig.note_offset
                    if midi_note < 128:
                        pianoroll[track_index, pos, midi_note] = 100

            # Check for chord change
            if time_step_in_chord >= current_chord_duration:
                time_step_in_chord = 0
                if chord_tokens_queue:
                    # Load the next chord into the context
                    current_chord_duration = timings_queue.popleft()
                    next_chord_tokens = chord_tokens_queue.popleft()
                    context_sequence.extend(next_chord_tokens)
                    generated_sequence_cache.clear()  # Flush cache after chord change

    # MIDI conversion
    pr = []
    for i, t in enumerate(EncodingConfig.tracks):
        pr.append(pypianoroll.StandardTrack(
            pianoroll=pianoroll[i],
            program=EncodingConfig.programs[t],
            is_drum=(t == 'Drums'),
            name=f'Program: {t}, ({EncodingConfig.programs[t]})'
        ))

    mt = pypianoroll.Multitrack(tracks=pr, tempo=np.full(pianoroll.shape[1], tempo), resolution=4)

    mt.write(output)

    return output
