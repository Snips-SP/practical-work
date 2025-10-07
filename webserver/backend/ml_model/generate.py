from backend.ml_model.helper import chord2tokens, load_latest_checkpoint, EncodingConfig, get_device
import numpy as np
import torch
import pypianoroll
from tqdm import trange
import os
from collections import deque


def sliding_window_generate(model, context, max_tokens=1024, window_size=1024, step_size=256, temperature=0.7, top_k=0, top_p=0.45):
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
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
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


def generate_from_chords(chords: list, timings: list, tempo: int,  model_dir: str, output: str = 'output.mid', temperature=1, top_k=0, top_p=0.45):
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
    :param float temperature: The sampling temperature for token generation. A higher value results in higher random
                        Defaults to 0.7.
    :param int top_k: The number of top-k most likely tokens to consider at each
                        Defaults to 0.
    :param int top_p: The cumulative probability for top-p sampling.
                        Defaults to 0.45.
    :raises FileNotFoundError: If `model_dir` does not exist or if the required
                               model and config files are not found within it.
    :raises IndexError: If the number of chords does not match the number of
                        timing durations, leading to an attempt to access a
                        non-existent chord.
    :returns: The file path of the generated MIDI file.
    :rtype: str
    """
    # Use appropriate xpu, cude or cpu device
    device = get_device('xpu')

    # Check if the run folder exists
    if not os.path.isdir(model_dir):
        raise FileNotFoundError('Could not find run directory to load model state dict and config file from')

    # Check if chords and timings are of equal length
    if len(chords) != len(timings):
        raise ValueError('The number of chords must be equal to the number of timings')

    # Load model from the folder
    model, _, _, _, _, _, _, _, _, _ = load_latest_checkpoint(model_dir)
    # Move to device
    model.to(device)
    model.eval()

    # Encode chords into tokens
    chord_tokens_queue = deque([chord2tokens(c) for c in chords])
    timings_queue = deque(timings)

    # Use a fixed-size deque for the context window to prevent unbounded memory growth
    # The size should be based on the model's window size.
    context_window_size = model.config.max_position_embeddings  # e.g., 1024
    context_sequence = deque(maxlen=context_window_size)

    # Create an empty pianoroll array
    total_steps = sum(timings)

    # Initialize context with special token and the first chord
    context_sequence.append(EncodingConfig.begin_note)
    context_sequence.extend(chord_tokens_queue.popleft())

    steps_in_current_chord = timings_queue.popleft()
    generated_sequence_cache = deque()

    step = EncodingConfig.pianoroll_resolution // EncodingConfig.encoding_resolution

    # Create an empty pianoroll array with resolution 24
    pianoroll = np.zeros((len(EncodingConfig.encoding_order), total_steps * step, 128))

    current_tick = 0
    i = 0
    # Decode it again
    with torch.no_grad(), trange(total_steps) as progress_bar:
        # Prevent infinite loop
        while i <= 10_000:
            i += 1
            # If the queue is empty, create new tokens
            if not generated_sequence_cache:
                new_tokens = sliding_window_generate(model, list(context_sequence), max_tokens=1024, temperature=temperature, top_k=top_k, top_p=top_p)
                generated_sequence_cache.extend(new_tokens)

            note = generated_sequence_cache.popleft()
            # Update the context sequence
            context_sequence.append(note)
            if note == EncodingConfig.time_note:
                # Either way update the position by 6 (step)
                current_tick += step
                # Update the progress bar by a step
                progress_bar.update(1)

                # Check if we have spent enough time on the chord and need to switch to the next one
                if steps_in_current_chord <= 1:
                    # Check if we still have a next chord
                    if chord_tokens_queue:
                        # Load the next chord into the context
                        steps_in_current_chord = timings_queue.popleft()
                        # Add the basic piano chord to the context queue
                        context_sequence.extend(chord_tokens_queue.popleft())
                        # Flush cache after chord change it will be filled at the beginning of next loop
                        generated_sequence_cache.clear()
                    else:
                        # We can terminate
                        pass
                else:
                    steps_in_current_chord -= 1

                # End the generation if we have reached the end
                if current_tick >= pianoroll.shape[1]:
                    break
            else:
                if note < EncodingConfig.instrument_bases['Drums']:
                    # It is a melodic note

                    # Calculate the trc the note belongs to
                    # 0 = Bass -> 3 = Bass
                    # 1 = Piano -> 1 = Piano
                    # 2 = Guitar -> 2 = Guitar
                    # 3 = String -> 4 Strings
                    trc = EncodingConfig.trc_idx[note // EncodingConfig.note_size]
                    # Calculate the midi note value
                    midi_value = note % EncodingConfig.note_size + EncodingConfig.note_offset
                    # Position is normal since melodic notes can only be placed on steps of 6
                    pianoroll[trc, current_tick:min(pianoroll.shape[1] - 1, current_tick + step), midi_value] = 100
                elif note < EncodingConfig.instrument_bases['Microtimings']:
                    # It is a drum note
                    trc = EncodingConfig.midi_tracks.index('Drums')
                    midi_value = EncodingConfig.drum_token_to_pitch[note]

                    # Check the next note if its a microtimings note
                    # Check if the queue is empty
                    if not generated_sequence_cache:
                        # Extend the queue with newly generated tokens
                        new_tokens = sliding_window_generate(model, list(context_sequence), max_tokens=1024, temperature=temperature, top_k=top_k, top_p=top_p)
                        generated_sequence_cache.extend(new_tokens)

                    if EncodingConfig.instrument_bases['Microtimings'] <= generated_sequence_cache[0] < \
                            EncodingConfig.instrument_bases['Special']:
                        # It is a microtiming, so we have to adjust the position by the offset
                        microtiming = generated_sequence_cache.popleft()
                        # Update the context sequence
                        context_sequence.append(microtiming)

                        offset = EncodingConfig.microtiming_token_to_delta[microtiming]
                        # Add offset to the position if it does not fall out of the length otherwise we discard it
                        if current_tick + offset < pianoroll.shape[1]:
                            pianoroll[trc, current_tick + offset, midi_value] = 100
                    else:
                        # It is not a microtiming but some other note, which means we do not have an offset
                        pianoroll[trc, current_tick, midi_value] = 100

                elif note < EncodingConfig.instrument_bases['Special']:
                    # It is a Microtimings token. They should not appear on their own. Only paired with drum tokens
                    # where they should always be filtered out by the above function
                    print(f'Microtimings token found: {note}')
                    raise ValueError('Microtimings token found')
                else:
                    # It is either padding or and end note
                    continue

    if i >= 10_000:
        print(f'Maximum number of iterations reached. Terminated.')

    # MIDI conversion
    pr = []
    for i, t in enumerate(EncodingConfig.midi_tracks):
        pr.append(pypianoroll.StandardTrack(
            pianoroll=pianoroll[i],
            program=EncodingConfig.programs[t],
            is_drum=(t == 'Drums'),
            name=f'Program: {t}, ({EncodingConfig.programs[t]})'
        ))
    # Create the multitrack object and write it to the output
    mt = pypianoroll.Multitrack(tracks=pr, tempo=np.full(pianoroll.shape[1], tempo), resolution=EncodingConfig.pianoroll_resolution)
    mt.write(output)

    return output
