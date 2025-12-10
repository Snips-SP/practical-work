import os
import re
import time
import numpy as np
import pypianoroll
import torch
from backend.ml_model.dataloader import OnTheFlyMidiDataset
from backend.ml_model.encoding import EncodingConfig
from backend.ml_model.helper import get_device, load_latest_checkpoint
from tqdm import tqdm


def create_seed_from_drum_midi(midi_path: str, chunk_size: int = 256):
    # Load the mid as multi track object
    m = pypianoroll.read(midi_path)

    # Beat resolution is the number of steps a measure is divided into
    # Calculate how many time steps a quarter note takes to fill a measure
    step = m.resolution // 4

    # Convert it to a numpy array of shape (num_tracks, num_time_steps, num_pitches=128)
    pr = m.stack()
    # (num_time_steps, num_pitches=128, num_tracks)
    pr = np.transpose(pr, (1, 2, 0))

    # Init token list
    tokens = []
    tick = 0
    tokens.append(EncodingConfig.begin_note)

    seq_buffer = [
        # Anchor tick, buffer
        (tick, []),
        (tick + step, [])
    ]
    while tick < pr.shape[0]:
        # Update our valid positions
        if tick % step == 0 and tick != 0:
            # Handle the edge case where both buffers are going to be the same
            if not tick + step == seq_buffer[1][0]:
                # We have advanced to the next valid sixteenth note,
                # thus we rotate out the last sequence and write it to seq
                _, last_seq = seq_buffer[0]
                last_seq.append(EncodingConfig.time_note)
                tokens.extend(EncodingConfig.reorder_current(last_seq))
                # Remove the last sequence and add new next sequence
                seq_buffer.pop(0)
                seq_buffer.append((tick + step, []))

        # active has shape (N, 2) with columns [pitch, track]
        active = np.argwhere(pr[tick] != 0)
        for pitch, track in active:
            # -------------------
            # Handle drum events
            # -------------------
            if pitch not in EncodingConfig.drum_pitches:
                continue

            # Determine to which sixteenth note this note snaps to (either next or last)
            offset = tick - seq_buffer[0][0]
            if offset <= 3:
                buffer = 0
            else:
                offset = tick - seq_buffer[1][0]
                buffer = 1

            # Map pitch
            drum_token = EncodingConfig.drum_pitch_to_token[pitch]

            # Add base drum token
            seq_buffer[buffer][1].append(drum_token)

            # Add timing offset if not 0
            if offset != 0:
                offset_token = EncodingConfig.microtiming_delta_to_token[offset]
                seq_buffer[buffer][1].append(offset_token)

        # Check if we have enough tokens otherwise just keep going
        if chunk_size - 2 <= len(seq_buffer[0][1]) + len(seq_buffer[1][1]) + len(tokens):
            break
        # Increase tick
        tick += 1

    # Write all sequences to seq
    for _, sub_seq in seq_buffer:
        tokens.extend(EncodingConfig.reorder_current(sub_seq))

    return np.array(tokens)


def get_dyad_midi(chord_name, root_octave=2):
    # Map Note Names to Semitone Offsets (C = 0)
    # Includes both Sharps (#) and Flats (b)
    note_map = {
        'C': 0, 'C#': 1, 'Db': 1,
        'D': 2, 'D#': 3, 'Eb': 3,
        'E': 4,
        'F': 5, 'F#': 6, 'Gb': 6,
        'G': 7, 'G#': 8, 'Ab': 8,
        'A': 9, 'A#': 10, 'Bb': 10,
        'B': 11
    }

    # Map Chord Qualities to Characteristic Intervals (in semitones)
    # Based on the strategy of selecting the most "defining" note.
    quality_map = {
        # --- Basic Triads ---
        '': 4,  # Major: The Major 3rd (4) defines the major quality.
        'm': 3,  # Minor: The Minor 3rd (3) defines the minor quality.

        # --- Suspended / Add ---
        'sus4': 5,  # Suspended: The Perfect 4th (5) replaces the 3rd.
        'add9': 2,  # Add9: The Major 2nd (2) is the added color.


        # --- Diminished / Augmented ---
        'dim': 6,  # Diminished: The Tritone (6) is the defining unstable interval.
        'aug': 8,  # Augmented: The Minor 6th/Aug 5th (8) is the defining tension.

        # --- 7th Chords ---
        'M7': 11,  # Major 7: The Major 7th (11) is the essential "jazz" tone.
        'maj7': 11,  # (Alias for M7)
        'mM7': 11,  # Minor-Major 7: The Major 7th (11) creates the "Spy" sound against the minor 3rd.

        'min7': 10,  # Minor 7: The Minor 7th (10) defines the function.
        'm7': 10,  # (Alias for min7)

        '7': 10,  # Dominant 7: The Minor 7th (10) is the tritone resolution point.

        # --- 6th Chords ---
        '6': 9,  # Major 6: The Major 6th (9) is the characteristic color.
        'm6': 9,  # Minor 6: The Major 6th (9) creates the "Dorian" sound.

        # --- Altered 7ths ---
        'M7-5': 6,  # Maj7b5: The Tritone (6) is more defining here than the M7.
        '7-5': 6,  # 7b5: The Tritone (6) is the core altered tone.
        'm7-5': 6,  # Half-Dim: The Tritone (6) defines the half-diminished sound.
        '7#5': 8,  # 7#5 (Alt): The Augmented 5th (8) creates the specific tension.

        # --- Extended (9ths) ---
        '7(9)': 14,  # Dom9: Checks for the Major 9th (14).
        'm7(9)': 14,  # Min9: Checks for the Major 9th (14).
        'maj9': 14,  # Maj9: We check for the 9th (2) to distinguish from standard maj7.
        'M9': 14,
        '9': 14,    # Dominant9
        'min9': 14, # Min9
        'm9': 14, # Min9
    }

    # Parse the String using Regex
    # Looks for: Start -> (A-G followed by optional # or b) -> (Rest of string)
    match = re.match(r"^([A-G][#b]?)(.*)$", chord_name)

    if not match:
        raise Exception(f'Error: Could not parse chord {chord_name}')

    root_str = match.group(1)
    quality_str = match.group(2)

    # Calculate Pitches
    if root_str not in note_map:
        raise Exception(f'Error: Invalid root note {root_str}')

    # Get Root MIDI
    root_pitch_class = note_map[root_str]
    root_midi = (root_octave + 1) * 12 + root_pitch_class  # MIDI 60 is C4 (Octave 4)

    # Get Interval Offset
    offset = quality_map.get(quality_str)

    if offset is None:
        raise Exception(f'Quality {quality_str} not found.')

    interval_midi = root_midi + offset

    return [root_midi, interval_midi]


def generate_sixteenth_notes(model, context, num_sixteenth_notes=16, window_size=2048, step_size=256, temperature=0.7, top_k=0, top_p=0.45, device='cpu'):
    """
    Generates a sequence of tokens using a sliding window approach,
    optimized for a torch.compiled model by using a fixed input shape.
    """
    steady_state_input_len = window_size - step_size
    target_gen_length = steady_state_input_len + step_size

    context_tokens = torch.tensor(context, device=device, dtype=torch.int64).unsqueeze(0)

    # Pad the initial context
    if context_tokens.shape[1] < steady_state_input_len:
        pad_len = steady_state_input_len - context_tokens.shape[1]
        padding = torch.full((1, pad_len), EncodingConfig.padding_token, device=device, dtype=torch.int64)
        context_tokens = torch.cat([padding, context_tokens], dim=1)

    context_tokens = context_tokens[:, -steady_state_input_len:]

    generated_token_chunks = []
    new_sixteenth_notes = 0

    with torch.no_grad():
        while new_sixteenth_notes <= num_sixteenth_notes:
            # current_context will always have shape [1, steady_state_input_len]
            current_context = context_tokens[:, -steady_state_input_len:]

            # Create a mask that ignores padding tokens.
            attention_mask = (current_context != EncodingConfig.padding_token).long()

            context_tokens = model.generate(
                current_context,
                attention_mask=attention_mask,
                max_length=target_gen_length,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                do_sample=True,
                use_cache=True,
                pad_token_id=EncodingConfig.padding_token,
            )

            # Grab only the new tokens
            new_tokens = context_tokens[:, current_context.shape[1]:]

            # Append the new tokens to our list and update the full context
            generated_token_chunks.append(new_tokens.cpu())
            # Count up how many sixteenth notes we generated
            new_sixteenth_notes += (new_tokens == EncodingConfig.time_note).sum()

    # Concatenate all generated chunks at once
    all_generated_tokens = torch.cat(generated_token_chunks, dim=1)

    # Return the generated tokens
    return all_generated_tokens.cpu().squeeze(0).numpy()


def continue_piece(chord, num_sixteenth_notes, model, piece, instrument='Guitar', keep_context=True, temperature=1, top_k=0, top_p=0.45, device='xpu'):
    # Find last sixteenth note in the context
    last_time_note = (piece.size - 1 - np.argmax(piece[::-1] == EncodingConfig.time_note))

    # This should work if the last note is not a time note
    # Split into context and anticipation
    context = piece[:last_time_note]
    anticipation = piece[last_time_note:]

    # Extract only the drums from the last sixteenth note
    drum_interval = EncodingConfig.instrument_intervals['Drum+Micro']
    mask = ((drum_interval[0] <= anticipation) & (anticipation <= drum_interval[1]))
    anticipation = list(anticipation[mask])

    # Get the most important pitches for the chord identity
    root_pitch, interval_pitch = get_dyad_midi(chord, root_octave=2)

    # Add the root note as the bass to the anticipation
    anticipation.append(EncodingConfig.instrument_intervals['Bass'][0] - 1 + root_pitch - EncodingConfig.note_offset)

    # Add the interval pitch either to the guitar track of piano track
    if instrument == 'Guitar':
        # Get the pitch into Octave 3 by adding +12
        anticipation.append(EncodingConfig.instrument_intervals['Guitar'][0] - 1 + root_pitch + 12 - EncodingConfig.note_offset)
    elif instrument == 'Piano':
        # Get the pitch into Octave 4 by adding +24
        anticipation.append(EncodingConfig.instrument_intervals['Piano'][0] - 1 + root_pitch + 24 - EncodingConfig.note_offset)
    else:
        raise ValueError(f'Unknown Instrument {instrument}')

    # Concatenate the context and the correctly ordered anticipation together
    piece = np.concatenate((context, [EncodingConfig.time_note], EncodingConfig.reorder_current(anticipation), [EncodingConfig.time_note]))
    new_notes = generate_sixteenth_notes(
        model,
        # Add the context with the anticipation together
        context=piece,
        num_sixteenth_notes=num_sixteenth_notes,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        device=device
    )

    # Cut any tokens that may have been created after the last time note
    time_notes = np.where(new_notes == EncodingConfig.time_note)[0]
    new_tokens = new_notes[:time_notes[num_sixteenth_notes - 1]]

    if keep_context:
        # We include the piece and the new tokens
        piece = np.concatenate([piece, new_tokens])
    else:
        # We dont want the drum seed included in the piece
        piece = np.concatenate(([EncodingConfig.begin_note], new_tokens))

    return piece


def decode_to_midi(piece: np.ndarray, total_length: int, tempo: int, output: str = 'output.mid'):
    # Define step size
    step = EncodingConfig.pianoroll_resolution // EncodingConfig.encoding_resolution

    # Create an empty pianoroll array with resolution 24
    pianoroll = np.zeros((len(EncodingConfig.encoding_order), total_length * step, 128))

    # Prepare intervals
    melodic_interval = EncodingConfig.instrument_intervals['Melodic']
    drum_interval = EncodingConfig.instrument_intervals['Drums']
    bass_interval = EncodingConfig.instrument_intervals['Bass']
    guitar_interval = EncodingConfig.instrument_intervals['Guitar']
    strings_interval = EncodingConfig.instrument_intervals['Strings']
    piano_interval = EncodingConfig.instrument_intervals['Piano']
    microtiming_interval = EncodingConfig.instrument_intervals['Microtimings']

    current_tick = 0
    i = 0
    # Decode it again
    while i <= 100_000:
        if i >= len(piece):
            print('We didnt produce enough tokens')
            break
        note = piece[i]
        i += 1

        if note == EncodingConfig.time_note:
            # Either way update the position by 6 (step)
            current_tick += step

            # End the generation if we have reached the end
            if current_tick >= pianoroll.shape[1]:
                break
        else:
            if melodic_interval[0] <= note <= melodic_interval[1]:
                # It is a melodic note
                if bass_interval[0] <= note <= bass_interval[1]:
                    track = 'Bass'
                elif guitar_interval[0] <= note <= guitar_interval[1]:
                    track = 'Guitar'
                elif strings_interval[0] <= note <= strings_interval[1]:
                    track = 'Strings'
                elif piano_interval[0] <= note <= piano_interval[1]:
                    track = 'Piano'
                else:
                    raise ValueError('Note not in melodic range')

                # Calculate the midi note value
                midi_value = note + EncodingConfig.note_offset - (EncodingConfig.instrument_intervals[track][0] - 1)
                trc_index = EncodingConfig.encoding_order.index(track)

                # Position is normal since melodic notes can only be placed on steps of 6
                pianoroll[trc_index, current_tick:min(pianoroll.shape[1] - 1, current_tick + step), midi_value] = 100
            elif drum_interval[0] <= note <= drum_interval[1]:
                # It is a drum note
                trc = EncodingConfig.midi_tracks.index('Drums')
                midi_value = EncodingConfig.drum_token_to_pitch[note]

                if i + 1 < len(piece) and microtiming_interval[0] <= piece[i + 1] <= microtiming_interval[1]:
                    # It is a microtiming, so we have to adjust the position by the offset
                    i += 1
                    microtiming = piece[i]

                    offset = EncodingConfig.microtiming_token_to_delta[microtiming]
                    # Add offset to the position if it does not fall out of the length otherwise we discard it
                    if current_tick + offset < pianoroll.shape[1]:
                        pianoroll[trc, current_tick + offset, midi_value] = 100
                else:
                    # It is not a microtiming but some other note, which means we do not have an offset
                    pianoroll[trc, current_tick, midi_value] = 100
            else:
                # Its a different note we can ignore end_note, padding_token, or microtiming
                continue

    # MIDI conversion
    pr = []
    for i, t in enumerate(EncodingConfig.encoding_order):
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


def generate_backing_track(chords: list, timings: list, tempo: int,  model_dir: str, drum_seed_midi_file: str, output: str = 'output.mid',  temperature=1, top_k=0, top_p=0.45):
    # Use appropriate xpu, cude or cpu device
    device = get_device('xpu')

    # Check if the run folder exists
    if not os.path.isdir(model_dir):
        raise FileNotFoundError('Could not find run directory to load model state dict and config file from')

    # Check if chords and timings are of equal length
    if len(chords) != len(timings):
        raise ValueError('The number of chords must be equal to the number of timings')

    # Load model from the folder
    model = load_latest_checkpoint(model_dir, model_only=True)
    # Move to device
    model.to(device)
    model.eval()

    # Create seed from midi file
    drum_seed = create_seed_from_drum_midi(drum_seed_midi_file)
    # Cut it down to only one bar
    time_notes = np.where(drum_seed == EncodingConfig.time_note)[0]
    piece = drum_seed[:time_notes[15]]
    keep_context = False

    # Create music per chord for every sixteenth note in timing
    for chord, timing in tqdm(zip(chords, timings), desc=f'Generating {", ".join(chords)} chords', total=len(chords)):
        # We have to combine the new tokens and the piece
        piece = continue_piece(chord, timing, model, piece, 'Guitar', keep_context=keep_context, temperature=temperature, top_k=top_k, top_p=top_p, device=device)
        keep_context = True


    # Decode back to midi format and return the path to the output file
    return decode_to_midi(piece, sum(timings), tempo=tempo, output=output)
