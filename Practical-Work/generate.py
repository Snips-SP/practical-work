from helper import chord2tokens
import numpy as np
import torch
from transformers import GPT2LMHeadModel
from train import NetworkConfig
from helper import EncodingConfig
import pypianoroll


def generate_from_context(model, context, device):
    # Move to right device and generate
    input_ids = torch.tensor(context, device=device, dtype=torch.int64).unsqueeze(0)
    # TODO: Implement sliding window function to generate new tokens while slowly fading out the old context
    output_ids = model.generate(
        input_ids,
        max_length=1024,  # Generate 1024 NEW tokens, the generated sequence will include the input tokens as well
        temperature=1.0,  # Sampling randomness
        top_k=0,  # Controls diversity (higher means more randomness)
        top_p=0.7,  # Nucleus sampling
        do_sample=True  # Enables stochastic sampling
    )
    # Only keep new tokens
    new_tokens = output_ids[0, input_ids.shape[1]:]

    return new_tokens.numpy()


def generate_from_chords(chords: list, timings: list, num_bars: int, tempo: int, output: str = 'output.mid'):
    # Use appropriate gpu or cpu
    device = ('xpu' if torch.xpu.is_available() else
              'cuda' if torch.cuda.is_available() else
              'cpu')
    device = 'cpu'

    assert np.sum(timings) // 16 == num_bars, 'Chord timings dont add up to number of bars'

    model = GPT2LMHeadModel(NetworkConfig.config)
    model.load_state_dict(torch.load('gpt_model_state_dict.ph', weights_only=True, map_location=device))
    model.to(device)

    # Encode chords into tokens
    tokens = [chord2tokens(chord) for chord in chords]

    # Create empty pianoroll array
    pianoroll = np.zeros((len(EncodingConfig.tracks), num_bars * 16, 128))

    # Retrieve first chord
    chord = tokens.pop(0)

    # Will be used as context for the neural network
    context_sequence = [EncodingConfig.end_note]
    context_sequence.extend(chord)

    # TODO: Get rid of this assumption
    # We always generate 1024 tokens. Minus the tokens which we started from,
    # if we dont implement sliding window generation.
    # It could be however that the chord is to be repeated
    # More times than we can provide with these tokens.
    # Generate a sequence from our current context using the neural network
    sequence_from_chord = generate_from_context(model, context_sequence, device)

    # FOR DEBUGGING
    # Set every 15th element to 420
    sequence_from_chord[14::15] = EncodingConfig.time_note

    # Loop over all the 1/16 notes in our pianoroll file
    pos = 0
    timings_pos = 0
    while pos < pianoroll.shape[1]:
        for note in sequence_from_chord:
            # Add the current note to the context
            context_sequence.append(note)

            if note == EncodingConfig.time_note:
                # We have hit a time note, thus we have to advance to the next 1/16th note in the song
                if timings[timings_pos] == 0:
                    # The chord was repeated the correct amount of times
                    # Time for a chord change, which means we generate new sequence from our context plus the new chord
                    chord = tokens.pop(0)
                    context_sequence.extend(chord)
                    # Placeholder for generating a new sequence using the one we already have.
                    sequence_from_chord = generate_from_context(model, context_sequence, device)

                    # FOR DEBUGGING
                    # Set every 15th element to 420
                    sequence_from_chord[14::15] = EncodingConfig.time_note
                else:
                    # TODO: Get rid of assumption
                    # We take notes from the same sequence generated earlier
                    # We assume that the network will not just switch chord on it own in such a quick manner
                    timings[timings_pos] -= 1
            elif note < EncodingConfig.time_note:
                # We have hit an actually generated note, thus we put it into its right place in our pianoroll array
                # Calculate which track the note belongs to
                trc = EncodingConfig.trc_idx.index(note // EncodingConfig.note_size)
                # Calculate the midi note value
                mid = note % EncodingConfig.note_size + EncodingConfig.note_offset
                # Set the volume to 100 for the note in the piano roll array
                if mid < 128:
                    pianoroll[trc, pos, mid] = 100

    pr = []
    for i, (t, p) in enumerate(zip(EncodingConfig.tracks, [0, 0, 24, 32, 40])):
        pr.append(pypianoroll.Track(pianoroll=pianoroll[i], program=p, is_drum=(t == 'Drums')))
    mt = pypianoroll.Multitrack(tracks=pr, tempo=tempo, beat_resolution=4)
    mt.write(output)


generate_from_chords(['F', 'D', 'B'], [16, 32, 16], 4, 80)
