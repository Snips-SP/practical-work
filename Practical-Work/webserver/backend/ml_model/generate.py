from .helper import chord2tokens
from .train import NetworkConfig, EncodingConfig
import numpy as np
import torch
from transformers import GPT2LMHeadModel
import pypianoroll
from tqdm import trange


def sliding_window_generate(model, context, max_tokens=1024, window_size=1024, step_size=128):
    device = model.device
    generated_tokens = torch.tensor(context, device=device, dtype=torch.int64).unsqueeze(0)

    while generated_tokens.shape[1] < max_tokens:
        # The current context is the last n tokens which where generated
        # where n is the window_size - step_size, since we need exactly step_size space
        # to generate step_size new tokens
        current_context = generated_tokens[:, -(window_size - step_size):]

        output = model.generate(
            current_context,
            # We generate step_size new tokens
            max_length=current_context.shape[1] + step_size,
            temperature=1.0,
            top_k=0,
            top_p=0.7,
            do_sample=True
        )
        # Grap only the new tokens
        new_tokens = output[:, current_context.shape[1]:]
        # Append the new tokens to ALL generated tokens
        generated_tokens = torch.cat([generated_tokens, new_tokens], dim=1)

    # Ignore input tokens by returning only the last max_tokens generated tokens
    return generated_tokens[:, -max_tokens:].cpu().squeeze(0).numpy()


def generate_from_context(model, context, device):
    # Move to right device and generate
    input_ids = torch.tensor(context, device=device, dtype=torch.int64).unsqueeze(0)

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

    return new_tokens.cpu().squeeze(0).numpy()


def generate_from_chords(chords: list, timings: list, tempo: int,  model_path: str, output: str = 'output.mid'):
    # Use appropriate gpu or cpu
    device = ('xpu' if torch.xpu.is_available() else
              'cuda' if torch.cuda.is_available() else
              'cpu')

    model = GPT2LMHeadModel(NetworkConfig.config)
    model.load_state_dict(torch.load(model_path, weights_only=True, map_location=device))
    model.to(device)

    # Encode chords into tokens
    tokens = [chord2tokens(chord) for chord in chords]

    # Create empty pianoroll array
    pianoroll = np.zeros((len(EncodingConfig.tracks), np.sum(timings), 128))

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
    # We would have to generate tokens until we have counted enough time_notes...
    # But for now we just generate a bunch and hope for the best

    # Generate a sequence from our current context using the neural network
    sequence_from_chord = sliding_window_generate(model, context_sequence, max_tokens=1024)

    # FOR DEBUGGING
    # Set every 15th element to 420
    # sequence_from_chord[14::15] = EncodingConfig.time_note

    # Loop over all the 1/16 notes in our pianoroll file
    pos = 0
    timings_pos = 0
    with trange(pos, pianoroll.shape[1]) as progress_bar:
        while pos < pianoroll.shape[1]:
            for note in sequence_from_chord:
                # Add the current note to the context
                context_sequence.append(note)

                if note == EncodingConfig.time_note:
                    # We have hit a time note, thus we have to advance to the next 1/16th note in the song
                    if timings[timings_pos] == 0:
                        timings_pos += 1
                        # The chord was repeated the correct amount of times
                        # Time for a chord change, which means we generate new sequence from our context plus the new chord
                        chord = tokens.pop(0)
                        context_sequence.extend(chord)
                        # Placeholder for generating a new sequence using the one we already have.
                        sequence_from_chord = sliding_window_generate(model, context_sequence, max_tokens=1024)

                        # FOR DEBUGGING
                        # Set every 15th element to 420
                        # sequence_from_chord[14::15] = EncodingConfig.time_note
                    else:
                        # TODO: Get rid of assumption
                        # We take notes from the same sequence generated earlier
                        # We assume that the network will not just switch chord on it own in such a quick manner
                        timings[timings_pos] -= 1
                    # Either way update the position by 1
                    pos += 1
                    if pos >= pianoroll.shape[1]:
                        break
                    progress_bar.update(1)  # Update tqdm
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
    for i, t in enumerate(EncodingConfig.tracks):
        pr.append(pypianoroll.StandardTrack(pianoroll=pianoroll[i], program=EncodingConfig.programs[t], is_drum=(t == 'Drums')))
    mt = pypianoroll.Multitrack(tracks=pr, tempo=np.full(pianoroll.shape[1], tempo), resolution=4)

    mt.write(output)

    return output

