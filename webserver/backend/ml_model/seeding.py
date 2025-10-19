import os
import pickle
from backend.ml_model.encode import Runner
from backend.ml_model.helper import EncodingConfig

def create_seed_from_midi(midi_path: str, overwrite=False):
    # Encoded file
    encoded_file_path = midi_path + f'.0.tmp'

    if not os.path.exists(encoded_file_path) or overwrite is True:
        # Reuse the same runner to decode the midi file into a pickle file
        runner = Runner(0, None, overwrite=True)
        runner._run(midi_path)

    # Open encoded file as a numpy array
    with open(encoded_file_path, mode='rb') as f:
        midi_tokens = pickle.load(f)

    return midi_tokens


def join_seeds(seeds: list):
    joined_seed = []
    current_time_step = []
    current_seed = 0
    seed = seeds[current_seed]
    while True:
        token = seed.pop(0)

        if token == EncodingConfig.time_note or token == EncodingConfig.end_note:
            # Flush current time step to joined seed
            joined_seed.extend(EncodingConfig.reorder_current(current_time_step))
            current_time_step = []

            if token == EncodingConfig.time_note:
                # Switch to the next seed
                current_seed += 1
            else:
                # This seed is finished, so we remove it from the list, which automatically switches to the next seed
                seeds.remove(seed)
                # Remove if we have exhausted all seeds
                if len(seeds) == 0:
                    break

            # If all seeds have been joined on this time step go to the next time step on the first seed
            if current_seed >= len(seeds):
                current_seed = 0
                joined_seed.append(EncodingConfig.time_note)

            # Otherwise get all tokens from the current time step of the next seed
            seed = seeds[current_seed]
        else:
            # Add the current token to the current time step
            current_time_step.append(token)

    return joined_seed

