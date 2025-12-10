from torch.utils.data import Dataset
from backend.ml_model.encoding import EncodingConfig
import pypianoroll
import torch
import numpy as np
import random

class OnTheFlyMidiDataset(Dataset):
    """
    On-the-fly MIDI dataloader
    """

    def __init__(self, datafiles: list, n_modulations: int = 0, chunk_size: int = 1024, warmup_steps: int = 128):
        self.filepaths = datafiles
        self.chunk_size = chunk_size
        self.warmup_steps = warmup_steps

        # All possible pitch shifts, excluding 0.
        possible_shifts = list(range(-5, 0)) + list(range(1, 7))

        # Raise an exception if n_modulations is too high
        if n_modulations > len(possible_shifts):
            raise ValueError(f'Requested n_modulations ({n_modulations}) is larger than the number of available shifts ({len(possible_shifts)}).')

        selected_shifts = random.sample(possible_shifts, n_modulations)
        self.augmentation_range = [0] + selected_shifts

    def __len__(self):
        return len(self.filepaths)

    @staticmethod
    def encode_midi(chunk_size:int, file_path: str, aug_value:int):
        # Load the npz as midi file
        m = pypianoroll.load(file_path)

        # Beat resolution is the number of steps a measure is divided into
        # Calculate how many time steps a quarter note takes to fill a measure
        step = m.resolution // 4

        # Convert it to a numpy array of shape (num_tracks, num_time_steps, num_pitches=128)
        pr = m.stack()
        # (num_time_steps, num_pitches=128, num_tracks)
        pr = np.transpose(pr, (1, 2, 0))
        # Change ordering to get consistent indexing
        pr = pr[:, :, EncodingConfig.trc_idx]

        # Init token list
        tokens = []

        # Chose a random starting tick in the sequence
        tick = random.randint(0, pr.shape[0])
        if tick == 0:
            tokens.append(EncodingConfig.begin_note)

        # Snap to the last step
        tick = tick - (tick % step)

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
                track = EncodingConfig.encoding_order[track]
                if track == 'Drums':
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

                # ------------------------
                # Handle pitched instruments
                # ------------------------
                # Only accept notes on grid 0,6,12,18 and only write to the current sequence
                elif tick % step == 0:
                    # Adjust the pitch of the note with the shift
                    # If modulation is 0 (default) then we don't change the pitch at all
                    pitch = pitch + aug_value

                    # Apply shift
                    if pitch < 0:
                        pitch += 12
                    if pitch > 127:
                        pitch -= 12

                    # Offset and clip
                    pitch -= EncodingConfig.note_offset
                    if pitch < 0:
                        pitch = 0
                    if pitch > EncodingConfig.note_size - 1:
                        pitch = EncodingConfig.note_size - 1

                    # Encode token
                    note = (EncodingConfig.instrument_intervals[track][0] - 1) + pitch

                    if note >= EncodingConfig.vocab_size:
                        raise ValueError('Note out of vocabulary.')

                    # Write the note to the current sequence of this sixteenth note
                    seq_buffer[0][1].append(note)

            # Check if we have enough tokens otherwise just keep going
            if chunk_size - 2 <= len(seq_buffer[0][1]) + len(seq_buffer[1][1]) + len(tokens):
                break
            # Increase tick
            tick += 1

        # Write all sequences to seq
        for _, sub_seq in seq_buffer:
            tokens.extend(EncodingConfig.reorder_current(sub_seq))

        if len(tokens) > chunk_size:
            # We truncate to the exact chunk size.
            tokens = tokens[:chunk_size]
        else:
            # Check if we have room for the EOS
            if len(tokens) < chunk_size:
                tokens.append(EncodingConfig.end_note)

        tokens = np.array(tokens)
        seq_len = len(tokens)
        if seq_len < chunk_size:
            # We need to pad the sequence
            pad_len = chunk_size - seq_len

            # Create padding
            padding_arr = np.full((pad_len,), EncodingConfig.padding_token, dtype=np.int64)

            # Create the final token sequence
            tokens = np.concatenate([tokens, padding_arr])

            # 1 for real tokens, 0 for padding
            mask = np.concatenate([np.ones(seq_len, dtype=np.int64), np.zeros(pad_len, dtype=np.int64)])
        else:
            # Otherwise the mask is just full of ones
            mask = np.ones_like(tokens, dtype=np.int64)

        return tokens, mask


    def __getitem__(self, idx):
        midi_path = self.filepaths[idx]
        aug_value = random.choice(self.augmentation_range)
        # Get tokens and mask from midi file
        tokens, mask = self.encode_midi(self.chunk_size, midi_path, aug_value)

        # Create a copy of tokens to serve as the target labels
        labels = tokens.copy()

        # Anywhere the attention mask is 0, the label should be -100 (ignore)
        labels[mask == 0] = -100

        # Apply the Warm-up Mask by overwriting the first 64 entries with -100.
        labels[:self.warmup_steps] = -100

        # Convert to torch tensors
        return torch.from_numpy(tokens).to(torch.long), torch.from_numpy(mask).to(torch.long), torch.from_numpy(labels).to(torch.long)

