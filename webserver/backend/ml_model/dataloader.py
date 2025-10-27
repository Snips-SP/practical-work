import threading
import glob
import os
import pypianoroll
import torch
import numpy as np
import random
from torch.utils.data import Dataset


class MidiDataset(Dataset):
    """A PyTorch Dataset for loading chunked midi training data with efficient memory management.

        This dataset class is designed to handle large-scale tokenized music data that has been
        pre-processed and split into multiple compressed chunk files. It implements an efficient
        loading strategy that keeps only one chunk in memory at a time while preloading the next
        chunk in a background thread to minimize I/O bottlenecks during training. The dataset
        automatically cycles through all chunks and handles the mapping from global indices to
        chunk-specific indices.

        The dataset expects the data directory to contain numbered .npz files (e.g., 000.npz,
        001.npz, etc.) where each file contains a compressed NumPy array of tokenized sequences.

        :param str dataset_path: Path to the directory containing the chunked .npz files.
        :raises NotADirectoryError: If the specified dataset_path is not a valid directory.
        :raises IndexError: If an invalid index is accessed via __getitem__.

        Attributes:
            dataset_path (str): Path to the dataset directory.
            chunk_files (dict): Mapping from file numbers to file paths.
            file_lengths (dict): Mapping from file numbers to the number of sequences in each file.
            length (int): Total number of sequences across all chunks.
            current_file_index (int): Index of the currently loaded chunk file.
            current_data (np.ndarray): Currently loaded chunk data.
            next_data (np.ndarray): Next chunk data being preloaded.
            lock (threading.Lock): Thread lock for synchronizing data access.
            preload_event (threading.Event): Event for coordinating preloading operations.
            restart (bool): Flag indicating whether to restart from the first chunk.
    """

    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        self.chunk_files = {}
        self.file_lengths = {}
        self.length = 0
        self.current_file_index = 0
        self.current_data = []
        self.next_data = []
        self.lock = threading.Lock()
        self.preload_event = threading.Event()
        self.restart = False

        # Validate dataset path
        if not os.path.isdir(self.dataset_path):
            raise NotADirectoryError(f'The specified dataset path is not a directory: {self.dataset_path}')

        # Load chunk file paths and determine total length
        for file_number, file_path in enumerate(sorted(glob.glob(os.path.join(dataset_path, '*.npz')))):
            # Extract file number and map it to the file path
            self.chunk_files[file_number] = file_path

            # Get chunk length
            chunk = np.load(file_path)
            self.file_lengths[file_number] = len(chunk[chunk.files[0]])
            self.length += self.file_lengths[file_number]
            chunk.close()

        # Preload the first two files
        self.current_data = self._load_file(0)
        if len(self.chunk_files) > 1:
            self._preload_next(1)

    def _load_file(self, file_index):
        """Load a chunk file by index."""
        file_path = self.chunk_files[file_index]
        chunk = np.load(file_path)
        data = chunk[chunk.files[0]]
        chunk.close()
        return data

    def _preload_next(self, idx):
        """Preload the next file in a separate thread."""

        def _worker():
            with self.lock:
                # Load the file in another thread
                self.next_data = self._load_file(idx)

            # Signal that the preload is complete
            self.preload_event.set()

        # Reset the event and start the preload worker thread
        self.preload_event.clear()
        threading.Thread(target=_worker).start()

    def _switch_to_next(self):
        """Switch to the preloaded next data."""
        # Wait for the preload to complete
        self.preload_event.wait()

        self.current_data = self.next_data
        self.next_data = []
        if self.restart:
            self.current_file_index = 0
            self.restart = False
        else:
            self.current_file_index += 1
        if self.current_file_index + 1 < len(self.chunk_files):
            self._preload_next(self.current_file_index + 1)
        else:
            # Preload the first file again to loop all the samples
            self._preload_next(0)
            self.restart = True

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # Calculate which chunk file this idx belongs to
        cumulative_length = 0
        for file_number, file_length in self.file_lengths.items():
            if cumulative_length + file_length > idx:
                # Determine the index within the current file
                file_idx = idx - cumulative_length
                # Switch to the correct file if necessary
                if file_number != self.current_file_index:
                    self._switch_to_next()

                # Return the requested data
                tokens = self.current_data[file_idx]

                # Entire mask is 1
                mask = np.ones(len(tokens), dtype=tokens.dtype)

                return tokens, mask

            cumulative_length += file_length

        # Raise an error if the index is out of range
        raise IndexError('Index out of range.')


class MidiRAMDataset(Dataset):
    """A PyTorch Dataset that loads all chunked midi file training data into RAM for fast access.

        This dataset class loads the entire tokenized music dataset into system memory during
        initialization, providing maximum training speed at the cost of high memory usage.
        It is suitable for smaller datasets or systems with abundant RAM where I/O latency
        needs to be completely eliminated during training. All compressed chunk files are
        loaded and concatenated into a single NumPy array for direct indexing.

        The dataset expects the data directory to contain .npz files where each file contains
        a compressed NumPy array of tokenized sequences. All chunks are loaded sequentially
        and concatenated along the first axis.

        :param str dataset_path: Path to the directory containing the chunked .npz files.
        :raises NotADirectoryError: If the specified dataset_path is not a valid directory.

        Attributes:
            dataset_path (str): Path to the dataset directory.
            data (np.ndarray): Concatenated array containing all tokenized sequences.
    """
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        self.data = []
        self.length = 0

        # Validate dataset path
        if not os.path.isdir(self.dataset_path):
            raise NotADirectoryError(f'The specified dataset path is not a directory: {self.dataset_path}')

        for file_path in glob.glob(os.path.join(dataset_path, '*.npz')):
            # Load all into ram
            chunk = np.load(file_path)
            self.data.append([chunk[file] for file in chunk.files][0])
            chunk.close()
        self.data = np.concatenate(self.data, axis=0)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        tokens = self.data[idx]
        # Entire mask is 1
        mask = np.ones(len(tokens), dtype=int)
        return tokens, mask


class OnTheFlyMidiDataset(Dataset):
    """
    On-the-fly MIDI dataloader
    """

    def __init__(self, datafiles: list, encodingConfig, n_modulations: int = 0, chunk_size: int = 1024):
        self.encodingConfig = encodingConfig
        self.filepaths = datafiles
        self._effective_chunk_size = chunk_size - 2
        self.chunk_size = chunk_size

        # All possible pitch shifts, excluding 0.
        possible_shifts = list(range(-5, 0)) + list(range(1, 7))

        # Raise an exception if n_modulations is too high
        if n_modulations > len(possible_shifts):
            raise ValueError(f'Requested n_modulations ({n_modulations}) is larger than the number of available shifts ({len(possible_shifts)}).')

        selected_shifts = random.sample(possible_shifts, n_modulations)
        self.augmentation_range = [0] + selected_shifts

    def __len__(self):
        return len(self.filepaths)


    def load_midi_to_array(self, file_path: str):
        archive = np.load(file_path)

        info = archive['info']
        beat_resolution = info['beat_resolution']

        array = []

        for key in info.keys():
            if key != 'name' and key != 'beat_resolution':
                track_info = info[key]
                track_index = int(key)

                ### TODO: Finish this loading using the encoding config
                # We need to get the right arrays from the archive, probably 'pianoroll_{index}_csc_data'
                # Then put them into the right order which is defined in encoding config
                # Then compare them with m = pypianoroll.load(file_path); pr = m.stack()




# b'{"name": "b0493abb18a9b7da99cf3ac222b2c41d", "1": {"is_drum": false, "program": 0, "name": "Piano"}, "0": {"is_drum": true, "program": 0, "name": "Drums"}, "3": {"is_drum": false, "program": 32, "name": "Bass"}, "2": {"is_drum": false, "program": 24, "name": "Guitar"}, "4": {"is_drum": false, "program": 48, "name": "Strings"}, "beat_resolution": 24}'

    def encode_midi(self, file_path: str, aug_value):
        # Load it as a multitrack object
        m = pypianoroll.load(file_path)
        ### TODO: Figure out if we can just load the file as an numpy array to remove overhead from the pypianoroll library

        # https://hermandong.com/pypianoroll/doc.html#pypianoroll.read
        # https://hermandong.com/pypianoroll/_modules/pypianoroll/inputs.html#load
        # https://hermandong.com/pypianoroll/_modules/pypianoroll/inputs.html#read



        # Beat resolution is the number of steps a measure is divided into
        # Calculate how many time steps a quarter note takes to fill a measure
        step = m.resolution // 4
        # Convert it to a numpy array of shape (num_tracks, num_time_steps, num_pitches=128)
        pr = m.stack()
        # Transpose the array to have shape (num_time_steps, num_pitches, num_tracks)
        pr = np.transpose(pr, (1, 2, 0))
        # Change ordering to get consistent indexing
        pr = pr[:, :, self.encodingConfig.trc_idx]

        # Init token list with bos token
        tokens = [self.encodingConfig.begin_note]

        # Chose a random starting tick in the sequence
        tick = random.randint(0, pr.shape[0])
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
                    last_seq.append(self.encodingConfig.time_note)
                    tokens.extend(self.encodingConfig.reorder_current(last_seq))
                    # Remove the last sequence and add new next sequence
                    seq_buffer.pop(0)
                    seq_buffer.append((tick + step, []))

            # active has shape (N, 2) with columns [pitch, track]
            active = np.argwhere(pr[tick] != 0)
            for pitch, track in active:
                # -------------------
                # Handle drum events
                # -------------------
                track = self.encodingConfig.encoding_order[track]
                if track == 'Drums':
                    if pitch not in self.encodingConfig.drum_pitches:
                        continue

                    # Determine to which sixteenth note this note snaps to (either next or last)
                    offset = tick - seq_buffer[0][0]
                    if offset <= 3:
                        buffer = 0
                    else:
                        offset = tick - seq_buffer[1][0]
                        buffer = 1

                    # Map pitch
                    drum_token = self.encodingConfig.drum_pitch_to_token[pitch]

                    # Add base drum token
                    seq_buffer[buffer][1].append(drum_token)

                    # Add timing offset if not 0
                    if offset != 0:
                        offset_token = self.encodingConfig.microtiming_delta_to_token[offset]
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
                    pitch -= self.encodingConfig.note_offset
                    if pitch < 0:
                        pitch = 0
                    if pitch > self.encodingConfig.note_size - 1:
                        pitch = self.encodingConfig.note_size - 1

                    # Encode token
                    note = (self.encodingConfig.instrument_intervals[track][0] - 1) + pitch

                    if note >= self.encodingConfig.vocab_size:
                        raise ValueError('Note out of vocabulary.')

                    # Write the note to the current sequence of this sixteenth note
                    seq_buffer[0][1].append(note)

            # Check if we have enough tokens otherwise just keep going
            if self._effective_chunk_size <= len(seq_buffer[0][1]) + len(seq_buffer[1][1]) + len(tokens):
                break
            # Increase tick
            tick += 1

        # Write all sequences to seq
        for _, sub_seq in seq_buffer:
            tokens.extend(self.encodingConfig.reorder_current(sub_seq))

        # Cut the tokens down if we processed too many. Usually its only 3-5 tokens more
        if len(tokens) > self.chunk_size - 1:
            tokens = tokens[:self.chunk_size - 1]

        # Append eos token
        tokens.append(self.encodingConfig.end_note)

        tokens = np.array(tokens)
        seq_len = len(tokens)
        if seq_len < self.chunk_size:
            # We need to pad the sequence
            pad_len = self.chunk_size - seq_len

            # Create padding
            padding_arr = np.full((pad_len,), self.encodingConfig.padding_token, dtype=np.int64)

            # Create the final token sequence
            tokens = np.concatenate([tokens, padding_arr])

            # Create mask: 0 for real tokens, 1 for padding
            mask = np.concatenate([np.zeros(seq_len, dtype=np.int64), np.ones(pad_len, dtype=np.int64)])
        else:

            # Otherwise the mask is just full of zeros
            mask = np.zeros_like(tokens, dtype=np.int64)
        return tokens, mask


    def __getitem__(self, idx):
        midi_path = self.filepaths[idx]
        aug_value = random.choice(self.augmentation_range)
        # Get tokens and mask from midi file
        tokens, mask = self.encode_midi(midi_path, aug_value)

        # Convert to torch tensors
        return torch.from_numpy(tokens).to(torch.long), torch.from_numpy(mask).to(torch.long)

