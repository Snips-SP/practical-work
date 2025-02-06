import threading
import os
import glob
import numpy as np
from torch.utils.data import Dataset


class CustomEncodingVocabulary:
    tokens = []
    padding_token = None

    @classmethod
    def initialize(cls):
        if not cls.tokens:  # Prevent re-initialization
            # Drums: [0 * 84, 0 * 84 + 83] = [0, 83]
            # Piano: [1 * 84, 1 * 84 + 83] = [84, 167]
            # Guitar: [2 * 84, 2 * 84 + 83] = [168, 251]
            # Bass: [3 * 84, 3 * 84 + 83] = [252, 335]
            # Strings: [4 * 84, 4 * 84 + 83] = [336, 419]
            cls.tokens.extend(range(0, 83))  # Drum tokens [0, 83]
            cls.tokens.extend(range(83, 167))  # Piano tokens [84, 167]
            cls.tokens.extend(range(167, 251))  # Guitar tokens [168, 251]
            cls.tokens.extend(range(251, 335))  # Bass tokens [252, 335]
            cls.tokens.extend(range(335, 419))  # Strings tokens [336, 419]
            cls.tokens.append(cls.tokens[-1] + 1) ### TODO: What is this token, why is it used, why do I even exist (420)
            cls.tokens.append(cls.tokens[-1] + 1)  # Add the token which represents a pause in the music (421)
            cls.padding_token = cls.tokens[-1] + 1  # Add the token which represents the end of the sequence (422)


class GPT2Dataset(Dataset):
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

        # Validate dataset path
        if not os.path.isdir(self.dataset_path):
            raise NotADirectoryError(f'The specified dataset path is not a directory: {self.dataset_path}')

        # Load chunk file paths and determine total length
        for file_number, file_path in enumerate(sorted(glob.glob(os.path.join(dataset_path, '*.npz')))):
            # Extract file number and map it to the file path
            file_name = os.path.basename(file_path)  # e.g., '01' -> 1
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
        self.current_file_index += 1
        if self.current_file_index + 1 < len(self.chunk_files):
            self._preload_next(self.current_file_index + 1)

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
                mask = np.ones(len(tokens), dtype=int)

                return tokens, mask

            cumulative_length += file_length

        # Raise an error if the index is out of range
        raise IndexError('Index out of range.')


class GPT2RAMDataset(Dataset):
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        self.max_length = None
        self.data = []
        self.length = 0

        # Validate dataset path
        if not os.path.isdir(self.dataset_path):
            raise NotADirectoryError(f'The specified dataset path is not a directory: {self.dataset_path}')

        for file_path in glob.glob(os.path.join(dataset_path, '*.npz')):
            # Load all into ram
            chunk = np.load(file_path)
            self.data.extend([chunk[file] for file in chunk.files])
            chunk.close()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        tokens = self.data[idx]
        # Perform padding if needed
        if len(tokens) < self.max_length:
            # Create a padded array of size max_length filled with padding token (e.g., 0)
            padded_tokens = np.zeros(self.max_length, dtype=int)
            padded_tokens[:len(tokens)] = tokens
            mask = np.zeros(self.max_length, dtype=int)
            # Set mask to 1 for valid tokens
            mask[:len(tokens)] = 1
        else:
            # No padding needed, just use the original tokens
            padded_tokens = tokens[:self.max_length]
            # Entire mask is 1
            mask = np.ones(self.max_length, dtype=int)
        return padded_tokens, mask
