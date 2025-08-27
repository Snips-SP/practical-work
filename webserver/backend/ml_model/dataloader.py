import threading
import os
import glob
import numpy as np
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
