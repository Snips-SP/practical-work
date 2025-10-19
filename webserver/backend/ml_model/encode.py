from collections import defaultdict

from backend.ml_model.helper import EncodingConfig
import pypianoroll
from tqdm import tqdm
import numpy as np
import pickle
from multiprocessing import Pool
import multiprocessing as mp
from typing import List
import glob
import gc
import os
import argparse

EncodingConfig.initialize()


def get_split_files(sample_keys, all_groups, endings):
    """
    For a given list of sample keys, collects the corresponding files
    that have the desired modulation endings.
    """
    final_files = []
    for key in sample_keys:
        # Get all files available for this sample
        all_mod_files = all_groups[key]
        # Filter them to keep only the ones we want
        filtered_files = [
            f for f in all_mod_files
            if any(f.endswith(ending) for ending in endings)
        ]
        final_files.extend(filtered_files)
    return final_files


def process_files_worker(files_to_process: List[str], results_queue: mp.Queue, sequence_length: int):
    """
    This is the producer function. It reads a subset of .tmp files,
    creates sequences, and puts them on the queue for the writer process.
    """
    # Number of sequences to batch before sending to the writer.
    # This reduces communication overhead.
    WORKER_BATCH_SIZE = 1000

    chunks = []
    current_chunk = []

    for file_path in files_to_process:
        if not os.path.exists(file_path):
            continue

        try:
            with open(file_path, mode='rb') as f:
                file_tokens = pickle.load(f)
        except (pickle.UnpicklingError, EOFError):
            # Skip corrupted or empty temp files
            continue

        for note in file_tokens:
            current_chunk.append(note)
            if len(current_chunk) >= sequence_length:
                chunks.append(np.array(current_chunk, dtype=np.uint16))
                current_chunk = []

                if len(chunks) >= WORKER_BATCH_SIZE:
                    results_queue.put(chunks)
                    chunks = []

    if chunks:
        results_queue.put(chunks)


def save_chunks_writer(output_path: str, results_queue: mp.Queue, total_sequences_approx: int, chunk_size: int,
                       sentinel: None):
    """
    This is the consumer function. It receives batches of sequences from workers,
    accumulates them, and saves them to .npz files when a full chunk is ready.
    """
    all_chunks = []
    num_files_saved = 0

    pbar = tqdm(total=total_sequences_approx, desc=f'Chunking and saving files for {os.path.basename(output_path)}')

    while True:
        worker_batch = results_queue.get()

        if worker_batch is sentinel:
            break

        all_chunks.extend(worker_batch)
        pbar.update(len(worker_batch))

        while len(all_chunks) >= chunk_size:
            data_to_save = np.array(all_chunks[:chunk_size])
            np.savez_compressed(
                os.path.join(output_path, f'{num_files_saved:03d}.npz'),
                data=data_to_save
            )
            num_files_saved += 1
            all_chunks = all_chunks[chunk_size:]

    # Save the final, smaller chunk that remains
    if all_chunks:
        print(f'\nSaving final chunk with {len(all_chunks)} sequences...')
        data_to_save = np.array(all_chunks)
        np.savez_compressed(
            os.path.join(output_path, f'{num_files_saved:03d}.npz'),
            data=data_to_save
        )
    pbar.close()


class Runner:
    def __init__(self, number_of_modulations, trc_avg, overwrite=True):
        self.number_of_modulations = number_of_modulations
        self.trc_avg = trc_avg
        self.overwrite = overwrite

    def _run(self, file_path):
        """Processes a single MIDI file and converts it to tokenized sequences with data augmentation.

            This method takes a MIDI pianoroll file and transforms it into tokenized sequences suitable
            for training generative models. The process includes track reordering (bass first), pitch
            modulation for data augmentation, temporal quantization to sixteenth notes, and encoding
            of notes into a compact token representation. Multiple augmented versions of the same piece
            are generated through pitch transposition.

            The encoding process:
            1. Loads the MIDI file and transposes tracks to prioritize bass
            2. Calculates per-track pitch averages for intelligent modulation
            3. Generates multiple transposed versions (0 to self.number_of_modulations semitones)
            4. Quantizes timing to sixteenth note grid
            5. Encodes each note as a single token combining track and pitch information
            6. Adds timing and end-of-sequence markers

            :param str file_path: Path to the input MIDI pianoroll (.npz) file to process.
            :returns: None. The method saves the tokenized sequences to a temporary pickle file
                     with '.tmp' extension added to the original file path.
            :rtype: None

            Note:
                The method applies intelligent pitch shifting that considers the global dataset
                pitch averages per track to avoid shifting notes too far out of their natural
                range. Drum tracks (track index 1) are not pitch-shifted during augmentation.
                All generated token sequences are saved to '{file_path}.{modulation}.tmp' for later chunking.
        """
        # If already have all the modulated tmp files and dont want to overwrite them we are done
        if len(list(glob.glob(os.path.join(os.path.dirname(file_path), '*.tmp')))) >= 12 and self.overwrite is False:
            return

        # Load it as a multitrack object
        m = pypianoroll.load(file_path)
        # Beat resolution is the number of steps a measure is divided into
        # Calculate how many time steps a quarter note takes to fill a measure
        step = m.resolution // 4
        # Convert it to a numpy array of shape (num_tracks, num_time_steps, num_pitches=128)
        pr = m.stack()
        # Transpose the array to have shape (num_time_steps, num_pitches, num_tracks)
        pr = np.transpose(pr, (1, 2, 0))
        # Change ordering to get the bass first and have consistent indexing
        pr = pr[:, :, EncodingConfig.trc_idx]
        # This changes the indexing to:
        # 0 = Bass (Program: 32)
        # 1 = Drums (Program: Drums (0))
        # 2 = Piano (Program: 0)
        # 3 = Guitar (Program: 24)
        # 4 = String (Program: 48)

        # Again get all time steps of notes being played
        p = np.where(pr != 0)
        # Calculate the average pitch of this song per track
        cur_avg_c = np.zeros((len(EncodingConfig.encoding_order), 2))
        for i in range(len(p[0])):
            # Track of the note at timestep i
            track = p[2][i]
            # The pitch of the note
            pitch = p[1][i]
            # Save them into our array
            cur_avg_c[track, 0] += pitch
            cur_avg_c[track, 1] += 1

        # Replace count of 0 to 1
        cur_avg_c[:, 1] = np.where(cur_avg_c[:, 1] == 0, 1, cur_avg_c[:, 1])
        # Perform the division safely
        cur_avg = np.where(cur_avg_c[:, 1] > 0, cur_avg_c[:, 0] / cur_avg_c[:, 1], 60)
        # Create a list with possible modulations
        possible_shifts = list(range(-5, 0)) + list(range(1, 7))
        modulation = [0] + possible_shifts[:self.number_of_modulations]

        for s in modulation:
            # Only create modulation if we really need it
            if os.path.exists(file_path + f'.{s}.tmp') and self.overwrite is False:
                continue

            seq = []
            # last, next buffer
            seq_buffer = [
                # Anchor tick, buffer
                (0, []),
                (step, [])
            ]

            for tick in range(pr.shape[0]):
                # Update our valid positions
                if tick % step == 0 and tick != 0:
                    # We have advanced to the next valid sixteenth note,
                    # thus we rotate out the last sequence and write it to seq
                    _, last_seq = seq_buffer[0]
                    last_seq.append(EncodingConfig.time_note)
                    seq.extend(EncodingConfig.reorder_current(last_seq))
                    # Remove the last sequence and add new next sequence
                    seq_buffer.pop(0)
                    seq_buffer.append((tick + step, []))

                # active has shape (N, 2) with columns [pitch, track]
                active = np.argwhere(pr[tick] != 0)
                for pitch, track in active:

                    # -------------------
                    # Handle drum events
                    # -------------------
                    if track == EncodingConfig.encoding_order.index('Drums'):
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
                    else:
                        # Only accept notes on grid 0,6,12,18 and only write to the current sequence
                        if tick % step != 0:
                            continue

                        # Decide if we apply a shift of one octave downwards when modulating
                        if self.trc_avg is None or cur_avg[track] + s < self.trc_avg[track] + 6:
                            shift = s
                        else:
                            shift = s - 12

                        # Adjust the pitch of the note with the shift
                        # If modulation is 0 (default) then we don't change the pitch at all
                        pitch = pitch + shift

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
                        note = track * EncodingConfig.note_size + pitch
                        # Write the note to the current sequence of this sixteenth note
                        seq_buffer[0][1].append(note)

            # Write all sequences to seq
            for _, sub_seq in seq_buffer:
                seq.extend(EncodingConfig.reorder_current(sub_seq))

            seq.append(EncodingConfig.end_note)

            # Store all the sequences, lists of tokens, as a pickle file
            with open(file_path + f'.{s}.tmp', mode='wb') as f:
                pickle.dump(seq, f)


    # Use a wrapper if encoding a file delivers an exception
    def safe_run(self, file):
        try:
            self._run(file)
        except Exception as e:
            print(f'Error processing {file}: {e}')


def encode_dataset(
        output,
        dataset: str = 'lpd_5',
        num_workers: int = 4,
        da: int = 5,
        sequence_length: int = 1024,
        chunk_size: int = 400_000,
        encode_from_tmp: bool = False
):
    """Encodes the Lakh Pianoroll Dataset into tokenized sequences for machine learning training.

        This function processes MIDI pianoroll data from the LPD-5 dataset, converting it into
        tokenized sequences suitable for training generative models. The function performs three
        main stages: (1) calculates average pitch values per track across the dataset for
        normalization, (2) encodes MIDI files into token sequences using multiprocessing, and
        (3) chunks the tokenized data into compressed NumPy archives for efficient loading during
        training. The process includes data augmentation through pitch modulation and track
        reordering to prioritize bass instruments.

        :param str output: Output directory path where the encoded dataset chunks will be saved.
        :param str dataset: Dataset path,
                            defaults to 'lpd_5'.
        :param int num_workers: Number of parallel processes to use for encoding,
                           defaults to 4.
        :param int da: Applies da random pitch modulations to each track (from -5 to +6 semitones),
                      defaults to 5.
        :param int sequence_length: Length of individual token sequences to generate,
                                   defaults to 1024.
        :param int chunk_size: Number of sequences to include in each output chunk file,
                              defaults to 400,000.
        :param bool encode_from_tmp: If True, skips encoding and creates chunks from existing
                                    temporary files. If False, performs full encoding pipeline,
                                    defaults to False.
        :raises SystemExit: If the specified dataset is not found or is invalid.
        :raises AssertionError: If dataset is not 'lpd_5' (when called from command line).
        :returns: None. The function saves encoded chunks as compressed .npz files and
                 track ordering information to the output directory.
        :rtype: None
    """
    if os.path.isdir(dataset):
        if os.path.basename(dataset) == 'lpd_5':
            dataset_path = os.path.join(dataset, 'lpd_5_cleansed/*/*/*/*/*.npz')
        else:
            raise AssertionError(f'Invalid dataset: {dataset}')
    else:
        raise FileNotFoundError(f'Dataset not found: {dataset}')

    # Get all midi files from dataset structure
    pianoroll_files = list(glob.glob(dataset_path))

    if encode_from_tmp:
        print('Skipping getting an average pitch value over the entire dataset. (1/3)')
        print('Skipping encode midi files as token encoding. (2/3)')
    else:
        if os.path.exists(os.path.join(dataset, f'trc_avg.pkl')):
            print('Get average pitch value per track over entire dataset from file. (1/3)')
            # Load the pickle file
            with open(os.path.join(dataset, 'trc_avg.pkl'), mode='rb') as f:
                trc_avg = pickle.load(f)
        else:
            # Get an average of each pitch value per track over all datapoints
            trc_avg_c = np.zeros((len(EncodingConfig.midi_tracks), 2))
            for file_path in tqdm(pianoroll_files, desc='Get average pitch value per track over entire dataset. (1/3)'):
                # Load it as a multitrack object
                m = pypianoroll.load(file_path)
                # Convert it to a numpy array of shape (num_time_steps, num_pitches=128, num_tracks)
                pr = m.stack()
                # Transpose the array to have shape (num_time_steps, num_pitches, num_tracks)
                pr = np.transpose(pr, (1, 2, 0))
                # Change ordering to get the bass first and have consistent indexing
                pr = pr[:, :, EncodingConfig.trc_idx]
                # Get all time steps of notes being played
                p = np.where(pr != 0)
                for i in range(len(p[0])):
                    # Track of the note at timestep i
                    track = p[2][i]
                    # The pitch of the note
                    pitch = p[1][i]
                    # Save them into our array
                    trc_avg_c[track, 0] += pitch
                    trc_avg_c[track, 1] += 1
                del m, pr, p

            # Replace 0 in cur_avg_c[:, 1] with 1
            # In order to avoid runtime errors
            trc_avg_c[:, 1] = np.where(trc_avg_c[:, 1] == 0, 1, trc_avg_c[:, 1])

            # Calculate the average note value per track
            trc_avg = np.where(trc_avg_c[:, 1] > 0, trc_avg_c[:, 0] / trc_avg_c[:, 1], 60)

            # Store all the trc avg, as a pickle file since it takes 40min to calculate
            with open(os.path.join(dataset, f'trc_avg.pkl'), mode='wb') as f:
                pickle.dump(trc_avg, f)

            del trc_avg_c
            gc.collect()

        # Create a runner class to transfer read-only variables to the processes
        runner = Runner(da, trc_avg, not encode_from_tmp)

        # Encode the midi files as token encodings
        with tqdm(total=len(pianoroll_files), desc='Encode midi files as token encoding. (2/3)') as t:
            with Pool(num_workers) as p:
                for _ in p.imap_unordered(runner.safe_run, pianoroll_files):
                    t.update(1)

    # Make folder structure
    output_folder = f'{output}_da_{da}'
    train_folder = os.path.join(f'{output}_da_{da}', 'train')
    valid_folder = os.path.join(f'{output}_da_{da}', 'valid')
    test_folder = os.path.join(f'{output}_da_{da}', 'test')
    for folder in [output_folder, train_folder, valid_folder, test_folder]:
        if not os.path.exists(folder):
            os.makedirs(folder)

    print('Finding and grouping files by sample...')
    all_tmp_files = glob.glob(os.path.join(dataset, 'lpd_5_cleansed/*/*/*/*/*.npz.*.tmp'))

    # A dictionary to hold lists of files
    sample_groups = defaultdict(list)
    for f in all_tmp_files:
        # Extract the base name to identify the unique sample
        base_name = f.split('.npz.')[0] + '.npz'
        sample_groups[base_name].append(f)

    print(f'Found {len(sample_groups)} unique samples.')

    # Split the list of unique samples into train, valid, and test
    unique_samples = list(sample_groups.keys())
    np.random.shuffle(unique_samples)  # Shuffle for a random split

    # Calculate split indices for an 80/10/10 split
    n_samples = len(unique_samples)
    train_end = int(0.8 * n_samples)
    valid_end = int(0.9 * n_samples)

    # Assign the unique sample keys to each split
    train_sample_keys = unique_samples[:train_end]
    valid_sample_keys = unique_samples[train_end:valid_end]
    test_sample_keys = unique_samples[valid_end:]

    print(f'Splitting samples: {len(train_sample_keys)} train, {len(valid_sample_keys)} validation, {len(test_sample_keys)} test.')

    # Build the final file lists with the desired modulations
    # Define which modulations to use based on the da parameter
    possible_shifts = list(range(-5, 0)) + list(range(1, 7))
    np.random.shuffle(possible_shifts)

    shifts_to_use = [0] + possible_shifts[:da]
    desired_endings = {f'.npz.{s}.tmp' for s in shifts_to_use}
    print(f'Including original (0) and {da} other modulations: {shifts_to_use}')

    # Create the final lists of file paths for each split
    train_files = get_split_files(train_sample_keys, sample_groups, desired_endings)
    valid_files = get_split_files(valid_sample_keys, sample_groups, desired_endings)
    test_files = get_split_files(test_sample_keys, sample_groups, desired_endings)

    print('\n--- Final Split ---')
    print(f'Training files:   {len(train_files)}')
    print(f'Validation files: {len(valid_files)}')
    print(f'Test files:       {len(test_files)}\n')

    # Chunk all those files into their folders
    print(f'Starting chunking stage with {num_workers} worker processes. (3/3)')

    for output_folder, files in zip([train_folder, valid_folder, test_folder], [train_files, valid_files, test_files]):
        results_queue = mp.Queue()
        # A signal to tell the writer when all workers are done
        sentile = None

        # Split the list of files among the worker processes
        file_chunks = np.array_split(files, num_workers)

        # Create and start the single writer process
        writer_process = mp.Process(
            target=save_chunks_writer,  # An approximation of the number of sequences
            args=(output_folder, results_queue, 300_000 * (da + 1), chunk_size, sentile)
        )
        writer_process.start()

        # Create and start the worker processes
        worker_processes = []
        for i in range(num_workers):
            p = mp.Process(
                target=process_files_worker,
                args=(file_chunks[i].tolist(), results_queue, sequence_length)
            )
            worker_processes.append(p)
            p.start()

        # Wait for all workers to finish their file processing
        for p in worker_processes:
            p.join()

        # Once all workers are done, send the sentinel to tell the writer to stop
        results_queue.put(sentile)

        # Wait for the writer process to finish saving everything
        writer_process.join()

    # Save the ordering of the tracks
    with open(os.path.join(output_folder, 'tracks.trc'), 'w') as f:
        f.write(','.join([EncodingConfig.encoding_order[t] for t in EncodingConfig.trc_idx]))

    print('\nDataset encoding finished successfully.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', help='source dir', default='lpd_5')
    parser.add_argument('--process', help='num process', type=int, default=4)
    parser.add_argument('--da', help='modulation for data augmentation (0-11)', type=int, default=5)
    parser.add_argument('--output', help='output name', required=True)
    parser.add_argument('--sequence_length', help='Length of the individual sequences', default=1024)
    parser.add_argument('--chunk_size', help='Amount of sequences in one chunk', default=400_000)
    parser.add_argument('--encode_from_tmp',
                        help='Grap the encodings and chunk them together from already encoded tmp files', type=bool,
                        default=False)
    args = parser.parse_args()

    encode_dataset(args.output,
                   args.dataset,
                   args.process,
                   args.da,
                   args.sequence_length,
                   args.chunk_size,
                   args.encode_from_tmp)
