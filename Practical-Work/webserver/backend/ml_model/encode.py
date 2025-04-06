from .train import EncodingConfig
import pypianoroll
import glob
import pickle
import gc
import os
from tqdm import tqdm
import numpy as np
from multiprocessing import Pool
import argparse
EncodingConfig.initialize()


class Runner:
    def __init__(self, trc_idx, number_of_modulations, trc_avg):
        self.trc_idx = trc_idx
        self.number_of_modulations = number_of_modulations
        self.trc_avg = trc_avg

    def _run(self, file_path):
        seq = []
        # Load it as a multitrack object
        m = pypianoroll.load(file_path)
        # Beat resolution is the number of steps a measure is divided into
        # Calculate how many time steps a quarter note takes to fill a measure
        step = m.resolution // 4
        # Convert it to a numpy array of shape (num_time_steps, num_pitches=128, num_tracks)
        pr = m.stack()
        # Transpose the array to have shape (num_time_steps, num_pitches, num_tracks)
        pr = np.transpose(pr, (1, 2, 0))
        # Change ordering to get the bass first and have consistent indexing
        pr = pr[:, :, self.trc_idx]
        # Again get all time steps of notes being played
        p = np.where(pr != 0)
        # Calculate the average pitch of this song per track
        cur_avg_c = np.zeros((len(EncodingConfig.tracks), 2))
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
        # Create list with [0, random permutation of the numbers 1 - 12]
        modulation = [0] + (np.random.permutation(11) + 1).tolist()

        current_seq = []
        # Do data augmentation according to specified parameter 'da' in range [0-11]
        # All sequences are appended to the same list
        for s in modulation[:self.number_of_modulations + 1]:
            pos = 0
            # Create an encoding sequence for each modulation
            for i in range(len(p[0])):
                # Ignore smaller notes then sixteenth notes or notes that are not on the gird
                if p[0][i] % step != 0:
                    continue
                # Only if we have advanced in the position will the current cur_seq be written down
                # More note encodings can be placed on the same sixteenth note since they can be
                # from different instruments or an instrument can play more notes at once
                if pos < p[0][i]:
                    # From last position to next note occurrence write down the notes which are played
                    # by the instruments
                    for _ in range(pos, p[0][i], step):
                        seq.extend(self._reorder_current(current_seq))
                        seq.append(EncodingConfig.time_note)
                        current_seq = []
                # Set current position to the last note occurrence
                pos = p[0][i]
                # Get current pitch
                pitch = p[1][i]
                # Get current track
                track = p[2][i]

                shift = 0
                # Do this for every track expect drums
                if track != 1:
                    # Decide if we apply a shift of one octave downwards when modulating if
                    # The hypothetical pitch average of the current track after applying modulation s
                    # <
                    # A threshold based on the datasets global pitch average for this track plus a buffer of 6
                    if cur_avg[track] + s < self.trc_avg[track] + 6:
                        shift = s
                    else:
                        shift = s - 12

                # Adjust the pitch of the note with the shift
                # If modulation is 0 (default) then we don't change the pitch at all
                pitch = pitch + shift

                # Bring the pitch back up one octave if for whatever reason we shifted
                # the note to far down
                if pitch < 0:
                    pitch += 12
                # Bring the pitch down if we exceed 127. The highest value of midi files (0x7F)
                if pitch > 127:
                    pitch -= 12
                # Apply our offset
                pitch -= EncodingConfig.note_offset

                # Do some checks again to ensure that the note is within our interval of wanted notes
                # [0, note_size (84)]
                # Since notes above 84 are not used, and we want to keep the dictionary concise
                if pitch < 0:
                    pitch = 0
                if pitch > EncodingConfig.note_size:
                    pitch = EncodingConfig.note_size - 1

                # Finally calculate the number which represents the note track and pitch all together
                note = track * EncodingConfig.note_size + pitch
                # Append it to our current sequence
                current_seq.append(note)

            # Fill in the last note which is played
            seq.extend(self._reorder_current(current_seq))
            # And a last time_note with the end_note as well
            seq.append(EncodingConfig.time_note)
            seq.append(EncodingConfig.end_note)
            current_seq = []
        # Store all the sequences, lists of tokens, as a pickle file
        with open(file_path + '.tmp', mode='wb') as f:
            pickle.dump(seq, f)

    def _reorder_current(self, cur_seq):
        # Reorder the sequence so that piano is the last instrument
        # Drums: [0 * 84, 0 * 84 + 83] = [0, 83]
        # Piano: [1 * 84, 1 * 84 + 83] = [84, 167]
        # Guitar: [2 * 84, 2 * 84 + 83] = [168, 251]
        # Bass: [3 * 84, 3 * 84 + 83] = [252, 335]
        # Strings: [4 * 84, 4 * 84 + 83] = [336, 419]

        ### TODO: Something seems wrong here, '# Bass, Piano, etc..., Drums' seems to imply a different ordering
        cur = []
        for c in sorted(cur_seq):
            # Checks if a note c is not in the range [84, 168)
            # i.e. if the instrument is one of the following
            # Bass, Drums, Guitar, Strings
            if not (c >= EncodingConfig.note_size and c < EncodingConfig.note_size * 2):
                cur.append(c)
        for c in sorted(cur_seq):
            # Checks if a note c is in the range [84, 168)
            # i.e. if the instrument is piano
            if (c >= EncodingConfig.note_size and c < EncodingConfig.note_size * 2):
                cur.append(c)

        return cur  # Bass, Piano, etc..., Drums

    # Use a wrapper if encoding a file delivers an exception
    def safe_run(self, file):
        try:
            self._run(file)
        except Exception as e:
            print(f'Error processing {file}: {e}')


def encode_dataset(output,
                   dataset: str ='lpd_5',
                   process: int = 4,
                   da: int = 5,
                   sequence_length: int = 1024,
                   chunk_size: int = 400_000,
                   encode_from_tmp: bool = False):
    if dataset == 'lpd_5':
        tracks = ['Drums', 'Piano', 'Guitar', 'Bass', 'Strings']
        if os.path.isdir('lpd_5/lpd_5_full'):
            dataset_path = 'lpd_5/lpd_5_full/*/*.npz'
        elif os.path.isdir('lpd_5/lpd_5_cleansed'):
            dataset_path = 'lpd_5/lpd_5_cleansed/*/*/*/*/*.npz'
        else:
            print('invalid dataset')
            exit()
    else:
        print('invalid dataset')
        exit()

    # Check if the directory exists
    if not os.path.exists(output):
        # Create the directory
        os.makedirs(output)

    trc_len = len(tracks)
    # Push Bass index in front of leaving others as is
    # [3, 0, 1, 2, 4]
    # [Bass, Drums, Piano, Guitar, Strings]
    trc_idx = sorted(list(range(trc_len)), key=lambda x: 0 if tracks[x] == 'Bass' else 1)

    if encode_from_tmp is False:
        pianoroll_files = []
        if not os.path.exists(os.path.join('lpd_5', f'trc_avg.pkl')):
            # Get an average of each pitch value per track over all datapoints
            trc_avg_c = np.zeros((len(tracks), 2))
            for file_path in tqdm(glob.glob(dataset_path),
                                  desc='Get average pitch value per track over entire dataset. (1/3)'):
                # Save npz path into list
                pianoroll_files.append(file_path)
                # Load it as a multitrack object
                m = pypianoroll.load(file_path)
                # Convert it to a numpy array of shape (num_time_steps, num_pitches=128, num_tracks)
                pr = m.stack()
                # Transpose the array to have shape (num_time_steps, num_pitches, num_tracks)
                pr = np.transpose(pr, (1, 2, 0))
                # Change ordering to get the bass first and have consistent indexing
                pr = pr[:, :, trc_idx]
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
            with open(os.path.join('lpd_5', f'trc_avg.pkl'), mode='wb') as f:
                pickle.dump(trc_avg, f)

            del trc_avg_c
            gc.collect()
        else:
            print('Get average pitch value per track over entire dataset from file. (1/3)')
            # Load the pickle file
            with open(os.path.join('lpd_5', 'trc_avg.pkl'), mode='rb') as f:
                trc_avg = pickle.load(f)
            # Search for the file paths with glob
            pianoroll_files = list(glob.glob(dataset_path))

        # Create a runner class to transfer read only variables to the processes
        runner = Runner(trc_idx, da, trc_avg)

        # Encode the midi files as token encodings
        results = []
        with tqdm(total=len(pianoroll_files), desc='Encode midi files as token encoding. (2/3)') as t:
            with Pool(process) as p:
                for _ in p.imap_unordered(runner.safe_run, pianoroll_files):
                    t.update(1)
    else:
        pianoroll_files = list(glob.glob(dataset_path))

    numfiles = 0
    chunks = []
    current_chunk = []
    for file_path in tqdm(pianoroll_files, desc='Create chunks from the temporary files. (3/3)'):
        # Open temporary the encoding file
        with open(file_path + '.tmp', mode='rb') as f:
            file_chunk = pickle.load(f)
        # Loop through the tokens in the sequence and split them into chunks of combine size
        for note in file_chunk:
            # Fill current sub chunk
            current_chunk.append(note)
            if len(current_chunk) > int(sequence_length) - 1:
                # Save full chunk in list
                chunks.append(np.stack(current_chunk))
                # Start a new sub chunk
                current_chunk = []
        # Save the chunk if its size exceeds, for example, 50000 sequences
        # A chunk with 50000 sequences of length 4096 will take up around 390MB in memory
        # It will take around 0.48 seconds to load it into memory
        # Good enough if we load it in a second threat
        if len(chunks) > int(chunk_size):
            # Cast them to uint16 for less memory usage
            optimized_chunks = np.array([np.array(chunk, dtype=np.uint16) for chunk in chunks])
            np.savez_compressed(os.path.join(output, f'{numfiles:03d}.npz'), data=optimized_chunks)
            numfiles += 1
            chunks = []

    # Save last chunk even if it's not finished yet
    # Nope, we ignore the last chunk since it would have a different size and could not be stored in our matrix
    ### TODO Implement a padding function for the last sequence in the last chunk
    # chunks.append(current_chunk)
    # Cast them to uint16 for less memory usage
    optimized_chunks = np.array([np.array(chunk, dtype=np.uint16) for chunk in chunks])
    np.savez_compressed(os.path.join(output, f'{numfiles:03d}.npz'), data=optimized_chunks)
    chunks = []
    # Save the ordering of the tracks
    with open(os.path.join(output, 'tracks.trc'), 'w') as f:
        f.write(','.join([tracks[t] for t in trc_idx]))


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

    assert args.dataset == 'lpd_5', 'Dataset required lpd_5'

    encode_dataset(args.output,
                   args.dataset,
                   args.process,
                   args.da,
                   args.sequence_length,
                   args.chunk_size,
                   args.encode_from_tmp)
