import glob
import os
import pickle
import numpy as np
import pypianoroll
from backend.ml_model.helper import EncodingConfig


def _reorder_current(cur_seq):
    # Reorder the sequence so that piano is the last instrument
    # Bass: [0 * 84, 0 * 84 + 83] = [0, 83]
    # Drums: [1 * 84, 1 * 84 + 83] = [84, 167]
    # Piano: [2 * 84, 2 * 84 + 83] = [168, 251]
    # Guitar: [3 * 84, 3 * 84 + 83] = [252, 335]
    # Strings: [4 * 84, 4 * 84 + 83] = [336, 419]

    cur = []
    for c in sorted(cur_seq):
        # Checks if a note c is not in the range [84, 168)
        # i.e. if the instrument is not drums
        if not (c >= EncodingConfig.note_size and c < EncodingConfig.note_size * 2):
            cur.append(c)
    for c in sorted(cur_seq):
        # Checks if a note c is in the range [84, 168)
        # i.e. if the instrument is drums
        if (c >= EncodingConfig.note_size and c < EncodingConfig.note_size * 2):
            cur.append(c)

    return cur  # Bass, Piano, etc..., Drums


def test_encoding_decoding():
    script_dir = os.path.dirname(os.path.abspath(__file__))

    resolution = 8

    # Change the working directory to the script's directory
    os.chdir(script_dir)
    file_path = 'K/J/M/TRKJMLE128F92E5B12/08548ae294e0f8f0aa6fde2ba0454026.npz'
    # Grap one of the encoded midi files
    original_file_path = os.path.join('..', 'lpd_5', 'lpd_5_cleansed', file_path)
    encoded_file_path = os.path.join('..', 'lpd_5', 'lpd_5_cleansed', f'{file_path}.tmp')

    # Remove old one if it still exists
    if os.path.exists(encoded_file_path):
        os.remove(encoded_file_path)

    # Encode the file and save it in the same place just with .tmp added
    percent_of_discarded_notes = encode(original_file_path, 'lpd_5/lpd_5_cleansed/*/*/*/*/*.npz', resolution)

    print(f'During encoding we discarded: {percent_of_discarded_notes}% of notes.')

    # Open original as a Multitrack object
    mt_original = pypianoroll.load(original_file_path)

    tempo = int(mt_original.tempo[0])
    # Get length in time steps (probably quarter notes)
    length = mt_original.get_max_length()

    # Decode it again
    mt_decoded = decode(encoded_file_path, length, tempo, resolution)

    # Set a few flags to making working in the daw easier
    for track in mt_original.tracks:
        if not track.is_drum:
            track.name = f'Program: {track.program}'
        else:
            track.name = f'Program: 0 (Drums)'

    mt_original.write(os.path.join('../tmp', 'original.mid'))

    for track in mt_decoded.tracks:
        if not track.is_drum:
            track.name = f'Program: {track.program}'
        else:
            track.name = f'Program: 0 (Drums)'

    mt_decoded.write(os.path.join('../tmp', 'decoded.mid'))

    print('fin')


def encode(file_path, dataset_path, encoding_resolution, number_of_modulations=0):
    tracks = ['Drums', 'Piano', 'Guitar', 'Bass', 'Strings']
    trc_len = len(tracks)
    trc_idx = sorted(list(range(trc_len)), key=lambda x: 0 if tracks[x] == 'Bass' else 1)

    if os.path.exists(os.path.join('../lpd_5', 'trc_avg.pkl')):
        with open(os.path.join('../lpd_5', 'trc_avg.pkl'), mode='rb') as f:
            trc_avg = pickle.load(f)
    else:
        # Get an average of each pitch value per track over all datapoints
        trc_avg_c = np.zeros((len(tracks), 2))
        for fp in glob.glob(dataset_path):
            # Load it as a multitrack object
            m = pypianoroll.load(fp)
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
            del m, pr, p, track, pitch

        # Replace 0 in cur_avg_c[:, 1] with 1
        # In order to avoid runtime errors
        trc_avg_c[:, 1] = np.where(trc_avg_c[:, 1] == 0, 1, trc_avg_c[:, 1])

        # Calculate the average note value per track
        trc_avg = np.where(trc_avg_c[:, 1] > 0, trc_avg_c[:, 0] / trc_avg_c[:, 1], 60)
        del trc_avg_c

    seq = []
    # Load it as a multitrack object
    m = pypianoroll.load(file_path)
    # Quarter note resolution is the number of time steps per quater note
    # Calculate how many time steps a quarter note takes to fill a measure
    step = m.resolution // encoding_resolution
    # Convert it to a numpy array of shape (num_tracks, num_time_steps, num_pitches=128)
    pr = m.stack()

    for i, track in enumerate(m.tracks):
        print(f'Track {i}: {track.name}, Program: {track.program}, Is Drum: {track.is_drum}')

    # Transpose the array to have shape (num_time_steps, num_pitches, num_tracks)
    pr = np.transpose(pr, (1, 2, 0))
    # Change ordering to get the bass first and have consistent indexing
    pr = pr[:, :, trc_idx]
    # This changes the indexing to:
    # 0 = Bass (Program: 32)
    # 1 = Drums (Program: Drums (0))
    # 2 = Piano (Program: 0)
    # 3 = Guitar (Program: 24)
    # 4 = String (Program: 48)

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

    # Count discarded notes
    discarded_notes = 0

    current_seq = []
    # Do data augmentation according to specified parameter 'da' in range [0-11]
    # All sequences are appended to the same list
    for s in modulation[:number_of_modulations + 1]:
        pos = 0
        # Create an encoding sequence for each modulation
        for i in range(len(p[0])):
            # Ignore smaller notes then sixteenth notes or notes that are not on the gird
            # only look at indices 0, 6, 12, 18, 24
            if p[0][i] % step != 0:
                discarded_notes += 1
                continue
            # Only if we have advanced in the position will the current cur_seq be written down
            # More note encodings can be placed on the same sixteenth note since they can be
            # from different instruments or an instrument can play more notes at once
            if pos < p[0][i]:
                # From last position to next note occurrence write down the notes which are played
                # by the instruments
                for _ in range(pos, p[0][i], step):
                    seq.extend(_reorder_current(current_seq))
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
                if cur_avg[track] + s < trc_avg[track] + 6:
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
            # Apply our offset to encode less notes [0, 83]
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
        seq.extend(_reorder_current(current_seq))
        # And a last time_note with the end_note as well
        seq.append(EncodingConfig.time_note)
        seq.append(EncodingConfig.end_note)
        current_seq = []
    # Store all the sequences, lists of tokens, as a pickle file
    with open(file_path + '.tmp', mode='wb') as f:
        pickle.dump(seq, f)

    # Return % of discarded notes for analysis
    return (discarded_notes / len(p[0])) * 100


def decode(encoded_file_path, length, tempo, resolution):
    # Open encoded file as numpy array
    with open(encoded_file_path, mode='rb') as f:
        file_chunk = pickle.load(f)

    # Create empty pianoroll array
    pianoroll = np.zeros((len(EncodingConfig.tracks), length, 128))

    pos = 0
    # Decode it again
    for note in file_chunk:
        if note == EncodingConfig.time_note:
            # Either way update the position by 1
            pos += 1
            if pos >= pianoroll.shape[1]:
                break
        elif note < EncodingConfig.time_note:
            # We have hit an actually generated note, thus we put it into its right place in our pianoroll array
            # Calculate which track the note belongs to
            # 0 = Bass -> 3 = Bass
            # 1 = Drums -> 0 = Drums
            # 2 = Piano -> 1 = Piano
            # 3 = Guitar -> 2 = Guitar
            # 4 = String -> 4 Strings
            trc = EncodingConfig.trc_idx[note // EncodingConfig.note_size]
            # Calculate the midi note value
            mid = note % EncodingConfig.note_size + EncodingConfig.note_offset
            # Set the volume to 100 for the note in the piano roll array
            if mid < 128:
                pianoroll[trc, pos, mid] = 100

    pr = []
    for i, t in enumerate(EncodingConfig.tracks):
        pr.append(pypianoroll.StandardTrack(pianoroll=pianoroll[i], program=EncodingConfig.programs[t],
                                            is_drum=(t == 'Drums')))
    mt_decoded = pypianoroll.Multitrack(tracks=pr, tempo=np.full(pianoroll.shape[1], tempo), resolution=resolution)

    return mt_decoded


if __name__ == '__main__':
    test_encoding_decoding()
