import glob
import os
import pickle
import numpy as np
import pypianoroll
from backend.ml_model.helper import EncodingConfig
from backend.ml_model.dataloader import OnTheFlyMidiDataset

from typing import Dict, List

from torch.utils.data import DataLoader


def test_encoding_decoding():
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Change the working directory to the script's directory
    os.chdir(script_dir)
    file_path = 'M/M/M/TRMMMQN128F4238509/a04d6f748582e3bb25fc2d1108f71573.npz'
    # Grap one of the encoded midi files
    original_file_path = os.path.join('..', 'lpd_5', 'lpd_5_cleansed', file_path)

    # Create dataset with only one midi file
    dataset = OnTheFlyMidiDataset([original_file_path], 11)

    # Encode the midi file
    tokens, mask = dataset.__getitem__(0)

    # Open original as a Multitrack object
    mt_original = pypianoroll.load(original_file_path)

    tempo = int(mt_original.tempo[0])
    # Get length in time steps (probably quarter notes)
    length = mt_original.get_max_length()

    # Decode it again
    mt_decoded = new_decode(tokens.numpy(), length, tempo, encoding_resolution=4)

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


def new_decode(seq, length, tempo, encoding_resolution=4):
    resolution = 24
    step = resolution // encoding_resolution

    # Create empty pianoroll array with resolution 24
    pianoroll = np.zeros((len(EncodingConfig.encoding_order), length, 128))

    melodic_interval = EncodingConfig.instrument_intervals['Melodic']
    drum_interval = EncodingConfig.instrument_intervals['Drums']
    bass_interval = EncodingConfig.instrument_intervals['Bass']
    guitar_interval = EncodingConfig.instrument_intervals['Guitar']
    strings_interval = EncodingConfig.instrument_intervals['Strings']
    piano_interval = EncodingConfig.instrument_intervals['Piano']

    current_tick = 0
    i = 0
    # Decode it again
    while True:
        if i >= len(seq):
            break
        note = seq[i]
        i += 1

        # Update the context sequence
        if note == EncodingConfig.time_note:
            # Either way update the position by 6 (step)
            current_tick += step

            # End the generation if we have reached the end
            if current_tick >= pianoroll.shape[1]:
                break
        else:
            if melodic_interval[0] <= note <= melodic_interval[1]:
                # It is a melodic note
                if bass_interval[0] <= note <= bass_interval[1]:
                    track = 'Bass'
                elif guitar_interval[0] <= note <= guitar_interval[1]:
                    track = 'Guitar'
                elif strings_interval[0] <= note <= strings_interval[1]:
                    track = 'Strings'
                elif piano_interval[0] <= note <= piano_interval[1]:
                    track = 'Piano'
                else:
                    raise ValueError('Note not in melodic range')

                # Calculate the midi note value
                midi_value = note + EncodingConfig.note_offset - (EncodingConfig.instrument_intervals[track][0] - 1)
                trc_index = EncodingConfig.encoding_order.index(track)

                # Position is normal since melodic notes can only be placed on steps of 6
                pianoroll[trc_index, current_tick:min(pianoroll.shape[1] - 1, current_tick + step), midi_value] = 100
            elif drum_interval[0] <= note <= drum_interval[1]:
                # It is a drum note
                trc = EncodingConfig.midi_tracks.index('Drums')
                midi_value = EncodingConfig.drum_token_to_pitch[note]

                microtiming_interval = EncodingConfig.instrument_intervals['Microtimings']
                if microtiming_interval[0] <= seq[i+1] <= microtiming_interval[1]:
                    # It is a microtiming, so we have to adjust the position by the offset
                    i += 1

                    offset = EncodingConfig.microtiming_token_to_delta[seq[i]]
                    # Add offset to the position if it does not fall out of the length otherwise we discard it
                    if current_tick + offset < pianoroll.shape[1]:
                        pianoroll[trc, current_tick + offset, midi_value] = 100
                else:
                    # It is not a microtiming but some other note, which means we do not have an offset
                    pianoroll[trc, current_tick, midi_value] = 100
            else:
                # Its a different note we can ignore end_note, padding_token, or microtiming
                continue

    pr = []
    for i, t in enumerate(EncodingConfig.encoding_order):
        pr.append(pypianoroll.StandardTrack(pianoroll=pianoroll[i], program=EncodingConfig.programs[t],  is_drum=(t == 'Drums')))
    mt_decoded = pypianoroll.Multitrack(tracks=pr, tempo=np.full(pianoroll.shape[1], tempo), resolution=24)

    return mt_decoded


def step_to_pos(x, resolution=24, beats_per_bar=4):
    return f'Bar: {(x // resolution) // beats_per_bar + 1}, Beat in bar: {(x // resolution) % beats_per_bar + 1}, Sixteenth note: {(x % resolution) // (resolution // 4) + 1}'


if __name__ == '__main__':
    test_encoding_decoding()
