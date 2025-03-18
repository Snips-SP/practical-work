import os
import re
import subprocess
from pydub.silence import split_on_silence
from pydub import AudioSegment
from pydub.utils import which

# Finds ffmpeg in system path
ffmpeg_path = which('ffmpeg')
if ffmpeg_path is None:
    # Manually specify if needed
    ffmpeg_path = r'C:\Users\brunner4\anaconda3\envs\TorchGPU\Library\bin\ffmpeg.exe'

AudioSegment.converter = ffmpeg_path


class EncodingConfig:
    # List of all used tokens (excluding padding token)
    tokens: list = []
    # Token used for padding
    padding_token: int = None
    # The length of all tokens (tokens and padding)
    vocab_size: int = None

    # All the instruments which are used in our encoding
    tracks = ['Drums', 'Piano', 'Guitar', 'Bass', 'Strings']
    programs = {
        'Drums': 0,  # Program is not important here since we set the flag is_drum when creating a track
        'Piano': 0,  # Program for Acoustic Grand Piano
        'Guitar': 32,  # Program for 32 for Electric Guitar
        'Bass': 35,  # Program for Electric Bass (picked)
        'Strings': 49  # Program for String Ensemble 1
    }

    # The offsets between the instruments and range of notes
    note_size: int = 84
    note_offset: int = 24

    # Tokens for time note and end note
    time_note: int = None
    end_note: int = None

    # List with prioritised index where Bass is placed in front
    trc_idx: list = None

    @classmethod
    def initialize(cls):
        if not cls.tokens:  # Prevent re-initialization
            # Drums: [0 * 84, 0 * 84 + 83] = [0, 83]
            # Piano: [1 * 84, 1 * 84 + 83] = [84, 167]
            # Guitar: [2 * 84, 2 * 84 + 83] = [168, 251]
            # Bass: [3 * 84, 3 * 84 + 83] = [252, 335]
            # Strings: [4 * 84, 4 * 84 + 83] = [336, 419]
            cls.tokens.extend(range(0, 420))
            cls.time_note = cls.tokens[-1] + 1
            cls.tokens.append(cls.time_note)  # Add the token which represents a pause in the music (420)
            cls.end_note = cls.tokens[-1] + 1
            cls.tokens.append(cls.end_note)  # Add the token which represents the end of the sequence (421)
            cls.padding_token = cls.tokens[-1] + 1  # Add the padding token to the mix (422)
            cls.vocab_size = cls.padding_token + 1  # We need to add one to the total size since 0 is included

            cls.trc_idx = sorted(list(range(len(cls.tracks))), key=lambda x: 0 if cls.tracks[x] == 'Bass' else 1)


# Function to get the next available index for a new run folder
def get_next_run_folder(name, base_dir='runs'):
    # List all folders in the base directory
    existing_folders = os.listdir(base_dir)

    # Regex to capture 'run_<index>' format
    run_pattern = re.compile(fr'^{name}_(\d+)$')

    # Find the highest index
    max_index = 0
    for folder in existing_folders:
        match = run_pattern.match(folder)
        if match:
            # Extract the index from folder name
            index = int(match.group(1))
            max_index = max(max_index, index)

    # Increase the index by 1 for the next run
    new_run_name = f'{name}_{max_index + 1}'
    new_run_path = os.path.join(base_dir, new_run_name)

    return new_run_path


def get_latest_checkpoint(directory, name):
    # Define a regex pattern to extract the epoch number
    pattern = re.compile(rf'{name}(\d+)\.ph')

    latest_epoch = -1
    latest_file = None

    # Iterate through files in the directory
    for filename in os.listdir(directory):
        match = pattern.match(filename)
        if match:
            epoch = int(match.group(1))  # Extract epoch number
            if epoch > latest_epoch:
                latest_epoch = epoch
                latest_file = filename

    if latest_file:
        return os.path.join(directory, latest_file)
    else:
        return None  # No valid checkpoint found


def mid_to_mp3(mid_file: str, sf2_file: str, output_file: str = 'output.mp3'):
    assert os.path.isfile(mid_file), 'Mid file not found'
    assert os.path.isfile(sf2_file), 'sf2 file not found'

    # Create temporary wav file
    tmp_wav = 'tmp.wav'
    subprocess.run(["fluidsynth", "-qni", sf2_file, mid_file, "-F", tmp_wav])

    # Convert wav to mp3
    audio = AudioSegment.from_wav(tmp_wav)

    # TODO:
    # FluidSynth created long periods of silence after the notes of the mid file have finished playing
    # I cannot figure out why this is happening but I will just remove it during the conversion to mp3

    # Parameters for silence detection
    # Silence threshold in dBFS
    silence_thresh = audio.dBFS - 14
    # Minimum silence duration in milliseconds
    min_silence_len = 500

    # Split the audio based on silence
    segments = split_on_silence(audio, min_silence_len=min_silence_len, silence_thresh=silence_thresh)

    # Join all non-silent segments
    trimmed_audio = segments[0]  # Start with the first segment

    for segment in segments[1:]:
        trimmed_audio += segment  # Concatenate the segments back together

    trimmed_audio.export(output_file, format='mp3')

    # Delete tmp wav file
    os.remove(tmp_wav)


def chord2tokens(chord):
    if chord is None or chord == 'auto':
        return [EncodingConfig.time_note]
    else:
        base = ['E', 'F', 'G', 'A', 'B', 'C', 'D'].index(chord[0])
        basenote = [4, 5, 7, 9, 11, 12, 14][base]  # Bass
        chordtype = chord[1:]
        if len(chord) > 1 and chord[1] == '#':
            basenote += 1
            chordtype = chord[2:]
        offset = basenote + EncodingConfig.note_size * 2 + 24  # Piano notes
        if len(chordtype) == 0:
            return [basenote, offset, offset + 4, offset + 7]
        elif chordtype == 'm':
            return [basenote, offset, offset + 3, offset + 7]
        elif chordtype == '7':
            return [basenote, offset, offset + 4, offset + 7, offset + 10]
        elif chordtype == 'm7':
            return [basenote, offset, offset + 3, offset + 7, offset + 10]
        elif chordtype == 'M7':
            return [basenote, offset, offset + 4, offset + 7, offset + 11]
        elif chordtype == 'm7-5':
            return [basenote, offset, offset + 3, offset + 6, offset + 10]
        elif chordtype == 'dim':
            return [basenote, offset, offset + 3, offset + 6, offset + 9]
        elif chordtype == 'sus4':
            return [basenote, offset, offset + 5, offset + 7]
        elif chordtype == '7sus4':
            return [basenote, offset, offset + 5, offset + 7, offset + 10]
        elif chordtype == 'aug':
            return [basenote, offset, offset + 4, offset + 8]
        elif chordtype == 'm6':
            return [basenote, offset, offset + 3, offset + 7, offset + 9]
        elif chordtype == '7(9)':
            return [basenote, offset, offset + 4, offset + 7, offset + 10, offset + 14]
        elif chordtype == 'm7(9)':
            return [basenote, offset, offset + 3, offset + 7, offset + 10, offset + 14]
        elif chordtype == 'add9':
            return [basenote, offset, offset + 4, offset + 7, offset + 14]
        elif chordtype == '6':
            return [basenote, offset, offset + 4, offset + 7, offset + 9]
        elif chordtype == 'mM7':
            return [basenote, offset, offset + 3, offset + 7, offset + 11]
        elif chordtype == '7-5':
            return [basenote, offset, offset + 4, offset + 6, offset + 10]
        else:
            return [basenote]
