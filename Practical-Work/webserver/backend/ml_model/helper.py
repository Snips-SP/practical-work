import os
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
