import os
import re
from encode import EncodingParameters

EncodingParameters.initialize()


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


def chord2tokens(chord):
    if chord is None or chord == 'auto':
        return [EncodingParameters.time_note]
    else:
        base = ['E', 'F', 'G', 'A', 'B', 'C', 'D'].index(chord[0])
        basenote = [4, 5, 7, 9, 11, 12, 14][base]  # Bass
        chordtype = chord[1:]
        if len(chord) > 1 and chord[1] == '#':
            basenote += 1
            chordtype = chord[2:]
        offset = basenote + EncodingParameters.note_size * 2 + 24  # Piano notes
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