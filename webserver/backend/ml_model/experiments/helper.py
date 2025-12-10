import pypianoroll
import numpy as np
from backend.ml_model.encoding import EncodingConfig

def calculate_statistics_per_file(file_path):
    """Process a single file and return its statistics per track"""
    try:
        # Load it as a multitrack object
        m = pypianoroll.load(file_path)

        # Collect resolution
        resolution = m.resolution

        # Get the maximum length across all tracks
        max_length = m.get_max_length()

        # Convert to numpy array for easier processing
        pr = m.stack()  # Shape: (num_tracks, num_timesteps, num_pitches)

        # Identify track types based on program and drum flag
        track_types = []
        for track in m.tracks:
            if track.is_drum:
                track_types.append('Drums')
            elif track.program == 0:
                track_types.append('Piano')
            elif track.program == 24:
                track_types.append('Guitar')
            elif track.program == 32:
                track_types.append('Bass')
            elif track.program == 48:
                track_types.append('Strings')
            else:
                # Handle unexpected programs
                track_types.append(f'Unknown_Program_{track.program}')

        # Initialize resolution steps counter for each track type
        track_resolution_steps = {
            'Drums': {step: 0 for step in range(25)},
            'Piano': {step: 0 for step in range(25)},
            'Guitar': {step: 0 for step in range(25)},
            'Bass': {step: 0 for step in range(25)},
            'Strings': {step: 0 for step in range(25)}
        }

        # Count resolution steps for each track
        for timestep in range(max_length):
            step_in_resolution = timestep % m.resolution
            if step_in_resolution < 25:  # Only count steps 0-24
                # Check each track individually
                for track_idx, track_type in enumerate(track_types):
                    if track_idx < pr.shape[0]:  # Make sure track exists
                        # --- Note Onset Logic ---

                        # Get a boolean array of active pitches at the current step
                        current_notes_on = pr[track_idx, timestep, :] > 0

                        # Get a boolean array of active pitches at the previous step
                        if timestep == 0:
                            # At timestep 0, any note is an onset (no previous step)
                            prev_notes_on = np.zeros_like(current_notes_on, dtype=bool)
                        else:
                            prev_notes_on = pr[track_idx, timestep - 1, :] > 0

                        # Find onsets: notes that are ON now but were OFF previously
                        # (This is a boolean array operation: A and not B)
                        onsets = current_notes_on & (~prev_notes_on)

                        # Check if *any* pitch had an onset at this timestep
                        has_onset = np.any(onsets)

                        if has_onset:
                            track_resolution_steps[track_type][step_in_resolution] += 1

        return {
            'success': True,
            'resolution': resolution,
            'max_length': max_length,
            'track_types': track_types,
            'track_resolution_steps': track_resolution_steps
        }

    except Exception as e:
        return {
            'success': False,
            'error': str(e),
            'file_path': file_path
        }


def average_notes_per_timestep_for_file(file_path):
    """
    Process a single encoded midi file and return the total notes and total time steps per track.
    This version correctly calculates totals for the standard "average notes per time step" metric.
    """
    try:
        sequence = np.load(file_path)

        # Initialize counts: [total_notes, total_timesteps]
        file_counts = {
            'Drums': [0, 0], 'Piano': [0, 0], 'Guitar': [0, 0],
            'Bass': [0, 0], 'Strings': [0, 0], 'Others': [0, 0],
        }

        # Find all time step boundaries
        time_note_indices = np.where(sequence == EncodingConfig.time_note)[0]

        # If there are no time steps, return zeros
        if time_note_indices.size == 0:
            return {'success': True, 'counts': file_counts}

        boundaries = np.append(time_note_indices, len(sequence))

        for i in range(len(boundaries) - 1):
            start_index = boundaries[i] + 1
            end_index = boundaries[i + 1]
            timestep_tokens = sequence[start_index:end_index]

            if timestep_tokens.size == 0:
                continue

            # Use instrument bases for clear range checking
            b = EncodingConfig.instrument_bases

            # Calculate the number of notes for each instrument
            file_counts['Drums'][0] += np.sum(timestep_tokens < b['Bass'])
            file_counts['Bass'][0] += np.sum((b['Bass'] <= timestep_tokens) & (timestep_tokens < b['Piano']))
            file_counts['Piano'][0] += np.sum((b['Piano'] <= timestep_tokens) & (timestep_tokens < b['Guitar']))
            file_counts['Guitar'][0] += np.sum((b['Guitar'] <= timestep_tokens) & (timestep_tokens < b['Strings']))
            file_counts['Strings'][0] += np.sum((b['Strings'] <= timestep_tokens) & (timestep_tokens < b['Special']))
            file_counts['Others'][0] += np.sum(b['Special'] <= timestep_tokens)

        # The total number of time steps is the same for every track in the file.
        num_timesteps = len(time_note_indices)
        for track in file_counts:
            file_counts[track][1] = num_timesteps

        return {
            'success': True,
            'counts': file_counts,
        }

    except Exception as e:
        return {
            'success': False,
            'error': str(e),
            'file_path': file_path
        }