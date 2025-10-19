import glob
import os
import pickle
from multiprocessing import Pool
import multiprocessing as mp
import matplotlib
import numpy as np
import pypianoroll
from matplotlib import pyplot as plt
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
from collections import defaultdict, Counter
import glob


matplotlib.use('TkAgg')


### TODO: Clean this up and create plots for the statistics
# Create an average notes per bar statistic so we can see how many notes on average the network can have in memory


def process_file(file_path):
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
                        # Check if this specific track has a note at this timestep
                        has_note = np.any(pr[track_idx, timestep, :] > 0)
                        if has_note and track_type in track_resolution_steps:
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


def get_dataset_statistics_per_track(num_processes=8):
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Change the working directory to the script's directory
    os.chdir(script_dir)

    # Get all files
    file_paths = glob.glob('lpd_5/lpd_5_cleansed/*/*/*/*/*.npz')
    print(f'Found {len(file_paths)} files to process')

    if num_processes is None:
        num_processes = mp.cpu_count()

    print(f'Using {num_processes} processes')

    # Initialize result containers
    resolutions = set()
    total_files = 0
    total_timesteps = 0

    # Track-specific resolution steps counters
    track_resolution_steps = {
        'Drums': {step: 0 for step in range(25)},
        'Piano': {step: 0 for step in range(25)},
        'Guitar': {step: 0 for step in range(25)},
        'Bass': {step: 0 for step in range(25)},
        'Strings': {step: 0 for step in range(25)}
    }

    # Track type occurrence counter
    track_type_counts = {
        'Drums': 0,
        'Piano': 0,
        'Guitar': 0,
        'Bass': 0,
        'Strings': 0
    }

    # Process files using multiprocessing
    with Pool(processes=num_processes) as pool:
        results = list(tqdm(
            pool.imap_unordered(process_file, file_paths, chunksize=50),
            total=len(file_paths),
            desc='Processing files'
        ))

    # Aggregate results
    failed_files = []
    for result in results:
        if result['success']:
            total_files += 1
            resolutions.add(result['resolution'])
            total_timesteps += result['max_length']

            # Count track types in this file
            for track_type in result['track_types']:
                if track_type in track_type_counts:
                    track_type_counts[track_type] += 1

            # Aggregate track-specific resolution steps
            for track_type, steps_dict in result['track_resolution_steps'].items():
                if track_type in track_resolution_steps:
                    for step, count in steps_dict.items():
                        track_resolution_steps[track_type][step] += count
        else:
            failed_files.append((result['file_path'], result['error']))

    # Print any failed files
    if failed_files:
        print(f'\n{len(failed_files)} files failed to process:')
        for file_path, error in failed_files[:10]:
            print(f'  {file_path}: {error}')
        if len(failed_files) > 10:
            print(f'  ... and {len(failed_files) - 10} more')

    # Print statistics
    print('\n=== Dataset Statistics ===')
    print(f'Total files processed: {total_files}')
    print(f'Total timesteps: {total_timesteps}')
    print(f'Average timesteps per file: {total_timesteps / total_files if total_files > 0 else 0:.2f}')

    print(f'\nUnique resolutions found: {sorted(resolutions)}')

    print('\nTrack type occurrences:')
    for track_type, count in track_type_counts.items():
        print(f'{track_type}: {count} tracks')

    # Print resolution steps per track
    for track_type in ['Bass', 'Drums', 'Piano', 'Guitar', 'Strings']:
        print(f'\n{track_type} - Resolution steps distribution:')
        steps_dict = track_resolution_steps[track_type]
        total_notes = sum(steps_dict.values())
        if total_notes > 0:
            for step in range(25):
                count = steps_dict[step]
                if count > 0:
                    percentage = (count / total_notes) * 100
                    print(f'  Step {step:2d}: {count:8d} occurrences ({percentage:5.2f}%)')
        else:
            print(f'  No notes found for {track_type}')

    # Save statistics to file
    stats = {
        'total_files': total_files,
        'total_timesteps': total_timesteps,
        'resolutions': list(resolutions),
        'track_resolution_steps': track_resolution_steps,
        'track_type_counts': track_type_counts,
        'failed_files': failed_files
    }

    with open('../tmp/dataset_statistics_per_track.pkl', 'wb') as f:
        pickle.dump(stats, f)

    print(f'\nStatistics saved to dataset_statistics_per_track.pkl')

    # Create visualization for each track
    fig, axes = plt.subplots(3, 2, figsize=(20, 15))
    axes = axes.flatten()

    track_types = ['Bass', 'Drums', 'Piano', 'Guitar', 'Strings']

    for idx, track_type in enumerate(track_types):
        ax = axes[idx]
        steps = list(range(25))
        counts = [track_resolution_steps[track_type][step] for step in steps]

        # Only plot if there are notes for this track
        if sum(counts) > 0:
            ax.bar(steps, counts)
            ax.set_title(f'{track_type} - Resolution Steps Distribution')
            ax.set_xlabel('Step within Resolution')
            ax.set_ylabel('Number of Occurrences')
            ax.set_xticks(range(0, 25, 2))
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, f'No notes found\nfor {track_type}',
                    ha='center', va='center', transform=ax.transAxes, fontsize=12)
            ax.set_title(f'{track_type} - No Data')

    # Remove the extra subplot
    axes[5].remove()

    plt.tight_layout()
    plt.savefig('../tmp/track_resolution_statistics.png', dpi=300, bbox_inches='tight')
    plt.show()

    return stats


def calculate_discarded_notes_statistics():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    # Load the pickle file containing dataset statistics
    pickle_path = os.path.join('..', 'tmp', 'dataset_statistics_per_track.pkl')

    with open(pickle_path, 'rb') as f:
        stats = pickle.load(f)

    # Extract the relevant data
    track_resolution_steps = stats['track_resolution_steps']
    # track_type_counts = stats['track_type_counts']

    print('Track Resolution Steps Analysis')
    print('=' * 50)

    # Define encoding resolutions to test
    encoding_resolutions = [2, 4, 8, 12, 24]

    for encoding_resolution in encoding_resolutions:
        print(f'\nEncoding Resolution: {encoding_resolution}')
        print('-' * 30)

        # Calculate step size: 24 // encoding_resolution
        step = 24 // encoding_resolution

        # Note the dataset size in steps
        dataset_size = 0
        covered_dataset_size = 0

        # For each track, calculate what percentage of steps will be covered
        for track_name in ['Drums', 'Piano', 'Guitar', 'Bass', 'Strings']:
            track_steps = track_resolution_steps[track_name]

            # Sum up steps where the index can be cleanly divided by step
            covered_steps = 0
            total_steps = 0

            for step_index in range(25):  # 0 to 24
                count = track_steps[step_index]
                total_steps += count

                # Check if step_index is divisible by step
                if step_index % step == 0:
                    covered_steps += count

            covered_dataset_size += covered_steps
            dataset_size += total_steps

            # Calculate percentage
            if total_steps > 0:
                percentage = (covered_steps / total_steps) * 100
                print(f'{track_name:8}: {percentage:6.2f}% ({covered_steps:,}/{total_steps:,} steps)')
            else:
                print(f'{track_name:8}: No data available')

        percentage = (covered_dataset_size / dataset_size) * 100
        print(f'Complete size: {percentage:6.2f}% ({covered_dataset_size:,}/{dataset_size:,} steps)')
        print('-' * 30)


# if __name__ == '__main__':
#    calculate_discarded_notes_statistics()


def load_cleansed_ids(cleansed_ids_file):
    print(f'Loading cleansed IDs from {cleansed_ids_file}...')
    msd_ids = set()

    with open(cleansed_ids_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            if line_num % 100000 == 0:  # Progress indicator
                print(f'  Processed {line_num:,} lines...')

            line = line.strip()
            if line:
                parts = line.split()
                if len(parts) >= 2:
                    msd_id = parts[1].strip()
                    if msd_id.startswith('TR') and len(msd_id) == 18:
                        msd_ids.add(msd_id)

    print(f'Loaded {len(msd_ids):,} unique MSD IDs')
    return msd_ids


def process_label_provider_folder(folder_path, msd_ids_set, provider_name):
    """
    Process a label provider folder and count genre occurrences for our songs.

    Args:
        folder_path (str): Path to the label provider folder
        msd_ids_set (set): Set of our MSD IDs for fast lookup
        provider_name (str): Name of the label provider for display

    Returns:
        dict: Genre counts for our songs
    """
    print(f'\nProcessing {provider_name} folder: {folder_path}')

    genre_counts = defaultdict(int)
    id_list_files = glob.glob(os.path.join(folder_path, 'id_list_*.txt'))

    if not id_list_files:
        print(f'  No id_list_*.txt files found in {folder_path}')
        return genre_counts

    labeled_songs = set()  # Track unique songs that have at least one label
    for file_path in id_list_files:
        # Extract genre name from filename
        filename = os.path.basename(file_path)
        genre = filename.replace('id_list_', "").replace('.txt', "")

        # Count how many of our songs are in this genre
        count = 0
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    msd_id = line.strip()
                    if msd_id in msd_ids_set:
                        count += 1
                        labeled_songs.add(msd_id)  # Add to set of labeled songs

            if count > 0:
                genre_counts[genre] = count
                print(f'    {genre}: {count} songs')

        except Exception as e:
            print(f'    Error processing {file_path}: {e}')

    print(
        f'\n  Total unique songs with labels: {len(labeled_songs)} out of {len(msd_ids_set)} ({len(labeled_songs) / len(msd_ids_set) * 100:.1f}%)')
    return dict(genre_counts)


def create_pie_chart(genre_counts, provider_name, output_dir='genre_charts'):
    """
    Create a pie chart for genre distribution.

    Args:
        genre_counts (dict): Genre counts
        provider_name (str): Name of the label provider
        output_dir (str): Output directory for charts
    """
    if not genre_counts:
        print(f'No data to plot for {provider_name}')
        return

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Sort genres by count (descending) for better visualization
    sorted_genres = sorted(genre_counts.items(), key=lambda x: x[1], reverse=True)

    # Limit to top genres if there are too many (optional)
    max_genres = 15
    if len(sorted_genres) > max_genres:
        top_genres = sorted_genres[:max_genres]
        other_count = sum(count for _, count in sorted_genres[max_genres:])
        if other_count > 0:
            top_genres.append(('Other', other_count))
        sorted_genres = top_genres

    genres, counts = zip(*sorted_genres)
    total_songs = sum(counts)

    # Create pie chart
    plt.figure(figsize=(12, 8))
    colors = plt.cm.Set3(range(len(genres)))

    wedges, texts, autotexts = plt.pie(
        counts,
        labels=genres,
        autopct='%1.1f%%',
        colors=colors,
        startangle=90
    )

    # Improve text formatting
    for autotext in autotexts:
        autotext.set_color('black')
        autotext.set_fontsize(9)
        autotext.set_weight('bold')

    for text in texts:
        text.set_fontsize(10)

    plt.title(f'Genre Distribution - {provider_name}\n({total_songs:,} total matched labels)',
              fontsize=14, fontweight='bold', pad=20)

    plt.axis('equal')
    plt.tight_layout()

    # Save the chart
    output_file = os.path.join(output_dir, f'{provider_name.replace(" ", "_").lower()}_genre_distribution.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f'Saved chart: {output_file}')

    # Also show the chart
    plt.show()

    # Print summary statistics
    print(f'\n{provider_name} Summary:')
    print(f'  Total songs with genre labels: {total_songs:,}')
    print(f'  Number of different genres: {len(genre_counts)}')
    print(f'  Top 5 genres:')
    for i, (genre, count) in enumerate(sorted_genres[:5], 1):
        percentage = (count / total_songs) * 100
        print(f'    {i}. {genre}: {count:,} songs ({percentage:.1f}%)')


def analyze_genre_distributions(cleansed_ids_file, label_provider_folders):
    """
    Main function to analyze genre distributions across label providers.

    Args:
        cleansed_ids_file (str): Path to cleansed_ids.txt
        label_provider_folders (dict): Dictionary mapping provider names to folder paths
    """
    # Load our song IDs for fast lookup
    msd_ids_set = load_cleansed_ids(cleansed_ids_file)

    if not msd_ids_set:
        print('No valid MSD IDs found in cleansed_ids.txt')
        return

    # Process each label provider
    for provider_name, folder_path in label_provider_folders.items():
        if not os.path.exists(folder_path):
            print(f'Warning: Folder {folder_path} does not exist, skipping {provider_name}')
            continue

        # Count genres for this provider
        genre_counts = process_label_provider_folder(folder_path, msd_ids_set, provider_name)

        # Create pie chart
        create_pie_chart(genre_counts, provider_name)


#if __name__ == '__main__':
#    # Configuration
#    cleansed_ids_file = '../lpd_5/cleansed_ids.txt'  # Path to your cleansed_ids.txt file
#
#    # Define your label provider folders
#    label_provider_folders = {
#        'Last.fm Dataset': 'lpd_5/labels/lastfm',
#        'Million Song Dataset Benchmarks': 'lpd_5/labels/amg',
#        'Tagtraum Genre Annotations': 'lpd_5/labels/tagtraum'
#    }
#
#    # Run the analysis
#    analyze_genre_distributions(cleansed_ids_file, label_provider_folders)


def get_drum_pitch_counts():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)

    file_paths = glob.glob('../lpd_5/lpd_5_cleansed/*/*/*/*/*.npz')
    drum_pitch_counts = Counter()

    for file_path in tqdm(file_paths):
        try:
            m = pypianoroll.load(file_path)

            drum_tracks = [t for t in m.tracks if t.is_drum]
            if not drum_tracks:
                continue

            for drum_track in drum_tracks:
                pr = drum_track.pianoroll
                pitch_occurrences = np.sum(pr > 0, axis=0)

                for pitch, count in enumerate(pitch_occurrences):
                    if count > 0:
                        drum_pitch_counts[pitch] += count

        except Exception as e:
            print(f'Error processing {file_path}: {e}')

    # Calculate coverage-based selection
    total_notes = sum(drum_pitch_counts.values())
    sorted_pitches = sorted(drum_pitch_counts.items(), key=lambda x: x[1], reverse=True)

    # Output
    print('\nSorted pitch counts:')
    for pitch, count in sorted_pitches:
        print(f'Pitch {pitch}: {count} occurrences')

    for coverage_percentage in [0.9, 0.95, 0.99]:

        coverage_limit = coverage_percentage * total_notes
        selected_pitches = []
        running_total = 0

        for pitch, count in sorted_pitches:
            selected_pitches.append(pitch)
            running_total += count
            if running_total >= coverage_limit:
                break

        print(f'\nNumber of unique drum pitches: {len(drum_pitch_counts)}')
        print(f'Selected pitches for {coverage_percentage * 100}% coverage: {selected_pitches}')
        print(f'Total number of selected pitches: {len(selected_pitches)}')
        print(f'')


if __name__ == '__main__':
    get_drum_pitch_counts()