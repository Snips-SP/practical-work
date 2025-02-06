import os
import re


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