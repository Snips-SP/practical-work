import os
import glob
import time

import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm
from backend.ml_model.dataloader import OnTheFlyMidiDataset
from backend.ml_model.experiments.performance_tests import plot_with_avg
from torch.utils.data import DataLoader


def verify_encoded_files():
    # Change the working directory to the script's directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    dataset_path = '../lpd_5'

    print(f'Starting scan in: {dataset_path}')

    # Construct the search pattern for the original pianoroll files
    original_npz_files = glob.glob(os.path.join(dataset_path, 'lpd_5_cleansed/*/*/*/*/*.npz'))

    tmp_file_count = 0
    missing_tmp_files = []

    # Iterate through the found .npz files and check for their .tmp counterparts
    for npz_path in tqdm(original_npz_files, desc='Verifying encoded files'):
        expected_tmp_path = npz_path + '.tmp'
        if os.path.exists(expected_tmp_path):
            tmp_file_count += 1
        else:
            missing_tmp_files.append(npz_path)

    npz_file_count = len(original_npz_files)

    # --- Print the final report ---
    print(f'Total original pianoroll files (.npz): {npz_file_count}')
    print(f'Total encoded temporary files (.npz.tmp): {tmp_file_count}')

    if npz_file_count == tmp_file_count:
        print('All original .npz files have a corresponding .tmp file.')
    else:
        print(f'There are {len(missing_tmp_files)} file(s) missing a .tmp version.')

        # List the first 10 missing files as examples
        if missing_tmp_files:
            for i, missing_file in enumerate(missing_tmp_files[:10]):
                # Print the path relative to the dataset directory for cleaner output
                relative_path = os.path.relpath(missing_file, dataset_path)
                print(f'  - {relative_path}')
            if len(missing_tmp_files) > 10:
                print(f'  ... and {len(missing_tmp_files) - 10} more.')


def dataset_loading_times():
    batch_size = 32
    num_workers = 0 # os.cpu_count()
    dataset_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'lpd_5')
    sample_files = glob.glob(os.path.join(dataset_path, 'lpd_5_cleansed/*/*/*/*/*.npz'))

    # Create dataset
    train_dataset = OnTheFlyMidiDataset(sample_files, 11, chunk_size=1024)

    # Create dataloader
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    timings = []
    last_time = time.time()
    num_batches_to_test = 200

    for i, batch in enumerate(train_dataloader):
        # Calculate time taken for this batch
        current_time = time.time()
        time_taken = current_time - last_time
        timings.append(time_taken)
        last_time = current_time

        # Stop after N batches
        if i >= num_batches_to_test - 1:
            break

    print('Plotting results...')
    timings_to_plot = timings[1:]
    first_batch_time = timings[0]
    print(f'First batch (outlier) time: {first_batch_time:.3f}s')

    x_data = np.arange(len(timings_to_plot))
    y_data = np.array(timings_to_plot)

    # Create the plot
    fig, ax = plt.subplots(figsize=(12, 6))

    # Use your provided plotting function
    plot_with_avg(
        ax, x_data, y_data,
        label=f'Batch Load Time (num_workers={num_workers})',
        avg_fraction=0.2,
        unit='s',
        alpha=0.7,
        linewidth=2
    )

    # Add labels and title
    ax.set_xlabel('Batch Index (excluding first batch)')
    ax.set_ylabel('Time per Batch (s)')
    ax.set_title(f'DataLoader Speed Test (Batch Size={batch_size})')
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    dataset_loading_times()