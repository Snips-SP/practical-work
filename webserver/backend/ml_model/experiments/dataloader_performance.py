import os
import glob
import pickle
import time
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm
from backend.ml_model.dataloader import OnTheFlyMidiDataset
from torch.utils.data import DataLoader


def verify_encoded_files():
    # Change the working directory to the script's directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    dataset_path = '../lpd_5'

    print(f'Starting scan in: {dataset_path}')

    # Construct the search pattern for the original pianoroll files
    original_npz_files = glob.glob(os.path.join(dataset_path, 'lpd_5_cleansed/*/*/*/*/*.npz'))

    missing_tmp_files = []
    # Iterate through the found .npz files and check for their .npy counterparts
    for npz_path in tqdm(original_npz_files, desc='Verifying encoded files'):
        path_name = os.path.dirname(npz_path)
        filename = os.path.splitext(os.path.basename(npz_path))[0]

        for i in range(-5, 7):
            tmp_file_name = f'{filename}.{i}.npy'

            if not os.path.exists(os.path.join(dataset_path, path_name, tmp_file_name)):
                missing_tmp_files.append(os.path.join(dataset_path, path_name, tmp_file_name))

    print(f'Found {len(missing_tmp_files)} missing files')


def dataset_loading_times():
    batch_size = 32
    num_workers_list = [0, 1]
    results_filepath = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'plots', 'dataloader_benchmarks')

    if not os.path.exists(f'{results_filepath}.pkl'):
        all_timings = []
        for num_workers in num_workers_list:
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

            for i, batch in tqdm(enumerate(train_dataloader), desc=f'Testing loading speed with {num_workers} workers', total=num_batches_to_test):
                # Calculate time taken for this batch
                current_time = time.time()
                time_taken = current_time - last_time
                timings.append(time_taken)
                # Simulate working for 1 batch
                time.sleep(0.1)

                last_time = current_time

                # Stop after N batches
                if i >= num_batches_to_test - 1:
                    break

            first_batch_time = timings[0]
            print(f'First batch (outlier) time: {first_batch_time:.3f}s')

            all_timings.append(timings[1:])

        with open(f'{results_filepath}.pkl', 'wb') as f:
            pickle.dump(all_timings, f)

        print('Plotting results..')

    else:
        print('Plotting from stored results..')

        with open(f'{results_filepath}.pkl', 'rb') as f:
            all_timings = pickle.load(f)

    num_runs = len(num_workers_list)
    fig, axes = plt.subplots(
        nrows=num_runs,
        ncols=1,
        figsize=(12, 6 * num_runs),
        sharex=True  # Share the x-axis for easier comparison
    )

    for ax, num_workers, timing_list in zip(axes, num_workers_list, all_timings):
        x_data = np.arange(len(timing_list))
        # Remove the 0.1 seconds we have simulated from the loading times
        y_data = np.array(timing_list) - 0.1

        # Calculate average and standard deviation
        avg_time = np.mean(y_data)
        std_dev = np.std(y_data)

        # Plot the raw batch timings (with transparency)
        ax.plot(x_data, y_data, label='Individual Batch Time', alpha=0.5)

        # Plot the average line
        ax.axhline(avg_time,color='red',linestyle='--',label=f'Average: {avg_time:.4f} s')

        # Plot the standard deviation as a shaded region
        ax.fill_between(x_data,avg_time - std_dev,avg_time + std_dev,color='red',alpha=0.2,label=f'Std Dev: {std_dev:.4f} s')

        # Add labels and title for this specific subplot
        ax.set_ylabel('Time per Batch (s)')
        ax.set_title(f'Run with num_workers={num_workers}')
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.6)

    # Set common X-label for the bottom-most plot
    axes[-1].set_xlabel('Batch Index (excluding first batch)')

    # Add a title to the figure
    fig.suptitle(
        f'DataLoader Speed Test (batch_size={batch_size})\n'
        f'(Note: 0.1s artificial wait time subtracted from all batch times)',
        fontsize=16,
        y=1.02  # Adjust position to be above subplots
    )

    plt.tight_layout()
    # Adjust layout
    fig.subplots_adjust(top=0.94)

    plt.savefig(f'{results_filepath}.png', dpi=300, bbox_inches='tight')


if __name__ == '__main__':
    dataset_loading_times()