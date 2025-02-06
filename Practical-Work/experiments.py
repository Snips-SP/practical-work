import numpy as np
from tqdm import tqdm
import os
import torch
from dataloader import GPT2Dataset
import time
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')


def test_chunk_sizes():
    # Define the directory path
    directory_path = 'tmp'

    # Create the directory
    os.makedirs(directory_path, exist_ok=True)

    # Test different sizes of the chunks
    for size in [50000, 100000]:
        # Generate random data with tqdm progress bar
        optimized_chunks = []
        for _ in tqdm(range(size), desc='Generating random chunks'):
            optimized_chunks.append(np.random.randint(0, 1000, (4096,), dtype=np.uint16))

        # Save compressed .npz file
        np.savez_compressed(os.path.join(directory_path, f'{size}.npz'), *optimized_chunks)

        # Function to estimate memory usage
        def estimate_npz_memory_with_timing(file_path):
            # Time how long it will take to load the file into memory
            start_time = time.time()
            chunk = np.load(file_path)
            end_time = time.time()

            total_size = 0
            for key in tqdm(chunk.files, desc='Calculating memory usage'):
                array = chunk[key]
                total_size += array.nbytes  # Calculate size in bytes
            chunk.close()

            elapsed_time = end_time - start_time  # Calculate elapsed time

            print(f"Time taken to load and calculate memory usage: {elapsed_time:.2f} seconds")
            return total_size

        estimated_size = estimate_npz_memory_with_timing(os.path.join(directory_path, f'{size}.npz'))
        print(f'Estimated memory usage: {estimated_size / (1024 ** 2):.2f} MB')

    # 50000 yields around 390MB which seems to be a good compromise between memory usage and I/O activity
    # A load takes around 0.48 seconds

    # 100000 yields around 781MB
    # A load takes around 0.85 seconds


def dataloader_loading_times():
    dataset = GPT2Dataset('ldp_5_dataset')

    # List to store loading times
    loading_times = []

    # Iterate through the dataset and measure time
    for i in range(len(dataset)):
        start_time = time.perf_counter()  # Start timing

        sequence, mask = dataset.__getitem__(i)

        end_time = time.perf_counter()  # End timing

        loading_times.append(end_time - start_time)  # Store elapsed time

        del sequence, mask  # Free memory if needed

    # Plot the loading times
    plt.figure(figsize=(10, 5))
    plt.plot(loading_times, label="Load Time per Sample", marker="o", linestyle="-")
    plt.xlabel("Sample Index")
    plt.ylabel("Time (seconds)")
    plt.title("Dataset Loading Time Analysis")
    plt.legend()
    plt.grid(True)
    plt.show()

    # Its okay that we have the loading spikes at the times where the file switches.
    # In this simulated example we are not processing the sequence in the network
    # which means that the preloading is not even fast enough. But in real application it should work.


def dataloader_tests():
    dataset = GPT2Dataset('ldp_5_dataset')

    print(f'Length of dataset: {len(dataset)}')

    # Create a DataLoader from the dataset
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=8,  # Number of samples per batch
        shuffle=False,  # Shuffle the data
        num_workers=0,  # Number of subprocesses for data loading
    )
    count = 0
    # Iterate through the DataLoader
    for batch in dataloader:

        input_ids = batch[0]
        attention_mask = batch[1]
        count += len(input_ids)

        # print('Input IDs shape:', input_ids.shape)
        # print('Attention Mask shape:', attention_mask.shape)

        del input_ids
        del attention_mask

    print(f'Count: {count}')


dataloader_tests()