import time

import numpy as np
from tqdm import tqdm
import os

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
