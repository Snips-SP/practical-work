from backend.ml_model.train import NetworkConfig, train
from backend.ml_model.generate import generate_from_chords, sliding_window_generate
from backend.ml_model.dataloader import GPT2Dataset
from backend.ml_model.helper import chord2tokens, mid_to_mp3, EncodingConfig
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from transformers import GPT2LMHeadModel, GPT2Config
import torch
import os
import time

EncodingConfig.initialize()


def test_chunk_sizes():
    # Define the directory path
    directory_path = '../tmp'
    sequence_length = 1024

    # Create the directory
    os.makedirs(directory_path, exist_ok=True)

    # Test different sizes of the chunks
    for size in [50_000, 100_000, 200_000, 400_000]:
        # Generate random data with tqdm progress bar
        optimized_chunks = np.random.randint(0, 400, (size, sequence_length), dtype=np.uint16)

        # Save compressed .npz file
        np.savez_compressed(os.path.join(directory_path, f'{size}_{sequence_length}.npz'), optimized_chunks)

        # Time how long it will take to load the file into memory
        start_time = time.perf_counter()
        chunk = np.load(os.path.join(directory_path, f'{size}_{sequence_length}.npz'))
        end_time = time.perf_counter()

        # Calculate size of the content of the chunk in bytes
        total_size = 0
        for key in tqdm(chunk.files, desc='Calculating memory usage'):
            array = chunk[key]
            total_size += array.nbytes
        chunk.close()

        # Calculate elapsed time
        elapsed_time = end_time - start_time

        print(f'Load time of chunk from disk: {elapsed_time} seconds')
        print(f'Estimated memory usage: {total_size / (1024 ** 2):.2f} MB')


def dataloader_loading_times():
    dataset = GPT2Dataset('ldp_5_dataset')

    # List to store loading times
    loading_times = []

    # Verify that our encoding works
    token_set = set()

    # Iterate through the dataset and measure time
    for i in range(len(dataset)):
        start_time = time.perf_counter()  # Start timing

        sequence, mask = dataset.__getitem__(i)

        end_time = time.perf_counter()  # End timing

        token_set.update(sequence)

        loading_times.append(end_time - start_time)  # Store elapsed time

        del sequence, mask  # Free memory if needed

    print(sorted(token_set))

    # Plot the loading times
    plt.figure(figsize=(10, 5))
    plt.plot(loading_times, label='Load Time per Sample', marker='o', linestyle='-')
    plt.xlabel('Sample Index')
    plt.ylabel('Time (seconds)')
    plt.title('Dataset Loading Time Analysis')
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


def calculate_model_memory_usage():
    vocabulary = EncodingConfig.tokens
    padding_token = EncodingConfig.padding_token

    print(f'Vocabulary size: {len(vocabulary)}')

    config = GPT2Config(
        vocab_size=len(vocabulary),  # Size of your vocabulary (adjust to match your tokenizer)
        n_positions=4096,  # Maximum sequence length
        n_ctx=1024,  # Context window size
        n_embd=768,  # Embedding size
        n_layer=12,  # Number of transformer layers
        n_head=12,  # Number of attention heads
        pad_token_id=padding_token,  # Set padding token ID (e.g., same as eos_token)
    )

    model = GPT2LMHeadModel(config)

    total_params = sum(p.numel() for p in model.parameters())  # Total number of parameters
    param_size = total_params * 4  # Assuming float32 (4 bytes per parameter)

    print(f'Model parameters size (estimation): {param_size / (1024 ** 2):.2f} MB')  # Convert to MB

    torch.xpu.empty_cache()  # Clear cache to get an accurate reading

    before_mem = torch.xpu.memory_allocated()
    model.to('xpu')
    after_mem = torch.xpu.memory_allocated()

    print(f'Model memory usage: {(after_mem - before_mem) / (1024 ** 2):.2f} MB')


if __name__ == '__main__':
    pass
    # get_dataset_statistics_per_track()
    # testing_generation_function(False, 'GPT2_Small_3')
