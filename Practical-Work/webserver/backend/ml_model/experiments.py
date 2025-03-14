from .helper import EncodingConfig, mid_to_mp3
from .generate import generate_from_chords, sliding_window_generate, generate_from_context
from .dataloader import GPT2Dataset
import numpy as np
from tqdm import tqdm
import os
import torch
import time
import matplotlib
import matplotlib.pyplot as plt
from transformers import GPT2LMHeadModel, GPT2Config
from .train import NetworkConfig
from .helper import chord2tokens
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from datasets import Dataset
from torch.optim import AdamW

EncodingConfig.initialize()

matplotlib.use('TkAgg')


def training_test():
    # Set training parameters
    num_epochs = 1

    # Use appropriate gpu or cpu
    device = ('xpu' if torch.xpu.is_available() else
              'cuda' if torch.cuda.is_available() else
              'cpu')
    print('Using device:', device)

    # Dummy dataset: Repeating a simple sentence
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    text_data = ['Hello world! This is a test sentence.'] * 1000
    tokenized_data = tokenizer(text_data, truncation=True, padding='max_length', max_length=20, return_tensors='pt')

    # Convert to Hugging Face Dataset format
    dataset = Dataset.from_dict(
        {'input_ids': tokenized_data['input_ids'], 'attention_mask': tokenized_data['attention_mask']})

    # Instantiate GPT-2 model
    config = NetworkConfig.config
    config.vocab_size = tokenizer.vocab_size
    model = GPT2LMHeadModel(config)

    # Training loop
    num_training_steps = num_epochs * len(dataset)
    progress_bar = tqdm(range(num_training_steps), desc='Training Progress')

    # Make adjustment to the model
    model.train()
    model.to(device)

    # Set right padding token
    model.config.pad_token_id = EncodingConfig.padding_token

    # Define optimizer and learning rate scheduler
    optimizer = AdamW(model.parameters(), lr=5e-6)

    train_loss = []
    for epoch in range(num_epochs):
        total_loss = 0
        for batch_idx, batch in enumerate(dataset):
            input_ids = torch.tensor(batch['input_ids'], device=device).long()
            attention_mask = torch.tensor(batch['attention_mask'], device=device).long()

            outputs = model(input_ids=input_ids,
                            attention_mask=attention_mask,
                            labels=input_ids)

            invalid_tokens = (input_ids >= model.config.vocab_size).any()
            if invalid_tokens:
                print('WARNING: Input contains out-of-range token IDs!')

            # Check if there are any NaN values
            if torch.isnan(outputs.logits).any():
                print('NaN detected in logits!')

            # Zero gradients before the backward pass (best practice for pytorch)
            optimizer.zero_grad()

            # GPT-2 directly computes the loss if labels are provided
            loss = outputs.loss

            if torch.isnan(loss):
                print('NaN detected in loss!')

            for name, param in model.named_parameters():
                if param.grad is not None and torch.isnan(param.grad).any():
                    print(f'NaN detected in gradients of {name}')

            # Backward pass
            loss.backward()

            # Gradient Clipping to prevent exploding gradients
            total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)

            # Optimizer step
            optimizer.step()

            # Log some statistics
            detached_loss = loss.detach().cpu().item()

            total_loss += detached_loss

            if detached_loss > 100:
                raise Exception('Loss became to large!!!')

            progress_bar.set_postfix({
                'Loss': f'{detached_loss:.4f}',
            })

            progress_bar.update(1)
        train_loss.append(total_loss / len(dataset))

    # Test the network if it learned our dummy dataset
    prompt = 'Hello'
    input_ids = tokenizer(prompt, return_tensors='pt').input_ids.to(device)

    output = model.generate(input_ids, max_length=1000, num_return_sequences=1)
    print(f'Test Prompt: {prompt}')
    print(tokenizer.decode(output[0], skip_special_tokens=True))
    print('Training completed!')


def dataloader_test():
    dataset = GPT2Dataset(os.path.join('backend', 'ml_model', 'ldp_5_dataset'))

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=192,  # Number of samples per batch
        shuffle=False,  # This would fuck up our preloading
        num_workers=0,  # This would fuck up our preloading as well...
    )

    for epoch in range(4):
        total_loss = 0
        for batch_idx, batch in enumerate(dataloader):
            print(f'batch idx: {batch_idx}')


def testing_generation():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    device = 'cpu'

    model = GPT2LMHeadModel(NetworkConfig.config)
    model.load_state_dict(torch.load(os.path.join(script_dir, 'tmp', 'gpt_model_state_dict_0.ph'), weights_only=True,
                                     map_location=device))
    model.to(device)

    # Encode chords into tokens
    tokens = [chord2tokens(chord) for chord in ['A', 'D']]

    # Create empty pianoroll array
    pianoroll = np.zeros((len(EncodingConfig.tracks), np.sum([32, 32]), 128))

    # Retrieve first chord
    chord = tokens.pop(0)

    # Will be used as context for the neural network
    context_sequence = [EncodingConfig.end_note]
    context_sequence.extend(chord)

    input_ids = torch.tensor([context_sequence], dtype=torch.long)

    # Manually generating in order to inspect
    output = model(input_ids)
    logits = output.logits  # Extract logits

    print('Logits:', logits)  # Check for abnormal values

    probs = torch.nn.functional.softmax(logits[:, -1, :], dim=-1)  # Apply softmax
    print('Probabilities:', probs)  # Check if they sum to 1 and are valid

    next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)  # Sample tokens

    generated_sequence_1 = generate_from_context(model, context_sequence, device)

    generated_sequence_2 = sliding_window_generate(model, context_sequence, max_tokens=1024)


def testing_generation_function():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    mid_location = generate_from_chords(['A', 'D'],
                                        [32, 32],
                                        80,
                                        os.path.join(script_dir, 'tmp', 'gpt_model_state_dict_0.ph'),
                                        os.path.join(script_dir, 'tmp', 'output.mid'))
    print('gen fin')


def testing_conversion():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    mid_to_mp3(os.path.join(script_dir, 'tmp', 'output.mid'),
               os.path.join(script_dir, 'tmp', 'tmp/FluidR3_GM_GS.sf2'),
               os.path.join(script_dir, 'tmp', 'output.mp3'))
    print('convert fin')


def test_chunk_sizes():
    # Define the directory path
    directory_path = 'tmp'
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


def custom_vocabulary_test():
    EncodingConfig.initialize()

    tmp = EncodingConfig.tokens
    tmp = EncodingConfig.padding_token


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
