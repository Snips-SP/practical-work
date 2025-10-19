import os
import pickle
import random
import numpy as np
from matplotlib import pyplot as plt
from torch.optim import AdamW
from transformers import AutoTokenizer, Phi3Config, Phi3ForCausalLM
from datasets import load_dataset
from tqdm import tqdm
import time
import warnings
import torch


def run_training_test(attention_implementation, compile=False, torch_dtype=torch.float32, seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.xpu.is_available():
        torch.xpu.manual_seed(seed)

    print(f'\n--- Running test for "{attention_implementation}" attention with compile={compile} and dtype={torch_dtype}---')

    # --- Setup Environment and Parameters ---
    num_epochs = 2
    batch_size = 32
    model_id = 'microsoft/Phi-3-mini-4k-instruct'
    device = 'xpu'

    # Use appropriate gpu or cpu
    print(f'Using device: {device}, dtype: {torch_dtype}')

    # --- Load Tokenizer and Prepare Dataset ---
    print('Loading tokenizer and preparing dataset...')
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Using a standard dummy dataset for simplicity and reproducibility
    raw_dataset = load_dataset('dair-ai/emotion', split='train')

    # We only need a small subset for a quick benchmark
    subset_dataset = raw_dataset.select(range(2048))

    def tokenize_function(examples):
        return tokenizer(
            examples['text'],
            truncation=True,
            padding='max_length',
            max_length=256,
            return_tensors='pt'
        )

    tokenized_dataset = subset_dataset.map(tokenize_function, batched=True, remove_columns=['text', 'label'])
    tokenized_dataset.set_format(type='torch')

    # --- 3. Configure and Instantiate the Model ---
    print(f'Configuring model with {attention_implementation}...')
    config = Phi3Config(
        # Using a smaller configuration for faster local testing
        hidden_size=768,
        num_hidden_layers=12,
        intermediate_size=2048,
        num_attention_heads=12,
        num_key_value_heads=4,
        max_position_embeddings=2048,
        rope_theta=10000.0,
        vocab_size=len(tokenizer),
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        hidden_act='silu',
        initializer_range=0.02,
        layer_norm_eps=1e-5,
        tie_word_embeddings=False,
        attention_implementation='sdpa',
    )

    model = Phi3ForCausalLM(config)
    model.to(device, dtype=torch_dtype)

    if compile:
        print('Compiling model...')
        start_time = time.time()
        model = torch.compile(model)
        print(f'Model compiled successfully. It took: {time.time() - start_time:.2f} seconds to compile.')

    # --- Training Loop ---
    print('Starting training loop...')
    model.train()
    optimizer = AdamW(model.parameters(), lr=5e-5)

    # Use a DataLoader for proper batching
    train_loader = torch.utils.data.DataLoader(tokenized_dataset, batch_size=batch_size, shuffle=True)

    num_training_steps = num_epochs * len(train_loader)

    total_loss = 0.0
    total_times = []
    with tqdm(range(num_training_steps), desc=f'Training ({attention_implementation})') as progress_bar:
        for epoch in range(num_epochs):
            for batch in train_loader:
                # Move batch to device
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)

                # Forward pass
                start_time = time.time()
                optimizer.zero_grad()

                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=input_ids
                )

                loss = outputs.loss

                if torch.isnan(loss):
                    print('NaN loss detected! Stopping training.')
                    return float('inf')

                # Backward pass and optimization
                loss.backward()

                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                # Time the forward and backward pass of every batch
                end_time = time.time()
                total_times.append(end_time - start_time)

                total_loss += loss.item()

                progress_bar.update(1)
                progress_bar.set_postfix({'Loss': f'{loss.item():.4f}'})

    print(f'Training for "{attention_implementation}" completed in {sum(total_times):.2f} seconds, with a total loss of {total_loss/len(train_loader):.4f}.')

    # --- Quick Inference Test ---
    prompt = 'I feel happy because'
    inputs = tokenizer(prompt, return_tensors='pt').to(device)
    input_ids = inputs.input_ids
    attention_mask = inputs.attention_mask

    print('\n--- Inference Test ---')
    print(f'Prompt: "{prompt}"')

    # Generate text
    output = model.generate(input_ids, attention_mask=attention_mask, max_length=50, num_return_sequences=1, do_sample=True, top_k=0, top_p=0.95)
    decoded_output = tokenizer.decode(output[0], skip_special_tokens=True)

    print(f'Generated Text: {decoded_output}')
    print('-----------------------\n')

    return total_times


if __name__ == '__main__':
    # Run the benchmark for both implementations
    try:
        results_filename = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'plots', 'training_benchmark_results.pkl')
        all_results = {}

        # Check if the results file already exists
        if os.path.exists(results_filename):
            print(f'Found existing results file: "{results_filename}". Loading data...')
            with open(results_filename, 'rb') as f:
                all_results = pickle.load(f)
            print('Data loaded successfully.')

        else:
            # If the file doesn't exist, run all the tests
            print('No results file found. Running benchmark tests...')

            # Run all 8 configurations
            eager_times = run_training_test('eager', False, torch.float32)
            sdpa_times = run_training_test('sdpa', False, torch.float32)
            eager_times_bfloat16 = run_training_test('eager', False, torch.bfloat16)
            sdpa_times_bfloat16 = run_training_test('sdpa', False, torch.bfloat16)
            eager_times_compiled = run_training_test('eager', True, torch.float32)
            sdpa_times_compiled = run_training_test('sdpa', True, torch.float32)
            eager_times_compiled_bfloat16 = run_training_test('eager', True, torch.bfloat16)
            sdpa_times_compiled_bfloat16 = run_training_test('sdpa', True, torch.bfloat16)

            all_results = {
                'Eager': eager_times,
                'SDPA': sdpa_times,
                'Eager bfloat16': eager_times_bfloat16,
                'SDPA bfloat16': sdpa_times_bfloat16,
                'Eager Compiled': eager_times_compiled,
                'SDPA Compiled': sdpa_times_compiled,
                'Eager Compiled bfloat16': eager_times_compiled_bfloat16,
                'SDPA Compiled bfloat16': sdpa_times_compiled_bfloat16,
            }

            # Save the newly generated results to a file
            print(f'\nTests complete. Saving results to "{results_filename}"...')
            with open(results_filename, 'wb') as f:
                pickle.dump(all_results, f)
            print('Results saved successfully.')

        # Separate the first data point (startup) from the rest (steady-state)
        startup_times = {name: times[0] for name, times in all_results.items()}
        steady_state_times = {name: times[1:] for name, times in all_results.items()}
        steps = range(len(list(steady_state_times.values())[0]))

        def plot_with_avg(ax, x_data, y_data, label, avg_fraction=0.2, **kwargs):
            """
            Plots a line and adds a point at the end representing the average of the last
            portion of the data.
            """
            # Plot the full line
            line, = ax.plot(x_data, y_data, label=label, **kwargs)

            # Calculate the average of the last fraction of the sequence
            num_points = len(y_data)
            # Ensure we have points to average
            if num_points > 0:
                avg_start_index = int(num_points * (1 - avg_fraction))
                avg_value = np.mean(y_data[avg_start_index:])

                # Get the coordinate for the last point
                last_x_point = x_data[-1]

                # Plot the average point on top of the line
                ax.scatter(last_x_point, avg_value, color=line.get_color(),
                           s=100, zorder=5, edgecolor='black', linewidth=1.5)

                # Add a text label for the average value
                ax.text(last_x_point + 3, avg_value, f'{avg_value:.3f}s',
                        va='center', fontsize=9,
                        bbox=dict(facecolor='white', alpha=0.6, edgecolor='none', boxstyle='round,pad=0.2'))


        # --- Create the 2x2 Subplots ---
        fig, axes = plt.subplots(2, 2, figsize=(18, 14))
        fig.suptitle('Analysis of Training Performance', fontsize=20)

        # Plot Eager Implementations
        ax = axes[0, 0]
        plot_with_avg(ax, steps, steady_state_times['Eager'], label='Eager (FP32)')
        plot_with_avg(ax, steps, steady_state_times['Eager bfloat16'], label='Eager (bfloat16)')
        plot_with_avg(ax, steps, steady_state_times['Eager Compiled'], label='Eager Compiled (FP32)')
        plot_with_avg(ax, steps, steady_state_times['Eager Compiled bfloat16'], label='Eager Compiled (bfloat16)')
        ax.set_title('Eager Implementations (Steady-State)', fontsize=14)
        ax.set_ylabel('Time per Batch (seconds)')
        ax.legend()

        # Plot SDPA Implementations
        ax = axes[0, 1]
        plot_with_avg(ax, steps, steady_state_times['SDPA'], label='SDPA (FP32)')
        plot_with_avg(ax, steps, steady_state_times['SDPA bfloat16'], label='SDPA (bfloat16)')
        plot_with_avg(ax, steps, steady_state_times['SDPA Compiled'], label='SDPA Compiled (FP32)')
        plot_with_avg(ax, steps, steady_state_times['SDPA Compiled bfloat16'], label='SDPA Compiled (bfloat16)')
        ax.set_title('SDPA Implementations (Steady-State)', fontsize=14)
        ax.legend()

        # Plot Top Performers
        ax = axes[1, 0]
        plot_with_avg(ax, steps, steady_state_times['Eager Compiled bfloat16'], label='Eager Compiled (bfloat16)')
        plot_with_avg(ax, steps, steady_state_times['SDPA Compiled bfloat16'], label='SDPA Compiled (bfloat16)')
        ax.set_title('Top Performers Comparison (Steady-State)', fontsize=14)
        ax.set_xlabel('Training Step (after first batch)')
        ax.set_ylabel('Time per Batch (seconds)')
        ax.legend()

        # Plot Startup Time Comparison (Bar Chart)
        ax = axes[1, 1]
        top_performers_startup = {
            'Eager Comp bfloat16': startup_times['Eager Compiled bfloat16'],
            'SDPA Comp bfloat16': startup_times['SDPA Compiled bfloat16'],
            'Eager FP32': startup_times['Eager'],
            'SDPA FP32': startup_times['SDPA'],
        }
        bars = ax.bar(top_performers_startup.keys(), top_performers_startup.values(),
                      color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
        ax.set_title('Startup Time (First Batch)', fontsize=14)
        ax.set_xlabel('Implementation')
        ax.set_ylabel('Time (seconds)')
        ax.bar_label(bars, fmt='%.2f')

        # --- Finalize and Save ---
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        # plt.show()
        plt.savefig(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'plots', 'training_benchmark_results.png'), dpi=300)

    except Exception as e:
        print(f'\nAn error occurred: {e}')

