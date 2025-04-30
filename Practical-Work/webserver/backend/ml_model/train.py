import re
from .dataloader import GPT2Dataset, GPT2RAMDataset
from transformers import GPT2LMHeadModel, GPT2Config, get_scheduler
from tqdm import tqdm
from torch.optim import AdamW
import torch
import os
from torch.utils.tensorboard import SummaryWriter
import math
from torch.nn import CrossEntropyLoss


class EncodingConfig:
    # List of all used tokens (excluding padding token)
    tokens: list = []
    # Token used for padding
    padding_token: int = None
    # The length of all tokens (tokens and padding)
    vocab_size: int = None

    # All the instruments which are used in our encoding
    tracks = ['Drums', 'Piano', 'Guitar', 'Bass', 'Strings']
    programs = {
        'Drums': 0,  # Program is not important here since we set the flag is_drum when creating a track
        'Piano': 0,  # Program for Acoustic Grand Piano
        'Guitar': 32,  # Program for 32 for Electric Guitar
        'Bass': 35,  # Program for Electric Bass (picked)
        'Strings': 49  # Program for String Ensemble 1
    }

    # The offsets between the instruments and range of notes
    note_size: int = 84
    note_offset: int = 24

    # Tokens for time note and end note
    time_note: int = None
    end_note: int = None

    # List with prioritised index where Bass is placed in front
    trc_idx: list = None

    @classmethod
    def initialize(cls):
        if not cls.tokens:  # Prevent re-initialization
            # Bass: [0 * 84, 0 * 84 + 83] = [0, 83]
            # Drums: [1 * 84, 1 * 84 + 83] = [84, 167]
            # Piano: [2 * 84, 2 * 84 + 83] = [168, 251]
            # Guitar: [3 * 84, 3 * 84 + 83] = [252, 335]
            # Strings: [4 * 84, 4 * 84 + 83] = [336, 419]
            cls.tokens.extend(range(0, 420))
            cls.time_note = cls.tokens[-1] + 1
            cls.tokens.append(cls.time_note)  # Add the token which represents a pause in the music (420)
            cls.end_note = cls.tokens[-1] + 1
            cls.tokens.append(cls.end_note)  # Add the token which represents the end of the sequence (421)
            cls.padding_token = cls.tokens[-1] + 1  # Add the padding token to the mix (422)
            cls.vocab_size = cls.padding_token + 1  # We need to add one to the total size since 0 is included

            cls.trc_idx = sorted(list(range(len(cls.tracks))), key=lambda x: 0 if cls.tracks[x] == 'Bass' else 1)


EncodingConfig.initialize()


class NetworkConfig:
    # All the instruments which are used in our encoding
    config = GPT2Config(
        vocab_size=EncodingConfig.vocab_size,  # 423
        n_positions=1024,  # Maximum sequence length
        n_ctx=256,  # Context window size
        n_embd=256,  # Embedding size
        n_layer=2,  # Number of transformer layers
        n_head=2,  # Number of attention heads
        pad_token_id=EncodingConfig.padding_token,  # 422
    )


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


def get_latest_checkpoint(directory, name):
    # Define a regex pattern to extract the epoch number
    pattern = re.compile(rf'{name}(\d+)\.ph')

    latest_epoch = -1
    latest_file = None

    # Iterate through files in the directory
    for filename in os.listdir(directory):
        match = pattern.match(filename)
        if match:
            epoch = int(match.group(1))  # Extract epoch number
            if epoch > latest_epoch:
                latest_epoch = epoch
                latest_file = filename

    if latest_file:
        return os.path.join(directory, latest_file)
    else:
        return None  # No valid checkpoint found


def train(continue_from: str = None):
    # We assume the root path is the current script path
    root_path = os.path.dirname(os.path.abspath(__file__))
    # Set training parameters
    file_name = 'gpt_model_state_dict_epoch_'
    num_epochs = 2
    batch_size = 16

    print(f'Training for {num_epochs} epochs with batch size {batch_size}.')

    # Use appropriate gpu or cpu
    device = ('xpu' if torch.xpu.is_available() else
              'cuda' if torch.cuda.is_available() else
              'cpu')
    print('Using device:', device)

    # Instantiate GPT-2 model
    model = GPT2LMHeadModel(NetworkConfig.config)

    # Define a function to initialize weights
    def init_weights(module):
        if isinstance(module, torch.nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, torch.nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    # Apply the weight initialization
    model.apply(init_weights)

    # Search for ph file to continue training from this file
    if continue_from is not None:
        if os.path.isdir(continue_from):
            model_path = get_latest_checkpoint(continue_from, file_name)
            print(f'Continuing from directory: {continue_from}')
            print(f'With state dict: {model_path}')

            model.load_state_dict(torch.load(model_path, weights_only=True, map_location=device))
        else:
            raise FileNotFoundError('Directory for loading cannot be found')

    # Get dataset and dataloader
    dataset = GPT2RAMDataset(os.path.join(root_path, 'ldp_5_dataset'))

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    log_dir = get_next_run_folder('GPT2_Model', base_dir=os.path.join(root_path, 'runs'))

    # Create the directory if it doesn't exist
    os.makedirs(log_dir, exist_ok=True)

    # Initialize SummaryWriter with the new log directory
    writer = SummaryWriter(log_dir=log_dir)
    print(f'Logging to: {log_dir}')

    # Training loop
    num_training_steps = num_epochs * len(dataloader)
    progress_bar = tqdm(range(num_training_steps), desc='Training Progress')

    # Make adjustment to the model
    model.train()
    model.to(device)

    # Define loss function
    loss_fn = CrossEntropyLoss(ignore_index=EncodingConfig.padding_token)

    # Define optimizer and learning rate scheduler
    optimizer = AdamW(model.parameters(), lr=1e-4)

    # Compile model for additional training speed
    # Torch compile uses the triton backend, which I have not installed.
    # As it turns out its easy to install via pip, but not for intel arc gpus.
    # I will have to dual boot my pc into ubuntu 22.04 in order to install the intel xpu backend for triton.
    # As I like the pycharm environment and I am used to Windows pcs,
    # I will set up ubuntu server and use it as a remote development server and access it via my laptop.
    # This is not the first time I have done this. When it works it works greate,
    # but it takes a lot of time to get running.
    # model = torch.compile(model)

    # Disable caching as it conflicts with gradient_checkpointing
    # model.config.use_cache = False

    # Enable memory optimizations (we can get away with less memory)
    # model.gradient_checkpointing_enable()

    # Cosine Annealing with Warmup as learning rate scheduler
    lr_scheduler = get_scheduler(
        'cosine',
        optimizer=optimizer,
        num_warmup_steps=500,
        num_training_steps=num_training_steps
    )

    train_loss = []
    for epoch in range(num_epochs):
        total_loss = 0
        for batch_idx, batch in enumerate(dataloader):
            input_ids = batch[0].to(device).long()
            # We dont need padding since our data is always 1024 tokens long
            # attention_mask = batch[1].to(device).long()

            # Zero gradients before the backward pass (best practice for pytorch)
            optimizer.zero_grad()

            # If we only give the inputs and not the labels the hugging face model will not calculate
            # the loss on its own, so we can use our own loss function
            outputs = model(input_ids=input_ids)

            # Remove last timestep, because it does not predict anything
            shift_logits = outputs.logits[..., :-1, :].contiguous()
            # Remove the first timestep because it does not have a previous to predict
            shift_labels = input_ids[..., 1:].contiguous()

            loss = loss_fn(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

            # Check if there are any NaN values
            if torch.isnan(outputs.logits).any():
                print('WARNING: NaN detected in logits!')
                raise Exception

            if torch.isnan(loss):
                print('WARNING: NaN detected in loss!')
                raise Exception

            # Backward pass
            loss.backward()

            # Gradient Clipping to prevent exploding gradients
            total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)

            if total_norm > 1e6:
                print('WARNING: Exploding gradients detected!')
                raise Exception

            # Optimizer step
            optimizer.step()

            # Update learning rate
            lr_scheduler.step()

            # Log some statistics
            detached_loss = loss.detach().cpu().item()

            total_loss += detached_loss
            global_step = epoch * len(dataloader) + batch_idx
            writer.add_scalar('Training Loss', detached_loss, global_step)
            writer.add_scalar('Learning Rate', lr_scheduler.get_last_lr()[0], global_step)
            writer.add_scalar('Gradient Norm', total_norm, global_step)
            # Add if statement to prevent numerical overflow
            perplexity = math.exp(detached_loss) if detached_loss < 20 else float('inf')
            writer.add_scalar('Perplexity', perplexity, global_step)

            progress_bar.set_postfix({
                'Batch': batch_idx,
                'Loss': f'{detached_loss:.4f}',
                'LR': f'{lr_scheduler.get_last_lr()[0]:.6f}',
                'GradNorm': f'{total_norm:.2f}',
                'Perplexity': f'{perplexity:.2f}'
            })

            progress_bar.update(1)

        train_loss.append(total_loss / len(dataset))
        torch.save(model.state_dict(), os.path.join(log_dir, f'{file_name}{epoch}.ph'))

    print('Training completed!')
    writer.close()


if __name__ == '__main__':
    train('.')
