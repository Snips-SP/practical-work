from backend.ml_model.dataloader import GPT2Dataset, GPT2RAMDataset
from transformers import GPT2LMHeadModel, GPT2Config, get_scheduler
import torch
from torch.optim import AdamW
from torch.utils.tensorboard import SummaryWriter
from torch.nn import CrossEntropyLoss
from tqdm import tqdm
import argparse
import re
import math
import os


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
        n_layer=4,  # Number of transformer layers
        n_head=4,  # Number of attention heads
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


def get_device(preferred: str = None) -> str:
    # Normalize and check if a preferred device was given
    preferred = preferred.lower() if preferred else None

    # Check if the preferred device is available
    if preferred == 'xpu' and hasattr(torch, 'xpu') and torch.xpu.is_available():
        return 'xpu'
    elif preferred == 'cuda' and torch.cuda.is_available():
        return 'cuda'
    elif preferred == 'cpu':
        return 'cpu'

    # Fallback to automatic selection
    if hasattr(torch, 'xpu') and torch.xpu.is_available():
        return 'xpu'
    elif torch.cuda.is_available():
        return 'cuda'
    else:
        return 'cpu'


def train(num_epochs: int,
          batch_size: int = 16,
          learning_rate: float = 1e-4,
          lr_scheduler: str = 'cosine',
          gradient_checkpointing: bool = False,
          RAM_dataset: bool = False,
          device: str = None,
          continue_from: str = None,
          config=None):
    # We assume the root path is the current script path
    root_path = os.path.dirname(os.path.abspath(__file__))
    # Name for state directories
    state_dict_file_name = 'gpt_model_state_dict_epoch_'

    # Use referred device from user or chose it automatically
    device = get_device(device)
    print('Using device:', device)

    # Search for config and ph file to continue training from this file

    if continue_from is not None:
        continue_from = os.path.join(root_path, continue_from)
        # Check if all relevant files and folders exist
        if not os.path.isdir(continue_from):
            raise FileNotFoundError('Directory for loading cannot be found.')

        model_path = get_latest_checkpoint(continue_from, state_dict_file_name)
        config_path = os.path.join(continue_from, f'config.json')

        if model_path is None:
            raise FileNotFoundError('No state dictionary not found in folder.')
        if not os.path.exists(config_path):
            raise FileNotFoundError('No config file found in folder.')

        print(f'Continuing from directory: {continue_from}')
        print(f'With state dict: {model_path}')
        print(f'And config from: {config_path}')

        # Load config
        config = GPT2Config.from_json_file(config_path)
        # Create model from loaded configuration
        model = GPT2LMHeadModel(config)
        # Load model weights
        model.load_state_dict(torch.load(model_path, weights_only=True, map_location=device))
    else:
        print(f'Training from scratch.')

        if config is None:
            config = NetworkConfig.config

        # Instantiate GPT-2 model
        model = GPT2LMHeadModel(config)

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

    # Print out relevant gpt2 configurations
    print('GPT2 config: ')
    for key in ['n_positions', 'n_ctx', 'n_embd', 'n_layer', 'n_head']:
        print(f'   {key} = {config.to_dict()[key]}')
    print('')

    # Get dataset and dataloader
    if RAM_dataset:
        dataset = GPT2RAMDataset(os.path.join(root_path, 'ldp_5_dataset'))
    else:
        dataset = GPT2Dataset(os.path.join(root_path, 'ldp_5_dataset'))
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    log_dir = get_next_run_folder('GPT2_Model', base_dir=os.path.join(root_path, 'runs'))

    # Create the directory if it doesn't exist
    os.makedirs(log_dir, exist_ok=True)

    # Save current config into new log directory (it will be named config.json)
    config.save_pretrained(log_dir)

    # Initialize SummaryWriter with the new log directory
    writer = SummaryWriter(log_dir=log_dir)
    print(f'Logging to: {log_dir}')

    # Define progress bar
    num_training_steps = num_epochs * len(dataloader)

    # Set model to train and move it to device
    model.train()
    model.to(device)

    # Define loss function
    loss_fn = CrossEntropyLoss(ignore_index=EncodingConfig.padding_token)

    # Define optimizer and learning rate scheduler
    optimizer = AdamW(model.parameters(), lr=learning_rate)

    # Compile model for additional training speed
    # Torch compile uses the triton backend, which I have not installed.
    # As it turns out its easy to install via pip, but not for intel arc gpus.
    # I will have to dual boot my pc into ubuntu 22.04 in order to install the intel xpu backend for triton.
    # As I like the pycharm environment and I am used to Windows pcs,
    # I will set up ubuntu server and use it as a remote development server and access it via my laptop.
    # This is not the first time I have done this. When it works it works greate,
    # but it takes a lot of time to get running.
    # model = torch.compile(model)

    if gradient_checkpointing:
        # Enable memory optimizations (we can get away with less memory)
        model.gradient_checkpointing_enable()

        # Disable caching as it conflicts with gradient_checkpointing
        model.config.use_cache = False

    if lr_scheduler == 'cosine':
        # Cosine Annealing with Warmup as learning rate scheduler
        lr_scheduler = get_scheduler(
            'cosine',
            optimizer=optimizer,
            num_warmup_steps=500,
            num_training_steps=num_training_steps
        )
    else:
        raise NotImplemented(f'Learning rate scheduler {lr_scheduler} not implemented.')

    # Start training
    print(f'Training for {num_epochs} epochs with batch size {batch_size}.')
    print(' ')
    progress_bar = tqdm(range(num_training_steps), desc='Training Progress')
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

            if lr_scheduler is not None:
                # Update learning rate
                lr_scheduler.step()

            # Calculate and log stats
            detached_loss = loss.detach().cpu().item()

            total_loss += detached_loss
            global_step = epoch * len(dataloader) + batch_idx
            writer.add_scalar('Training Loss', detached_loss, global_step)
            if lr_scheduler is not None:
                writer.add_scalar('Learning Rate', lr_scheduler.get_last_lr()[0], global_step)
            writer.add_scalar('Gradient Norm', total_norm, global_step)
            # Add if statement to prevent numerical overflow
            perplexity = math.exp(detached_loss) if detached_loss < 20 else float('inf')
            writer.add_scalar('Perplexity', perplexity, global_step)

            display_stats = {
                'Batch': batch_idx,
                'Loss': f'{detached_loss:.4f}',
                'GradNorm': f'{total_norm:.2f}',
                'Perplexity': f'{perplexity:.2f}'
            }
            if lr_scheduler is not None:
                display_stats['LR'] = f'{lr_scheduler.get_last_lr()[0]:.6f}'

            # Display stats in progress bar
            progress_bar.set_postfix(display_stats)
            progress_bar.update(1)

        # Log total train loss over epoch
        train_loss.append(total_loss / len(dataset))
        writer.add_scalar('Total Training Loss', total_loss / len(dataset), global_step)
        # Save state dict after each epoch
        torch.save(model.state_dict(), os.path.join(log_dir, f'{state_dict_file_name}{epoch}.ph'))

    print('Training completed!')
    writer.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_epochs', type=int, required=True, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate for optimizer')
    parser.add_argument('--lr_scheduler', type=str, default='cosine', choices=['cosine'], help='Learning rate scheduler type')
    parser.add_argument('--gradient_checkpointing', action='store_true', help='Enable gradient checkpointing to save memory')
    parser.add_argument('--RAM_dataset', action='store_true', help='Load entire dataset into memory')
    parser.add_argument('--device', type=str, default=None, help='Device to use: "cpu", "cuda", or "xpu" (auto-select if None)')
    parser.add_argument('--continue_from', type=str, default=None, help='Path to a directory to continue training from a checkpoint')

    args = parser.parse_args()
    
    if True:
        # Train with same parameters but different configs
        for c_path in ['runs/GPT2_Model_6', 'runs/GPT2_Model_7', 'runs/GPT2_Model_8']:
            train(num_epochs=args.num_epochs,
                  batch_size=args.batch_size,
                  learning_rate=args.learning_rate,
                  lr_scheduler=args.lr_scheduler,
                  gradient_checkpointing=args.gradient_checkpointing,
                  RAM_dataset=args.RAM_dataset,
                  device=args.device,
                  continue_from=c_path)
    else:
        config1 = GPT2Config(
            vocab_size=EncodingConfig.vocab_size,  # 423
            n_positions=1024,  # Maximum sequence length
            n_ctx=256,  # Context window size
            n_embd=256,  # Embedding size
            n_layer=2,  # Number of transformer layers
            n_head=2,  # Number of attention heads
            pad_token_id=EncodingConfig.padding_token,  # 422
        )

        config2 = GPT2Config(
            vocab_size=EncodingConfig.vocab_size,  # 423
            n_positions=1024,  # Maximum sequence length
            n_ctx=256,  # Context window size
            n_embd=256,  # Embedding size
            n_layer=4,  # Number of transformer layers
            n_head=4,  # Number of attention heads
            pad_token_id=EncodingConfig.padding_token,  # 422
        )

        config3 = GPT2Config(
            vocab_size=EncodingConfig.vocab_size,  # 423
            n_positions=1024,  # Maximum sequence length
            n_ctx=256,  # Context window size
            n_embd=240,  # Embedding size
            n_layer=6,  # Number of transformer layers
            n_head=6,  # Number of attention heads
            pad_token_id=EncodingConfig.padding_token,  # 422
        )

        # Train with same parameters but different configs
        for config in [config1, config2, config3]:
            train(num_epochs=args.num_epochs,
                  batch_size=args.batch_size,
                  learning_rate=args.learning_rate,
                  lr_scheduler=args.lr_scheduler,
                  gradient_checkpointing=args.gradient_checkpointing,
                  RAM_dataset=args.RAM_dataset,
                  device=args.device,
                  continue_from=args.continue_from,
                  config=config)

