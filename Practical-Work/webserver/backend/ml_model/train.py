from .dataloader import GPT2Dataset, GPT2RAMDataset
from .helper import get_next_run_folder, EncodingConfig, get_latest_checkpoint
from transformers import GPT2LMHeadModel, GPT2Config, get_scheduler
from tqdm import tqdm
from torch.optim import AdamW
import torch
import os
from torch.utils.tensorboard import SummaryWriter
import math

EncodingConfig.initialize()

class NetworkConfig:
    # All the instruments which are used in our encoding
    config = GPT2Config(
        vocab_size=EncodingConfig.vocab_size,
        n_positions=1024,  # Maximum sequence length
        n_ctx=256,  # Context window size
        n_embd=256,  # Embedding size
        n_layer=2,  # Number of transformer layers
        n_head=2,  # Number of attention heads
        pad_token_id=EncodingConfig.padding_token,
    )


def train_simple(root_path):
    # Set training parameters
    name = 'gpt_model_state_dict_epoch_'
    num_epochs = 1
    batch_size = 100
    early_stopping = 1000

    print(f'Training simply for {num_epochs} epochs with batch size {batch_size}.')

    # Use appropriate gpu or cpu
    device = ('xpu' if torch.xpu.is_available() else
              'cuda' if torch.cuda.is_available() else
              'cpu')

    print('Using device:', device)

    # Instantiate GPT-2 model
    model = GPT2LMHeadModel(NetworkConfig.config)

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

    # Set right padding token
    model.config.pad_token_id = EncodingConfig.padding_token

    # Define optimizer and learning rate scheduler
    optimizer = AdamW(model.parameters(), lr=5e-6)

    train_loss = []
    for epoch in range(num_epochs):
        total_loss = 0
        for batch_idx, batch in enumerate(dataloader):
            input_ids = batch[0].to(device).long()
            attention_mask = batch[1].to(device).long()

            invalid_tokens = (input_ids >= model.config.vocab_size).any()
            if invalid_tokens:
                print('WARNING: Input contains out-of-range token IDs!')

            # Zero gradients before the backward pass (best practice for pytorch)
            optimizer.zero_grad()

            outputs = model(input_ids=input_ids,
                            attention_mask=attention_mask,
                            labels=input_ids)
            # GPT-2 directly computes the loss if labels are provided
            loss = outputs.loss

            # Check if there are any NaN values
            if torch.isnan(outputs.logits).any():
                print('WARNING: NaN detected in logits!')
                raise Exception

            if torch.isnan(loss):
                print('WARNING: NaN detected in loss!')
                raise Exception

            for name, param in model.named_parameters():
                if param.grad is not None and torch.isnan(param.grad).any():
                    print(f'WARNING: NaN detected in gradients of {name}')
                    raise Exception

            # Backward pass
            loss.backward()

            # Gradient Clipping to prevent exploding gradients
            total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)

            # Optimizer step
            optimizer.step()

            # Log some statistics
            detached_loss = loss.detach().cpu().item()

            total_loss += detached_loss
            global_step = epoch * len(dataloader) + batch_idx
            writer.add_scalar('Training Loss', detached_loss, global_step)
            # writer.add_scalar('Learning Rate', lr_scheduler.get_last_lr()[0], global_step)
            writer.add_scalar('Gradient Norm', total_norm, global_step)
            # Add if statement to prevent numerical overflow
            perplexity = math.exp(detached_loss) if detached_loss < 20 else float('inf')
            writer.add_scalar('Perplexity', perplexity, global_step)

            progress_bar.set_postfix({
                'Batch': batch_idx,
                'Loss': f'{detached_loss:.4f}',
                # 'LR': f'{lr_scheduler.get_last_lr()[0]:.6f}',
                'GradNorm': f'{total_norm:.2f}',
                'Perplexity': f'{perplexity:.2f}'
            })

            progress_bar.update(1)

        train_loss.append(total_loss / len(dataset))
        torch.save(model.state_dict(), os.path.join(log_dir, f'{name}{epoch}.ph'))

    print('Training completed!')
    writer.close()


def train(root_path, continue_from=None):
    # Set training parameters
    name = 'gpt_model_state_dict_epoch_'
    num_epochs = 1
    batch_size = 2

    print(f'Training for {num_epochs} epochs with batch size {batch_size}.')

    # Use appropriate gpu or cpu
    device = ('xpu' if torch.xpu.is_available() else
              'cuda' if torch.cuda.is_available() else
              'cpu')
    device = 'cpu'

    print('Using device:', device)

    # Instantiate GPT-2 model
    model = GPT2LMHeadModel(NetworkConfig.config)

    if continue_from is not None:
        if os.path.isdir(continue_from):
            model_path = get_latest_checkpoint(continue_from, name)
            print(f'Continuing from directory: {continue_from}')
            print(f'With state dict: {model_path}')

            model.load_state_dict(torch.load(model_path, weights_only=True, map_location=device))
        else:
            raise FileNotFoundError('Directory for loading cannot be found')

    # Get dataset and dataloader
    dataset = GPT2RAMDataset(os.path.join(root_path, 'ldp_5_dataset'))

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,  # Number of samples per batch
        shuffle=False,  # This would fuck up our preloading
        num_workers=0,  # This would fuck up our preloading as well...
    )

    # Create tensorboard logger in a new folder, so I have everything logged everytime,
    # since I often forget, and then it writes multiple runs into one folder which is a pain to separate.
    # Get the new folder path
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
    model.config.use_cache = False

    # Enable memory optimizations (we can get away with less memory)
    model.gradient_checkpointing_enable()

    # Define optimizer and learning rate scheduler
    optimizer = AdamW(model.parameters(), lr=5e-4)

    # Cosine Annealing with Warmup as learning rate scheduler
    lr_scheduler = get_scheduler(
        'cosine', optimizer=optimizer, num_warmup_steps=500, num_training_steps=num_training_steps
    )

    train_loss = []
    for epoch in range(num_epochs):
        total_loss = 0
        for batch_idx, batch in enumerate(dataloader):
            input_ids = batch[0].to(device).long()
            attention_mask = batch[1].to(device).long()
            # Create the labels which are just the inputs shifted to the right with a padding token at the end
            labels = torch.cat(
                [input_ids[:, 1:],
                 torch.full((len(input_ids), 1), EncodingConfig.padding_token, device=device, dtype=torch.long)],
                dim=1
            )
            # Zero gradients before the backward pass (best practice for pytorch)
            optimizer.zero_grad()

            # Forward pass using half precision to get away with even less memory
            with torch.autocast(device_type=device, dtype=torch.float16):
                outputs = model(input_ids=input_ids,
                                attention_mask=attention_mask,
                                labels=labels)
                # GPT-2 directly computes the loss if labels are provided
                loss = outputs.loss

                logits = outputs.logits.detach().cpu()

                # Backward pass
            loss.backward()

            # Gradient Clipping to prevent exploding gradients
            total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

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

            if detached_loss > 100:
                raise Exception('Loss became to large!!!')

            progress_bar.set_postfix({
                'Batch': batch_idx,
                'Loss': f'{detached_loss:.4f}',
                'LR': f'{lr_scheduler.get_last_lr()[0]:.6f}',
                'GradNorm': f'{total_norm:.2f}',
                'Perplexity': f'{perplexity:.2f}'
            })

            progress_bar.update(1)

        train_loss.append(total_loss / len(dataloader))

        torch.save(model.state_dict(), os.path.join(log_dir, f'{name}{epoch}.ph'))

    print('Training completed!')
    writer.close()


if __name__ == '__main__':
    train('.')