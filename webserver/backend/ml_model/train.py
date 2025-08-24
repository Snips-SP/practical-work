from backend.ml_model.helper import get_device, load_latest_checkpoint, get_next_run_folder, EncodingConfig
from backend.ml_model.dataloader import GPT2Dataset, GPT2RAMDataset
from transformers import GPT2LMHeadModel, GPT2Config, get_scheduler
import torch
from torch.optim import AdamW
from torch.utils.tensorboard import SummaryWriter
from torch.nn import CrossEntropyLoss
from tqdm import tqdm
import argparse
import math
import os


class NetworkConfig:
    # All the instruments which are used in our encoding
    config = GPT2Config(
        vocab_size=EncodingConfig.vocab_size,  # 423
        n_positions=1024,  # Maximum sequence length
        n_ctx=128,  # Context window size
        n_embd=256,  # Embedding size
        n_layer=1,  # Number of transformer layers
        n_head=1,  # Number of attention heads
        pad_token_id=EncodingConfig.padding_token,  # 422
    )


def train(num_epochs: int,
          batch_size: int = 16,
          learning_rate: float = 1e-4,
          lr_scheduler: str = 'cosine',
          gradient_checkpointing: bool = False,
          RAM_dataset: bool = False,
          device: str = None,
          continue_from: str = None,
          model_name: str = 'GPT2_Model',
          modulation: int = 0,
          config=None):
    """Trains a GPT-2 language model on the Lakh Pianoroll Dataset.

        This function performs end-to-end training of a GPT-2 model for music generation.
        It supports training from scratch or resuming from a checkpoint, with configurable
        hyperparameters and optimization settings. The function handles dataset loading,
        model initialization, gradient optimization with clipping, learning rate scheduling,
        and comprehensive logging of training metrics including loss, perplexity, and
        gradient norms.

        :param int num_epochs: Number of training epochs to run.
        :param int batch_size: Batch size for training, defaults to 16.
        :param float learning_rate: Learning rate for the AdamW optimizer, defaults to 1e-4.
        :param str lr_scheduler: Learning rate scheduler type, defaults to 'cosine'.
                               Currently only 'cosine' is supported.
        :param bool gradient_checkpointing: Enable gradient checkpointing to reduce memory usage
                                          at the cost of computational overhead, defaults to False.
        :param bool RAM_dataset: Load the entire dataset into RAM for faster access,
                               defaults to False.
        :param str device: Device to use for training ('cpu', 'cuda', 'xpu'). If None,
                          automatically selects the best available device, defaults to None.
        :param str continue_from: Path to a checkpoint directory to resume training from.
                                If None, starts training from scratch, defaults to None.
        :param str model_name: Name used for creating the logging directory,
                             defaults to 'GPT2_Model'.
        :param int modulation: Number of modulations for each midi file. Has to be created first with encode.py,
                            defaults to 0.
        :param config: Custom model configuration object. If None, uses the default
                      NetworkConfig.config, defaults to None.
        :raises NotImplementedError: If an unsupported learning rate scheduler is specified.
        :raises Exception: If NaN values are detected in logits, loss, or if exploding
                          gradients are detected (gradient norm > 1e6).
        :returns: None. The function saves checkpoints and logs training progress to disk.
        :rtype: None
    """
    # We assume the root path is the current script path
    root_path = os.path.dirname(os.path.abspath(__file__))

    # Use a preferred device from a user or chose it automatically
    device = get_device(device)
    print('Using device:', device)

    optimizer_kwargs = {
        'lr': learning_rate
    }

    start_epoch = 0
    # Search checkpoint file and load it
    if continue_from is not None:
        continue_from = os.path.join(root_path, 'runs', continue_from)
        model, optimizer, start_epoch, global_step = load_latest_checkpoint(continue_from,
                                                                            device=device,
                                                                            optimizer_class=AdamW,
                                                                            **optimizer_kwargs)

        # Move to device
        model.to(device)
        model.train()

        log_dir = continue_from
        print(f'Continuing from epoch {start_epoch}')
    else:
        print(f'Training from scratch.')
        # Init global step as 0
        global_step = 0
        # Use custom config if provided
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

        # Define optimizer
        optimizer = AdamW(model.parameters(), **optimizer_kwargs)

        # Create new logging dir
        log_dir = get_next_run_folder(model_name, base_dir=os.path.join(root_path, 'runs'))

        # Create the directory if it doesn't exist
        os.makedirs(log_dir, exist_ok=True)

    # Get dataset and dataloader
    if RAM_dataset:
        dataset = GPT2RAMDataset(os.path.join(root_path, f'lpd_5_dataset_da_{modulation}'))
    else:
        dataset = GPT2Dataset(os.path.join(root_path, f'lpd_5_dataset_da_{modulation}'))
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    # Initialize SummaryWriter with the log directory
    writer = SummaryWriter(log_dir=log_dir)
    print(f'Logging to: {log_dir}')

    # Define progress bar
    num_training_steps = num_epochs * len(dataloader)

    # Set the model to train and move it to the right device
    model.train()
    model.to(device)

    # Define loss function
    loss_fn = CrossEntropyLoss(ignore_index=EncodingConfig.padding_token)

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
        raise NotImplementedError(f'Learning rate scheduler {lr_scheduler} not implemented.')

    # Print out relevant gpt2 configurations
    print('GPT2 config: ')
    for key in ['n_positions', 'n_ctx', 'n_embd', 'n_layer', 'n_head']:
        print(f'   {key} = {model.config.to_dict()[key]}')
    print('')

    # Start training
    print(f'Training for {num_epochs} epochs with batch size {batch_size}.')
    print(' ')
    progress_bar = tqdm(range(num_training_steps), desc='Training Progress')
    train_loss = []
    for epoch in range(start_epoch, start_epoch + num_epochs):
        total_loss = 0
        for batch_idx, batch in enumerate(dataloader):
            input_ids = batch[0].to(device).long()
            # We dont need padding since our data is always 1024 tokens long
            # attention_mask = batch[1].to(device).long()

            # Zero gradients before the backward pass (best practice for pytorch)
            optimizer.zero_grad()

            # If we only give the inputs and not the labels, the hugging face model will not calculate
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

            global_step += 1

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

            # Display stats in the progress bar
            progress_bar.set_postfix(display_stats)
            progress_bar.update(1)

        # Log total train loss over epoch
        train_loss.append(total_loss / len(dataset))
        writer.add_scalar('Total Training Loss', total_loss / len(dataset), global_step)
        
        # Save state dict and other parameters after each epoch
        checkpoint_data = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'global_step': global_step,
            'config': model.config.to_dict()
        }

        torch.save(checkpoint_data, os.path.join(log_dir, f'checkpoint_{epoch}.ph'))

    print('Training completed!')
    writer.close()


def run_trainings_from_code():
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
        n_embd=384,  # Embedding size
        n_layer=6,  # Number of transformer layers
        n_head=6,  # Number of attention heads
        pad_token_id=EncodingConfig.padding_token,  # 422
    )
    num_epochs = 2
    batch_size = 16
    learning_rate = 1e-4
    lr_scheduler = 'cosine'
    gradient_checkpointing = False
    RAM_dataset = True
    device = 'xpu'
    modulation = 0

    # Train with same parameters but different configs
    for name, config, continue_from_path in [
        ('GPT2_Medium', config3, 'GPT2_Medium_1'),
        ('GPT2_Small', config2, 'GPT2_Small_1'),
        ('GPT2_Tiny', config1, 'GPT2_Tiny_1')
    ]:
        if name == 'GPT2_Medium':
            batch_size = 4
        else:
            batch_size = 16
        train(num_epochs=num_epochs,
              batch_size=batch_size,
              learning_rate=learning_rate,
              lr_scheduler=lr_scheduler,
              gradient_checkpointing=gradient_checkpointing,
              RAM_dataset=RAM_dataset,
              device=device,
              continue_from=continue_from_path,
              model_name=name,
              modulation=modulation,
              config=config)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_epochs', type=int, required=True, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate for optimizer')
    parser.add_argument('--lr_scheduler', type=str, default='cosine', choices=['cosine'], help='Learning rate scheduler type')
    parser.add_argument('--gradient_checkpointing', action='store_true', help='Enable gradient checkpointing to save memory')
    parser.add_argument('--RAM_dataset', action='store_true', help='Load entire dataset into memory')
    parser.add_argument('--modulation', type=int, default=0, help='Number of modulations for each midi file')
    parser.add_argument('--device', type=str, default=None, help='Device to use: cpu, cuda, or xpu (auto-select if None)')
    parser.add_argument('--continue_from', type=str, default=None, help='Path to a directory to continue training from a checkpoint')
    parser.add_argument('--run_trainings_from_code', type=bool, default=False,
                        help='Ignores command line interface arguments and runs the training function.')

    args = parser.parse_args()

    if args.run_trainings_from_code:
        run_trainings_from_code()
    else:
        train(num_epochs=args.num_epochs,
              batch_size=args.batch_size,
              learning_rate=args.learning_rate,
              lr_scheduler=args.lr_scheduler,
              gradient_checkpointing=args.gradient_checkpointing,
              RAM_dataset=args.RAM_dataset,
              continue_from=args.continue_from,
              modulation=args.modulation,
              device=args.device)



