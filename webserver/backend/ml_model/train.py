import glob
import json
import random

from backend.ml_model.helper import get_device, load_latest_checkpoint, EncodingConfig
from backend.ml_model.dataloader import MidiDataset, MidiRAMDataset, OnTheFlyMidiDataset
from torch.utils.data import DataLoader
from transformers import get_scheduler, Phi3ForCausalLM, Phi3Config
import torch
from torch.optim import AdamW
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import argparse
import math
import os

from typing import List, Tuple

# Temp
from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorForLanguageModeling

MODEL_CONFIGURATIONS = {
    'Phi-3-head-dim-64': {
        # --- Config V1: head_dim=64 ---
        'config': Phi3Config(
            # --- Core architecture ---
            hidden_size=512,
            num_hidden_layers=8,
            intermediate_size=2048,  # 4x MLP expansion ratio

            # --- Attention: Using a head_dim of 64 ---
            num_attention_heads=8,  # Derived from hidden_size / 64
            num_key_value_heads=2,  # GQA with a 4:1 ratio (8/2)

            # --- Standard parameters for a Phi-3 model ---
            max_position_embeddings=2048,
            rope_theta=10000.0,
            vocab_size=EncodingConfig.vocab_size,
            bos_token_id=EncodingConfig.begin_note,
            eos_token_id=EncodingConfig.end_note,
            pad_token_id=EncodingConfig.padding_token,
            hidden_act='silu',
            initializer_range=0.02,
            layer_norm_eps=1e-5,
            tie_word_embeddings=False,
        ),

        'hyperparameters': {
            'num_epochs': 2,
            'batch_size': 6,
            'learning_rate': 5e-4,
            'lr_scheduler': 'cosine',
            'num_estimated_epochs': 50, ### TODO: Find right amount of estimated epochs
            'patience': 5,
            'accumulation_steps': 1,
            'n_modulations': 11,
            'num_workers': -1,  # Use all the workers
            'attention_implementation': 'sdpa',
            'model_dtype': 'bfloat16',
            'compile_model': True,
            'gradient_checkpointing': False,
            'device': 'xpu',
            'model_name': 'Phi-3-head-dim-64',
        },
    },
    'Phi-3-head-dim-32': {
        # --- Config V2: head_dim=32 ---
        'config': Phi3Config(
            # --- Core architecture ---
            hidden_size=512,
            num_hidden_layers=8,
            intermediate_size=2048,  # 4x MLP expansion ratio

            # --- Attention: Using a head_dim of 32 ---
            num_attention_heads=16,  # Derived from hidden_size / 32
            num_key_value_heads=4,  # GQA ratio (16/4)

            # --- Standard parameters for a Phi-3 model ---
            max_position_embeddings=2048,
            rope_theta=10000.0,
            vocab_size=EncodingConfig.vocab_size,
            bos_token_id=EncodingConfig.begin_note,
            eos_token_id=EncodingConfig.end_note,
            pad_token_id=EncodingConfig.padding_token,
            hidden_act='silu',
            initializer_range=0.02,
            layer_norm_eps=1e-5,
            tie_word_embeddings=False,
        ),
        'hyperparameters': {
            'num_epochs': 2,
            'batch_size': 6,
            'learning_rate': 5e-4,
            'lr_scheduler': 'cosine',
            'num_estimated_epochs': 50, ### TODO: Find right amount of estimated epochs
            'patience': 5,
            'accumulation_steps': 1,
            'n_modulations': 11,
            'num_workers': -1,  # Use all the workers
            'attention_implementation': 'sdpa',
            'model_dtype': 'bfloat16',
            'compile_model': True,
            'gradient_checkpointing': False,
            'device': 'xpu',
            'model_name': 'Phi-3-head-dim-32',
        },
    }
}


class NetworkConfig:
    # All the instruments which are used in our encoding
    config = Phi3Config(
        # --- Core architecture ---
        hidden_size=512,
        num_hidden_layers=8,
        intermediate_size=2048,  # 4x MLP expansion ratio

        # --- Attention: Using a head_dim of 64 ---
        num_attention_heads=8,  # Derived from hidden_size / 64
        num_key_value_heads=2,  # GQA with a 4:1 ratio (8/2)

        # --- Standard parameters for a Phi-3 model ---
        max_position_embeddings=2048,
        rope_theta=10000.0,
        vocab_size=EncodingConfig.vocab_size,
        bos_token_id=EncodingConfig.begin_note,
        eos_token_id=EncodingConfig.end_note,
        pad_token_id=EncodingConfig.padding_token,
        hidden_act='silu',
        initializer_range=0.02,
        layer_norm_eps=1e-5,
        tie_word_embeddings=False,
        _attn_implementation='sdpa'
    )


def train(
        # Training Hyperparameters
        num_epochs: int, batch_size: int = 4,
        learning_rate: float = 1e-4, lr_scheduler: str = 'cosine',
        num_estimated_epochs: int = 100,
        patience: int = 4,

        # Dataset parameters
        split_ratios: Tuple[float, float, float] = (0.8, 0.1, 0.1),
        dataset_path: str = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'lpd_5'),
        n_modulations: int = 0,
        num_workers: int = 0,
        random_seed: int = 42,

        # Hardware related parameters
        attention_implementation: str = 'sdpa',
        accumulation_steps: int = 4,
        gradient_checkpointing: bool = False,
        device: str = None,
        model_dtype=torch.bfloat16,
        compile_model: bool = False,

        # Other parameters
        model_name: str = 'Phi-2_Model',
        runs_path: str = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'runs'),
        config=None,
        debug: bool = False,
):
    torch.autograd.set_detect_anomaly(debug)
    device = get_device(device)
    print(f'Using device: {device}\n')

    if not abs(sum(split_ratios) - 1.0) < 1e-8:
        raise ValueError(f'Split ratios must sum to 1.0, but got {sum(split_ratios)}')

    # Check if the right dataset is used
    if not os.path.isdir(os.path.join(dataset_path, 'lpd_5_cleansed')):
        raise Exception(f'Dataset {dataset_path}, does not contain the expected folder "lpd_5_cleansed".')

    # Path to model log dir
    log_dir = os.path.join(runs_path, model_name)
    if not os.path.isdir(log_dir):
        # =================
        # = Start new run =
        # =================
        start_epoch = 1
        global_step = 0
        patience_counter = 0
        training_loss_per_epoch = []
        validation_loss_per_epoch = []
        os.makedirs(log_dir, exist_ok=True)

        # Create train, valid and test splits from scratch
        sample_files = glob.glob(os.path.join(dataset_path, 'lpd_5_cleansed/*/*/*/*/*.npz'))

        random.seed(random_seed)
        random.shuffle(sample_files)

        # Calculate the split indices
        total_size = len(sample_files)
        train_end = int(total_size * split_ratios[0])
        valid_end = train_end + int(total_size * split_ratios[1])

        # Slice the shuffled list to create the splits
        train_files = sample_files[:train_end]
        valid_files = sample_files[train_end:valid_end]
        test_files = sample_files[valid_end:]

        with open(os.path.join(log_dir, 'train_valid_test_split.json'), 'w') as f:
            json.dump({
                'train': train_files,
                'valid': valid_files,
                'test': test_files
            }, f, indent=4)

        train_dataset = OnTheFlyMidiDataset(train_files, n_modulations, chunk_size=1024)
        valid_dataset = OnTheFlyMidiDataset(valid_files, 0, chunk_size=1024)

        print('Training from scratch.')
        if config is None:
            print('Using default config.\n')
            config = NetworkConfig.config
        else:
            print('Using provided config.\n')

        # Initializing model weights is done automatically by providing initializer_range=0.02 in the config
        config.initializer_range = 0.02

        # Turn on flash attention or sdpa if wanted
        if device == 'cuda' and (attention_implementation == 'flash_attention_2' or attention_implementation == 'sdpa'):
            config._attn_implementation = attention_implementation
        elif device == 'xpu' and attention_implementation == 'sdpa':
            config._attn_implementation = attention_implementation
        else:
            raise ValueError(f'Invalid attention implementation {attention_implementation} for device {device}.')

        # Set the attention implementation

        # Create a model from config
        model = Phi3ForCausalLM(config)

        # Keep the kwargs as a variable to save them to checkpoints later
        optimizer_kwargs = {
            'lr': learning_rate,
        }

        # Get fresh optimizers
        optimizer = AdamW(model.parameters(), **optimizer_kwargs)

        if lr_scheduler == 'cosine':
            # Keep the kwargs as a variable to save them to checkpoints later
            num_warmup_steps = 500 // accumulation_steps
            lr_scheduler_kwargs = {
                'name': 'cosine',
                'num_warmup_steps': num_warmup_steps,
                'num_training_steps': num_estimated_epochs * len(train_files) // batch_size
            }
            lr_scheduler = get_scheduler(optimizer=optimizer, **lr_scheduler_kwargs)
        elif lr_scheduler is not None:
            raise NotImplementedError(f'Learning rate scheduler {lr_scheduler} not implemented.')
    else:
        # =================================
        # = Continue from last checkpoint =
        # =================================
        model, training_loss_per_epoch, validation_loss_per_epoch, patience_dict, start_epoch, global_step, optimizer, optimizer_kwargs, lr_scheduler, lr_scheduler_kwargs, train_files, valid_files, test_files = load_latest_checkpoint(
            log_dir,
            device=device,
            optimizer_class=AdamW,
            learning_rate_scheduler_class=get_scheduler,
        )
        patience = patience_dict['patience']
        patience_counter = patience_dict['patience_counter']
        # Create datasets from saved train valid test split
        train_dataset = OnTheFlyMidiDataset(train_files, n_modulations, chunk_size=1024)
        valid_dataset = OnTheFlyMidiDataset(valid_files, 0, chunk_size=1024)
        print(f'Continuing training from epoch {start_epoch}\n')

    model.to(device, dtype=model_dtype)

    # Compile the model to use graph like representation
    if compile_model:
        model = torch.compile(model)

    # Enable gradient checkpointing
    if gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False

    # Create dataloader
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    valid_dataloader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    # Log every 20 accumulated steps
    logging_interval = 5
    writer = SummaryWriter(log_dir=log_dir)

    print('--- Training Initiated with the following parameters ---')
    for param, value in vars().items():
        if param != 'config':
            print(f"{param:<25}: {value}")
    print('------------------------------------------------------')

    # ===================
    # = Debugging model =
    # ===================
    if debug:
        # Dictionary to store the mean absolute value of activations from hooks
        activation_stats = {}

        def get_activation(name):
            def hook(model, input, output):
                activation_stats[name] = output.detach().abs().mean().cpu().item()

            return hook

        # List all layer names
        # print('\n'.join([f'{name}' for name, object in model.named_modules()]))

        # Register hooks on some key layers
        for i, layer in enumerate(model.model.layers):
            # Hook the MLP up-projection
            # This layer is responsible for up-projecting our input
            # Since it acts like an amplifier, any instability in the input would
            # cause huge instability in the network
            layer.mlp.fc1.register_forward_hook(get_activation(f'layer_{i}/mlp_fc1'))

            # Hook the query projection in self-attention
            # This layer is the query of the self-attention mechanism, if the value inside
            # the query vector becomes excessively large, the dot-product attention score could
            # also explode, leading to unstable softmax outputs.
            # layer.self_attn.q_proj.register_forward_hook(get_activation(f'layer_{i}/attn_q_proj'))

            # Hook the final output layer
            # The final linear layer which projects the models output into logits. If we experience
            # Nans in the logits this layer could be the culprit.

        # model.lm_head.register_forward_hook(get_activation('lm_head'))

        def log_logit_stats(name):
            def hook(model, input, output):
                # Log the max and min values of the logits
                writer.add_scalar(f'Per-Step/Logits/{name}_max', output.detach().max().cpu().item(), global_step)
                writer.add_scalar(f'Per-Step/Logits/{name}_min', output.detach().min().cpu().item(), global_step)

            return hook

        # Hook the final output layer
        model.lm_head.register_forward_hook(log_logit_stats('lm_head'))

    # ======================
    # = Main training loop =
    # ======================
    print(f'Starting training for {num_epochs} epochs...')
    progress_bar = tqdm(initial=0, total=num_epochs * len(train_dataloader), desc='Training Progress')
    for epoch in range(start_epoch, start_epoch + num_epochs):
        # --- Training Step ---
        model.train()
        total_train_loss = 0.0
        accumulated_loss = 0.0

        for i, batch in enumerate(train_dataloader):
            # Our dataset returns a tensor of size (16, 1024) of dtype int64
            input_ids = batch[0].to(device)

            # Make a forward pass
            # The model will automatically shift the labels and calculate the loss
            outputs = model(input_ids=input_ids, labels=input_ids)

            # The loss is already calculated and available in the output object
            loss = outputs.loss

            # Accumulate the gradient to make use of our full physical VRAM
            # and making our effective batch size more stable due to averaging
            detached_loss = loss.detach().cpu().item()
            accumulated_loss += detached_loss
            # Scale the loss
            loss = loss / accumulation_steps
            # Calculate the current gradient
            loss.backward()

            if torch.isnan(loss): raise Exception('NaN detected in training loss!')

            # Update weights only every 'accumulation_steps'
            if (i + 1) % accumulation_steps == 0:
                # Perform gradient clipping and log model wide gradient norm
                total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)

                if debug:
                    # ===================
                    # = Debugging Model =
                    # ===================
                    # Stop training at the first sings of infinite gradients
                    flag = False

                    # Log weights and their gradients
                    for name, param in model.named_parameters():
                        # Log Weight Distributions (with safety check)
                        if param.data is not None and torch.isfinite(param.data).all():
                            writer.add_histogram(f'Per-Step/Weights/{name}', param.data, global_step)
                        else:
                            flag = True
                            print(f'Warning: Skipping weight histogram for {name} due to non-finite values.')

                        # Log Per-Layer Gradient Norms (with safety check)
                        if param.requires_grad and param.grad is not None:
                            if torch.isfinite(param.grad).all():
                                grad_norm = param.grad.data.norm(2)
                                writer.add_scalar(f'Per-Step/Gradients/{name}_norm', grad_norm.item(), global_step)
                            else:
                                flag = True
                                print(f'Warning: Skipping gradient norm for {name} due to non-finite values.')

                    # Log mean absolute activation captured by hooks (as requested)
                    for name, value in activation_stats.items():
                        writer.add_scalar(f'Per-Step/Activations/{name}_mean_abs', value, global_step)

                if total_norm > 1e6: raise Exception('Exploding gradients detected!')

                # Make a learning step with the accumulated gradients
                optimizer.step()
                optimizer.zero_grad()

                if lr_scheduler is not None:
                    # Update learning rate
                    lr_scheduler.step()
                    last_lr = lr_scheduler.get_last_lr()[0]
                else:
                    last_lr = learning_rate

                # --- Per-Step Logging & Progress Bar ---
                avg_accumulated_loss = accumulated_loss / accumulation_steps
                total_train_loss += avg_accumulated_loss
                accumulated_loss = 0.0
                global_step += 1
                progress_bar.update(accumulation_steps)

                if global_step % logging_interval == 0:
                    writer.add_scalar('Per-Step/Training Loss', avg_accumulated_loss, global_step)
                    writer.add_scalar('Per-Step/Learning Rate', last_lr, global_step)
                    writer.add_scalar('Per-Step/Gradient Norm', total_norm, global_step)
                    progress_bar.set_postfix({'Loss': f'{detached_loss:.4f}', 'LR': f'{last_lr:.6f}'})

                if debug and flag:
                    raise Exception('Aborting due to non-finite values in gradient.')

        # --- Per-Epoch Training Logging ---
        # avg_train_loss is calculated over optimizer steps, not batches anymore
        num_optimizer_steps = len(train_dataloader) // accumulation_steps
        avg_train_loss = total_train_loss / num_optimizer_steps if num_optimizer_steps > 0 else 0
        training_loss_per_epoch.append(avg_train_loss)
        writer.add_scalar('Per-Epoch/Training Loss', avg_train_loss, epoch)
        writer.add_scalar('Per-Epoch/Training Perplexity', math.exp(avg_train_loss), epoch)

        # --- Validation Step ---
        model.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for batch in tqdm(valid_dataloader, desc=f'Epoch {epoch} Validation'):
                input_ids = batch[0].to(device).long()
                outputs = model(input_ids=input_ids, labels=input_ids)
                total_val_loss += outputs.loss.detach().cpu().item()

        # --- Per-Epoch Validation Logging ---
        avg_val_loss = total_val_loss / len(valid_dataloader)
        validation_loss_per_epoch.append(avg_val_loss)

        writer.add_scalar('Per-Epoch/Validation Loss', avg_val_loss, epoch)
        writer.add_scalar('Per-Epoch/Validation Perplexity', math.exp(avg_val_loss), epoch)
        print(f'\nEpoch {epoch}: Train Loss={avg_train_loss:.4f} | Val Loss={avg_val_loss:.4f}')

        # Check if we should increase patience counter or reset it
        best_model = False
        best_val_loss = min(validation_loss_per_epoch) if validation_loss_per_epoch else float('inf')
        if avg_val_loss <= best_val_loss:
            patience_counter = 0
            best_model = True
        else:
            patience_counter += 1

        # --- Checkpointing ---
        checkpoint_data = {
            'epoch': epoch,
            'training_loss_per_epoch': training_loss_per_epoch,
            'validation_loss_per_epoch': validation_loss_per_epoch,
            'patience': {
                'patience': patience,
                'patience_counter': patience_counter
            },
            'global_step': global_step,
            'model_state_dict': model.state_dict(),
            'config': model.config.to_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'optimizer_kwargs': optimizer_kwargs,
            'lr_scheduler_state_dict': lr_scheduler.state_dict() if lr_scheduler is not None else None,
            'lr_scheduler_kwargs': lr_scheduler_kwargs if lr_scheduler is not None else None,
        }
        torch.save(checkpoint_data, os.path.join(log_dir, f'checkpoint_epoch_{epoch}.ph'))
        if best_model:
            # Save as current as the best model
            torch.save(checkpoint_data, os.path.join(log_dir, f'checkpoint_best.ph'))

        # Early stop
        if patience_counter >= patience:
            print(f'\nEpoch {epoch}: Validation Loss has not improved in {patience} epochs. Stopping training.')
            writer.close()
            return True

        if epoch >= num_estimated_epochs:
            # Stop at estimated epochs
            writer.close()
            return True

    print('Training completed!')
    writer.close()
    return False


def training_manager(epochs_per_session=1, progress_file='runs/progress.json'):
    """
    Manages the training schedule by automatically selecting and training
    the model with the fewest completed epochs.
    """
    while True:
        root_path = os.path.dirname(os.path.abspath(__file__))
        progress_file = os.path.join(root_path, progress_file)
        # Load or initialize training progress
        try:
            with open(progress_file, 'r') as f:
                progress = json.load(f)
            print(f'Loaded progress from {progress_file}')
        except FileNotFoundError:
            print(f'{progress_file} not found. Initializing new progress tracker.')
            progress = {name: {'progress': 0, 'done': False} for name in MODEL_CONFIGURATIONS.keys()}

        print('Current training status:', progress)

        #  Select the model with the least training
        if all(details['done'] for details in progress.values()):
            print('All models are done. Exiting training manager.')
            return

        # Chose model with minimum epochs trained from the ones which are not done yet
        unfinished_models = {
            name: details for name, details in progress.items() if not details['done']
        }

        # Check if any unfinished models exist and find the one with min progress
        if unfinished_models:
            model_to_train = min(unfinished_models, key=lambda model: unfinished_models[model]['progress'])
            print(f'Model with least progress to train next: {model_to_train})')
        else:
            print('All models are done. No models to train.')
            print('')
            print('----------------------------------')
            print('!!!NO MORE TRAINING IS NEEDED!!!')
            print('----------------------------------')
            return

        # Prepare and run the training
        model_data = MODEL_CONFIGURATIONS[model_to_train]
        config = model_data['config']
        # Use a copy to avoid modifying the original dict
        hp = model_data['hyperparameters'].copy()
        hp['num_epochs'] = epochs_per_session

        # train the model
        done = train(config=config, debug=False, **hp)

        # Update the progress file if training was successful
        progress[model_to_train]['progress'] += epochs_per_session
        progress[model_to_train]['done'] = done

        with open(progress_file, 'w') as f:
            json.dump(progress, f, indent=4)

        print(f'\nProgress file updated. {model_to_train} now has {progress[model_to_train]["progress"]} total epochs.')
        print('')


if __name__ == '__main__':
    # Using ArgumentDefaultsHelpFormatter shows the default values in the --help message
    parser = argparse.ArgumentParser(
        description='Train a language model for music generation.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # --- Group arguments for better readability ---
    train_params = parser.add_argument_group('Training Parameters')
    data_params = parser.add_argument_group('Dataset Parameters')
    hardware_params = parser.add_argument_group('Hardware Parameters')
    run_params = parser.add_argument_group('Run Management Parameters')

    # --- Training Parameters ---
    train_params.add_argument('--num_epochs', type=int, required=True,
                              help='Number of epochs for this training session.')
    train_params.add_argument('--batch_size', type=int, default=4,
                              help='Number of samples per batch.')
    train_params.add_argument('--learning_rate', type=float, default=1e-4,
                              help='Initial learning rate for the optimizer.')
    train_params.add_argument('--lr_scheduler', type=str, default='cosine', choices=['cosine'],
                              help='Learning rate scheduler type.')
    train_params.add_argument('--num_estimated_epochs', type=int, default=100,
                              help='Total planned epochs for creating a consistent LR schedule.')
    train_params.add_argument('--patience', type=int, default=4,
                              help='Epochs to wait for validation loss improvement before stopping.')
    train_params.add_argument('--accumulation_steps', type=int, default=4,
                              help='Number of steps to accumulate gradients before an optimizer update.')

    # --- Dataset Parameters ---
    data_params.add_argument('--dataset_path', type=str,
                             default=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'lpd_5'),
                             help='Path to the root directory of the dataset.')
    data_params.add_argument('--split_ratios', type=float, nargs=3, default=[0.8, 0.1, 0.1],
                             metavar=('TRAIN', 'VALID', 'TEST'),
                             help='Train, validation, and test split ratios (must sum to 1).')
    data_params.add_argument('--n_modulations', type=int, default=0,
                             help='Number of random pitch augmentations per sample.')
    data_params.add_argument('--num_workers', type=int, default=0,
                             help='Number of worker processes for data loading. Use -1 for all CPU cores.')
    data_params.add_argument('--random_seed', type=int, default=42,
                             help='Seed for random operations to ensure reproducibility.')

    # --- Hardware Parameters ---
    hardware_params.add_argument('--device', type=str, default=None,
                                 help='Device to use: cpu, cuda, or xpu (auto-selects if None).')
    hardware_params.add_argument('--gradient_checkpointing', action='store_true',
                                 help='Enable gradient checkpointing to save memory at the cost of speed.')
    hardware_params.add_argument('--dtype', type=str, default='float32',
                                 help='Datatype of the model. (float32, bfloat16, or float16).')
    hardware_params.add_argument('--attention_implementation', type=str, default='sdpa',
                                 help='Attention implementation of the model. (flash_attention_2, sdpa or eager).')

    # Mutually exclusive group for compile for clear on/off control
    compile_group = hardware_params.add_mutually_exclusive_group()
    compile_group.add_argument('--compile', action='store_true', dest='compile',
                               help='Compiles model for faster training. Requires compatible hardware and libraries.')
    compile_group.add_argument('--no-compile', action='store_false', dest='compile', default=True,
                               help='Disable compile.')

    # --- Run Management Parameters ---
    run_params.add_argument('--model_name', type=str, default='Phi-2_Model',
                            help='Base name for the run and logging directory.')
    run_params.add_argument('--runs_path', type=str,
                            default=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'runs'),
                            help='Directory to save model checkpoints and logs.')
    run_params.add_argument('--debug', action='store_true',
                            help='Run in debug mode with a small subset of data.')
    run_params.add_argument('--training_manager', action='store_true',
                            help='If specified, ignores other CLI arguments and runs a hardcoded training session.')

    args = parser.parse_args()

    # --- Execute the appropriate function ---
    if args.training_manager:
        training_manager()
    else:
        if args.dtype == 'float32':
            dtype = torch.float32
        elif args.dtype == 'bfloat16':
            dtype = torch.bfloat16
        elif args.dtype == 'float16':
            dtype = torch.float16
        else:
            raise ValueError(f'Invalid dtype: {args.dtype}')

        # Pass all the relevant arguments to the train function
        train(
            num_epochs=args.num_epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            lr_scheduler=args.lr_scheduler,
            num_estimated_epochs=args.num_estimated_epochs,
            patience=args.patience,
            accumulation_steps=args.accumulation_steps,
            split_ratios=tuple(args.split_ratios),
            dataset_path=args.dataset_path,
            n_modulations=args.n_modulations,
            num_workers=args.num_workers,
            random_seed=args.random_seed,
            attention_implementation=args.attention_implementation,
            model_dtype=dtype,
            compile_model=args.compile,
            gradient_checkpointing=args.gradient_checkpointing,
            device=args.device,
            model_name=args.model_name,
            runs_path=args.runs_path,
            debug=args.debug
        )
