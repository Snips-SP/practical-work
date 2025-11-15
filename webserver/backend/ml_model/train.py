import os
import shutil

from backend.ml_model.helper import get_device, load_latest_checkpoint, EncodingConfig
from backend.ml_model.dataloader import OnTheFlyMidiDataset
import glob
import json
import random
from torch.utils.data import DataLoader
from transformers import get_scheduler, Phi3ForCausalLM, Phi3Config, GPT2Config, GPT2LMHeadModel
import torch
from torch.optim import AdamW
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import argparse
import math
from typing import Tuple


MODEL_CONFIGURATIONS = {
    #'Phi-3-v2-head-dim-64-n-mod-0-lr-3e-4-grad-acc-16': {
    #    # ~85M params
    #    # --- Config V2: head_dim=64 ---
    #    'config': Phi3Config(
    #        # --- Core architecture ---
    #        hidden_size=768,
    #        num_hidden_layers=12,
    #        intermediate_size=3072,  # 4x MLP expansion ratio
#
    #        # --- Attention: Using a head_dim of 64 ---
    #        num_attention_heads=12,  # Derived from hidden_size / 64
    #        num_key_value_heads=4,  # GQA ratio (16/4)
#
    #        # -- Advanced hyperparameters --
    #        hidden_act='silu',
    #        partial_rotary_factor=1.0,
    #        # Percentage of the query and keys which will have rotary embedding.
#
    #        # --- Standard parameters for a Phi-3 model ---
    #        max_position_embeddings=2048,
    #        vocab_size=EncodingConfig.vocab_size,
    #        bos_token_id=EncodingConfig.begin_note,
    #        eos_token_id=EncodingConfig.end_note,
    #        pad_token_id=EncodingConfig.padding_token,
    #        initializer_range=0.02,
    #        tie_word_embeddings=False,
    #    ),
    #    'hyperparameters': {
    #        'num_epochs': 2,
    #        'batch_size': 12,
    #        'learning_rate': 3e-4,
    #        'lr_scheduler': 'cosine',
    #        'num_estimated_epochs': 120,
    #        'patience': 5, # Patience per evaluation
    #        'accumulation_steps': 16,
    #        'n_modulations': 0,
    #        'chunk_size': 2048,
    #        'num_workers': 8,
    #        'attention_implementation': 'eager',
    #        'model_dtype': 'bfloat16',
    #        'compile_model': True,
    #        'gradient_checkpointing': False,
    #        'device': 'xpu',
    #        'model_name': 'Phi-3-v2-head-dim-64-n-mod-0-lr-3e-4-grad-acc-16',
    #    },
    #},
    'debugging': {
        # --- Config V2: head_dim=64 ---

        # ~33M params
        'config': Phi3Config(
            # --- Core architecture ---
            hidden_size=512,
            num_hidden_layers=8,
            intermediate_size=2048,  # 4x MLP expansion ratio

            # --- Attention: Using a head_dim of 64 ---
            num_attention_heads=8,  # Derived from hidden_size / 64
            num_key_value_heads=8,  # GQA ratio (16/4)

            # -- Advanced hyperparameters --
            hidden_act='silu',
            partial_rotary_factor=1.0,
            # Percentage of the query and keys which will have rotary embedding.
            rope_scaling=None,

            # --- Standard parameters for a Phi-3 model ---
            max_position_embeddings=2048,
            original_max_position_embeddings=2048,
            vocab_size=EncodingConfig.vocab_size,
            bos_token_id=EncodingConfig.begin_note,
            eos_token_id=EncodingConfig.end_note,
            pad_token_id=EncodingConfig.padding_token,
            initializer_range=0.02,
            tie_word_embeddings=True,
        ),

        'hyperparameters': {
            'batch_size': 8,
            'accumulation_steps': 8, # i.e. effective batch size of 64
            'learning_rate': 5e-3,
            'lr_scheduler': 'cosine',
            'num_estimated_epochs': 1,
            'patience': 5,

            'n_modulations': 0,
            'chunk_size': 2048,
            'num_workers': 4,
            'warmup_steps': 128, # About 1 bar

            'attention_implementation': 'eager',
            'model_dtype': 'bfloat16',
            'compile_model': True,
            'device': 'xpu',
            'model_name': 'debugging',
        },
    },
}


class NetworkConfig:
    config = Phi3Config(
        # --- Core architecture ---
        hidden_size=512,
        num_hidden_layers=8,
        intermediate_size=2048,  # 4x MLP expansion ratio

        # --- Attention: Using a head_dim of 64 ---
        num_attention_heads=8,  # Derived from hidden_size / 64
        num_key_value_heads=2,  # GQA with a 4:1 ratio (8/2)

        # --- Standard parameters for a Phi-3 model ---
        max_position_embeddings=1024,
        rope_theta=10000.0,
        vocab_size=EncodingConfig.vocab_size,
        bos_token_id=EncodingConfig.begin_note,
        eos_token_id=EncodingConfig.end_note,
        pad_token_id=EncodingConfig.padding_token,
        initializer_range=0.02,
        tie_word_embeddings=False,
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
        chunk_size: int = 2048,
        warmup_steps: int = 128,

        # Hardware related parameters
        attention_implementation: str = 'eager',
        accumulation_steps: int = 4,
        device: str = None,
        model_dtype='bfloat16',
        compile_model: bool = False,

        # Other parameters
        model_name: str = 'Phi-3_Model',
        checkpointing_per_epochs: int = 10,
        runs_path: str = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'runs'),
        config=None,
        debug: bool = False,
):
    torch.autograd.set_detect_anomaly(debug)

    if model_dtype == 'float32':
        model_dtype = torch.float32
    elif model_dtype == 'bfloat16':
        model_dtype = torch.bfloat16
    elif model_dtype == 'float16':
        model_dtype = torch.float16
    else:
        raise ValueError(f'Invalid dtype: {model_dtype}')

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

        train_dataset = OnTheFlyMidiDataset(train_files, encodingConfig=EncodingConfig, n_modulations=n_modulations, chunk_size=chunk_size, warmup_steps=warmup_steps)
        valid_dataset = OnTheFlyMidiDataset(valid_files, encodingConfig=EncodingConfig, n_modulations=0, chunk_size=chunk_size, warmup_steps=warmup_steps)

        print('Training from scratch.\n')
        if config is None:
            config = NetworkConfig.config

        # Turn on flash attention or sdpa if wanted
        if attention_implementation != 'eager':
            if device == 'cuda' and (attention_implementation == 'flash_attention_2' or attention_implementation == 'sdpa'):
                config._attn_implementation = attention_implementation
            elif device == 'xpu' and attention_implementation == 'sdpa':
                config._attn_implementation = attention_implementation
            else:
                raise ValueError(f'Invalid attention implementation {attention_implementation} for device {device}.')

        # Create a model from config
        model = Phi3ForCausalLM(config)

        # Keep the kwargs as a variable to save them to checkpoints later
        optimizer_kwargs = {
            'lr': learning_rate,
        }

        # Get fresh optimizers
        optimizer = AdamW(model.parameters(),  **optimizer_kwargs)

        if lr_scheduler == 'cosine':
            # Keep the kwargs as a variable to save them to checkpoints later
            if accumulation_steps <= 0:
                accumulation_steps = 1
            # Calculate the true number of optimizer update steps
            # We use batch_size * accumulation_steps because the optimizer only steps once per accumulation cycle.
            num_update_steps_per_epoch = len(train_dataset) // (batch_size * accumulation_steps)
            num_training_steps = num_estimated_epochs * num_update_steps_per_epoch

            # Calculate 5% warmup
            num_warmup_steps = int(num_training_steps * 0.05)

            # 3. Define the args
            lr_scheduler_kwargs = {
                'name': 'cosine',
                'num_warmup_steps': num_warmup_steps,
                'num_training_steps': num_training_steps
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
        train_dataset = OnTheFlyMidiDataset(train_files, encodingConfig=EncodingConfig, n_modulations=n_modulations, chunk_size=chunk_size, warmup_steps=warmup_steps)
        valid_dataset = OnTheFlyMidiDataset(valid_files, encodingConfig=EncodingConfig, n_modulations=0, chunk_size=chunk_size, warmup_steps=warmup_steps)
        print(f'Continuing training from epoch {start_epoch}.\n')

    model.to(device, dtype=model_dtype)

    # We need to keep track of the uncompiled model since it holds the uncompiled weights
    # which we want to store in the checkpoint files
    if compile_model:
        # Compile the model to use graph like representation
        print('Compiling model...')
        model_to_train = torch.compile(model)
    else:
        model_to_train = model

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

    # Log every epoch
    writer = SummaryWriter(log_dir=log_dir)

    # ======================
    # = Main training loop =
    # ======================
    progress_bar = tqdm(initial=0, total=num_epochs * len(train_dataloader), desc=f'Training {model_name} for {num_epochs} epochs.')
    for epoch in range(start_epoch, start_epoch + num_epochs):
        # --- Training Step ---
        model_to_train.train()
        total_train_loss = 0.0
        accumulated_loss = 0.0

        for i, (input_ids, attention_mask, labels) in enumerate(train_dataloader):
            # Make a forward pass
            # The model will automatically shift the labels and calculate the loss
            outputs = model_to_train(
                    input_ids=input_ids.to(device),
                    attention_mask=attention_mask.to(device),
                    labels=labels.to(device)
                )

            # The loss is already calculated and available in the output object
            loss = outputs.loss

            # Accumulate the gradient making our effective batch size more stable due to averaging
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
                total_norm = torch.nn.utils.clip_grad_norm_(model_to_train.parameters(), max_norm=1)

                if total_norm > 1e6: raise Exception('Exploding gradients detected!')

                # Make a learning step with the accumulated gradients
                optimizer.step()
                optimizer.zero_grad()

                if lr_scheduler is not None:
                    # Update learning rate
                    lr_scheduler.step()

                # --- Per-Step Logging & Progress Bar ---
                avg_accumulated_loss = accumulated_loss / accumulation_steps
                total_train_loss += avg_accumulated_loss
                accumulated_loss = 0.0
                global_step += 1
                progress_bar.update(accumulation_steps)
                progress_bar.set_postfix({'Training loss': f'{avg_accumulated_loss:.3f}'})
                writer.add_scalar('Per-Step/Train_loss', avg_accumulated_loss, global_step)
                writer.add_scalar('Per-Step/Learning_rate', optimizer.param_groups[0]['lr'], global_step)
                writer.add_scalar('Per-Step/Gradient_norm', total_norm.item(), global_step)

        # --- Per-Epochs Training Logging ---
        # avg_train_loss is calculated over optimizer steps, not batches anymore
        num_optimizer_steps = len(train_dataloader) // accumulation_steps
        avg_train_loss = total_train_loss / num_optimizer_steps if num_optimizer_steps > 0 else 0
        training_loss_per_epoch.append(avg_train_loss)
        writer.add_scalar('Per-Epoch/Training_loss', avg_train_loss, epoch)
        writer.add_scalar('Per-Epoch/Training_perplexity', math.exp(avg_train_loss), epoch)

    # --- Validation Step ---
    model_to_train.eval()
    total_val_loss = 0.0
    with torch.no_grad():
        for input_ids, attention_mask, labels in tqdm(valid_dataloader, desc=f'Epoch {epoch} Validation'):
            outputs = model_to_train(
               input_ids=input_ids.to(device),
               attention_mask=attention_mask.to(device),
               labels=labels.to(device)
            )

            total_val_loss += outputs.loss

    # --- Per-Epoch Validation Logging ---
    avg_val_loss = (total_val_loss.item() / len(valid_dataloader))
    validation_loss_per_epoch.append(avg_val_loss)

    writer.add_scalar('Per-Epoch/Validation_loss', avg_val_loss, epoch)
    writer.add_scalar('Per-Epoch/Validation_perplexity', math.exp(avg_val_loss), epoch)
    print(f'\nEpoch {epoch}: Train Loss={avg_train_loss:.4f} | Val Loss={avg_val_loss:.4f}')

    # Check if we should increase patience counter or reset it
    best_model = False
    best_val_loss = min(validation_loss_per_epoch) if validation_loss_per_epoch else float('inf')
    if avg_val_loss <= best_val_loss:
        patience_counter = 0
        best_model = True
    else:
        patience_counter += 1

    # Make a checkpoint every few epochs or when we aboard due to early stopping or if its the last epoch
    if epoch % checkpointing_per_epochs == 0 or patience_counter >= patience or epoch + 1 >= start_epoch + num_epochs:
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
            'model_dtype': model_dtype,
            'model_state_dict': model.state_dict(),
            'config': model.config.to_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'optimizer_kwargs': optimizer_kwargs,
            'lr_scheduler_state_dict': lr_scheduler.state_dict() if lr_scheduler is not None else None,
            'lr_scheduler_kwargs': lr_scheduler_kwargs if lr_scheduler is not None else None,
        }
        torch.save(checkpoint_data, os.path.join(log_dir, f'checkpoint_epoch_{epoch}.ph'))
        if best_model:
            # Save as current best model
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


def training_manager(epochs_per_session=1, progress_file='runs\progress.json'):
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
            print(f'Model with least progress to train next: {model_to_train}')
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
    train_params.add_argument('--num_epochs', type=int,
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
    run_params.add_argument('--model_name', type=str, default='Phi-3_Model',
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
            model_dtype=args.dtype,
            compile_model=args.compile,
            gradient_checkpointing=args.gradient_checkpointing,
            device=args.device,
            model_name=args.model_name,
            runs_path=args.runs_path,
            debug=args.debug
        )
