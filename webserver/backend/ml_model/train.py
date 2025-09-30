import json

from backend.ml_model.helper import get_device, load_latest_checkpoint, get_next_run_folder, EncodingConfig
from backend.ml_model.dataloader import MidiDataset, MidiRAMDataset
from transformers import get_scheduler, PhiConfig, PhiForCausalLM
import torch
from torch.optim import AdamW
from torch.utils.tensorboard import SummaryWriter
from torch.nn import CrossEntropyLoss
from tqdm import tqdm
import argparse
import math
import os

# Temp
from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorForLanguageModeling


class NetworkConfig:
    # All the instruments which are used in our encoding
    config = PhiConfig(
        vocab_size=EncodingConfig.vocab_size,  # 423
        n_positions=1024,  # Maximum sequence length
        n_ctx=128,  # Context window size
        n_embd=256,  # Embedding size
        n_layer=1,  # Number of transformer layers
        n_head=1,  # Number of attention heads
        pad_token_id=EncodingConfig.padding_token,  # 422
    )


def train(
        # Training Hyperparameters
        num_epochs: int, batch_size: int = 4,
        learning_rate: float = 1e-4, lr_scheduler: str = 'cosine',
        num_estimated_epochs: int = 100,
        patience: int = 4,

        # Dataset parameters
        modulation: int = 0,
        fit_dataset_in_ram: bool = False,
        num_workers: int = 0,

        # Hardware related parameters
        accumulation_steps: int = 4,
        gradient_checkpointing: bool = False,
        device: str = None,

        # Other parameters
        model_name: str = 'Phi-2_Model',
        continue_from: str = None,
        config=None,
        debug: bool = False,
):
    '''Trains a Phi-2-2 language model for music generation on the Lakh Pianoroll Dataset.

    This function provides a complete end-to-end pipeline for training, validation,
    and checkpointing. It handles training from scratch or resuming from a previous
    run, with a configurable cosine learning rate scheduler and comprehensive logging
    to TensorBoard. It also supports early stopping to prevent overfitting and save
    computation time.

    The function expects the dataset to be organized into 'train' and 'valid'
    subdirectories to prevent data leakage during validation. It saves checkpoints
    after every epoch, which include all necessary states (model, optimizer, scheduler)
    and hyperparameters to ensure that training can be reliably paused and resumed.

    :param int num_epochs: The number of epochs to run for this specific training session.
    :param int patience: The number of consecutive epochs to wait for an improvement in
                         validation loss before stopping the training early. Defaults to 4.
    :param int batch_size: The number of sequences in each training batch. Defaults to 4.
    :param int accumulation_steps: The number of gradient accumulation steps. Default to 4.
    :param float learning_rate: The initial learning rate for the AdamW optimizer. Defaults to 1e-4.
    :param str lr_scheduler: The learning rate scheduler type. Currently, only 'cosine' is supported. Defaults to 'cosine'.
    :param int num_estimated_epochs: The total number of epochs you *plan* to train for across all sessions.
                                     This is crucial for creating a consistent learning rate schedule that
                                     doesn't reset when you resume training. Defaults to 100.

    :param int modulation: The number of data augmentations (key modulations) per MIDI file.
                           This must match the pre-processed dataset. Defaults to 0.
    :param bool fit_dataset_in_ram: If True, loads the entire dataset into RAM for faster access using
                                    Phi-2RAMDataset. Requires significant memory. Defaults to False.
    :param int num_workers: The number of workers to use for data loading. Defaults to 0.

    :param bool gradient_checkpointing: If True, enables gradient checkpointing to save memory at the cost
                                        of a small computational overhead. Defaults to False.
    :param str device: The hardware device to use for training ('cpu', 'cuda', 'xpu'). If None, it will
                       be auto-detected. Defaults to None.

    :param str model_name: The base name used for creating the logging and checkpoint directory.
                           Defaults to 'Phi-2_Model'.
    :param str continue_from: The name of the run directory (e.g., 'Phi-2_Model_run1') to resume
                              training from. If None, starts a new run. Defaults to None.
    :param config: A custom Hugging Face Phi-2Config object. If None, a default configuration from
                   NetworkConfig is used. Defaults to None.
    :param bool debug: If True, enables additional debugging features. Defaults to False.

    :raises NotImplementedError: If an unsupported `lr_scheduler` is specified.
    :raises Exception: If NaN values are detected in the loss or if gradients explode.

    :returns: Bool. The function saves all outputs (logs, checkpoints) to disk.
    :rtype: Bool
    '''
    ### TODO: Figure out why the anomaly detection prevents exploding gradients when increasing the batch size from 6 to 16
    torch.autograd.set_detect_anomaly(debug)
    # =================
    # = Initial setup =
    # =================
    device = get_device(device)
    print(f'Using device: {device}\n')

    root_path = os.path.dirname(os.path.abspath(__file__))
    # Get dataset and dataloader
    if fit_dataset_in_ram:
        train_dataset = MidiRAMDataset(os.path.join(root_path, f'lpd_5_dataset_da_{modulation}', 'train'))
        valid_dataset = MidiRAMDataset(os.path.join(root_path, f'lpd_5_dataset_da_0', 'valid'))
    else:
        train_dataset = MidiDataset(os.path.join(root_path, f'lpd_5_dataset_da_{modulation}', 'train'))
        valid_dataset = MidiDataset(os.path.join(root_path, f'lpd_5_dataset_da_0', 'valid'))

    # Only shuffle the dataset if it fits in ram, otherwise the file loading mechanism would prevent shuffling
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=fit_dataset_in_ram,
        num_workers=num_workers if fit_dataset_in_ram else 0
    )
    valid_dataloader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=batch_size, shuffle=False, num_workers=0
    )

    if continue_from is not None and os.path.isdir(os.path.join(root_path, 'runs', continue_from)):
        # Continue a previous run
        log_dir = os.path.join(root_path, 'runs', continue_from)
        model, training_loss_per_epoch, validation_loss_per_epoch, patience_dict, start_epoch, global_step, optimizer, optimizer_kwargs, lr_scheduler, lr_scheduler_kwargs = load_latest_checkpoint(
            log_dir,
            device=device,
            optimizer_class=AdamW,
            learning_rate_scheduler_class=get_scheduler,
        )
        patience = patience_dict['patience']
        patience_counter = patience_dict['patience_counter']
        num_warmup_steps = lr_scheduler_kwargs['num_warmup_steps']
        model.to(device)
        print(f'Continuing training from epoch {start_epoch}\n')
    else:
        # Start a new run
        start_epoch = 1
        global_step = 0
        patience_counter = 0
        training_loss_per_epoch = []
        validation_loss_per_epoch = []

        print('Training from scratch.')
        if config is None:
            print('Using default config.\n')
            config = NetworkConfig.config
        else:
            print('Using provided config.\n')
        # Initializing model weights is done automatically by providing initializer_range=0.02 in the config
        model = PhiForCausalLM(config)
        model.to(device)

        # ==================
        # = Training setup =
        # ==================

        # Keep the kwargs as a variable to save them to checkpoints later
        optimizer_kwargs = {
            'lr': learning_rate,
        }

        # Get fresh optimizers and log dir
        optimizer = AdamW(model.parameters(), **optimizer_kwargs)
        log_dir = get_next_run_folder(model_name, base_dir=os.path.join(root_path, 'runs'))
        os.makedirs(log_dir, exist_ok=True)

        if lr_scheduler == 'cosine':
            # Keep the kwargs as a variable to save them to checkpoints later
            num_warmup_steps = 500 // accumulation_steps
            lr_scheduler_kwargs = {
                'name': 'cosine',
                'num_warmup_steps': num_warmup_steps,
                'num_training_steps': num_estimated_epochs * len(train_dataloader)
            }
            lr_scheduler = get_scheduler(optimizer=optimizer, **lr_scheduler_kwargs)
        elif lr_scheduler is not None:
            raise NotImplementedError(f'Learning rate scheduler {lr_scheduler} not implemented.')

    if gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False

    # Log every 20 accumulated steps
    logging_interval = 5
    writer = SummaryWriter(log_dir=log_dir)
    print(f'Logging to: {log_dir}\n')
    print('Phi2 Config:', {k: model.config.to_dict()[k] for k in
                           ['hidden_size', 'intermediate_size', 'num_hidden_layers', 'num_attention_heads']})
    print(
        f'Hyperparameters: lr={learning_rate}, batch_size={batch_size}, accumulation_steps={accumulation_steps}, num_warmup_steps={num_warmup_steps}\n')

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
            # Our dataset returns a tensor of size (16, 1024) of dtype int64 always...
            input_ids = batch[0].to(device).long()

            # Make a forward pass
            # outputs = model(input_ids=input_ids)
            # Remove the last timestep because it does not predict anything
            # shift_logits = outputs.logits[..., :-1, :].contiguous()
            # Remove the first timestep because it does not have a previous to predict
            # shift_labels = input_ids[..., 1:].contiguous()

            # loss = loss_fn(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

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
                global_step += 1
                progress_bar.update(accumulation_steps)

                if global_step % logging_interval == 0:
                    writer.add_scalar('Per-Step/Training Loss', avg_accumulated_loss, global_step)
                    writer.add_scalar('Per-Step/Learning Rate', last_lr, global_step)
                    writer.add_scalar('Per-Step/Gradient Norm', total_norm, global_step)
                    progress_bar.set_postfix({'Loss': f'{detached_loss:.4f}', 'LR': f'{last_lr:.6f}'})

                accumulated_loss = 0.0

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
            # For testing we dont stop until we have reached our estimated amount of epochs
            # print(f'\nEpoch {epoch}: Validation Loss has not improved in {patience} epochs. Stopping training.')
            # writer.close()
            # return True
            pass

        if epoch >= num_estimated_epochs:
            # Stop at estimated epochs
            writer.close()
            return True

    print('Training completed!')
    writer.close()
    return False


MODEL_CONFIGURATIONS = {
    'Phi-2_Small': {
        'config': PhiConfig(
            vocab_size=EncodingConfig.vocab_size, pad_token_id=EncodingConfig.padding_token,
            eos_token_id=EncodingConfig.end_note,
            max_position_embeddings=1024, hidden_size=256, intermediate_size=1024, num_hidden_layers=4,
            num_attention_heads=4,
            tie_word_embeddings=True, layer_norm_eps=1e-5, rope_theta=10000.0, initializer_range=0.02,
        ),
        'hyperparameters': {
            'num_epochs': 2, 'patience': 5, 'batch_size': 6, 'accumulation_steps': 1, 'learning_rate': 5e-4,
            'lr_scheduler': 'cosine', 'num_estimated_epochs': 20, 'modulation': 0, 'fit_dataset_in_ram': True,
            'num_workers': 0, 'gradient_checkpointing': False, 'device': 'xpu', 'model_name': 'Phi-2_Small',
            'continue_from': 'Phi-2_Small_1',
        }
    },
    'Phi-2_Medium': {
        'config': PhiConfig(
            vocab_size=EncodingConfig.vocab_size, pad_token_id=EncodingConfig.padding_token,
            eos_token_id=EncodingConfig.end_note,
            max_position_embeddings=1024, hidden_size=384, intermediate_size=1536, num_hidden_layers=6,
            num_attention_heads=6,
            tie_word_embeddings=True, layer_norm_eps=1e-5, rope_theta=10000.0, initializer_range=0.02,
        ),
        'hyperparameters': {
            'num_epochs': 2, 'patience': 5, 'batch_size': 6, 'accumulation_steps': 1, 'learning_rate': 5e-4,
            'lr_scheduler': 'cosine', 'num_estimated_epochs': 30, 'modulation': 0, 'fit_dataset_in_ram': True,
            'num_workers': 0, 'gradient_checkpointing': False, 'device': 'xpu', 'model_name': 'Phi-2_Medium',
            'continue_from': 'Phi-2_Medium_1',
        }
    },
    'Phi-2_Large': {
        'config': PhiConfig(
            vocab_size=EncodingConfig.vocab_size, pad_token_id=EncodingConfig.padding_token,
            eos_token_id=EncodingConfig.end_note,
            max_position_embeddings=1024, hidden_size=512, intermediate_size=2048, num_hidden_layers=8,
            num_attention_heads=8,
            tie_word_embeddings=True, layer_norm_eps=1e-5, rope_theta=10000.0, initializer_range=0.02,
        ),
        'hyperparameters': {
            'num_epochs': 2, 'patience': 5, 'batch_size': 6, 'accumulation_steps': 1, 'learning_rate': 5e-4,
            'lr_scheduler': 'cosine', 'num_estimated_epochs': 30, 'modulation': 0, 'fit_dataset_in_ram': True,
            'num_workers': 0, 'gradient_checkpointing': False, 'device': 'xpu', 'model_name': 'Phi-2_Large',
            'continue_from': 'Phi-2_Large_1',
        }
    },
    'Phi-2_Small_da_6': {
        'config': PhiConfig(
            vocab_size=EncodingConfig.vocab_size, pad_token_id=EncodingConfig.padding_token,
            eos_token_id=EncodingConfig.end_note,
            max_position_embeddings=1024, hidden_size=256, intermediate_size=1024, num_hidden_layers=4,
            num_attention_heads=4,
            tie_word_embeddings=True, layer_norm_eps=1e-5, rope_theta=10000.0, initializer_range=0.02,
        ),
        'hyperparameters': {
            'num_epochs': 2, 'patience': 5, 'batch_size': 6, 'accumulation_steps': 1, 'learning_rate': 5e-4,
            'lr_scheduler': 'cosine', 'num_estimated_epochs': 40, 'modulation': 6, 'fit_dataset_in_ram': True,
            'num_workers': 0, 'gradient_checkpointing': False, 'device': 'xpu', 'model_name': 'Phi-2_Small_da_6',
            'continue_from': 'Phi-2_Small_da_6_1',
        }
    },
    'Phi-2_Medium_da_6': {
        'config': PhiConfig(
            vocab_size=EncodingConfig.vocab_size, pad_token_id=EncodingConfig.padding_token,
            eos_token_id=EncodingConfig.end_note,
            max_position_embeddings=1024, hidden_size=384, intermediate_size=1536, num_hidden_layers=6,
            num_attention_heads=6,
            tie_word_embeddings=True, layer_norm_eps=1e-5, rope_theta=10000.0, initializer_range=0.02,
        ),
        'hyperparameters': {
            'num_epochs': 2, 'patience': 5, 'batch_size': 6, 'accumulation_steps': 1, 'learning_rate': 5e-4,
            'lr_scheduler': 'cosine', 'num_estimated_epochs': 50, 'modulation': 6, 'fit_dataset_in_ram': True,
            'num_workers': 0, 'gradient_checkpointing': False, 'device': 'xpu', 'model_name': 'Phi-2_Medium_da_6',
            'continue_from': 'Phi-2_Medium_da_6_1',
        }
    },
    'Phi-2_Large_da_6': {
        'config': PhiConfig(
            vocab_size=EncodingConfig.vocab_size, pad_token_id=EncodingConfig.padding_token,
            eos_token_id=EncodingConfig.end_note,
            max_position_embeddings=1024, hidden_size=512, intermediate_size=2048, num_hidden_layers=8,
            num_attention_heads=8,
            tie_word_embeddings=True, layer_norm_eps=1e-5, rope_theta=10000.0, initializer_range=0.02,
        ),
        'hyperparameters': {
            'num_epochs': 2, 'patience': 5, 'batch_size': 6, 'accumulation_steps': 1, 'learning_rate': 5e-4,
            'lr_scheduler': 'cosine', 'num_estimated_epochs': 60, 'modulation': 6, 'fit_dataset_in_ram': True,
            'num_workers': 0, 'gradient_checkpointing': False, 'device': 'xpu', 'model_name': 'Phi-2_Large_da_6',
            'continue_from': 'Phi-2_Large_da_6_1',
        }
    }
}


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
    data_params = parser.add_argument_group('Dataset & Hardware Parameters')
    run_params = parser.add_argument_group('Run Management Parameters')

    # --- Training Parameters ---
    train_params.add_argument('--num_epochs', type=int, required=True,
                              help='Number of epochs for this training session.')
    train_params.add_argument('--patience', type=int, default=4,
                              help='Epochs to wait for validation loss improvement before stopping.')
    train_params.add_argument('--batch_size', type=int, default=16,
                              help='Batch size for training.')
    train_params.add_argument('--learning_rate', type=float, default=1e-4,
                              help='Initial learning rate for the optimizer.')
    train_params.add_argument('--lr_scheduler', type=str, default='cosine', choices=['cosine'],
                              help='Learning rate scheduler type.')
    train_params.add_argument('--num_estimated_epochs', type=int, default=100,
                              help='Total planned epochs for creating a consistent LR schedule.')

    # --- Dataset & Hardware Parameters ---
    data_params.add_argument('--modulation', type=int, default=0,
                             help='Number of data augmentations (key modulations) per MIDI file.')
    data_params.add_argument('--fit_dataset_in_ram', action='store_true',
                             help='Load the entire dataset into RAM. Requires significant memory.')
    data_params.add_argument('--gradient_checkpointing', action='store_true',
                             help='Enable gradient checkpointing to save memory.')
    data_params.add_argument('--device', type=str, default=None,
                             help='Device to use: cpu, cuda, or xpu (auto-select if None).')

    # --- Run Management Parameters ---
    run_params.add_argument('--model_name', type=str, default='Phi-2_Model',
                            help='Base name for the run and logging directory.')
    run_params.add_argument('--continue_from', type=str, default=None,
                            help='Name of the run directory (e.g., "Phi-2_Model_run1") to resume training from.')
    run_params.add_argument('--training_manager', action='store_true',
                            help='If specified, ignores other CLI arguments and runs a hardcoded training session.')

    args = parser.parse_args()

    if args.training_manager:
        training_manager()
    else:
        # Pass all the relevant arguments to the train function
        train(num_epochs=args.num_epochs,
              patience=args.patience,
              batch_size=args.batch_size,
              learning_rate=args.learning_rate,
              lr_scheduler=args.lr_scheduler,
              num_estimated_epochs=args.num_estimated_epochs,
              modulation=args.modulation,
              fit_dataset_in_ram=args.fit_dataset_in_ram,
              gradient_checkpointing=args.gradient_checkpointing,
              device=args.device,
              model_name=args.model_name,
              continue_from=args.continue_from)
