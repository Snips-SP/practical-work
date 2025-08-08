import torch
from datasets import Dataset
from torch.optim import AdamW
from tqdm import tqdm
from transformers import GPT2Config, GPT2Tokenizer, GPT2LMHeadModel
from backend.ml_model.helper import EncodingConfig
from backend.ml_model.train import train
from backend.ml_model.train import NetworkConfig

EncodingConfig.initialize()

def train_many_configs():
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
    num_epochs = 2
    batch_size = 16
    learning_rate = 1e-4
    lr_scheduler = 'cosine'
    gradient_checkpointing = False
    RAM_dataset = True
    device = 'xpu'

    # Train with same parameters but different configs
    for name, config in [('tiny', config1), ('small', config2), ('medium', config3)]:
        train(num_epochs=num_epochs,
              batch_size=batch_size,
              learning_rate=learning_rate,
              lr_scheduler=lr_scheduler,
              gradient_checkpointing=gradient_checkpointing,
              RAM_dataset=RAM_dataset,
              device=device,
              continue_from=None,
              model_name=name,
              config=config)


def training_test():
    # Set training parameters
    num_epochs = 1
    batch_size = 8

    # Use appropriate gpu or cpu
    device = ('xpu' if torch.xpu.is_available() else
              'cuda' if torch.cuda.is_available() else
              'cpu')
    print('Using device:', device)

    # Dummy dataset: Repeating a simple sentence
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    text_data = ['Hello world! This is a test sentence.'] * 1000
    tokenized_data = tokenizer(text_data, truncation=True, padding='max_length', max_length=1024, return_tensors='pt')

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
        batch_input_ids = torch.zeros((batch_size, 1024))
        batch_attention_mask = torch.zeros((batch_size, 1024))
        for batch_idx, batch in enumerate(dataset):
            # Manual batching
            batch_input_ids[batch_idx % batch_size, :] = torch.tensor(batch['input_ids'])
            batch_attention_mask[batch_idx % batch_size, :] = torch.tensor(batch['attention_mask'])

            # If the current batch tensor is full we feed it to the network
            if batch_input_ids[-1, 0] != 0:
                # Move them to gpu
                batch_input_ids = batch_input_ids.to(device).long()
                batch_attention_mask = batch_attention_mask.to(device).long()

                outputs = model(input_ids=batch_input_ids,
                                attention_mask=batch_attention_mask,
                                labels=batch_input_ids)

                invalid_tokens = (batch_input_ids >= model.config.vocab_size).any()
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
                # Zero out batch for the next run
                batch_input_ids = torch.zeros((batch_size, 1024))
                batch_attention_mask = torch.zeros((batch_size, 1024))

            progress_bar.update(1)

        train_loss.append(total_loss / len(dataset))

    # Test the network if it learned our dummy dataset
    prompt = 'Hello'
    input_ids = tokenizer(prompt, return_tensors='pt').input_ids.to(device)

    output = model.generate(input_ids, max_length=1000, num_return_sequences=1)
    print(f'Test Prompt: {prompt}')
    print(tokenizer.decode(output[0], skip_special_tokens=True))
    print('Training completed!')



