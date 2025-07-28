from backend.ml_model.train import EncodingConfig
from backend.ml_model.generate import generate_from_chords, sliding_window_generate
from backend.ml_model.dataloader import GPT2Dataset
from backend.ml_model.train import NetworkConfig
from backend.ml_model.helper import chord2tokens, mid_to_mp3
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import pypianoroll
from transformers import GPT2Tokenizer, GPT2LMHeadModel, GPT2Config
from datasets import Dataset
import torch
from torch.optim import AdamW
import pickle
import os
import glob
import time
import requests

EncodingConfig.initialize()

matplotlib.use('TkAgg')


def generating_for_model_paths():
    model_paths = [
        'backend/ml_model/runs/GPT2_Tiny_1',
        'backend/ml_model/runs/GPT2_Tiny_2',
        'backend/ml_model/runs/GPT2_Tiny_3',
        'backend/ml_model/runs/GPT2_Small_1',
        'backend/ml_model/runs/GPT2_Small_2',
        'backend/ml_model/runs/GPT2_Small_3',
        'backend/ml_model/runs/GPT2_Medium_1',
        'backend/ml_model/runs/GPT2_Medium_2',
        'backend/ml_model/runs/GPT2_Medium_3',
    ]
    URL = 'http://localhost:5000/generate-music'
    SESSION_ID = 'dcecd20d-137a-4f53-af7f-5c3a8cf5ea94'
    BPM = 100

    CHORD_PROGRESSIONS = [
        'Am:32|C:32|D:32|F:32',  # House of the rising sun (simple)
        'Cm7:32|Fm7:32|Dm7-5:16|G7#5:16|Cm7:32'  # Blue bossa (complex)
    ]

    # Preserver session over all post requests
    session = requests.Session()

    def send_post(model_path, chord_progression):
        payload = {
            'session_id': SESSION_ID,
            'bpm': BPM,
            'model_path': model_path,
            'chord_progression': chord_progression
        }

        try:
            print(f'Sending post for {model_path} and {chord_progression}')
            response = session.post(URL, json=payload)
            if response.status_code == 200:
                print(f'Success: {model_path} | {chord_progression}')
            else:
                print(f'Error {response.status_code}: {response.text}')
            return response.status_code == 200
        except requests.RequestException as e:
            print(f'Request failed: {e}')
            return False

    # Generate chord progressions for each model
    for model_path in model_paths:
        for chord in CHORD_PROGRESSIONS:
            send_post(model_path, chord)


def test_encoding_decoding():
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Change the working directory to the script's directory
    os.chdir(script_dir)
    file_name = 'b97c529ab9ef783a849b896816001748'
    # Grap one of the encoded midi files
    original_file_path = os.path.join('lpd_5', 'lpd_5_cleansed', 'A', 'A', 'A', 'TRAAAGR128F425B14B',
                                      f'{file_name}.npz')
    encoded_file_path = os.path.join('lpd_5', 'lpd_5_cleansed', 'A', 'A', 'A', 'TRAAAGR128F425B14B',
                                     f'{file_name}.npz.tmp')
    # Remove old one if it still exists
    if os.path.exists(encoded_file_path):
        os.remove(encoded_file_path)

    # Encode the file and save it in the same place just with .tmp added
    encode(original_file_path, 'lpd_5/lpd_5_cleansed/*/*/*/*/*.npz')

    # Open original as a Multitrack object
    mt_original = pypianoroll.load(original_file_path)

    tempo = int(mt_original.tempo[0])
    # Get length in time steps (probably quarter notes)
    length = mt_original.get_max_length()

    # Open encoded file as numpy array
    with open(encoded_file_path, mode='rb') as f:
        file_chunk = pickle.load(f)

    # Create empty pianoroll array
    pianoroll = np.zeros((len(EncodingConfig.tracks), length, 128))

    pos = 0
    # Decode it again
    for note in file_chunk:
        if note == EncodingConfig.time_note:
            # Either way update the position by 1
            pos += 1
            if pos >= pianoroll.shape[1]:
                break
        elif note < EncodingConfig.time_note:
            # We have hit an actually generated note, thus we put it into its right place in our pianoroll array
            # Calculate which track the note belongs to
            # 0 = Bass -> 3 = Bass
            # 1 = Drums -> 0 = Drums
            # 2 = Piano -> 1 = Piano
            # 3 = Guitar -> 2 = Guitar
            # 4 = String -> 4 Strings
            trc = EncodingConfig.trc_idx[note // EncodingConfig.note_size]
            # Calculate the midi note value
            mid = note % EncodingConfig.note_size + EncodingConfig.note_offset
            # Set the volume to 100 for the note in the piano roll array
            if mid < 128:
                pianoroll[trc, pos, mid] = 100

    pr = []
    for i, t in enumerate(EncodingConfig.tracks):
        pr.append(pypianoroll.StandardTrack(pianoroll=pianoroll[i], program=EncodingConfig.programs[t],
                                            is_drum=(t == 'Drums')))
    mt_decoded = pypianoroll.Multitrack(tracks=pr, tempo=np.full(pianoroll.shape[1], tempo), resolution=4)

    # Set a few flags to making working in the daw easier
    for track in mt_original.tracks:
        if not track.is_drum:
            track.name = f'Program: {track.program}'
        else:
            track.name = f'Program: 0 (Drums)'

    mt_original.write(os.path.join('tmp', 'original.mid'))

    for track in mt_decoded.tracks:
        if not track.is_drum:
            track.name = f'Program: {track.program}'
        else:
            track.name = f'Program: 0 (Drums)'

    mt_decoded.write(os.path.join('tmp', 'decoded.mid'))


def _reorder_current(cur_seq):
    # Reorder the sequence so that piano is the last instrument
    # Bass: [0 * 84, 0 * 84 + 83] = [0, 83]
    # Drums: [1 * 84, 1 * 84 + 83] = [84, 167]
    # Piano: [2 * 84, 2 * 84 + 83] = [168, 251]
    # Guitar: [3 * 84, 3 * 84 + 83] = [252, 335]
    # Strings: [4 * 84, 4 * 84 + 83] = [336, 419]

    cur = []
    for c in sorted(cur_seq):
        # Checks if a note c is not in the range [84, 168)
        # i.e. if the instrument is not drums
        if not (c >= EncodingConfig.note_size and c < EncodingConfig.note_size * 2):
            cur.append(c)
    for c in sorted(cur_seq):
        # Checks if a note c is in the range [84, 168)
        # i.e. if the instrument is drums
        if (c >= EncodingConfig.note_size and c < EncodingConfig.note_size * 2):
            cur.append(c)

    return cur  # Bass, Piano, etc..., Drums


def encode(file_path, dataset_path):
    number_of_modulations = 0
    tracks = ['Drums', 'Piano', 'Guitar', 'Bass', 'Strings']
    trc_len = len(tracks)
    trc_idx = sorted(list(range(trc_len)), key=lambda x: 0 if tracks[x] == 'Bass' else 1)

    # Get an average of each pitch value per track over all datapoints
    trc_avg_c = np.zeros((len(tracks), 2))
    for fp in glob.glob(dataset_path):
        # Load it as a multitrack object
        m = pypianoroll.load(fp)
        # Convert it to a numpy array of shape (num_time_steps, num_pitches=128, num_tracks)
        pr = m.stack()
        # Transpose the array to have shape (num_time_steps, num_pitches, num_tracks)
        pr = np.transpose(pr, (1, 2, 0))
        # Change ordering to get the bass first and have consistent indexing
        pr = pr[:, :, trc_idx]
        # Get all time steps of notes being played
        p = np.where(pr != 0)
        for i in range(len(p[0])):
            # Track of the note at timestep i
            track = p[2][i]
            # The pitch of the note
            pitch = p[1][i]
            # Save them into our array
            trc_avg_c[track, 0] += pitch
            trc_avg_c[track, 1] += 1
        del m, pr, p, track, pitch

    # Replace 0 in cur_avg_c[:, 1] with 1
    # In order to avoid runtime errors
    trc_avg_c[:, 1] = np.where(trc_avg_c[:, 1] == 0, 1, trc_avg_c[:, 1])

    # Calculate the average note value per track
    trc_avg = np.where(trc_avg_c[:, 1] > 0, trc_avg_c[:, 0] / trc_avg_c[:, 1], 60)
    del trc_avg_c

    seq = []
    # Load it as a multitrack object
    m = pypianoroll.load(file_path)
    # Beat resolution is the number of steps a measure is divided into
    # Calculate how many time steps a quarter note takes to fill a measure
    step = m.resolution // 4
    # Convert it to a numpy array of shape (num_tracks, num_time_steps, num_pitches=128)
    pr = m.stack()

    for i, track in enumerate(m.tracks):
        print(f'Track {i}: {track.name}, Program: {track.program}, Is Drum: {track.is_drum}')

    # Transpose the array to have shape (num_time_steps, num_pitches, num_tracks)
    pr = np.transpose(pr, (1, 2, 0))
    # Change ordering to get the bass first and have consistent indexing
    pr = pr[:, :, trc_idx]
    # This changes the indexing to:
    # 0 = Bass (Program: 32)
    # 1 = Drums (Program: Drums (0))
    # 2 = Piano (Program: 0)
    # 3 = Guitar (Program: 24)
    # 4 = String (Program: 48)

    # Again get all time steps of notes being played
    p = np.where(pr != 0)
    # Calculate the average pitch of this song per track
    cur_avg_c = np.zeros((len(EncodingConfig.tracks), 2))
    for i in range(len(p[0])):
        # Track of the note at timestep i
        track = p[2][i]
        # The pitch of the note
        pitch = p[1][i]
        # Save them into our array
        cur_avg_c[track, 0] += pitch
        cur_avg_c[track, 1] += 1

    # Replace count of 0 to 1
    cur_avg_c[:, 1] = np.where(cur_avg_c[:, 1] == 0, 1, cur_avg_c[:, 1])
    # Perform the division safely
    cur_avg = np.where(cur_avg_c[:, 1] > 0, cur_avg_c[:, 0] / cur_avg_c[:, 1], 60)
    # Create list with [0, random permutation of the numbers 1 - 12]
    modulation = [0] + (np.random.permutation(11) + 1).tolist()

    current_seq = []
    # Do data augmentation according to specified parameter 'da' in range [0-11]
    # All sequences are appended to the same list
    for s in modulation[:number_of_modulations + 1]:
        pos = 0
        # Create an encoding sequence for each modulation
        for i in range(len(p[0])):
            # Ignore smaller notes then sixteenth notes or notes that are not on the gird
            # only look at indices 0, 6, 12, 18, 24
            if p[0][i] % step != 0:
                continue
            # Only if we have advanced in the position will the current cur_seq be written down
            # More note encodings can be placed on the same sixteenth note since they can be
            # from different instruments or an instrument can play more notes at once
            if pos < p[0][i]:
                # From last position to next note occurrence write down the notes which are played
                # by the instruments
                for _ in range(pos, p[0][i], step):
                    seq.extend(_reorder_current(current_seq))
                    seq.append(EncodingConfig.time_note)
                    current_seq = []
            # Set current position to the last note occurrence
            pos = p[0][i]
            # Get current pitch
            pitch = p[1][i]
            # Get current track
            track = p[2][i]

            shift = 0
            # Do this for every track expect drums
            if track != 1:
                # Decide if we apply a shift of one octave downwards when modulating if
                # The hypothetical pitch average of the current track after applying modulation s
                # <
                # A threshold based on the datasets global pitch average for this track plus a buffer of 6
                if cur_avg[track] + s < trc_avg[track] + 6:
                    shift = s
                else:
                    shift = s - 12

            # Adjust the pitch of the note with the shift
            # If modulation is 0 (default) then we don't change the pitch at all
            pitch = pitch + shift

            # Bring the pitch back up one octave if for whatever reason we shifted
            # the note to far down
            if pitch < 0:
                pitch += 12
            # Bring the pitch down if we exceed 127. The highest value of midi files (0x7F)
            if pitch > 127:
                pitch -= 12
            # Apply our offset to encode less notes [0, 83]
            pitch -= EncodingConfig.note_offset

            # Do some checks again to ensure that the note is within our interval of wanted notes
            # [0, note_size (84)]
            # Since notes above 84 are not used, and we want to keep the dictionary concise
            if pitch < 0:
                pitch = 0
            if pitch > EncodingConfig.note_size:
                pitch = EncodingConfig.note_size - 1

            # Finally calculate the number which represents the note track and pitch all together
            note = track * EncodingConfig.note_size + pitch
            # Append it to our current sequence
            current_seq.append(note)

        # Fill in the last note which is played
        seq.extend(_reorder_current(current_seq))
        # And a last time_note with the end_note as well
        seq.append(EncodingConfig.time_note)
        seq.append(EncodingConfig.end_note)
        current_seq = []
    # Store all the sequences, lists of tokens, as a pickle file
    with open(file_path + '.tmp', mode='wb') as f:
        pickle.dump(seq, f)


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

    generated_sequence_1 = torch.tensor([context_sequence], dtype=torch.long)

    for i in range(1024):
        # Cut the sequence if it grows above our context size minus 1
        generated_sequence_1 = generated_sequence_1[:, -(1024 - 1):]

        # Manually generating tokens one by one
        output = model(generated_sequence_1)
        logits = output.logits  # Extract logits

        probs = torch.nn.functional.softmax(logits[:, -1, :], dim=-1)  # Apply softmax

        next_tokens = torch.multinomial(probs, num_samples=1)  # Sample tokens

        # Append new token to our sequence
        generated_sequence_1 = torch.concatenate([generated_sequence_1, next_tokens], dim=-1)
    generated_sequence_1 = generated_sequence_1.squeeze(0)

    # Use predefined generation function
    generated_sequence_2 = generate_from_context(model, context_sequence, device)
    # Use predefined generation function with sliding window approach
    generated_sequence_3 = sliding_window_generate(model, context_sequence, max_tokens=1024)

    print(
        f'Max and min from method 1: {generated_sequence_1.max()}, {generated_sequence_1.min()}, Shape: {generated_sequence_1.shape}')
    print(generated_sequence_1[:50])

    print(
        f'Max and min from method 2: {generated_sequence_2.max()}, {generated_sequence_2.min()}, Shape: {generated_sequence_2.shape}')
    print(generated_sequence_2[:50])

    print(
        f'Max and min from method 3: {generated_sequence_3.max()}, {generated_sequence_3.min()}, Shape: {generated_sequence_3.shape}')
    print(generated_sequence_3.squeeze(0)[:50])

    print('fin')


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


def testing_generation_function(simple: bool = True, gpt_version: str = None):
    if gpt_version is None:
        raise ValueError('Chose valid gpt version from runs folder')

    script_dir = os.path.dirname(os.path.abspath(__file__))

    if simple:
        chord_progression = ['C', 'G', 'Am', 'F']
        chord_timings = [32, 32, 32, 32]
    else:
        chord_progression = ['Cm7', 'Fm7', 'Dm7-5', 'G7#5', 'Cm7']
        chord_timings = [32, 32, 32, 32, 32]

    file_name = '_'.join(chord_progression) + f'_{gpt_version}'

    midi_file = os.path.join(script_dir, 'tmp', f'{file_name}.mid')
    model_file = os.path.join(script_dir, 'runs', gpt_version)

    generate_from_chords(chord_progression, chord_timings, 100, model_file, midi_file)

    mid_to_mp3(midi_file, os.path.join(script_dir, 'tmp', 'SoundFont.sf2'), os.path.join(script_dir, 'tmp', f'{file_name}.mp3'))
    print('convert fin')


if __name__ == '__main__':
    testing_generation_function(False, 'GPT2_Small_3')
