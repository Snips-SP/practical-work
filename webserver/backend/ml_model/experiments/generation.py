import os
import numpy as np
import torch
from backend.ml_model.train import NetworkConfig
from backend.ml_model.generate import generate_from_context, sliding_window_generate, generate_from_chords
import requests
from transformers import GPT2LMHeadModel
from backend.ml_model.helper import chord2tokens, mid_to_mp3, EncodingConfig

EncodingConfig.initialize()


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

    mid_to_mp3(midi_file, os.path.join(script_dir, 'tmp', 'SoundFont.sf2'),
               os.path.join(script_dir, 'tmp', f'{file_name}.mp3'))
    print('convert fin')
