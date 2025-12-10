import requests
import os
import random


def get_random_drum_selection(base_dir):
    if not os.path.exists(base_dir):
        print(f"Warning: Drum directory {base_dir} not found.")
        return None, None

    # Get valid categories (folders that are not hidden)
    categories = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]

    if not categories:
        return None, None

    selected_category = random.choice(categories)
    category_path = os.path.join(base_dir, selected_category)

    # Get valid files (.mid or .midi)
    patterns = [f for f in os.listdir(category_path) if f.endswith('.mid') or f.endswith('.midi')]

    if not patterns:
        return selected_category, None

    selected_pattern = random.choice(patterns)

    return selected_category, selected_pattern


def generating_for_model_paths():
    # Configuration
    URL = 'http://localhost:5000/generate-music'
    DRUM_SEEDS_DIR = os.path.join('backend', 'ml_model', 'seeds', 'drums')

    model_paths = [
        'backend/ml_model/runs/Phi-3-33M-head-dim-32-GQA-ratio-16-8-ebs-64-lr-1e-3-epochs-100',
        'backend/ml_model/runs/Phi-3-33M-head-dim-64-GQA-ratio-1-ebs-64-lr-1e-3-epochs-100',
    ]

    # Define the musical structures (Chords + Timings + BPM)
    musical_contexts = [
        {
            'name': 'Simple', # Baseline check for simple chords
            'bpm': 100,
            'chords': ['Am', 'C', 'D', 'F'],
            'timings': [16, 16, 16, 16],
        },{
            'name': 'Complex', # Check for complex chords
            'bpm': 120,
            'chords': ['Cm7', 'Fm7', 'Dm7-5', 'G7#5', 'Cm7'],
            'timings': [16, 16, 8, 8, 16],
        },{
            'name': 'Cinematic', # For long a range coherency test
            'bpm': 70,
            'chords': ['Am', 'Fmaj7', 'C', 'G', 'Dm', 'Am', 'E7', 'Am9'],
            'timings': [32, 32, 32, 32, 16, 16, 32, 64],
        },{
            'name': 'Neo-Soul_LoFi', # Check if it can generate a popular genre
            'bpm': 85,
            'chords': ['Dbmaj9', 'Cm7', 'Fm9', 'Bbm7', 'Eb9'],
            'timings': [16, 16, 16, 8, 8],
        },{
            'name': 'Classical', # Check if it generates classical (more niece in our dataset)
            'bpm': 90,
            'chords': ['Dm', 'G7', 'Cmaj7', 'Fmaj7', 'Bm7-5', 'E7', 'Am'],
            'timings': [8, 8, 8, 8, 8, 8, 16],
        },{
            'name': 'Modulation', # Test if it can handle key changes
            'bpm': 110,
            'chords': ['C', 'Am', 'F', 'G', 'E7', 'A', 'F#m', 'E'],
            'timings': [16, 16, 16, 16, 16, 16, 16, 16],
        }
    ]

    # Define the generation hyperparameters (Temp, Top_K, Top_P)
    hyperparameters = [
        {'tag': 'Baseline', 'temperature': 0.7, 'top_k': 0, 'top_p': 0.99},
        {'tag': 'HighTemp', 'temperature': 0.8, 'top_k': 0, 'top_p': 0.99},
        {'tag': 'LowTopP', 'temperature': 0.9, 'top_k': 0, 'top_p': 0.99},
        {'tag': 'HighTopP', 'temperature': 1.0, 'top_k': 0, 'top_p': 0.99},
        {'tag': 'Experimental', 'temperature': 1.2, 'top_k': 0, 'top_p': 0.99}
    ]

    session = requests.Session()

    def send_post(model_path, context, params):
        # Pick a random drum beat for this specific generation
        drum_cat, drum_pat = get_random_drum_selection(DRUM_SEEDS_DIR)

        # Construct a name
        name = 'phi-3-hd-64' if 'head-dim-64' in model_path else 'phi-3-hd-32'
        song_name = f'{name}_{context["name"]}_T{params["temperature"]}_P{params["top_p"]}_{drum_cat}'

        payload = {
            'song_name': song_name,
            'bpm': context['bpm'],
            'model_path': model_path,
            'chords': context['chords'],
            'timings': context['timings'],
            'temperature': params['temperature'],
            'top_k': params['top_k'],
            'top_p': params['top_p'],
            'drum_category': drum_cat,
            'drum_pattern': drum_pat
        }

        try:
            print(f'Generating: {song_name} | Drums: {drum_cat}/{drum_pat}')
            response = session.post(URL, json=payload)
            if response.status_code != 200:
                print(f'Error {response.status_code}: {response.text}')
            return response.status_code == 200
        except requests.RequestException as e:
            print(f'Request failed: {e}')
            return False

    for model_path in model_paths:
        for context in musical_contexts:
            for params in hyperparameters:
                send_post(model_path, context, params)


if __name__ == '__main__':
    generating_for_model_paths()