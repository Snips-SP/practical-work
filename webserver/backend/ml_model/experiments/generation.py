import requests


def generating_for_model_paths():
    model_paths = [
        'backend/ml_model/runs/Phi-2_Large_da_6_1',
        'backend/ml_model/runs/Phi-2_Large_1',
        'backend/ml_model/runs/Phi-2_Medium_da_6_1',
        'backend/ml_model/runs/Phi-2_Medium_1',
        'backend/ml_model/runs/Phi-2_Small_da_6_1',
        'backend/ml_model/runs/Phi-2_Small_1',
    ]
    URL = 'http://localhost:5000/generate-music'

    payloads = [
        # Simple chord progression
        {
            'bpm': 100,
            'chords': ['Am', 'C', 'D', 'F'],
            'timings': [16, 16, 16, 16],
            # Baseline middle temp middle top p
            'temperature': 0.75,
            'top_k': 0,
            'top_p': 0.9,
        },  {
            'bpm': 100,
            'chords': ['Am', 'C', 'D', 'F'],
            'timings': [16, 16, 16, 16],
            # High temp
            'temperature': 0.95,
            'top_k': 0,
            'top_p': 0.9,
        }, {
            'bpm': 100,
            'chords': ['Am', 'C', 'D', 'F'],
            'timings': [16, 16, 16, 16],
            # Middle temp low top p
            'temperature': 0.75,
            'top_k': 0,
            'top_p': 0.7,
        }, {
            'bpm': 100,
            'chords': ['Am', 'C', 'D', 'F'],
            'timings': [16, 16, 16, 16],
            # Middle temp high top p
            'temperature': 0.75,
            'top_k': 0,
            'top_p': 0.98,
        },
        # Complex chord progression
        {
            'bpm': 120,
            'chords': ['Cm7', 'Fm7', 'Dm7-5', 'G7#5', 'Cm7'],
            'timings': [16, 16, 8, 8, 16],
            # Baseline middle temp middle top p
            'temperature': 0.75,
            'top_k': 0,
            'top_p': 0.9,
        }, {
            'bpm': 120,
            'chords': ['Cm7', 'Fm7', 'Dm7-5', 'G7#5', 'Cm7'],
            'timings': [16, 16, 8, 8, 16],
            # High temp
            'temperature': 0.95,
            'top_k': 0,
            'top_p': 0.9,
        }, {
            'bpm': 120,
            'chords': ['Cm7', 'Fm7', 'Dm7-5', 'G7#5', 'Cm7'],
            'timings': [16, 16, 8, 8, 16],
            # Middle temp low top p
            'temperature': 0.75,
            'top_k': 0,
            'top_p': 0.7,
        }, {
            'bpm': 120,
            'chords': ['Cm7', 'Fm7', 'Dm7-5', 'G7#5', 'Cm7'],
            'timings': [16, 16, 8, 8, 16],
            # Middle temp high top p
            'temperature': 0.75,
            'top_k': 0,
            'top_p': 0.98,
        },
    ]

    # Preserver session over all post requests
    session = requests.Session()

    def send_post(model_path, chords, timings, bpm, temperature, top_k, top_p):
        payload = {
            'song_name': f'{model_path.split("/")[-1][:-2][6:]}_{"_".join(chords)}_{temperature}_{top_k}_{top_p}',
            'bpm': bpm,
            'model_path': model_path,
            'chords': chords,
            'timings': timings,
            'temperature': temperature,
            'top_k': top_k,
            'top_p': top_p,
        }

        try:
            print(f'Sending post for {model_path}: {payload}')
            response = session.post(URL, json=payload)
            if response.status_code == 200:
                print(f'Success: {model_path}')
            else:
                print(f'Error {response.status_code}: {response.text}')
            return response.status_code == 200
        except requests.RequestException as e:
            print(f'Request failed: {e}')
            return False

    # Generate chord progressions for each model
    for model_path in model_paths:
        for kwargs in payloads:
            send_post(model_path=model_path, **kwargs)


if __name__ == '__main__':
    generating_for_model_paths()