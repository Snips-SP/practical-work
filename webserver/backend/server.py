from backend.ml_model.generate import generate_backing_track
from backend.ml_model.helper import mid_to_mp3
from flask import Flask, render_template, jsonify, request
import os.path
import json
import re

app = Flask(__name__, template_folder='../templates', static_folder='../static')
app.secret_key = 'secret_keyyyy'
DRUM_SEEDS_DIR = os.path.join('backend', 'ml_model', 'seeds', 'drums')

def sanitize_filename(name):
    # Replace spaces with underscores
    name = name.replace(' ', '_')
    # Remove all characters that are not alphanumeric, underscores, or hyphens
    name = re.sub(r'[^a-zA-Z0-9_-]', '', name)
    # Ensure it's not empty
    return name or 'untitled'


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/get-drum-seeds', methods=['GET'])
def get_drum_seeds():
    if not os.path.exists(DRUM_SEEDS_DIR):
        return jsonify({'categories': {}})

    seeds_data = {}

    # Loop through all folders (Categories)
    for category in os.listdir(DRUM_SEEDS_DIR):
        category_path = os.path.join(DRUM_SEEDS_DIR, category)

        if os.path.isdir(category_path):
            files = []
            # Loop through files in that folder (Items)
            for f in os.listdir(category_path):
                if f.endswith('.mid') or f.endswith('.midi'):
                    files.append(f)

            # Only add category if it has files
            if files:
                seeds_data[category] = files

    return jsonify({'categories': seeds_data})


@app.route('/play-song', methods=['GET'])
def play_song():
    filename = request.args.get('filename')
    if filename is None:
        return jsonify({'error': 'Filename has to be provided'}), 400

    songs = [os.path.splitext(f)[0] for f in os.listdir(os.path.join('static', 'music')) if f.endswith('.mp3')]

    if filename not in songs:
        return jsonify({'error': 'No song found'}), 400

    # Encode just the filename, not the entire path
    audio_url = f'static/music/{filename}.mp3'
    metadata_url = f'static/music/{filename}.json'

    return jsonify({'audio_url': audio_url, 'metadata_url': metadata_url})


@app.route('/get-songs', methods=['GET'])
def get_songs():
    # Get a list of all songs
    music_dir = os.path.join('static', 'music')

    if not os.path.exists(music_dir):
        return jsonify({'songs': []})
    # Get all songs as dictionaries with display name and filename
    songs = []
    for f in os.listdir(music_dir):
        if f.endswith('.json'):
            with open(os.path.join(music_dir, f), 'r') as file:
                data = json.load(file)
                songs.append({'name': data['name'], 'path': os.path.splitext(f)[0]})

    return jsonify({'songs': songs})


@app.route('/get-models', methods=['GET'])
def get_models():
    # Get a list of all models in the current runs directory
    runs_dir = os.path.join('backend', 'ml_model', 'runs')

    if not os.path.exists(runs_dir):
        return jsonify({'models': []})

    model_dirs = []
    for name in os.listdir(runs_dir):
        model_dir_path = os.path.join(runs_dir, name)
        # Check if the path exists
        if os.path.isdir(model_dir_path):
            model_dirs.append({'name': name, 'path': os.path.join(runs_dir, name)})

    return jsonify({'models': model_dirs})


@app.route('/generate-music', methods=['POST'])
def generate_music():
    # Check user input
    name = request.json.get('song_name', None)
    if name is None or name.strip() == '':
        return jsonify({'error': 'No name provided'}), 400

    bpm = request.json.get('bpm', None)
    if bpm is None:
        return jsonify({'error': 'No bpm provided'}), 400

    model_path = request.json.get('model_path', None)
    if model_path is None:
        return jsonify({'error': 'Model path directory not found'}), 404

    chords = request.json.get('chords', None)
    if chords is None or not isinstance(chords, list) or len(chords) == 0:
        return jsonify({'error': 'Chords have to be provided as a list with at least 1 element'}), 400

    timings = request.json.get('timings', None)
    if timings is None or not isinstance(timings, list) or len(timings) == 0:
        return jsonify({'error': 'Timings have to be provided as a list with at least 1 element'}), 400

    if len(chords) != len(timings):
        return jsonify({'error': 'The number of chords must be equal to the number of timings'}), 400

    for i in range(len(timings)):
        try:
            timings[i] = int(timings[i])
        except ValueError:
            return jsonify({'error': f'Invalid timing value at index {i}'}), 400

    temperature = request.json.get('temperature', None)
    if temperature is None:
        return jsonify({'error': 'No temperature provided'}), 400
    try:
        temperature = float(temperature)
    except ValueError:
        return jsonify({'error': 'Invalid temperature value'}), 400

    top_k = request.json.get('top_k', None)
    if top_k is None:
        return jsonify({'error': 'No top_k provided'}), 400
    try:
        top_k = int(top_k)
    except ValueError:
        return jsonify({'error': 'Invalid top_k value'}), 400

    top_p = request.json.get('top_p', None)
    if top_p is None:
        return jsonify({'error': 'No top_p provided'}), 400
    try:
        top_p = float(top_p)
    except ValueError:
        return jsonify({'error': 'Invalid top_p value'}), 400

    drum_category = request.json.get('drum_category')
    drum_pattern = request.json.get('drum_pattern')

    if not drum_category or not drum_pattern:
        return jsonify({'error': 'Drum style and pattern must be selected'}), 400

    # Construct the path securely
    drum_seed_path = os.path.join(DRUM_SEEDS_DIR, drum_category, drum_pattern)

    if not os.path.exists(drum_seed_path):
        return jsonify({'error': 'Selected drum seed file not found on server'}), 404

    # Get right folder for user
    music_dir = os.path.join('static', 'music')
    os.makedirs(music_dir, exist_ok=True)

    # Get generated songs and check if the name is already used
    filename = sanitize_filename(name)
    songs = [os.path.splitext(f)[0] for f in os.listdir(os.path.join('static', 'music')) if f.endswith('.mp3')]

    if filename in songs:
        return jsonify({'error': 'Song already exists'}), 400

    new_song_path = os.path.join(music_dir, f'{filename}.mp3')

    # Make a temporary folder and file location
    os.makedirs(os.path.join('backend', 'tmp'), exist_ok=True)
    tmp_mid_file = os.path.join('backend', 'tmp', f'{filename}.mid')

    # Generate midi file
    try:
        generate_backing_track(
            chords=chords,
            timings=timings,
            tempo=bpm,
            model_dir=model_path,
            drum_seed_midi_file=drum_seed_path,
            output=tmp_mid_file,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
        )
    except ValueError as e:
        return jsonify({'error': f'{e}'}), 400
    # Trim and convert to mp3
    mid_to_mp3(tmp_mid_file, os.path.join('backend', 'SoundFont.sf2'), new_song_path)

    # Save song metadata
    with open(os.path.join(music_dir, f'{filename}.json'), 'w') as f:
        json.dump({
            'name': name,
            'chords': chords,
            'timings': timings,
            'bpm': bpm,
            'model_path': model_path,
            'temperature': temperature,
            'top_k': top_k,
            'top_p': top_p,
        }, f)

    # Remove the temporary midi file (we only need to keep the mp3 file)
    os.remove(tmp_mid_file)

    return jsonify({'song': {
        'name': name,
        'path': filename
    }})


def run():
    app.run(host='0.0.0.0')
