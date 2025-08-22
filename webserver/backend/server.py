from backend.ml_model.generate import generate_from_chords
from backend.ml_model.helper import mid_to_mp3
from flask import Flask, render_template, jsonify, request
from urllib.parse import quote
import os.path

app = Flask(__name__, template_folder='../templates', static_folder='../static')
app.secret_key = 'secret_keyyyy'


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/play-song', methods=['GET'])
def play_song():
    song = request.args.get('song')
    if song:
        songs = [os.path.splitext(f)[0] for f in os.listdir(os.path.join('static', 'music')) if f.endswith('.mp3')]

        if song not in songs:
            return jsonify({'error': 'No song found'}), 400

        filename = f'{song}.mp3'
        # Encode just the filename, not the entire path
        encoded_filename = quote(filename)
        audio_url = f'/static/music/{encoded_filename}'

        return jsonify({'audio_url': audio_url})

    return jsonify({'error': 'No song found'}), 400


@app.route('/get-songs', methods=['GET'])
def get_songs():
    # Get a list of all songs
    music_dir = os.path.join('static', 'music')

    if not os.path.exists(music_dir):
        return jsonify({'songs': []})
    # Get all songs without their extension
    songs = [os.path.splitext(f)[0] for f in os.listdir(music_dir) if f.endswith('.mp3')]

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
    bpm = request.json.get('bpm', None)
    if bpm is None:
        return jsonify({'error': 'No bpm provided'}), 400

    model_path = request.json.get('model_path', None)
    if model_path is None:
        return jsonify({'error': 'Model path directory not found'}), 404

    textbox = request.json.get('chord_progression', None)
    if not textbox:
        return jsonify({'error': 'No chord progression provided'}), 400

    # Bring chords into right format
    # From A:16|B:32|C:16|D:8 -> [A, B, C, D], [16, 32, 16, 8]
    chords = []
    timings = []
    try:
        for chord_timing in textbox.split('|'):
            if ':' not in chord_timing:
                return jsonify(
                    {'error': f'Invalid chord-timing pair format: "{chord_timing}". Expected "Chord:Timing"'}), 400

            chord, timing_str = chord_timing.split(':', 1)
            chord = chord.strip()
            timing_str = timing_str.strip()

            if not chord:
                return jsonify({'error': f'Missing chord name in pair: "{chord_timing}"'}), 400

            if not timing_str.isdigit():
                return jsonify({'error': f'Invalid timing value (not an integer) in pair: "{chord_timing}"'}), 400

            chords.append(chord)
            timings.append(int(timing_str))

    except Exception as e:
        return jsonify({'error': f'Error parsing chord progression: {str(e)}'}), 400

    # Get right folder for user
    music_dir = os.path.join('static', 'music')
    os.makedirs(music_dir, exist_ok=True)

    # Get generated songs from user
    contents = os.listdir(music_dir)
    # Create a new name from the used gpt2 model and chords
    current_song_name = f'{os.path.basename(model_path)}_{"_".join(chords)}_{bpm}BPM_{len(contents) + 1}'
    new_song_path = os.path.join(music_dir, f'{current_song_name}.mp3')

    # Make a temporary folder and file location
    os.makedirs(os.path.join('backend', 'tmp'), exist_ok=True)
    tmp_mid_file = os.path.join('backend', 'tmp', f'{current_song_name}.mid')

    # Generate midi file
    try:
        generate_from_chords(chords=chords, timings=timings, tempo=bpm, model_dir=model_path, output=tmp_mid_file)
    except ValueError:
        return jsonify({'error': f'Unknown chord value found in input chord progression'}), 400
    # Trim and convert to mp3
    mid_to_mp3(tmp_mid_file, os.path.join('backend', 'SoundFont.sf2'), new_song_path)

    # Remove the temporary midi file (we only need to keep the mp3 file)
    # We are keeping them for now for analysis
    # os.remove(tmp_mid_file)

    return jsonify({'audio_url': new_song_path,
                    'song_name': current_song_name})


def run():
    app.run(host='0.0.0.0')
