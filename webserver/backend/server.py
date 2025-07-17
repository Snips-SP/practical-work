from backend.ml_model.generate import generate_from_chords
from backend.ml_model.helper import mid_to_mp3
from backend.ml_model.train import get_latest_checkpoint
from flask import Flask, render_template, jsonify, session, request
import uuid
import os.path

app = Flask(__name__, template_folder='../templates', static_folder='../static')
app.secret_key = 'secret_keyyyy'


@app.before_request
def set_session():
    # Create a session for the user if it doesn't already exist
    if 'user_id' not in session:
        session['user_id'] = str(uuid.uuid4())  # Generate a unique user ID


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/play-song', methods=['GET'])
def play_song():
    # Check for the song in the user session and get its full path
    song = request.args.get('song')
    if song:
        user_id = session.get('user_id')
        # Check 
        if user_id is None:
            return jsonify({'error': 'Session expired or invalid'}), 401

        session_dir = os.path.join('static', 'music', user_id)
        # Get generated songs from user
        songs = [os.path.splitext(f)[0] for f in os.listdir(session_dir) if f.endswith('.mp3')]
        # Check if the song exists
        if song not in songs:
            return jsonify({'error': 'No song found'}), 400
        # Create full song path
        full_song_path = os.path.join(session_dir, f'{song}.mp3')

        return jsonify({'audio_url': full_song_path})

    return jsonify({'error': 'No song found'}), 400


@app.route('/get-songs', methods=['GET'])
def get_songs():
    # Get a list of all songs in the current user directory
    user_id = session.get('user_id')
    if user_id is None:
        return jsonify({'error': 'Session expired or invalid'}), 401

    session_dir = os.path.join('static', 'music', user_id)

    if not os.path.exists(os.path.join('static', 'music', user_id)):
        return jsonify({'songs': []})
    # Get all songs without their extension
    songs = [os.path.splitext(f)[0] for f in os.listdir(session_dir) if f.endswith('.mp3')]

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
            # Check if both a state dict and a config file are present
            state_dict_file_name = 'gpt_model_state_dict_epoch_'

            model_path = get_latest_checkpoint(model_dir_path, state_dict_file_name)
            config_path = os.path.join(model_dir_path, f'config.json')

            if model_path is not None and os.path.exists(config_path):
                model_dirs.append({'name': name, 'path': os.path.join(runs_dir, name)})

    return jsonify({'models': model_dirs})


@app.route('/generate-music', methods=['POST'])
def generate_music():
    user_id = session.get('user_id')

    if user_id is None:
        return jsonify({'error': 'Session expired or invalid'}), 401

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
    session_dir = os.path.join('static', 'music', user_id)
    os.makedirs(session_dir, exist_ok=True)

    # Get generated songs from user
    contents = os.listdir(session_dir)
    # Create new name from used gpt2 model and chords
    current_song_name = f'{os.path.basename(model_path)}_{"_".join(chords)}_{bpm}BPM_{len(contents) + 1}'
    new_song_path = os.path.join(session_dir, f'{current_song_name}.mp3')

    # Make temporary folder and file location
    os.makedirs(os.path.join('backend', 'tmp'), exist_ok=True)
    tmp_mid_file = os.path.join('backend', 'tmp', f'{current_song_name}.mid')

    # Generate midi file
    try:
        generate_from_chords(chords=chords, timings=timings, tempo=bpm, model_dir=model_path, output=tmp_mid_file)
    except ValueError:
        return jsonify({'error': f'Unknown chord value found in input chord progression'}), 400
    # Trim and convert to mp3
    mid_to_mp3(tmp_mid_file, os.path.join('backend', 'SoundFont.sf2'), new_song_path)

    # Remove temporary midi file (we only need to keep the mp3 file)
    # os.remove(tmp_mid_file)
    # We are keeping them for now for analysis

    return jsonify({'audio_url': new_song_path,
                    'song_name': current_song_name})


def run():
    app.run(host='0.0.0.0')
