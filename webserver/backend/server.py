import os.path
from flask import Flask, render_template, jsonify, session, request
import uuid
from .ml_model.generate import generate_from_chords
from .ml_model.helper import mid_to_mp3

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
    # Get all songs without their extension
    songs = [os.path.splitext(f)[0] for f in os.listdir(session_dir) if f.endswith('.mp3')]

    return jsonify({'songs': songs})


@app.route('/generate-music', methods=['POST'])
def generate_music():
    user_id = session.get('user_id')

    if user_id is None:
        return jsonify({'error': 'Session expired or invalid'}), 401

    # Get chords from the user
    textbox = request.json.get('chord_progression', [])
    bpm = request.json.get('bpm', 80)

    # Bring chords into right format
    # From A:16|B:32|C:16|D:8 -> [A, B, C, D], [16, 32, 16, 8]
    chords = []
    timings = []
    for chord_timing in textbox.split('|'):
        chord, timing = chord_timing.split(':')
        chords.append(chord)
        timings.append(int(timing))

    # Get right folder for user
    session_dir = os.path.join('static', 'music', user_id)
    os.makedirs(session_dir, exist_ok=True)
    # Get generated songs from user
    contents = os.listdir(session_dir)
    current_song_name = f'Song_{len(contents) + 1}'
    new_song_path = os.path.join(session_dir, f'{current_song_name}.mp3')

    # Generate midi file
    os.makedirs(os.path.join('backend', 'tmp'), exist_ok=True)
    tmp_mid_file = os.path.join('backend', 'tmp', f'{user_id}_tmp.mid')
    generate_from_chords(chords,
                         timings,
                         bpm,
                         os.path.join('backend', 'gpt.ph'),
                         tmp_mid_file)
    # Trim and convert to mp3
    mid_to_mp3(tmp_mid_file,
               os.path.join('backend', 'SoundFont.sf2'),
               new_song_path)
    os.remove(tmp_mid_file)

    return jsonify({'audio_url': new_song_path,
                    'song_name': current_song_name})


def run():
    app.run()
