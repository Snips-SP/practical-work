import os.path

from flask import Flask, render_template, jsonify, session, request
import uuid
from ...generate import generate_from_chords
from ...helper import mid_to_mp3

app = Flask(__name__, template_folder='../templates', static_folder='../static')
app.secret_key = 'your_secret_key'  # Set a secret key for session management


@app.before_request
def set_session():
    # Create a session for the user if it doesn't already exist
    if 'user_id' not in session:
        session['user_id'] = str(uuid.uuid4())  # Generate a unique user ID


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/generate-music', methods=['POST'])
def generate_music():
    # You can access the session variable here
    user_id = session.get('user_id')

    if user_id is None:
        return jsonify({'error': 'Session expired or invalid'}), 401

    # Get chords from the user
    chords = request.json.get('chord_progression', [])

    # TODO
    # Bring chords into the right format
    # Make timings available to user somehow
    # Make tempo available as well

    # Get right folder for user
    session_dir = os.path.join('static', 'music', user_id)
    os.makedirs(session_dir, exist_ok=True)
    # Get generated songs from user
    contents = os.listdir(session_dir)
    new_song_path = os.path.join(session_dir, f'Song_{len(contents)+1}.mp3')

    # Generate midi file
    tmp_mid_file = os.path.join('tmp', f'{user_id}_tmp.mid')
    generate_from_chords(['A', 'D', 'F'],
                         [16, 32, 16],
                         4,
                         80,
                         'gpt_model_state_dict.ph',
                         tmp_mid_file)
    # Trim and convert to mp3
    mid_to_mp3(tmp_mid_file,
               'FluidR3_GM_GS.sf2',
               new_song_path)
    os.remove(tmp_mid_file)

    return jsonify({'audio_url': new_song_path})


def run():
    app.run()
