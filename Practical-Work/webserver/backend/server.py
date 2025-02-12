from flask import Flask, render_template, request, jsonify

app = Flask(__name__, template_folder="../templates", static_folder="../static")


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/generate-music', methods=['POST'])
def generate_music():
    data = request.json

    selected_options = data.get("chord_progression", [])
    print(f"Button Pressed! Selected Options: {selected_options}")

    # Placeholder: Python function to generate music from neural network

    return jsonify({"audio_url": "/static/music/test.mp3"})


def run():
    app.run()
