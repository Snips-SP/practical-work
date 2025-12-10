# Phi3-mini Music Generator

A web-based application that generates music using a custom-trained **Phi3-mini** model. Users can interact with the model via a web interface to generate MIDI and MP3 files, utilizing genre-specific drum seeds to enhance the output.

-----

## Installation & Setup

### 1\. Create Conda Environment

```bash
conda create --name MusicGeneration python=3.10.16 -y
conda activate MusicGeneration
```

### 2\. Install System Dependencies

You must install FFmpeg for audio processing.

```bash
conda install -c conda-forge ffmpeg -y
```

### 3\. Install PyTorch (XPU/GPU Support)

**For Intel Arc (XPU) Support:**

```bash
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/xpu
```

> **Note:** To use XPU compilation, please follow the instructions here:
> [PyTorch Inductor on Windows](https://docs.pytorch.org/tutorials/unstable/inductor_windows.html)

**For CUDA 11.8 (Nvidia):**

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### 4\. Install Python Libraries

Install the required Python packages:

```bash
pip install -r requirements.txt
```

### 5\. Install Fluidsynth

Fluidsynth is required to render MIDI files into audio.

  - **Windows** (via Chocolatey):

<!-- end list -->

```bash
choco install fluidsynth -y
```

  - **Linux** (Debian/Ubuntu):

<!-- end list -->

```bash
sudo apt install fluidsynth
```

-----

## Configuration

### Drum Seeds Directory

The application uses drum loops to enhance generation. You must configure the path to your drum seeds in `server.py`.

The directory should contain folders named after genres (e.g., `rock`, `jazz`), with drum loop files inside them.

**In `server.py`:**

```python
DRUM_SEEDS_DIR = os.path.join('backend', 'ml_model', 'seeds', 'drums')
```

Ensure your directory structure matches this path or update the variable to point to your custom location.

-----

## Training the Model

### 1\. Download Dataset

Download the **cleansed version** of the Lakh Pianoroll Dataset from:

🔗 [https://hermandong.com/lakh-pianoroll-dataset/dataset](https://hermandong.com/lakh-pianoroll-dataset/dataset)

Place the extracted dataset in the `lpd_5` directory in the same folder as `encode.py`.

### 2\. Start Training

```bash
python -m backend.ml_model.train --help
```

Use the train module to train new versions of the Phi3-mini model with varying hyperparameters.

#### Training Details

  - Model architecture can be configured in the `NetworkConfig` class in `train.py`.

### 3\. Training Progress

View training progress live or analyze it afterward using TensorBoard:

```bash
tensorboard --logdir runs
```

-----

## Running the Webserver

Run the following command to start the server:

```bash
python run.py
```

This launches a local web interface where users can generate music using the selected model from the run folder and listen to generated outputs in MP3.

### Note on SoundFonts

The `SoundFont.sf2` file (e.g., `FluidR3_GM_GS.sf2`) must be located in the `backend/` folder.
[Download SoundFont](https://musical-artifacts.com/artifacts/738)

-----

## Project Structure

```
webserver/
├── backend/
│   ├── ml_model/
│   │   ├── experiments/          # Experimental code for the model, dataset, encoding analysis and debugging
│   │   ├── lpd_5/                # Raw dataset containing lpd_5_cleansed folder
│   │   ├── runs/                 # Saves model weights per epoch per run (logs included)
│   │   ├── seeds/                
│   │   │   └── drums/            # GENRE FOLDERS GO HERE (Rock, Jazz, etc.)
│   │   ├── dataloader.py
│   │   ├── encoding.py           
│   │   ├── experiments.py
│   │   ├── generate.py
│   │   ├── helper.py
│   │   └── train.py              # Train your own model
│   ├── SoundFont.sf2             # SoundFont file for rendering MIDI to audio
│   └── server.py                 # Web server backend (Edit DRUM_SEEDS_DIR here)
│
├── static/
│   ├── music/                    # Generated output folder
│   ├── chord_editor.js
│   ├── scripts.js
│   └── visualizer.js
│
├── templates/
├── run.py                        # Starts the webserver
└── README.md
```

-----

## License

This project is for academic use. Please credit appropriately if reused.

-----

