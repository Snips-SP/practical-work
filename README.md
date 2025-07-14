# GPT2 Music Generator

A web-based application that generates music using a custom-trained GPT-2 model. Users can input chords in a specified format, and the model outputs MIDI and MP3 files based on the input.

---
## Installation & Setup

### 1. Create Conda Environment

```bash
conda create --name MusicGeneration python=3.10.16 -y
conda activate MusicGeneration
```

### 2. Install Dependencies

**Important:** To install PyTorch with GPU/XPU support, run:


**For CUDA 11.8:**

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

**For Intel Arc:**

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/test/xpu
```

**For the rest of the libraries:**

```bash
pip install -r requirements.txt
conda install -c conda-forge ffmpeg -y
```

### 3. Install Fluidsynth

- **Windows** (via Chocolatey):

```bash
choco install fluidsynth -y
```

- **Linux**:

```bash
sudo apt install fluidsynth
```

---

## Training the GPT-2 Model

### 1. Download Dataset

Download the **cleansed version** of the Lakh Pianoroll Dataset from:

🔗 https://hermandong.com/lakh-pianoroll-dataset/dataset

Place the extracted dataset in the same directory as `encode.py`.

### 2. Encode Dataset

```bash
python -m backend.ml_model.encode --help
```

With this module the `ldp_5_dataset/` directory containing the `.npz` files for training can be generated.

### 3. Start Training

```bash
python -m backend.ml_model.train --help
```

With the train module new versions from the GPT2 model can be trained with varying hyperparameters.
#### Training Details

- Model architecture can be changed in the `NetworkConfig` class in `train.py`.

---

## Running the Webserver

Run the following command to start the server:

```bash
python run.py
```

This launches a local web interface where users can:

- Input chord progressions
- Generate music using the selected GPT-2 model from the run folder
- Listen to generated outputs in MP3

### Note

- The `SoundFont.sf2` soundfont file (e.g., `FluidR3_GM_GS.sf2`) must be in the `backend/` folder. (https://musical-artifacts.com/artifacts/738)

---

## Chord Input Format

Input chords in the following format:

```
CHORD:DURATION|CHORD:DURATION|...
```

- **CHORD**: Any chord (major, minor, extended, etc.) with any root note.
- **DURATION**: Length of the chord in sixtheenth notes.

### Example

```
C:4|Cm:2|C7:2|Cdim:4
```

### Supported Chords

Example for root note `C` (all other root notes like `D`, `E`, `F#`, etc. are also valid):

```
C
Cm
C7
Cm7
CM7
Cm7-5
Cdim
Csus4
C7sus4
Caug
Cm6
C7(9)
Cm7(9)
Cadd9
C6
CmM7
C7-5
```

---

## Session Storage

Each user session creates a unique folder where all generated audio is stored temporarily.

---

## Project Structure

```
webserver/
├── backend/
│   ├── ml_model/
│   │   ├── ldp_5_dataset/        # Encoded dataset
│   │   ├── lpd_5/                # Raw dataset containing lpd_5_cleansed folder
│   │   ├── runs/                 # Saves model weights per epoch per run (including tensorboard logs)
│   │   ├── __init__.py
│   │   ├── dataloader.py
│   │   ├── encode.py             # Encode raw dataset
│   │   ├── experiments.py
│   │   ├── generate.py
│   │   ├── helper.py
│   │   └── train.py              # Train your own model
│   ├── SoundFont.sf2             # SoundFont file for rendering MIDI to audio
│   └── server.py                 # Web server backend
│
├── static/
│   ├── music/
│   │   └── ec841223-8294-4dea-aca1-f1bf43f1fcc0a/   # User session folder
│   ├── scripts.js
│   ├── style.css
│   └── visualizer.js
│
├── templates/
│   └── run.py                    # Starts the webserver
│
└── README.md
```
---

## License

This project is for academic use. Please credit appropriately if reused.

---
