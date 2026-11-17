# Behind Curtain - Robot Voice Repeater

Listens to your microphone, turns what you say into text (offline, with
faster-whisper), and says it back in a robot voice (offline, with Piper TTS).
It keeps listening while the robot is talking so nothing you say is missed.

- `robot_voice.py` - the simple version (about 70 lines)
- `robot_voice_full.py` - full version with command-line options, FX presets,
  Windows/Google fallbacks and pause-accurate playback

## Setup (Windows)

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Download the models (one time, about 650 MB total):

```powershell
.\.venv\Scripts\python.exe -c "from huggingface_hub import snapshot_download; snapshot_download('Systran/faster-whisper-small.en', local_dir='models/whisper-small.en')"
.\.venv\Scripts\python.exe -m piper.download_voices --download-dir models/piper en_US-ryan-medium en_US-amy-medium en_US-lessac-medium
```

## Run

```powershell
.\.venv\Scripts\python.exe robot_voice.py
```

Use headphones so the robot does not hear itself. Ctrl+C to stop.

Settings at the top of `robot_voice.py`:

- `THRESHOLD` - mic sensitivity (lower = more sensitive)
- `PAUSE_CHUNKS` - how long a pause ends a sentence
- `ROBOT_HZ` - robot buzz pitch (30 = deep, 120 = buzzy)
