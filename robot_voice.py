
import queue
import threading
import numpy as np
import sounddevice as sd
from faster_whisper import WhisperModel
from piper import PiperVoice

THRESHOLD = 0.02   # microphone sensitivity (lower = more sensitive)
PAUSE_CHUNKS = 38  # silence that ends a sentence, in 32 ms chunks (38 = ~1.2 s)
ROBOT_HZ = 60      # robot buzz: 30 = deep, 120 = buzzy

whisper = WhisperModel("models/whisper-small.en", device="cpu", compute_type="int8")
voice = PiperVoice.load("models/piper/en_US-ryan-medium.onnx")
sentences = queue.Queue()
robot_speaking = threading.Event()   # set while the robot is talking
recent_words = set()                 # words the robot said recently (to ignore its own echo)


def record_sentence(mic):
    """Wait for you to talk, then record until you pause."""
    recording = []
    before = []        # last few chunks before talking started, so the first word is not cut
    quiet = 0
    while True:
        chunk = mic.read(512)[0][:, 0]
        limit = THRESHOLD * 5 if robot_speaking.is_set() else THRESHOLD   # less sensitive while robot talks
        loud = np.abs(chunk).mean() > limit
        if loud:
            quiet = 0
        elif recording:
            quiet += 1
        if loud or recording:
            recording.append(chunk)
        else:
            before = (before + [chunk])[-6:]                 
        if quiet > PAUSE_CHUNKS:
            if len(recording) - quiet < 10:                   
                recording, quiet = [], 0
                continue
            audio = np.concatenate(before + recording)
            return audio / max(np.abs(audio).max(), 0.01) * 0.9   # normalise volume for whisper


def robot():                            # background thread: text from speech, then speak it as a robot.
    
    global recent_words
    while True:
        audio = sentences.get()
        segments, _ = whisper.transcribe(audio, language="en", beam_size=5, vad_filter=True)
        text = " ".join(s.text for s in segments).strip()
        words = set(w.strip(".,!?-").lower() for w in text.split())
        if not words:
            continue
        if len(words & recent_words) / len(words) > 0.6:   # mostly the robot's own words = echo
            print("(ignored echo:", text, ")")
            continue
        print("You said:", text)
        speech = np.concatenate([c.audio_float_array for c in voice.synthesize(text)])
        t = np.arange(len(speech)) / voice.config.sample_rate
        speech = speech * (0.6 + 0.4 * np.sin(2 * np.pi * ROBOT_HZ * t))   # robot buzz
        recent_words = words
        robot_speaking.set()
        sd.play(speech.astype(np.float32), voice.config.sample_rate, blocking=True)
        robot_speaking.clear()


threading.Thread(target=robot, daemon=True).start()
print("Ready - talk, pause, and the robot repeats you. Ctrl+C to stop.")
try:
    with sd.InputStream(samplerate=16000, channels=1, blocksize=512, dtype="float32") as mic:
        while True:
            sentences.put(record_sentence(mic))
except KeyboardInterrupt:
    print("Bye.")
