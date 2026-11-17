"""
Robot Voice Repeater  (speech -> text -> robotic text-to-speech)
----------------------------------------------------------------
1. Listens on the laptop microphone and detects when you speak.
2. Records each sentence until you pause, and keeps listening while the
   robot is talking - nothing you say is missed.
3. Converts each recording to text (speech recognition).
4. Speaks the text with a text-to-speech voice.
5. Runs the TTS audio through a robot effect and plays it, in order, with
   the same pauses between sentences that you left.

Run:
    python robot_voice.py

Options:
    python robot_voice.py --list-devices      show available audio devices
    python robot_voice.py --list-voices       show available TTS voices
    python robot_voice.py --device 1          use a specific input device
    python robot_voice.py --voice amy         pick a TTS voice by (part of) its name
    python robot_voice.py --tts windows       use the built-in Windows voices instead of piper
    python robot_voice.py --rate 150          TTS speaking speed (words per minute)
    python robot_voice.py --fx classic        robot strength: light (default), classic, heavy
    python robot_voice.py --carrier 80        change the robot "buzz" frequency
    python robot_voice.py --no-fx             plain TTS without the robot effect
    python robot_voice.py --threshold 0.03    fix the voice detection level (skips auto-calibration)
    python robot_voice.py --model medium.en   whisper model size: tiny.en, base.en, small.en, medium.en, large-v3
    python robot_voice.py --model D:/models/my-whisper   path to a downloaded model folder
    (default: the local folder models/whisper-small.en next to this script)
    python robot_voice.py --language en       spoken language code for whisper (default en)
    python robot_voice.py --duck 1.0          mic sensitivity while robot talks (1.0 = unchanged, for headphones)
    python robot_voice.py --max-gap 5         cap on the reproduced pause between sentences (seconds)
    python robot_voice.py --engine google      use Google's online recognizer instead of whisper

Speech recognition runs fully offline with faster-whisper using the model in
models/whisper-small.en. If you pass a model *name* instead (e.g. --model
medium.en) it is downloaded once and cached in %USERPROFILE%/.cache/huggingface.
Text-to-speech uses piper neural voices from models/piper (offline). The
built-in Windows voices are available as a fallback with --tts windows.

Press Ctrl+C to quit.
"""

import argparse
import difflib
import os
import queue
import sys
import tempfile
import threading
import time
import wave

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

import numpy as np
import sounddevice as sd
import pyttsx3

# ----------------------------------------------------------------------------
# Audio settings
# ----------------------------------------------------------------------------
SAMPLE_RATE = 16_000       # Hz for the microphone - what speech recognizers expect
BLOCK_SIZE = 512           # samples per callback (~32 ms)
CHANNELS = 1

# Whisper model used when --model is not given. If the local folder exists it is
# used (fully offline); otherwise the name is downloaded from Hugging Face once.
_HERE = os.path.dirname(os.path.abspath(__file__))
_LOCAL_MODEL_DIR = os.path.join(_HERE, "models", "whisper-small.en")
DEFAULT_MODEL = _LOCAL_MODEL_DIR if os.path.isdir(_LOCAL_MODEL_DIR) else "small.en"

# Folder holding piper TTS voices (*.onnx + *.onnx.json).
PIPER_VOICE_DIR = os.path.join(_HERE, "models", "piper")


# ----------------------------------------------------------------------------
# Robot voice effect
# ----------------------------------------------------------------------------
FX_PRESETS = {
    # name: (ring-mod wet mix, comb feedback, drive, echo gain, flanger depth)
    "light":   dict(ring_mix=0.30, comb_gain=0.45, drive=1.4, echo=0.20, flange=0.6),
    "classic": dict(ring_mix=0.75, comb_gain=0.00, drive=3.0, echo=0.35, flange=0.0),
    "heavy":   dict(ring_mix=0.95, comb_gain=0.60, drive=5.0, echo=0.40, flange=0.0),
}


def _comb_filter(x: np.ndarray, delay: int, gain: float) -> np.ndarray:
    """Feedback comb filter: y[n] = x[n] + gain * y[n - delay]. Gives a metallic ring."""
    if gain <= 0 or delay <= 0 or delay >= x.size:
        return x
    y = x.copy()
    for start in range(delay, x.size, delay):
        end = min(start + delay, x.size)
        y[start:end] += gain * y[start - delay:end - delay]
    return y


def _flanger(x: np.ndarray, sample_rate: int, depth: float) -> np.ndarray:
    """Slowly modulated short delay mixed with the dry signal - a 'synthetic' shimmer."""
    if depth <= 0:
        return x
    n = x.size
    t = np.arange(n, dtype=np.float32) / sample_rate
    base = 0.004 * sample_rate                      # 4 ms
    sweep = 0.0025 * sample_rate * depth            # +/- 2.5 ms * depth
    delay = base + sweep * np.sin(2 * np.pi * 0.35 * t)
    idx = np.arange(n, dtype=np.float32) - delay
    delayed = np.interp(idx, np.arange(n, dtype=np.float32), x, left=0.0, right=0.0)
    return 0.65 * x + 0.35 * delayed.astype(np.float32)


def robotize(audio: np.ndarray, sample_rate: int, carrier_hz: float = 60.0,
             preset: str = "light") -> np.ndarray:
    """
    Give a mono float32 voice signal in [-1, 1] a robotic character.

    Presets:
      light   - keeps the voice natural and human-like, adds a metallic
                resonance, a hint of ring-mod buzz and a synthetic shimmer.
      classic - the traditional heavy ring-modulated "Dalek" robot.
      heavy   - everything turned up: harsh, buzzy, machine-like.
    """
    if audio.size == 0:
        return audio
    p = FX_PRESETS[preset]

    audio = audio.astype(np.float32)
    t = np.arange(audio.size, dtype=np.float32) / sample_rate

    # 1. Ring modulation with a low sine carrier, blended with the dry voice.
    carrier = np.sin(2 * np.pi * carrier_hz * t).astype(np.float32)
    out = (1.0 - p["ring_mix"]) * audio + p["ring_mix"] * (audio * carrier)

    # 2. Metallic resonance (comb filter tuned near 220 Hz).
    out = _comb_filter(out, int(sample_rate / 220), p["comb_gain"])

    # 3. Synthetic shimmer (flanger).
    out = _flanger(out, sample_rate, p["flange"])

    # 4. Soft clipping for edge / saturation.
    out = np.tanh(out * p["drive"]) / np.tanh(p["drive"])

    # 5. Short echo (~45 ms) at low gain.
    delay = int(0.045 * sample_rate)
    if p["echo"] > 0 and delay < out.size:
        echo = np.zeros_like(out)
        echo[delay:] = out[:-delay] * p["echo"]
        out = out + echo

    # 6. Normalise to -1 dBFS.
    peak = np.max(np.abs(out))
    if peak > 0:
        out = out / peak * 0.89

    return out.astype(np.float32)


# ----------------------------------------------------------------------------
# Speech to text
# ----------------------------------------------------------------------------
class WhisperSpeechToText:
    """Offline speech recognition with faster-whisper (CTranslate2 on CPU)."""

    name = "whisper"

    def __init__(self, model: str = "base", language: str = "en", device: str = "cpu",
                 compute_type: str = "int8"):
        from faster_whisper import WhisperModel  # imported here so --engine google works without it

        self.language = language
        self.model_name = model
        where = "local folder" if os.path.isdir(model) else "cached download"
        print(f"Loading whisper model '{model}' ({where}) ...", flush=True)
        t0 = time.time()
        try:
            self.model = WhisperModel(model, device=device, compute_type=compute_type)
        except RuntimeError as e:
            if "alloc" in str(e).lower():
                raise SystemExit(
                    "Not enough free memory to load the speech model.\n"
                    "Close other programs (including any other running copy of this script), "
                    "or try a smaller model with --model tiny.en"
                ) from e
            raise
        print(f"  model ready in {time.time() - t0:.1f}s")

    def transcribe(self, audio: np.ndarray, sample_rate: int) -> str | None:
        """Return the recognised text, or None if nothing could be understood."""
        if sample_rate != 16_000:
            # whisper expects 16 kHz
            idx = np.arange(0, audio.size, sample_rate / 16_000)
            audio = np.interp(idx, np.arange(audio.size), audio)
        audio = audio.astype(np.float32)

        segments, _info = self.model.transcribe(
            audio,
            language=self.language,
            beam_size=5,
            vad_filter=True,          # trims silence so it doesn't hallucinate text
            condition_on_previous_text=False,
        )
        text = " ".join(seg.text.strip() for seg in segments).strip()
        return text or None


class GoogleSpeechToText:
    """Online speech recognition via Google's free web API (needs internet)."""

    name = "google"

    def __init__(self, language: str = "en-US"):
        import speech_recognition as sr  # optional dependency

        self._sr = sr
        self.language = language
        self.recognizer = sr.Recognizer()

    def transcribe(self, audio: np.ndarray, sample_rate: int) -> str | None:
        sr = self._sr
        pcm16 = (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16).tobytes()
        data = sr.AudioData(pcm16, sample_rate, 2)
        try:
            return self.recognizer.recognize_google(data, language=self.language)
        except sr.UnknownValueError:
            return None
        except sr.RequestError as e:
            print(f"[speech recognition error] {e} (is the internet connected?)", file=sys.stderr)
            return None


# ----------------------------------------------------------------------------
# Text to speech
# ----------------------------------------------------------------------------
class PiperTextToSpeech:
    """Natural-sounding neural TTS that runs offline (piper-tts, ONNX on CPU)."""

    name = "piper"

    def __init__(self, voice: str | None = None, rate: int = 150):
        from piper import PiperVoice, SynthesisConfig  # imported here so --tts windows works without it

        self.rate = rate
        self.model_path = self._find_voice(voice)
        self.voice_name = os.path.splitext(os.path.basename(self.model_path))[0]
        print(f"Loading piper voice '{self.voice_name}' ...", flush=True)
        self.voice = PiperVoice.load(self.model_path)
        # Piper speaks ~150 wpm at length_scale 1.0; longer scale = slower.
        self.syn_config = SynthesisConfig(length_scale=150.0 / max(rate, 40))
        # Warm up so the first real sentence isn't delayed.
        self.synthesize("ready")

    @staticmethod
    def available_models() -> list[str]:
        if not os.path.isdir(PIPER_VOICE_DIR):
            return []
        return sorted(
            os.path.join(PIPER_VOICE_DIR, f)
            for f in os.listdir(PIPER_VOICE_DIR) if f.endswith(".onnx")
        )

    @classmethod
    def list_voices(cls):
        models = cls.available_models()
        if not models:
            print(f"  (none found in {PIPER_VOICE_DIR})")
        for m in models:
            print(f"  {os.path.splitext(os.path.basename(m))[0]}")
        print("  Download more with:  python -m piper.download_voices --download-dir models/piper en_US-<name>-medium")
        print("  Browse voices at:    https://rhasspy.github.io/piper-samples/")

    @classmethod
    def _find_voice(cls, name: str | None) -> str:
        if name and os.path.isfile(name):
            return name
        models = cls.available_models()
        if not models:
            raise SystemExit(
                f"No piper voices found in {PIPER_VOICE_DIR}.\n"
                "Download one with:\n"
                "  python -m piper.download_voices --download-dir models/piper en_US-ryan-medium"
            )
        if not name:
            # Prefer ryan if present, otherwise the first available voice.
            for m in models:
                if "ryan" in os.path.basename(m).lower():
                    return m
            return models[0]
        for m in models:
            if name.lower() in os.path.basename(m).lower():
                return m
        print(f"[warning] no piper voice matching '{name}', using {os.path.basename(models[0])}",
              file=sys.stderr)
        return models[0]

    def synthesize(self, text: str) -> tuple[np.ndarray, int]:
        """Return (mono float32 audio, sample_rate) for the spoken text."""
        chunks = list(self.voice.synthesize(text, self.syn_config))
        if not chunks:
            return np.zeros(0, dtype=np.float32), self.voice.config.sample_rate
        audio = np.concatenate([c.audio_float_array for c in chunks]).astype(np.float32)
        return audio, chunks[0].sample_rate


class WindowsTextToSpeech:
    """Fallback TTS using the voices built into Windows (pyttsx3 / SAPI5)."""

    name = "windows"

    def __init__(self, voice: str | None = None, rate: int = 150):
        self.rate = rate
        self.voice_id = self._find_voice(voice)
        self.voice_name = voice or "default"

    @staticmethod
    def list_voices():
        engine = pyttsx3.init()
        for v in engine.getProperty("voices"):
            print(f"  {v.name}")
        engine.stop()

    @staticmethod
    def _find_voice(name: str | None) -> str | None:
        if not name:
            return None
        engine = pyttsx3.init()
        voices = engine.getProperty("voices")
        engine.stop()
        for v in voices:
            if name.lower() in v.name.lower() or name.lower() in v.id.lower():
                return v.id
        print(f"[warning] no TTS voice matching '{name}', using default", file=sys.stderr)
        return None

    def synthesize(self, text: str) -> tuple[np.ndarray, int]:
        """Return (mono float32 audio, sample_rate) for the spoken text."""
        # A fresh engine per utterance avoids pyttsx3's "second runAndWait
        # never returns" problems on Windows.
        engine = pyttsx3.init()
        engine.setProperty("rate", self.rate)
        if self.voice_id:
            engine.setProperty("voice", self.voice_id)

        fd, path = tempfile.mkstemp(suffix=".wav")
        os.close(fd)
        try:
            engine.save_to_file(text, path)
            engine.runAndWait()
            engine.stop()
            return self._read_wav(path)
        finally:
            try:
                os.remove(path)
            except OSError:
                pass

    @staticmethod
    def _read_wav(path: str) -> tuple[np.ndarray, int]:
        with wave.open(path, "rb") as w:
            rate = w.getframerate()
            channels = w.getnchannels()
            width = w.getsampwidth()
            frames = w.readframes(w.getnframes())

        if width == 2:
            data = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
        elif width == 1:
            data = (np.frombuffer(frames, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
        elif width == 4:
            data = np.frombuffer(frames, dtype=np.int32).astype(np.float32) / 2147483648.0
        else:
            raise ValueError(f"unsupported WAV sample width: {width}")

        if channels > 1:
            data = data.reshape(-1, channels).mean(axis=1)
        return data, rate


# ----------------------------------------------------------------------------
# Voice activity detection + main loop
# ----------------------------------------------------------------------------
class VoiceRepeater:
    def __init__(
        self,
        stt,
        tts,
        device=None,
        threshold=None,
        silence_duration: float = 0.8,
        min_speech_duration: float = 0.3,
        max_record_seconds: float = 15.0,
        carrier_hz: float = 60.0,
        fx_preset: str | None = "light",
        duck: float = 2.5,
        max_gap: float = 8.0,
    ):
        self.stt = stt
        self.tts = tts
        self.device = device
        self.threshold = threshold
        self.silence_duration = silence_duration
        self.min_speech_duration = min_speech_duration
        self.max_record_seconds = max_record_seconds
        self.carrier_hz = carrier_hz
        self.fx_preset = fx_preset  # None = no effect
        self.duck = duck
        self.max_gap = max_gap

        self._queue: "queue.Queue[np.ndarray]" = queue.Queue()       # mic blocks
        self._work_q: "queue.Queue[Utterance]" = queue.Queue()       # recorded -> worker
        self._play_q: "queue.Queue[Utterance]" = queue.Queue()       # synthesised -> player
        self._playing = threading.Event()                            # set while robot speaks
        self._recent_said: list[tuple[str, float]] = []              # (text, time finished) for echo rejection
        self._lock = threading.Lock()

    # Called by sounddevice from a separate thread for every audio block.
    def _callback(self, indata, frames, time_info, status):
        if status:
            print(f"[audio warning] {status}", file=sys.stderr)
        self._queue.put(indata[:, 0].copy())

    @staticmethod
    def _rms(block: np.ndarray) -> float:
        return float(np.sqrt(np.mean(block.astype(np.float64) ** 2)))

    def calibrate(self, seconds: float = 1.5):
        """Measure background noise and set the speech threshold above it."""
        print(f"Calibrating background noise for {seconds:.1f}s - please stay quiet...")
        noise = sd.rec(int(seconds * SAMPLE_RATE), samplerate=SAMPLE_RATE,
                       channels=CHANNELS, dtype="float32", device=self.device)
        sd.wait()
        ambient = self._rms(noise[:, 0])
        # Speech is normally several times louder than room noise.
        self.threshold = max(ambient * 3.0, 0.01)
        print(f"  ambient noise RMS: {ambient:.4f} -> threshold set to {self.threshold:.4f}")

    def run(self):
        print("Robot Voice Repeater  (speech -> text -> robot TTS)")
        if self.threshold is None:
            self.calibrate()
        print(f"  input device     : {self.device if self.device is not None else 'default'}")
        print(f"  voice threshold  : {self.threshold:.3f} (RMS)")
        print(f"  recognition      : {self.stt.name} ({self.stt.language})")
        print(f"  voice            : {self.tts.name} / {self.tts.voice_name} @ {self.tts.rate} wpm")
        fx = f"{self.fx_preset} (carrier {self.carrier_hz:.0f} Hz)" if self.fx_preset else "off"
        print(f"  robot effect     : {fx}")
        print(f"  while speaking   : mic threshold x{self.duck:.1f}, echo rejection on")
        print("Keep talking - every sentence is queued and repeated in order with your pauses.")
        print("Use headphones for best results. Ctrl+C to quit.\n")

        threading.Thread(target=self._worker, name="stt-tts", daemon=True).start()
        threading.Thread(target=self._player, name="player", daemon=True).start()

        stream = sd.InputStream(
            samplerate=SAMPLE_RATE,
            blocksize=BLOCK_SIZE,
            channels=CHANNELS,
            dtype="float32",
            device=self.device,
            callback=self._callback,
        )

        silence_blocks_needed = int(self.silence_duration * SAMPLE_RATE / BLOCK_SIZE)
        max_blocks = int(self.max_record_seconds * SAMPLE_RATE / BLOCK_SIZE)
        pre_roll_blocks = 4  # keep a little audio before detection so the first syllable isn't cut

        with stream:
            while True:
                utt = self._record_sentence(silence_blocks_needed, max_blocks, pre_roll_blocks)
                duration = utt.audio.size / SAMPLE_RATE
                if duration < self.min_speech_duration + self.silence_duration:
                    continue  # cough / click / noise burst
                self._work_q.put(utt)

    # -- listener -----------------------------------------------------------
    def _record_sentence(self, silence_blocks_needed, max_blocks, pre_roll_blocks) -> "Utterance":
        recording, pre_roll = [], []
        speaking = False
        silent_blocks = 0
        t_start = 0.0
        overlapped = False
        block_sec = BLOCK_SIZE / SAMPLE_RATE

        while True:
            block = self._queue.get()
            level = self._rms(block)
            playing = self._playing.is_set()
            # While the robot is talking, require a louder signal so that its own
            # voice coming out of the speakers is less likely to count as speech.
            threshold = self.threshold * (self.duck if playing else 1.0)

            if not speaking:
                pre_roll.append(block)
                if len(pre_roll) > pre_roll_blocks:
                    pre_roll.pop(0)
                if level > threshold:
                    speaking = True
                    overlapped = playing
                    t_start = time.monotonic() - len(pre_roll) * block_sec
                    recording = list(pre_roll) + [block]
            else:
                recording.append(block)
                overlapped = overlapped or playing
                silent_blocks = 0 if level > threshold else silent_blocks + 1
                if silent_blocks >= silence_blocks_needed or len(recording) >= max_blocks:
                    t_end = time.monotonic() - silent_blocks * block_sec
                    return Utterance(np.concatenate(recording), t_start, t_end, overlapped)

    # -- worker: speech -> text -> robot speech -----------------------------
    def _worker(self):
        while True:
            utt = self._work_q.get()
            text = self.stt.transcribe(utt.audio, SAMPLE_RATE)
            if not text:
                self._say("(could not understand that)")
                continue
            if utt.overlapped and self._is_echo(text):
                self._say(f"(ignored my own voice: {text})")
                continue

            speech, rate = self.tts.synthesize(text)
            if self.fx_preset:
                speech = robotize(speech, rate, self.carrier_hz, self.fx_preset)
            utt.text, utt.speech, utt.rate = text, speech, rate
            self._say(f"You said : {text}")
            self._play_q.put(utt)

    def _is_echo(self, text: str) -> bool:
        """True if `text` closely matches something the robot said in the last ~20 s."""
        now = time.monotonic()
        norm = lambda s: "".join(ch.lower() for ch in s if ch.isalnum() or ch == " ").split()
        words = norm(text)
        with self._lock:
            self._recent_said = [(t, ts) for t, ts in self._recent_said if now - ts < 20]
            for said, _ in self._recent_said:
                ratio = difflib.SequenceMatcher(None, words, norm(said)).ratio()
                if ratio >= 0.7:
                    return True
        return False

    # -- player: speak in order, reproducing the speaker's pauses -----------
    def _player(self):
        prev_user_end = None
        prev_play_end = None
        while True:
            utt = self._play_q.get()

            if prev_user_end is not None:
                gap = min(max(utt.t_start - prev_user_end, 0.0), self.max_gap)
                wait = prev_play_end + gap - time.monotonic()
                if wait > 0:
                    time.sleep(wait)

            queued = self._play_q.qsize() + self._work_q.qsize()
            self._say(f"Robot    : {utt.text}" + (f"   [{queued} more queued]" if queued else ""))
            self._playing.set()
            try:
                sd.play(utt.speech, utt.rate, blocking=True)
            finally:
                time.sleep(0.1)
                self._playing.clear()

            prev_play_end = time.monotonic()
            prev_user_end = utt.t_end
            with self._lock:
                self._recent_said.append((utt.text, prev_play_end))

    def _say(self, msg: str):
        with self._lock:
            print(msg, flush=True)


class Utterance:
    """One detected sentence travelling through the pipeline."""

    __slots__ = ("audio", "t_start", "t_end", "overlapped", "text", "speech", "rate")

    def __init__(self, audio: np.ndarray, t_start: float, t_end: float, overlapped: bool):
        self.audio = audio          # mic recording (16 kHz float32)
        self.t_start = t_start      # when the speaker started (monotonic seconds)
        self.t_end = t_end          # when the speaker stopped
        self.overlapped = overlapped  # robot was talking during this recording
        self.text = ""
        self.speech = None          # robot audio
        self.rate = 0


# ----------------------------------------------------------------------------
# Command line
# ----------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Repeat what you say as a robot (speech -> text -> robot TTS).")
    p.add_argument("--list-devices", action="store_true", help="list audio devices and exit")
    p.add_argument("--list-voices", action="store_true", help="list text-to-speech voices and exit")
    p.add_argument("--device", type=int, default=None, help="input device index (see --list-devices)")
    p.add_argument("--tts", choices=["piper", "windows"], default="piper",
                   help="text-to-speech engine: piper = natural neural voice, offline (default); "
                        "windows = built-in Windows voices")
    p.add_argument("--voice", type=str, default=None,
                   help="TTS voice name or part of it, e.g. 'ryan', 'amy', 'lessac' for piper, "
                        "'david' / 'zira' for windows (see --list-voices)")
    p.add_argument("--rate", type=int, default=150, help="TTS speaking rate in words per minute (default 150)")
    p.add_argument("--fx", choices=["light", "classic", "heavy"], default="light",
                   help="robot effect strength: light = human-like with robotic edge (default), "
                        "classic = traditional ring-mod robot, heavy = harsh machine")
    p.add_argument("--engine", choices=["whisper", "google"], default="whisper",
                   help="speech recognition engine: whisper = offline (default), google = online")
    p.add_argument("--model", type=str, default=DEFAULT_MODEL,
                   help="whisper model: tiny.en, base.en, small.en, medium.en, large-v3, or a path "
                        f"to a downloaded model folder (default: {DEFAULT_MODEL})")
    p.add_argument("--language", type=str, default=None,
                   help="spoken language. whisper: 'en', 'hi', 'fr'... (default en). "
                        "google: 'en-US', 'en-IN'... (default en-US)")
    p.add_argument("--threshold", type=float, default=None,
                   help="RMS level above which audio counts as speech. By default it is "
                        "calibrated automatically from background noise at startup. "
                        "Lower = more sensitive, raise it if noise triggers recording.")
    p.add_argument("--silence", type=float, default=0.8,
                   help="seconds of silence that end a sentence (default 0.8)")
    p.add_argument("--carrier", type=float, default=60.0,
                   help="robot buzz frequency in Hz (default 60). Try 30-120.")
    p.add_argument("--no-fx", action="store_true", help="disable the robot effect (plain TTS)")
    p.add_argument("--max-seconds", type=float, default=15.0,
                   help="maximum length of one recording (default 15)")
    p.add_argument("--duck", type=float, default=2.5,
                   help="multiply the mic threshold by this while the robot is speaking so its own "
                        "voice is not re-recorded (default 2.5; use 1.0 with headphones)")
    p.add_argument("--max-gap", type=float, default=8.0,
                   help="longest pause reproduced between sentences, in seconds (default 8)")
    return p.parse_args()


def main():
    args = parse_args()

    if args.list_devices:
        print(sd.query_devices())
        return
    if args.list_voices:
        print("Piper voices (--tts piper):")
        PiperTextToSpeech.list_voices()
        print("Windows voices (--tts windows):")
        WindowsTextToSpeech.list_voices()
        return

    if args.engine == "whisper":
        stt = WhisperSpeechToText(model=args.model, language=args.language or "en")
    else:
        stt = GoogleSpeechToText(language=args.language or "en-US")

    if args.tts == "piper":
        tts = PiperTextToSpeech(voice=args.voice, rate=args.rate)
    else:
        tts = WindowsTextToSpeech(voice=args.voice, rate=args.rate)

    repeater = VoiceRepeater(
        stt=stt,
        tts=tts,
        device=args.device,
        threshold=args.threshold,
        silence_duration=args.silence,
        max_record_seconds=args.max_seconds,
        carrier_hz=args.carrier,
        fx_preset=None if args.no_fx else args.fx,
        duck=args.duck,
        max_gap=args.max_gap,
    )

    try:
        repeater.run()
    except KeyboardInterrupt:
        print("\nBye.")
    except sd.PortAudioError as e:
        print(f"\nAudio device error: {e}", file=sys.stderr)
        print("Run with --list-devices to see available microphones, then pass --device N.",
              file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
