"""
asr_engine.py
--------------
Speech-to-text engine with two modes:
  - "offline": faster-whisper (CTranslate2 reimplementation of OpenAI
    Whisper) — same model weights and accuracy as openai-whisper, but
    2-4x faster on CPU thanks to quantized inference (int8/float16).
    Runs fully locally after the model is downloaded once.
  - "online": Google's free Web Speech API via the `SpeechRecognition`
    library. No local model needed, but requires internet and a specific
    language (no auto-detect).

Reference: faster-whisper - https://github.com/SYSTRAN/faster-whisper
Reference: OpenAI Whisper  - https://github.com/openai/whisper
"""

import os
import shutil
import subprocess
import warnings
from dataclasses import dataclass, field
from typing import Callable, List, Optional

import numpy as np
import soundfile as sf

warnings.filterwarnings("ignore")


def _resolve_ffmpeg() -> str:
    """
    Locate ffmpeg without a hardcoded, machine-specific path. Checks PATH
    first, then falls back to the `imageio-ffmpeg` pip package (a portable
    ffmpeg binary) so this works across machines without manual setup.
    """
    found = shutil.which("ffmpeg")
    if found:
        return found
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        raise RuntimeError(
            "ffmpeg was not found on PATH and the 'imageio-ffmpeg' fallback "
            "package isn't installed. Install ffmpeg system-wide, or run: "
            "pip install imageio-ffmpeg"
        )


FFMPEG_BIN = _resolve_ffmpeg()

SUPPORTED_LANGUAGES = {
    "Auto-detect": None,
    "English": "en",
    "Hindi": "hi",
    "Assamese": "as",
    "Bengali": "bn",
    "Spanish": "es",
    "French": "fr",
    "German": "de",
    "Chinese": "zh",
    "Japanese": "ja",
    "Korean": "ko",
    "Arabic": "ar",
    "Russian": "ru",
    "Portuguese": "pt",
    "Italian": "it",
    "Tamil": "ta",
    "Telugu": "te",
    "Urdu": "ur",
    "Marathi": "mr",
    "Gujarati": "gu",
    "Punjabi": "pa",
}

MODEL_SIZES = ["tiny", "base", "small", "medium", "large-v3"]
TARGET_SAMPLE_RATE = 16000

# Maps ISO-639-1 codes to locale codes Google's Web Speech API expects.
ONLINE_LOCALE_MAP = {
    "en": "en-US", "hi": "hi-IN", "as": "as-IN", "bn": "bn-IN", "es": "es-ES",
    "fr": "fr-FR", "de": "de-DE", "zh": "zh-CN", "ja": "ja-JP", "ko": "ko-KR",
    "ar": "ar-SA", "ru": "ru-RU", "pt": "pt-PT", "it": "it-IT", "ta": "ta-IN",
    "te": "te-IN", "ur": "ur-PK", "mr": "mr-IN", "gu": "gu-IN", "pa": "pa-IN",
}


@dataclass
class Segment:
    start: float
    end: float
    text: str


@dataclass
class TranscriptionResult:
    text: str
    language: str
    segments: List[Segment] = field(default_factory=list)


def _normalize_audio(input_path: str) -> str:
    """
    Re-encode audio into a clean 16kHz mono WAV via ffmpeg. Browser-recorded
    audio (WebM/Opus blobs) often plays fine in a browser's lenient player
    but has incomplete container metadata that trips up stricter decoders
    downstream. Used for both offline and online transcription paths.
    """
    output_path = input_path + "_normalized.wav"
    cmd = [
        FFMPEG_BIN, "-y", "-i", input_path,
        "-ar", str(TARGET_SAMPLE_RATE), "-ac", "1", "-f", "wav", output_path,
    ]
    proc = subprocess.run(cmd, capture_output=True)
    if proc.returncode != 0 or not os.path.exists(output_path) or os.path.getsize(output_path) == 0:
        raise ValueError(
            f"Could not normalize audio file (ffmpeg failed): "
            f"{proc.stderr.decode(errors='ignore')[:300]}"
        )
    return output_path


def _load_audio_array(wav_path: str) -> np.ndarray:
    """
    Decode the normalized WAV ourselves with `soundfile` and hand the model
    a numpy array directly, instead of a file path. This avoids relying on
    an internal, unchecked decode step that could silently return 0 samples
    for some browser-recorded files.
    """
    audio, sr = sf.read(wav_path, dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != TARGET_SAMPLE_RATE:
        raise ValueError(
            f"Normalized audio has sample rate {sr}Hz, expected {TARGET_SAMPLE_RATE}Hz."
        )
    if audio.size == 0:
        raise ValueError(
            f"Decoded 0 audio samples from '{wav_path}'. The recording is "
            "likely empty, silent, or corrupted. Try re-recording."
        )
    return audio


class ASREngine:
    """
    Wrapper around faster-whisper (offline) with an online fallback via
    Google's free Web Speech API.
    """

    def __init__(self, model_size: str = "small"):
        if model_size not in MODEL_SIZES:
            raise ValueError(f"model_size must be one of {MODEL_SIZES}")
        self.model_size = model_size
        self._model = None

    def _load_model(self, progress_callback: Optional[Callable[[str], None]] = None):
        if self._model is None:
            from faster_whisper import WhisperModel

            if progress_callback:
                progress_callback(
                    f"Loading Whisper '{self.model_size}' model — if this is the "
                    "first run, it's downloading the model files now (small ≈ "
                    "500MB) and this can take a while depending on your "
                    "connection. Subsequent runs load instantly from local cache."
                )

            # compute_type="int8" gives the biggest CPU speedup with a
            # negligible accuracy trade-off vs the original fp32 weights.
            # device="auto" picks GPU (CUDA) automatically if available,
            # otherwise falls back to CPU with int8 quantization.
            # cpu_threads is set explicitly - without it, CTranslate2 may
            # not use all available cores, which is a common cause of
            # "it just sits there" on CPU-only machines.
            self._model = WhisperModel(
                self.model_size,
                device="auto",
                compute_type="int8",
                cpu_threads=max(1, os.cpu_count() or 4),
            )
        return self._model

    def transcribe(
        self,
        audio_path: str,
        language: Optional[str] = None,
        task: str = "transcribe",
        mode: str = "offline",
        beam_size: int = 5,
        vad_filter: bool = True,
        progress_callback: Optional[Callable[[str], None]] = None,
    ) -> TranscriptionResult:
        """
        Args:
            audio_path: path to the audio file.
            language: language code (e.g. "en", "hi") or None to auto-detect
                      (offline mode only — online mode requires a specific
                      language and defaults to English if none given).
            task: "transcribe" or "translate" (offline mode only).
            mode: "offline" -> local faster-whisper model, fully private,
                  works without internet after the model is downloaded once.
                  "online"  -> Google's free Web Speech API. No local model
                  needed, but requires internet and sends audio to Google's
                  servers.
            beam_size: higher = more accurate but slower on CPU. Use 1
                       (greedy decoding) for the fastest offline results,
                       5 (default) for the best accuracy.
            vad_filter: skips silent stretches (faster + often more
                        accurate), but downloads a small separate VAD model
                        on first use. Disable to rule out VAD download as
                        the cause of a stall.
            progress_callback: optional function called with short status
                       strings at each stage, so a UI can show exactly
                       which step is currently running instead of one
                       static "loading" message.
        """
        def _report(msg: str) -> None:
            if progress_callback:
                progress_callback(msg)

        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        if os.path.getsize(audio_path) == 0:
            raise ValueError(
                f"Audio file is empty (0 bytes): {audio_path}. "
                "The recording may not have finished saving. Try re-recording."
            )

        _report("Normalizing audio with ffmpeg...")
        normalized_path = _normalize_audio(audio_path)

        if mode == "online":
            _report("Sending audio to Google's Web Speech API...")
            return _transcribe_online(normalized_path, language_code=language or "en")

        _report("Decoding audio into a waveform...")
        audio_array = _load_audio_array(normalized_path)

        model = self._load_model(progress_callback=_report)

        _report("Running Whisper transcription...")
        segments_gen, info = model.transcribe(
            audio_array,
            language=language,
            task=task,
            vad_filter=vad_filter,
            beam_size=beam_size,
        )

        # segments_gen is a lazy generator - nothing above actually runs
        # inference until we iterate it here, which is where most of the
        # wall-clock time is spent for longer audio.
        segments = [
            Segment(start=s.start, end=s.end, text=s.text.strip())
            for s in segments_gen
        ]
        full_text = " ".join(s.text for s in segments).strip()

        return TranscriptionResult(
            text=full_text,
            language=info.language or language or "unknown",
            segments=segments,
        )


def _transcribe_online(wav_path: str, language_code: str = "en") -> TranscriptionResult:
    """
    Online ASR using Google's free Web Speech API via `SpeechRecognition`.
    Requires internet. No timestamped segments are returned (Google's free
    endpoint doesn't provide them), so paragraph structuring will fall back
    to sentence-count windows for online transcripts.
    """
    import speech_recognition as sr

    locale = ONLINE_LOCALE_MAP.get(
        language_code, language_code if "-" in language_code else "en-US"
    )

    recognizer = sr.Recognizer()
    with sr.AudioFile(wav_path) as source:
        audio_data = recognizer.record(source)

    try:
        text = recognizer.recognize_google(audio_data, language=locale)
    except sr.UnknownValueError:
        text = ""
    except sr.RequestError as e:
        raise ConnectionError(
            f"Could not reach Google's speech API (no internet or service "
            f"unavailable): {e}. Try switching to Offline mode instead."
        )

    return TranscriptionResult(text=text.strip(), language=locale, segments=[])


def language_name_to_code(name: str) -> Optional[str]:
    return SUPPORTED_LANGUAGES.get(name)