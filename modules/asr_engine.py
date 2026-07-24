"""
asr_engine.py
--------------
Speech-to-text engine using faster-whisper (CTranslate2 reimplementation
of OpenAI Whisper) — same model weights and accuracy, but 2-4x faster
on CPU thanks to quantized inference (int8/float16).

Runs fully locally after the model is downloaded once. No internet,
no API keys, no credentials needed. Supports ~100 languages with
auto-detect, and optional translation to English.

This is the "offline ASR" component of the hybrid architecture:
  - ASR:        faster-whisper  (offline, local)
  - Processing: formatting, keywords, extractive summary (offline, local)
  - Summarize:  abstractive summary (online, optional — falls back to local)
  - Read aloud: gTTS (online, optional)
  - Export:     .txt/.docx/.pdf/.md (offline, local)

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
    downstream.
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
    Wrapper around faster-whisper for fully local, private speech-to-text.
    No internet needed after the model is downloaded once. No API keys
    or credentials required.
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
        beam_size: int = 5,
        vad_filter: bool = True,
        progress_callback: Optional[Callable[[str], None]] = None,
    ) -> TranscriptionResult:
        """
        Args:
            audio_path: path to the audio file.
            language: language code (e.g. "en", "hi") or None to auto-detect.
            task: "transcribe" (keep original language) or "translate"
                  (translate to English).
            beam_size: higher = more accurate but slower on CPU. Use 1
                       (greedy decoding) for the fastest results, 5
                       (default) for the best accuracy.
            vad_filter: skips silent stretches (faster + often more
                        accurate), but downloads a small separate VAD model
                        on first use. Disable to rule out VAD download as
                        the cause of a stall.
            progress_callback: optional function called with short status
                       strings at each stage, so a UI can show exactly
                       which step is currently running.
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


# ── gTTS language mapping ──
# Maps faster-whisper / ISO-639-1 language codes to gTTS language codes.
# gTTS uses Google Translate's language codes which differ slightly
# from ISO-639-1 (e.g. 'zh' → 'zh-CN').
GTTS_LANG_MAP = {
    "en": "en", "hi": "hi", "as": "en",  # Assamese not in gTTS → fallback English
    "bn": "bn", "es": "es", "fr": "fr", "de": "de",
    "zh": "zh-CN", "ja": "ja", "ko": "ko", "ar": "ar",
    "ru": "ru", "pt": "pt", "it": "it", "ta": "ta",
    "te": "te", "ur": "ur", "mr": "mr", "gu": "gu", "pa": "pa",
}


def get_gtts_language(lang_code: str) -> str:
    """
    Map a faster-whisper language code to a gTTS language code.
    Falls back to English if the language isn't supported by gTTS.
    """
    return GTTS_LANG_MAP.get(lang_code, "en")


def language_name_to_code(name: str) -> Optional[str]:
    return SUPPORTED_LANGUAGES.get(name)