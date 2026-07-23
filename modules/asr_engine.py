"""
asr_engine.py
--------------
Speech-to-text engine built on OpenAI Whisper (open-source, offline-capable,
multilingual ASR model supporting ~100 languages with automatic language
detection). This is the "Speech Recognition" component from the project's
proposed pipeline.

Reference: OpenAI Whisper - https://github.com/openai/whisper
"Robust Speech Recognition via Large-Scale Weak Supervision" (Radford et al., 2022)

--------------------------------------------------------------------------
FIX (see bottom of file for full explanation):
Previously this module normalized audio with its own ffmpeg call, then
passed the *file path* to whisper's model.transcribe(). Whisper then runs
a SECOND, internal ffmpeg subprocess to decode that same file. That second
decode could silently return 0 samples for some browser-recorded files,
which crashed deep inside the encoder with:
    "cannot reshape tensor of 0 elements into shape [1, 0, 12, -1]"
Fix: decode the normalized WAV ourselves with `soundfile` and pass the
resulting numpy array directly to transcribe(), skipping Whisper's
internal (unchecked) ffmpeg decode entirely.
--------------------------------------------------------------------------
"""

import os
import shutil
import subprocess
import warnings
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import soundfile as sf

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------
# Locate ffmpeg WITHOUT a hardcoded, machine-specific path.
# The previous version hardcoded a path under a specific Windows user
# profile (C:\Users\Lenovo\...) which only works on that one machine.
# This checks PATH first, and falls back to the `imageio-ffmpeg` pip
# package (which ships a portable ffmpeg binary) if ffmpeg isn't found
# on PATH - works cross-platform, no manual install required.
# ---------------------------------------------------------------------
def _resolve_ffmpeg() -> str:
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

# Whisper's built-in language codes -> human-readable names (subset shown in UI).
# Full list: https://github.com/openai/whisper/blob/main/whisper/tokenizer.py
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
    Re-encode audio into a clean 16kHz mono WAV via ffmpeg. This fixes the
    common case where browser-recorded audio (WebM/Opus blobs) plays fine
    in a browser's lenient player but has incomplete container metadata
    that makes ffmpeg's stricter decoder misbehave when read directly.
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
    THE CORE FIX.

    Decode the already-normalized WAV ourselves with `soundfile` and return
    a float32 mono numpy array, instead of handing the file path to
    whisper.transcribe(). Whisper would otherwise run its OWN internal
    ffmpeg subprocess to decode the file a second time - and that second,
    unchecked decode is what was silently producing 0 samples and crashing
    the encoder with the "reshape tensor of 0 elements" error.

    Reading the file ourselves means:
      - Only ONE ffmpeg decode happens in the whole pipeline (ours).
      - We can check the result BEFORE handing it to the model, and fail
        with a clear, actionable error instead of a cryptic tensor crash.
    """
    audio, sr = sf.read(wav_path, dtype="float32", always_2d=False)

    if audio.ndim > 1:
        audio = audio.mean(axis=1)  # collapse to mono if needed

    if sr != TARGET_SAMPLE_RATE:
        raise ValueError(
            f"Normalized audio has sample rate {sr}Hz, expected "
            f"{TARGET_SAMPLE_RATE}Hz. The ffmpeg normalization step may "
            "not have applied correctly."
        )

    if audio.size == 0:
        raise ValueError(
            f"Decoded 0 audio samples from '{wav_path}'. The recording is "
            "likely empty, silent, or the source file was corrupted. "
            "Try re-recording or re-uploading the file."
        )

    return audio


class ASREngine:
    """
    Thin wrapper around openai-whisper that:
      - Lazily loads the model (only once, cached across calls).
      - Supports automatic language detection or a user-forced language.
      - Returns text plus timestamped segments (segments are later used by
        text_processor.py to infer paragraph/topic breaks from pauses).
    """

    def __init__(self, model_size: str = "small"):
        if model_size not in MODEL_SIZES:
            raise ValueError(f"model_size must be one of {MODEL_SIZES}")
        self.model_size = model_size
        self._model = None

    def _load_model(self):
        if self._model is None:
            import whisper  # imported lazily so the rest of the app works
            # without whisper/torch installed (e.g. for UI development/tests).
            self._model = whisper.load_model(self.model_size)
        return self._model

    def transcribe(
        self,
        audio_path: str,
        language: Optional[str] = None,
        task: str = "transcribe",
    ) -> TranscriptionResult:
        """
        Transcribe an audio file (wav/mp3/m4a/etc - anything ffmpeg can read).

        Args:
            audio_path: path to the audio file.
            language: Whisper language code (e.g. "en", "hi") or None to
                      auto-detect from the first 30 seconds of audio.
            task: "transcribe" (keep original language) or "translate"
                  (translate speech into English).
        """
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        if os.path.getsize(audio_path) == 0:
            raise ValueError(
                f"Audio file is empty (0 bytes): {audio_path}. "
                "The recording may not have finished saving. Try re-recording."
            )

        # Step 1: normalize to a clean 16kHz mono WAV (our own ffmpeg call,
        # checked for success).
        normalized_path = _normalize_audio(audio_path)

        # Step 2: decode that WAV ourselves into a numpy array (also
        # checked for success) - this is the fix. We never again hand a
        # file path to whisper, so its internal ffmpeg decode never runs.
        audio_array = _load_audio_array(normalized_path)

        model = self._load_model()
        result = model.transcribe(
            audio_array,
            language=language,
            task=task,
            verbose=False,
            fp16=False,
        )

        segments = [
            Segment(start=s["start"], end=s["end"], text=s["text"].strip())
            for s in result.get("segments", [])
        ]

        return TranscriptionResult(
            text=result.get("text", "").strip(),
            language=result.get("language", language or "unknown"),
            segments=segments,
        )


def language_name_to_code(name: str) -> Optional[str]:
    return SUPPORTED_LANGUAGES.get(name)