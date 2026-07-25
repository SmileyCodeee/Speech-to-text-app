"""
asr_engine.py
-------------
Faster-Whisper based Automatic Speech Recognition engine.

CPU-only by design: this app targets low-spec local machines and hosts
like Render that have no GPU at all, so there's no CUDA detection code
to maintain, no ctranslate2 GPU dependency, and one less thing that can
fail. If you ever do deploy to a machine with an NVIDIA GPU, the only
change needed is device="cuda", compute_type="float16" in _load_model().

Features:
- CPU-only faster-whisper (int8 quantized — fastest accuracy/speed
  trade-off on CPU)
- ffmpeg-based audio normalization (handles browser WebM/Opus recordings
  with incomplete container metadata that can otherwise cause truncated
  or partial transcripts)
- Tuned VAD filtering that avoids clipping real speech
- Anti-hallucination decoding settings for longer recordings
- Audio-coverage tracking, so a UI can detect when part of a recording
  was silently skipped
- Language detection, transcription and translation
- Progress callback support
"""

import os
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List, Optional

import numpy as np
import soundfile as sf
from faster_whisper import WhisperModel


# --------------------------------------------------------------------------
# ffmpeg resolution
# --------------------------------------------------------------------------

def _resolve_ffmpeg() -> str:
    """
    Locate ffmpeg without a hardcoded, machine-specific path. Checks PATH
    first, then falls back to the `imageio-ffmpeg` pip package (a portable
    ffmpeg binary) so this works across machines (including Render) without
    manual setup.
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
TARGET_SAMPLE_RATE = 16000


# --------------------------------------------------------------------------
# Supported languages
# --------------------------------------------------------------------------

SUPPORTED_LANGUAGES = {
    "Auto-detect": None,
    "English": "en",
    "Assamese": "as",
    "Hindi": "hi",
    "Bengali": "bn",
    "Tamil": "ta",
    "Telugu": "te",
    "Malayalam": "ml",
    "Kannada": "kn",
    "Marathi": "mr",
    "Gujarati": "gu",
    "Punjabi": "pa",
    "Urdu": "ur",
    "Nepali": "ne",
    "French": "fr",
    "German": "de",
    "Spanish": "es",
    "Italian": "it",
    "Portuguese": "pt",
    "Japanese": "ja",
    "Korean": "ko",
    "Chinese": "zh",
    "Russian": "ru",
}


# --------------------------------------------------------------------------
# Faster-Whisper model sizes
# --------------------------------------------------------------------------
# On a low-spec CPU, "tiny" or "base" are strongly recommended. "small"
# and up will work but transcription will be noticeably slower - see the
# sidebar model-size help text in app.py.
MODEL_SIZES = [
    "tiny",
    "base",
    "small",
    "medium",
    "large-v3",
]

# Voice-activity-detection tuning. faster-whisper's default VAD parameters
# are tuned for speed and can misclassify quieter speech, accented speech,
# or speech immediately following a short pause as silence - dropping it
# from the transcript entirely with no error. These looser settings trade
# a bit of speed for not losing real speech:
#   - min_silence_duration_ms raised: a pause has to be longer before it's
#     treated as a cut point, so brief natural pauses inside a sentence
#     aren't chopped out.
#   - speech_pad_ms raised: extra audio is kept on each side of a detected
#     speech region, so word onsets/endings right at the boundary aren't
#     clipped.
VAD_PARAMETERS = {
    "min_silence_duration_ms": 1000,
    "speech_pad_ms": 400,
}

# Whisper sometimes transliterates certain languages into Latin script
# instead of their native script (e.g. Hindi as "aap kaise ho" instead of
# "आप कैसे हो"), especially on smaller models or ambiguous audio. Seeding
# the decoder with a short native-script prompt biases it toward
# continuing in the correct script for the rest of the transcript. Only
# applied when a specific language is selected (not Auto-detect) and the
# task is "transcribe" (script-seeding doesn't apply to "translate",
# which always outputs English).
SCRIPT_SEED_PROMPTS = {
    "hi": "यह हिंदी में एक सामान्य वाक्य है।",
    "as": "এইটো অসমীয়া ভাষাত এটা সাধাৰণ বাক্য।",
}


# --------------------------------------------------------------------------
# Language helper functions
# --------------------------------------------------------------------------

def language_name_to_code(language_name: str) -> Optional[str]:
    """
    Convert a language name into its language code.

    Example:
        "English" -> "en"
        "Assamese" -> "as"
        "Auto-detect" -> None
    """
    return SUPPORTED_LANGUAGES.get(language_name, None)


def get_gtts_language(language_code: str) -> str:
    """
    Convert detected language code into a gTTS-compatible code.
    """
    gtts_languages = {
        "en": "en", "as": "en", "hi": "hi", "bn": "bn", "ta": "ta",
        "te": "te", "ml": "ml", "kn": "kn", "mr": "mr", "gu": "gu",
        "pa": "pa", "ur": "ur", "ne": "ne", "fr": "fr", "de": "de",
        "es": "es", "it": "it", "pt": "pt", "ja": "ja", "ko": "ko",
        "zh": "zh-CN", "ru": "ru",
    }
    return gtts_languages.get(language_code, "en")


# --------------------------------------------------------------------------
# Result data structures
# --------------------------------------------------------------------------

@dataclass
class TranscriptionSegment:
    """Represents one segment of transcribed speech."""
    start: float
    end: float
    text: str


@dataclass
class TranscriptionResult:
    """
    Final transcription result. Compatible with app.py.
    """
    text: str
    language: str
    segments: List[TranscriptionSegment] = field(default_factory=list)
    # Total duration of the input audio, in seconds. None when unknown.
    audio_duration: Optional[float] = None

    @property
    def covered_duration(self) -> float:
        """How much of the audio timeline the returned segments span."""
        if not self.segments:
            return 0.0
        return self.segments[-1].end

    @property
    def coverage_ratio(self) -> Optional[float]:
        """
        Fraction of the audio's total duration covered by transcribed
        segments. None when audio_duration isn't known. A low ratio is a
        strong signal that VAD or Whisper's no-speech detection silently
        dropped part of the recording.
        """
        if not self.audio_duration or self.audio_duration <= 0:
            return None
        return min(1.0, self.covered_duration / self.audio_duration)


# --------------------------------------------------------------------------
# Audio normalization helpers
# --------------------------------------------------------------------------

def _normalize_audio(input_path: str) -> str:
    """
    Re-encode audio into a clean 16kHz mono WAV via ffmpeg. Browser-recorded
    audio (WebM/Opus blobs) often plays fine in a browser's lenient player
    but has incomplete container metadata that trips up stricter decoders
    downstream - re-encoding through ffmpeg avoids transcripts that are
    silently missing content near format-specific edge cases.
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
    or a truncated duration for some browser-recorded files.
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


# --------------------------------------------------------------------------
# ASR Engine
# --------------------------------------------------------------------------

class ASREngine:
    """
    Faster-Whisper ASR engine, CPU-only.

    int8 quantization gives the best speed/accuracy trade-off on CPU with
    a negligible accuracy cost vs. full precision. cpu_threads is set
    explicitly to the machine's core count, since without it CTranslate2
    may not use all available cores - a common cause of transcription
    feeling "stuck" on modest hardware.
    """

    def __init__(self, model_size: str = "base"):
        self.model_size = model_size
        self.model = None
        self.device = "cpu"
        self.compute_type = "int8"
        # Lazy-loaded on first transcribe() call (rather than in __init__)
        # so a caller can supply a progress_callback and have the model
        # download/load status actually reach the UI, instead of only the
        # server console.

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def _load_model(self, progress_callback: Optional[Callable[[str], None]] = None):
        """Load the Faster-Whisper model on CPU, if not already loaded."""
        if self.model is not None:
            return self.model

        def update(msg: str) -> None:
            print(msg)
            if progress_callback:
                try:
                    progress_callback(msg)
                except Exception:
                    pass

        update(
            f"Loading Faster-Whisper '{self.model_size}' model on CPU — if "
            "this is the first run, it's downloading the model files now "
            "and this can take a while depending on your connection. "
            "Subsequent runs load instantly from local cache."
        )

        try:
            self.model = WhisperModel(
                self.model_size,
                device="cpu",
                compute_type="int8",
                cpu_threads=max(1, os.cpu_count() or 4),
            )
            update("Faster-Whisper successfully loaded on CPU.")
        except Exception as e:
            raise RuntimeError(f"Could not load Faster-Whisper model. Error: {e}") from e

        return self.model

    # ------------------------------------------------------------------
    # Transcription
    # ------------------------------------------------------------------

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
        Transcribe or translate an audio file.

        Parameters
        ----------
        audio_path: path to the audio file.
        language: language code (e.g. "en"), or None to auto-detect.
        task: "transcribe" keeps the original language; "translate"
              translates speech to English.
        beam_size: higher = more accurate but slower on CPU. Use 1
                   (greedy decoding) for the fastest results on low-spec
                   machines, 5 (default) for the best accuracy.
        vad_filter: skips silent stretches (faster, and can reduce
                    hallucinated text in true silence), but can clip real
                    speech it misjudges as silence. Uses the loosened
                    VAD_PARAMETERS above to reduce that risk.
        progress_callback: optional callback for Streamlit status updates.
        """

        def update(msg: str) -> None:
            print(msg)
            if progress_callback:
                try:
                    progress_callback(msg)
                except Exception:
                    pass

        if task not in ("transcribe", "translate"):
            raise ValueError("Task must be either 'transcribe' or 'translate'.")

        if not audio_path:
            raise ValueError("No audio file was provided.")

        audio_file = Path(audio_path)
        if not audio_file.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_file}")
        if not audio_file.is_file():
            raise ValueError(f"Invalid audio file path: {audio_file}")

        file_size = audio_file.stat().st_size
        update(f"Audio file: {audio_file.name}")
        update(f"Audio size: {file_size / 1024:.2f} KB")
        if file_size == 0:
            raise ValueError("The audio file is empty.")

        # --------------------------------------------------------------
        # Normalize audio (fixes truncated/partial transcripts from
        # browser recordings with incomplete container metadata) and
        # decode it ourselves so we know its true duration up front.
        # --------------------------------------------------------------
        update("Normalizing audio with ffmpeg...")
        normalized_path = _normalize_audio(str(audio_file))

        update("Decoding audio into a waveform...")
        audio_array = _load_audio_array(normalized_path)
        audio_duration = len(audio_array) / TARGET_SAMPLE_RATE
        update(f"Audio duration: {audio_duration:.1f}s")

        model = self._load_model(progress_callback=progress_callback)

        update(f"Using device: {self.device} (compute type: {self.compute_type})")
        update("Starting Faster-Whisper transcription...")

        # Bias the decoder toward the correct native script for languages
        # Whisper sometimes transliterates into Latin script instead (e.g.
        # Hindi, Assamese). Only applies when a specific language is
        # selected and we're transcribing (not translating).
        initial_prompt = SCRIPT_SEED_PROMPTS.get(language) if task == "transcribe" else None
        if initial_prompt:
            update(f"Seeding decoder with native-script prompt for '{language}'...")

        try:
            segments_gen, info = model.transcribe(
                audio_array,
                language=language,
                task=task,
                beam_size=beam_size,
                vad_filter=vad_filter,
                vad_parameters=VAD_PARAMETERS if vad_filter else None,
                # Prevents the decoder from conditioning on its own
                # previous output. On longer audio,
                # condition_on_previous_text=True (faster-whisper's
                # default) can cause the model to drift after one
                # uncertain chunk and effectively give up transcribing
                # the rest of the file - a well-known cause of
                # transcripts much shorter than the source audio.
                condition_on_previous_text=False,
                initial_prompt=initial_prompt,
            )

            transcription_segments = []
            text_parts = []

            for segment in segments_gen:
                segment_text = segment.text.strip()
                if not segment_text:
                    continue

                transcription_segments.append(
                    TranscriptionSegment(
                        start=segment.start, end=segment.end, text=segment_text,
                    )
                )
                text_parts.append(segment_text)
                update(f"Transcribed {segment.start:.1f}s - {segment.end:.1f}s")

            final_text = " ".join(text_parts).strip()

            detected_language = getattr(info, "language", "unknown")
            language_probability = getattr(info, "language_probability", None)

            update(f"Detected language: {detected_language}")
            if language_probability is not None:
                update(f"Language confidence: {language_probability:.2%}")

            if not final_text:
                update("No speech was detected.")
            else:
                update("Transcription completed successfully.")

            result = TranscriptionResult(
                text=final_text,
                language=detected_language,
                segments=transcription_segments,
                audio_duration=audio_duration,
            )

            ratio = result.coverage_ratio
            if ratio is not None:
                update(f"Coverage: transcript spans {ratio * 100:.0f}% of the audio duration.")

            return result

        except Exception as e:
            print("Transcription failed.")
            print(f"Error: {e}")
            raise RuntimeError(f"Faster-Whisper transcription failed: {e}") from e