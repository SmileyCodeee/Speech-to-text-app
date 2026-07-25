"""
asr_engine.py
-------------
Faster-Whisper based Automatic Speech Recognition engine.

Features:
- Automatic CUDA detection
- Safe CPU fallback for Render and other CPU-only systems
- Faster-Whisper transcription
- Language detection
- Transcription and translation
- VAD filtering
- Progress callback support
- WAV and other audio file support
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import ctranslate2
from faster_whisper import WhisperModel


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

MODEL_SIZES = [
    "tiny",
    "base",
    "small",
    "medium",
    "large-v3",
]


# --------------------------------------------------------------------------
# Language helper functions
# --------------------------------------------------------------------------

def language_name_to_code(
    language_name: str,
) -> Optional[str]:
    """
    Convert a language name into its language code.

    Example:
        "English" -> "en"
        "Assamese" -> "as"
        "Auto-detect" -> None
    """

    return SUPPORTED_LANGUAGES.get(
        language_name,
        None,
    )


def get_gtts_language(
    language_code: str,
) -> str:
    """
    Convert detected language code into a gTTS-compatible code.
    """

    gtts_languages = {
        "en": "en",
        "as": "en",
        "hi": "hi",
        "bn": "bn",
        "ta": "ta",
        "te": "te",
        "ml": "ml",
        "kn": "kn",
        "mr": "mr",
        "gu": "gu",
        "pa": "pa",
        "ur": "ur",
        "ne": "ne",
        "fr": "fr",
        "de": "de",
        "es": "es",
        "it": "it",
        "pt": "pt",
        "ja": "ja",
        "ko": "ko",
        "zh": "zh-CN",
        "ru": "ru",
    }

    return gtts_languages.get(
        language_code,
        "en",
    )


# --------------------------------------------------------------------------
# Result data structures
# --------------------------------------------------------------------------

@dataclass
class TranscriptionSegment:
    """
    Represents one segment of transcribed speech.
    """

    start: float
    end: float
    text: str


@dataclass
class TranscriptionResult:
    """
    Final transcription result.

    This structure is compatible with app.py.
    """

    text: str
    language: str
    segments: list


# --------------------------------------------------------------------------
# ASR Engine
# --------------------------------------------------------------------------

class ASREngine:
    """
    Faster-Whisper ASR engine.

    CUDA is detected safely before attempting GPU initialization.

    On Render:
        CUDA devices = 0
        -> CPU
        -> int8

    On a supported NVIDIA GPU system:
        CUDA devices > 0
        -> CUDA
        -> float16
    """

    def __init__(
        self,
        model_size: str = "base",
    ):

        self.model_size = model_size

        self.model = None

        self.device = None

        self.compute_type = None

        # Load model
        self._load_model()

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def _load_model(self):
        """
        Load Faster-Whisper model.

        First checks CUDA availability using CTranslate2.

        This avoids trying to initialize CUDA on Render,
        which does not provide an NVIDIA GPU.
        """

        print("=" * 60)

        print(
            "Loading Faster-Whisper model"
        )

        print(
            f"Model size: {self.model_size}"
        )

        print("=" * 60)

        # --------------------------------------------------------------
        # Check CUDA safely
        # --------------------------------------------------------------

        try:

            cuda_device_count = (
                ctranslate2.get_cuda_device_count()
            )

            print(
                f"CUDA devices detected: "
                f"{cuda_device_count}"
            )

        except Exception as e:

            print(
                "CUDA availability check failed."
            )

            print(
                f"CUDA check error: {e}"
            )

            cuda_device_count = 0

        # --------------------------------------------------------------
        # GPU path
        # --------------------------------------------------------------

        if cuda_device_count > 0:

            print(
                "CUDA GPU detected."
            )

            print(
                "Trying GPU acceleration..."
            )

            try:

                self.model = WhisperModel(
                    self.model_size,
                    device="cuda",
                    compute_type="float16",
                )

                self.device = "cuda"

                self.compute_type = "float16"

                print(
                    "Faster-Whisper successfully "
                    "loaded on CUDA GPU."
                )

                return

            except Exception as e:

                print(
                    "GPU model loading failed."
                )

                print(
                    f"GPU error: {e}"
                )

                print(
                    "Falling back to CPU..."
                )

        # --------------------------------------------------------------
        # CPU path
        # --------------------------------------------------------------

        print(
            "No usable CUDA GPU detected."
        )

        print(
            "Loading Faster-Whisper on CPU..."
        )

        try:

            self.model = WhisperModel(
                self.model_size,
                device="cpu",
                compute_type="int8",
            )

            self.device = "cpu"

            self.compute_type = "int8"

            print(
                "Faster-Whisper successfully "
                "loaded on CPU."
            )

        except Exception as e:

            print(
                "Failed to load Faster-Whisper model."
            )

            raise RuntimeError(
                "Could not load Faster-Whisper model. "
                f"Error: {e}"
            ) from e

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
        progress_callback: Optional[
            Callable[[str], None]
        ] = None,
    ) -> TranscriptionResult:
        """
        Transcribe or translate an audio file.

        Parameters
        ----------
        audio_path:
            Path to the audio file.

        language:
            Language code, for example "en".
            None means automatic language detection.

        task:
            "transcribe" keeps the original language.
            "translate" translates speech to English.

        beam_size:
            Beam size for decoding.

        vad_filter:
            Skip silent sections.

        progress_callback:
            Optional callback for Streamlit status updates.
        """

        # --------------------------------------------------------------
        # Helper for progress updates
        # --------------------------------------------------------------

        def update(message: str):

            print(message)

            if progress_callback:

                try:

                    progress_callback(
                        message
                    )

                except Exception:

                    pass

        # --------------------------------------------------------------
        # Validate task
        # --------------------------------------------------------------

        if task not in (
            "transcribe",
            "translate",
        ):

            raise ValueError(
                "Task must be either "
                "'transcribe' or 'translate'."
            )

        # --------------------------------------------------------------
        # Validate audio path
        # --------------------------------------------------------------

        if not audio_path:

            raise ValueError(
                "No audio file was provided."
            )

        audio_file = Path(
            audio_path
        )

        if not audio_file.exists():

            raise FileNotFoundError(
                f"Audio file not found: "
                f"{audio_file}"
            )

        if not audio_file.is_file():

            raise ValueError(
                f"Invalid audio file path: "
                f"{audio_file}"
            )

        # --------------------------------------------------------------
        # File information
        # --------------------------------------------------------------

        file_size = (
            audio_file.stat().st_size
        )

        update(
            f"Audio file: "
            f"{audio_file.name}"
        )

        update(
            f"Audio size: "
            f"{file_size / 1024:.2f} KB"
        )

        if file_size == 0:

            raise ValueError(
                "The audio file is empty."
            )

        # --------------------------------------------------------------
        # Start transcription
        # --------------------------------------------------------------

        update(
            f"Using device: "
            f"{self.device}"
        )

        update(
            f"Compute type: "
            f"{self.compute_type}"
        )

        update(
            "Starting Faster-Whisper "
            "transcription..."
        )

        try:

            segments, info = (
                self.model.transcribe(
                    str(audio_file),
                    language=language,
                    task=task,
                    beam_size=beam_size,
                    vad_filter=vad_filter,
                )
            )

            # ----------------------------------------------------------
            # Collect segments
            # ----------------------------------------------------------

            transcription_segments = []

            text_parts = []

            for segment in segments:

                segment_text = (
                    segment.text.strip()
                )

                if not segment_text:

                    continue

                transcription_segments.append(
                    TranscriptionSegment(
                        start=segment.start,
                        end=segment.end,
                        text=segment_text,
                    )
                )

                text_parts.append(
                    segment_text
                )

                update(
                    f"Transcribed "
                    f"{segment.start:.1f}s - "
                    f"{segment.end:.1f}s"
                )

            # ----------------------------------------------------------
            # Combine text
            # ----------------------------------------------------------

            final_text = " ".join(
                text_parts
            ).strip()

            # ----------------------------------------------------------
            # Detected language
            # ----------------------------------------------------------

            detected_language = getattr(
                info,
                "language",
                "unknown",
            )

            language_probability = getattr(
                info,
                "language_probability",
                None,
            )

            update(
                f"Detected language: "
                f"{detected_language}"
            )

            if language_probability is not None:

                update(
                    f"Language confidence: "
                    f"{language_probability:.2%}"
                )

            # ----------------------------------------------------------
            # Empty transcription
            # ----------------------------------------------------------

            if not final_text:

                update(
                    "No speech was detected."
                )

            else:

                update(
                    "Transcription completed "
                    "successfully."
                )

            # ----------------------------------------------------------
            # Return result
            # ----------------------------------------------------------

            return TranscriptionResult(
                text=final_text,
                language=detected_language,
                segments=transcription_segments,
            )

        except Exception as e:

            print(
                "Transcription failed."
            )

            print(
                f"Error: {e}"
            )

            raise RuntimeError(
                "Faster-Whisper transcription "
                f"failed: {e}"
            ) from e