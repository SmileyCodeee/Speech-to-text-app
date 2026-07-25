"""
app.py
------
Speech-to-Text Note Taking Application
A multilingual, offline-capable pipeline that turns spoken audio into
structured, exportable notes.

Pipeline: Audio -> ASR (Whisper) -> Text Structuring -> Summarization
          -> Keyword Extraction -> Export (.txt / .docx / .pdf / .md)

Run with:  streamlit run app.py
"""

import importlib.util
import os
import tempfile
import time

import streamlit as st

from modules.asr_engine import ASREngine, SUPPORTED_LANGUAGES, MODEL_SIZES, language_name_to_code
from modules.text_processor import structure_from_segments, structure_from_text, extract_keywords
from modules.summarizer import summarize
from modules.exporter import export

st.set_page_config(page_title="Speech-to-Text Notes", page_icon="🎙️", layout="wide")

# --------------------------------------------------------------------------
# Detect whether abstractive summarization's dependencies are installed.
# On memory-constrained hosts (e.g. Streamlit Community Cloud's free tier),
# `transformers` + `torch` are deliberately left out of requirements.txt —
# loading the BART model there reliably exceeds the memory limit and
# crashes the whole app with no clear error. Rather than let the app crash
# when someone picks "Abstractive", we simply don't offer that option if
# the packages aren't present.
# --------------------------------------------------------------------------
ABSTRACTIVE_AVAILABLE = (
    importlib.util.find_spec("transformers") is not None
    and importlib.util.find_spec("torch") is not None
)

# --------------------------------------------------------------------------
# Session state
# --------------------------------------------------------------------------
defaults = {
    "transcript_text": "",
    "detected_language": "",
    "structured_notes": None,
    "summary": "",
    "keywords": [],
    "audio_path": None,
}
for k, v in defaults.items():
    st.session_state.setdefault(k, v)


@st.cache_resource(show_spinner=False)
def get_engine(model_size: str) -> ASREngine:
    return ASREngine(model_size=model_size)


def save_uploaded_audio(uploaded_file) -> str:
    suffix = os.path.splitext(uploaded_file.name)[1] or ".wav"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(uploaded_file.getvalue())
    tmp.close()
    return tmp.name


# --------------------------------------------------------------------------
# Sidebar - settings
# --------------------------------------------------------------------------
with st.sidebar:
    st.title("🎙️ Settings")

    st.subheader("Speech Recognition")
    model_size = st.selectbox(
        "Whisper model size",
        MODEL_SIZES,
        index=MODEL_SIZES.index("base"),
        help="Larger models are more accurate but use more memory. On "
             "memory-limited hosts (e.g. Streamlit Community Cloud's free "
             "tier), stick to 'tiny' or 'base' — 'small' and above risk "
             "crashing the app when the model loads.",
    )
    if model_size in ("medium", "large-v3"):
        st.warning(
            f"'{model_size}' needs significant RAM to load. This may crash "
            "the app on memory-limited hosts. Prefer 'tiny' or 'base' if "
            "you're on a free-tier deployment.",
            icon="⚠️",
        )

    transcription_mode = st.radio(
        "Transcription mode",
        ["offline", "online"],
        format_func=lambda x: "Offline (private, works without internet)" if x == "offline"
        else "Online (Google Web Speech API, needs internet)",
        help="Offline uses the local Whisper model. Online uses Google's "
             "free speech API — no model download or extra memory needed, "
             "but requires an internet connection and sends audio to Google.",
    )

    decoding_speed = st.radio(
        "Offline decoding",
        ["fast", "accurate"],
        format_func=lambda x: "Fast (greedy, quickest on CPU)" if x == "fast"
        else "Accurate (beam search, slower on CPU)",
        index=0,
        help="'Accurate' uses beam search (beam_size=5) — noticeably slower "
             "on CPU-only machines. 'Fast' uses greedy decoding and is the "
             "safer default on limited hosting.",
    )
    beam_size = 1 if decoding_speed == "fast" else 5

    vad_filter = st.checkbox(
        "Skip silent stretches (VAD)",
        value=True,
        help="Speeds up transcription by skipping silence, but downloads a "
             "small separate voice-activity-detection model on first use. "
             "Turn this off to test whether a stuck transcription is caused "
             "by that download rather than the main Whisper model.",
    )

    language_name = st.selectbox(
        "Spoken language",
        list(SUPPORTED_LANGUAGES.keys()),
        index=0,
        help="Choose 'Auto-detect' to let Whisper identify the language automatically, "
             "or pick a specific language for better accuracy.",
    )
    task = st.radio(
        "Task",
        ["transcribe", "translate"],
        format_func=lambda x: "Transcribe (keep original language)" if x == "transcribe"
        else "Translate to English",
        help="Whisper can translate any supported language directly into English.",
    )

    st.divider()
    st.subheader("Notes & Summary")

    summary_options = ["extractive"]
    if ABSTRACTIVE_AVAILABLE:
        summary_options.append("abstractive")

    summary_method = st.radio(
        "Summarization method",
        summary_options,
        format_func=lambda x: "Extractive (fast, offline, any language)" if x == "extractive"
        else "Abstractive (Hugging Face model, English-focused, needs internet first run)",
    )

    if not ABSTRACTIVE_AVAILABLE:
        st.caption(
            "ℹ️ Abstractive summarization is unavailable in this deployment "
            "(the `transformers`/`torch` packages aren't installed — likely "
            "intentional on a memory-limited host). Extractive summarization "
            "works fully offline and needs no extra setup."
        )

    num_summary_sentences = st.slider("Summary length (sentences)", 2, 10, 5)
    num_keywords = st.slider("Number of keywords", 3, 20, 8)

    st.divider()
    st.caption(
        "Runs locally using faster-whisper for ASR and lightweight, "
        "language-agnostic NLP for structuring/summarization/keywords. "
        "Offline mode keeps audio on your machine; online mode sends it to "
        "Google's speech API."
    )

# --------------------------------------------------------------------------
# Main layout
# --------------------------------------------------------------------------
st.title("Speech-to-Text Note Taking Application")
st.caption(
    "Convert lectures, meetings, and interviews into clean, structured, "
    "exportable notes — in real time or from a recording, in your own language."
)

tab_record, tab_upload, tab_paste = st.tabs(["🎤 Record", "📁 Upload Audio", "📋 Paste Transcript"])

audio_source_path = None

with tab_record:
    st.write("Record directly in the browser (no extra software needed).")
    audio_value = st.audio_input("Record your lecture / meeting / note")
    if audio_value is not None:
        tmp_path = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        tmp_path.write(audio_value.getvalue())
        tmp_path.close()
        audio_source_path = tmp_path.name
        st.audio(audio_value)

        st.download_button(
            "⬇️ Download recording (.wav)",
            data=audio_value.getvalue(),
            file_name="recording.wav",
            mime="audio/wav",
            use_container_width=True,
        )

with tab_upload:
    st.write("Upload a pre-recorded audio file (wav, mp3, m4a, ogg, flac...).")
    uploaded = st.file_uploader("Choose an audio file", type=["wav", "mp3", "m4a", "ogg", "flac", "webm"])
    if uploaded is not None:
        audio_source_path = save_uploaded_audio(uploaded)
        st.audio(uploaded)

        st.download_button(
            f"⬇️ Download original file ({uploaded.name})",
            data=uploaded.getvalue(),
            file_name=uploaded.name,
            mime=uploaded.type or "application/octet-stream",
            use_container_width=True,
        )

with tab_paste:
    st.write("Already have a transcript? Paste it here to skip straight to structuring/summarizing.")
    pasted_text = st.text_area("Paste transcript text", height=200)
    if st.button("Process pasted text", use_container_width=True) and pasted_text.strip():
        notes = structure_from_text(pasted_text)
        st.session_state.structured_notes = notes
        st.session_state.transcript_text = pasted_text
        st.session_state.detected_language = language_name_to_code(language_name) or "unknown"
        st.success("Transcript structured! Scroll down to view your notes.")

if audio_source_path:
    st.session_state.audio_path = audio_source_path
    if st.button("🔎 Transcribe Audio", type="primary", use_container_width=True):
        lang_code = language_name_to_code(language_name)

        # st.status gives a live, updating log of exactly which stage is
        # running (normalizing / downloading model / decoding), instead of
        # one static spinner message - this is what previously made a
        # slow-but-working transcription look identical to a hung one.
        with st.status(f"Transcribing with Whisper '{model_size}'...", expanded=True) as status:
            def _update(msg: str) -> None:
                status.write(msg)

            try:
                engine = get_engine(model_size)
                start = time.time()
                result = engine.transcribe(
                    audio_source_path,
                    language=lang_code,
                    task=task,
                    mode=transcription_mode,
                    beam_size=beam_size,
                    vad_filter=vad_filter,
                    progress_callback=_update,
                )
                elapsed = time.time() - start

                st.session_state.transcript_text = result.text
                st.session_state.detected_language = result.language

                # Online mode (and any path with no timestamps) returns an
                # empty segments list, so fall back to sentence-count-based
                # structuring instead of leaving paragraphs empty.
                if result.segments:
                    st.session_state.structured_notes = structure_from_segments(result.segments)
                else:
                    st.session_state.structured_notes = structure_from_text(result.text)

                if not result.text.strip():
                    status.update(label="Transcription finished, but no speech was detected.", state="error")
                    st.warning(
                        "Transcription finished but returned no text. The "
                        "recording may be silent, too quiet, or the wrong "
                        "language was selected — try re-recording, lowering "
                        "the mic distance, or switching modes."
                    )
                else:
                    status.update(
                        label=f"Transcribed in {elapsed:.1f}s. Detected language: {result.language}",
                        state="complete",
                    )
            except ImportError:
                status.update(label="Missing dependency", state="error")
                st.error(
                    "The `faster-whisper` package isn't installed in this "
                    "environment. Install with:\n\n`pip install -U faster-whisper`\n\n"
                    "You'll also need `ffmpeg` installed on your system (e.g. "
                    "`sudo apt install ffmpeg` / `brew install ffmpeg`), or the "
                    "`imageio-ffmpeg` package as a fallback."
                )
            except MemoryError:
                status.update(label="Out of memory", state="error")
                st.error(
                    "Transcription ran out of memory. Try a smaller Whisper "
                    "model size ('tiny' or 'base') from the sidebar, or "
                    "switch to Online mode, which doesn't load a local model."
                )
            except Exception as e:
                status.update(label="Transcription failed", state="error")
                st.error(f"Transcription failed: {e}")

# --------------------------------------------------------------------------
# Results
# --------------------------------------------------------------------------
if st.session_state.structured_notes and st.session_state.structured_notes.paragraphs:
    st.divider()
    st.header("📝 Structured Notes")

    left, right = st.columns([2, 1])

    with left:
        st.subheader("Notes")
        editable_notes = st.text_area(
            "Edit your notes below if needed:",
            value="\n\n".join(st.session_state.structured_notes.paragraphs),
            height=350,
        )
        current_paragraphs = [p.strip() for p in editable_notes.split("\n\n") if p.strip()]

    with right:
        st.subheader("Detected Language")
        st.info(st.session_state.detected_language or "N/A")

        if st.button("✨ Generate Summary", use_container_width=True):
            with st.spinner("Summarizing..."):
                try:
                    st.session_state.summary = summarize(
                        st.session_state.transcript_text,
                        method=summary_method,
                        num_sentences=num_summary_sentences,
                    )
                except MemoryError:
                    st.error(
                        "Summarization ran out of memory. Try Extractive mode "
                        "instead, which needs no large model."
                    )

        if st.button("🔑 Extract Keywords", use_container_width=True):
            with st.spinner("Extracting keywords..."):
                st.session_state.keywords = extract_keywords(
                    st.session_state.transcript_text,
                    language_code=st.session_state.detected_language,
                    top_n=num_keywords,
                )

    if st.session_state.summary:
        st.subheader("Summary")
        st.write(st.session_state.summary)

    if st.session_state.keywords:
        st.subheader("Keywords")
        st.write(" · ".join(f"`{k}`" for k in st.session_state.keywords))

    st.divider()
    st.header("📤 Export Notes")
    title = st.text_input("Note title", value="My Notes")
    fmt = st.selectbox("Format", ["docx", "pdf", "txt", "md"])

    if st.button("Generate export file", type="primary"):
        data = export(
            fmt, title, current_paragraphs,
            summary=st.session_state.summary,
            keywords=st.session_state.keywords,
        )
        mime_types = {
            "docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            "pdf": "application/pdf",
            "txt": "text/plain",
            "md": "text/markdown",
        }
        st.download_button(
            f"⬇️ Download {title}.{fmt}",
            data=data,
            file_name=f"{title}.{fmt}",
            mime=mime_types[fmt],
            use_container_width=True,
        )
else:
    st.info("Record or upload audio above, or paste a transcript, to get started.")