# Speech-to-Text Note Taking Application

A multilingual, privacy-first note-taking app that converts spoken audio
(live recording or uploaded files) into clean, structured, exportable notes —
with automatic summarization and keyword extraction.

Built to match the project brief: real-time ASR, automatic note structuring,
multilingual support, multiple export formats, and local/offline processing.

## Pipeline

```
Audio (mic / file) ──▶ Whisper ASR ──▶ Text Structuring ──▶ Summarization
                                              │                    │
                                              ▼                    ▼
                                        Keyword Extraction    Export (.txt/.docx/.pdf/.md)
```

| Stage              | Technique / Library                              |
|--------------------|---------------------------------------------------|
| Audio capture      | Streamlit `st.audio_input` (browser mic) / file upload |
| Speech recognition | [OpenAI Whisper](https://github.com/openai/whisper) — open-source, offline, ~100 languages, auto language detection |
| Text structuring   | Pause-based paragraph segmentation + regex sentence splitting |
| Summarization      | Extractive (frequency-based, offline, any language) or optional abstractive (Hugging Face Transformers, e.g. BART) |
| Keyword extraction | [YAKE](https://github.com/LIAAD/yake) — unsupervised, multilingual |
| Export             | `python-docx` (.docx), `reportlab` (.pdf), plain `.txt`, `.md` |
| Interface          | [Streamlit](https://streamlit.io/) |

## Features

- 🎙️ Record live from your browser microphone, or upload a pre-recorded file (wav/mp3/m4a/ogg/flac/webm).
- 🌍 **Multilingual**: auto-detect the spoken language or pick from 19+ languages (English, Hindi, Assamese, Bengali, Spanish, French, German, Chinese, Japanese, Korean, Arabic, Russian, and more — anything Whisper supports).
- 🔄 Optional translation of any language directly to English.
- 📝 Automatically restructures raw speech into readable paragraphs/bullet points based on natural pauses.
- ✨ One-click summary and keyword extraction for quick review.
- ✏️ Edit the generated notes before exporting.
- 📤 Export to `.txt`, `.docx`, `.pdf`, or `.md`.
- 🔒 Runs entirely on your machine — audio and text never need to leave your device.

## Setup

1. Install [ffmpeg](https://ffmpeg.org/) (required by Whisper to read audio):
   - Ubuntu/Debian: `sudo apt install ffmpeg`
   - macOS: `brew install ffmpeg`
   - Windows: download from ffmpeg.org and add to PATH

2. Create a virtual environment and install dependencies:
   ```bash
   python3 -m venv venv
   source venv/bin/activate        # Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. Run the app:
   ```bash
   streamlit run app.py
   ```

   The first time you transcribe audio, Whisper will download the chosen
   model (e.g. `small` ≈ 460 MB) — this requires an internet connection
   once; after that, everything runs fully offline.

## Choosing a Whisper model size

| Size    | Approx. speed | Approx. accuracy | Good for |
|---------|---------------|-------------------|----------|
| tiny    | fastest       | lowest            | quick drafts, low-resource machines |
| base    | fast          | ok                | short notes |
| small   | balanced      | good              | **recommended default** |
| medium  | slower        | very good         | important recordings |
| large-v3| slowest       | best              | maximum accuracy, needs a strong GPU |

## Project structure

```
speech_notes_app/
├── app.py                  # Streamlit UI - entry point
├── modules/
│   ├── asr_engine.py        # Whisper-based speech recognition
│   ├── text_processor.py    # Structuring + keyword extraction (YAKE)
│   ├── summarizer.py         # Extractive + optional abstractive summarization
│   └── exporter.py           # .txt / .docx / .pdf / .md export
├── requirements.txt
└── README.md
```

## Notes & possible extensions

- The "Paste Transcript" tab lets you test structuring/summarization/export
  without any audio at all — handy for quickly trying the app out.
- For faster inference (especially on CPU), consider swapping in
  [faster-whisper](https://github.com/SYSTRAN/faster-whisper) or
  [Vosk](https://alphacephei.com/vosk/) in `asr_engine.py` — both are
  mentioned as alternatives in the original project proposal.
- Abstractive summarization defaults to `facebook/bart-large-cnn`, which is
  English-only; for multilingual abstractive summaries, swap in a model like
  `csebuetnlp/mT5_multilingual_XLSum` in `summarizer.py`.
- Could be extended with speaker diarization, live streaming transcription,
  or collaborative/cloud sync for a future version.
