# Speech-to-Text Note Taking Application

A multilingual, privacy-first note-taking app that converts spoken audio
(live recording or uploaded files) into clean, structured, exportable notes —
with automatic summarization and keyword extraction.

Built to match the project brief: real-time ASR, automatic note structuring,
multilingual support, multiple export formats, and both offline and online
processing.

## Pipeline

Audio (mic / file) ──▶ Normalize (ffmpeg) ──▶ ASR (Whisper / Google) ──▶ Text Structuring ──▶ Summarization
│ │
▼ ▼
Keyword Extraction Export (.txt/.docx/.pdf/.md)

| Stage              | Technique / Library                              |
|--------------------|---------------------------------------------------|
| Audio capture      | Streamlit `st.audio_input` (browser mic) / file upload |
| Audio normalization| ffmpeg re-encode to 16kHz mono WAV, decoded via `soundfile` — avoids silent 0-sample decode failures on browser-recorded audio |
| Speech recognition | **Offline**: [OpenAI Whisper](https://github.com/openai/whisper) — open-source, ~100 languages, auto language detection, fully local. **Online**: Google's free Web Speech API via `SpeechRecognition` — no model download, but needs internet and a specific language |
| Text structuring   | Pause-based paragraph segmentation (offline mode) + regex sentence splitting with a word-window fallback for unpunctuated speech |
| Summarization      | Extractive (frequency-based, stopword-aware, offline, any language) or optional abstractive (Hugging Face Transformers, e.g. BART) |
| Keyword extraction | [YAKE](https://github.com/LIAAD/yake) — unsupervised, multilingual |
| Export             | `python-docx` (.docx), `reportlab` (.pdf), plain `.txt`, `.md` |
| Interface          | [Streamlit](https://streamlit.io/) |

## Features

- 🎙️ Record live from your browser microphone, or upload a pre-recorded file (wav/mp3/m4a/ogg/flac/webm).
- 🌐 **Offline or online transcription**: run Whisper fully locally (private, no internet needed after the model is downloaded once), or switch to Google's free Web Speech API for a lighter, no-download option when you have internet.
- 🌍 **Multilingual**: auto-detect the spoken language or pick from 19+ languages (English, Hindi, Assamese, Bengali, Spanish, French, German, Chinese, Japanese, Korean, Arabic, Russian, and more — anything Whisper supports). Online mode requires picking a specific language (no auto-detect).
- 🔄 Optional translation of any language directly to English (offline mode).
- 📝 Automatically restructures raw speech into readable paragraphs/bullet points based on natural pauses (offline mode) or sentence-count windows (online mode, since Google's free API doesn't return timestamps).
- ✨ One-click summary (extractive or abstractive) and keyword extraction for quick review.
- ✏️ Edit the generated notes before exporting.
- 📤 Export to `.txt`, `.docx`, `.pdf`, or `.md`.
- 🔒 Offline mode keeps audio and text entirely on your machine; online mode sends audio to Google's servers for that transcription only.

## Setup

1. Install [ffmpeg](https://ffmpeg.org/) (used to normalize audio before it reaches either transcription engine):
   - Ubuntu/Debian: `sudo apt install ffmpeg`
   - macOS: `brew install ffmpeg`
   - Windows: `winget install ffmpeg` (or download from ffmpeg.org and add to PATH)

   If ffmpeg isn't found on your system PATH, the app automatically falls
   back to the portable binary bundled with the `imageio-ffmpeg` pip
   package (installed via `requirements.txt`), so a system-wide install
   isn't strictly required.

2. Create a virtual environment and install dependencies:
```bash
   python3 -m venv venv
   source venv/bin/activate        # Windows: venv\Scripts\activate
   pip install -r requirements.txt
```

   Key packages: `streamlit`, `openai-whisper`, `torch`, `soundfile`,
   `numpy`, `SpeechRecognition` (online mode), `imageio-ffmpeg` (ffmpeg
   fallback), `yake`, `python-docx`, `reportlab`, and optionally
   `transformers` for abstractive summarization.

3. Run the app:
```bash
   streamlit run app.py
```

   In **Offline mode**, the first time you transcribe audio, Whisper will
   download the chosen model (e.g. `small` ≈ 460 MB) — this requires an
   internet connection once; after that, transcription runs fully offline.

   In **Online mode**, no model download is needed, but every transcription
   requires an active internet connection since audio is sent to Google's
   free Web Speech API.

   Toggle between the two anytime from the sidebar's "Transcription mode" setting.

## Choosing a Whisper model size (offline mode)

| Size    | Approx. speed | Approx. accuracy | Good for |
|---------|---------------|-------------------|----------|
| tiny    | fastest       | lowest            | quick drafts, low-resource machines |
| base    | fast          | ok                | short notes |
| small   | balanced      | good              | **recommended default** |
| medium  | slower        | very good         | important recordings |
| large-v3| slowest       | best              | maximum accuracy, needs a strong GPU |

Online mode has no model size setting — Google's API handles this server-side.

## Project structure

speech_notes_app/
├── app.py # Streamlit UI - entry point
├── modules/
│ ├── asr_engine.py # Whisper (offline) + Google Web Speech API (online) ASR, with ffmpeg-based audio normalization
│ ├── text_processor.py # Structuring + keyword extraction (YAKE), with a word-window fallback for unpunctuated speech
│ ├── summarizer.py # Extractive (stopword-aware) + optional abstractive summarization
│ └── exporter.py # .txt / .docx / .pdf / .md export
├── requirements.txt
└── README.md


## Troubleshooting

- **"Transcription failed: [WinError 2] The system cannot find the file specified"** — ffmpeg isn't visible on PATH from the process running Streamlit. Restart your terminal/IDE after installing ffmpeg (PATH changes don't apply to already-running processes), or rely on the automatic `imageio-ffmpeg` fallback.
- **"cannot reshape tensor of 0 elements..."** — usually caused by browser-recorded audio that plays fine but has incomplete container metadata. The app now normalizes and decodes audio itself before handing it to Whisper, which should prevent this; if it still occurs, try re-recording or re-uploading the file.
- **Online mode fails with a connection error** — check your internet connection, or switch back to Offline mode from the sidebar.
- **Summary looks identical to the transcript** — very short recordings (a few sentences) are summarized proportionally rather than cut down to a fixed count; for genuinely short notes, the "summary" may reasonably include most of the content.

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
- Online mode uses Google's *free*, unofficial Web Speech endpoint, which
  has no uptime or rate-limit guarantees — fine for personal/demo use, but
  not a substitute for a paid API in production.
- Could be extended with speaker diarization, live streaming transcription,
  or collaborative/cloud sync for a future version.
