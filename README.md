# Speech-to-Text Note Taking Application

A multilingual, privacy-first note-taking app that converts spoken audio (live recording or uploaded files) into clean, structured, exportable notes — with automatic summarization and keyword extraction.

**Offline-first** — every core feature (ASR, structuring, keyword extraction, extractive summarization, export) runs fully locally with no internet connection. Abstractive summarization is available as an optional add-on that needs internet on its first run only. Works on Windows, Linux, and macOS.

---

## Pipeline

```
                    AUDIO INPUT
                   /            \
          Microphone          Audio File
                   \            /
                    \          /
                     ▼        ▼
              Faster-Whisper
           (Local ASR — OFFLINE)
                     │
                     ▼
             Transcribed Text
                     │
        ┌────────────┴────────────┐
        ▼                         ▼
  Text Structuring          Summarization
     (OFFLINE)          ┌───────┴────────┐
  • Paragraph breaks     ▼                ▼
    from pause gaps  Extractive       Abstractive
  • Keywords (YAKE)   (OFFLINE)    (OPTIONAL, needs
                                    transformers+torch —
                                    see Setup below)
        │                         │
        └────────────┬────────────┘
                     ▼
              Structured Notes
                     │
                     ▼
               Export Notes
          TXT / DOCX / PDF / MD
                (OFFLINE)
```

| Stage | Technique / Library | Offline? |
|---|---|---|
| Audio capture | Streamlit `st.audio_input` (browser mic) / file upload — both offer a direct download button for the captured/uploaded audio | ✓ |
| Audio normalization | ffmpeg re-encode to 16kHz mono WAV, decoded via `soundfile` — avoids silent 0-sample decode failures on browser-recorded audio. Falls back to `imageio-ffmpeg` portable binary if system ffmpeg isn't found. | ✓ |
| Speech recognition | `faster-whisper` (CTranslate2) — same accuracy as OpenAI Whisper, 2–4× faster on CPU via int8 quantization. Auto-detect or 20+ selectable languages. No API keys needed. | ✓ |
| Text structuring | Paragraph breaks inferred from pause gaps between Whisper's timestamped segments (≥2s gap = new paragraph); regex sentence splitting with a word-window fallback for unpunctuated text | ✓ |
| Keyword extraction | YAKE (multilingual) + a ~300-word post-filter blacklist to remove generic/filler terms, plus deduplication of single words already covered by a surviving multi-word phrase | ✓ |
| Extractive summary | TextRank (graph centrality) over TF-IDF-weighted sentence vectors, plus a position bias favoring the first/last sentences. Stopwords are language-aware for English, Hindi, and Assamese specifically — other languages fall back to English stopwords. | ✓ |
| Abstractive summary | Hugging Face `facebook/bart-large-cnn`, loaded directly via `AutoModelForSeq2SeqLM`. **Not installed by default** — see Setup. Needs internet the first time to download (~1.6GB), then cached locally. Falls back to extractive automatically if unavailable or offline. | ✗ (first run) → ✓ (cached) |
| Export | `python-docx` (.docx), `reportlab` (.pdf), plain `.txt`, `.md` | ✓ |
| Interface | Streamlit | ✓ |

> **Bottom line:** ASR, structuring, keywords, extractive summarization, and export all work with zero internet connection, out of the box. Abstractive summarization is opt-in (see Setup) and needs internet once to download its model.

---

## Features

- 🎙️ Record live from your browser microphone, or upload a pre-recorded file (wav/mp3/m4a/ogg/flac/webm) — both let you download the captured/uploaded audio directly.
- 🔒 Offline ASR — `faster-whisper` runs locally, CPU-only. No audio or text is sent anywhere. No API keys needed.
- ⚡ Speed vs. accuracy control — "Fast" (greedy) or "Accurate" (beam search) decoding, and a toggle to skip silent stretches (VAD).
- 🌍 Multilingual — auto-detect the spoken language or pick from 20+ languages (English, Hindi, Assamese, Bengali, Tamil, Telugu, Malayalam, Kannada, Marathi, Gujarati, Punjabi, Urdu, Nepali, French, German, Spanish, Italian, Portuguese, Japanese, Korean, Chinese, Russian).
- 🔤 Native-script seeding for Hindi and Assamese — when one of these is explicitly selected (not Auto-detect), the decoder is biased with a short native-script prompt to avoid transliterating into Latin script. This only applies when explicitly selected, not under Auto-detect.
- 🔄 Optional translation of any language directly to English.
- 📝 Automatically restructures raw speech into paragraphs based on natural pauses in the audio.
- ✨ One-click summary — extractive (TextRank + TF-IDF, offline, strongest for English/Hindi/Assamese) or abstractive (BART, optional install, needs internet first run).
- 🔑 Keyword extraction with a comprehensive post-filter to remove generic/filler words.
- ✏️ Edit the generated notes before exporting.
- 📤 Export to `.txt`, `.docx`, `.pdf`, or `.md` — fully offline.

---

## Setup

### 1. Install ffmpeg

The app needs ffmpeg to re-encode audio before transcription. If it isn't on your system PATH, the app automatically falls back to the portable binary bundled with `imageio-ffmpeg`, so a system-wide install isn't strictly required — but recommended for reliability.

**Windows:**
```powershell
winget install ffmpeg
```
After installing, fully close and reopen your terminal / IDE — PATH changes only apply to newly started processes. Verify with `ffmpeg -version`.

**Linux (Debian/Ubuntu):**
```bash
sudo apt update && sudo apt install ffmpeg
```

**Linux (Fedora):**
```bash
sudo dnf install ffmpeg
```

**macOS:**
```bash
brew install ffmpeg
```

### 2. Create a virtual environment and install dependencies

There are **two requirements files** — pick one:

- **`requirements.txt`** — lightweight, cloud-friendly. Extractive summarization only. Recommended for free-tier hosts (Streamlit Community Cloud, Render free tier, ~512MB–1GB RAM) and for most local use.
- **`requirements-full.txt`** — everything in `requirements.txt`, plus `transformers`+`torch` (CPU-only) for abstractive summarization. Use this if you have several GB of free RAM and want the "Abstractive" summary option available.

**Windows (PowerShell):**
```powershell
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
# or, for abstractive summarization too:
pip install -r requirements-full.txt
```

**Linux / macOS:**
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
# or:
pip install -r requirements-full.txt
```

If `pip install` fails on Linux with a compiler error:
```bash
sudo apt install build-essential python3-dev libsndfile1
```

> The app detects at runtime whether `transformers`/`torch` are installed and simply hides the "Abstractive" option in the sidebar if they aren't — so installing the lightweight `requirements.txt` is always safe; nothing crashes, that option just won't appear.

### 3. Run the app

```bash
streamlit run app.py
```

The first time you transcribe audio, `faster-whisper` downloads the chosen model (e.g. `base` ≈ 150MB, `small` ≈ 500MB) — this requires internet once; after that, transcription is fully offline.

### Choosing a Whisper model size

| Size | Speed | Accuracy | Good for |
|---|---|---|---|
| tiny | fastest | lowest | quick drafts, low-resource machines |
| base | fast | ok | short notes — **recommended default on memory-limited hosts** |
| small | balanced | good | better accuracy, needs more RAM/time |
| medium | slower | very good | important recordings, needs significant RAM |
| large-v3 | slowest | best | maximum accuracy, needs a lot of RAM/CPU time or a GPU |

> `faster-whisper` runs CPU-only in this app (int8 quantized) by design, so it works reliably on hosts with no GPU at all, like most free-tier deployments.

---

## Project structure

```
speech_notes_app/
├── app.py                  # Streamlit UI — entry point
├── modules/
│   ├── asr_engine.py       # faster-whisper ASR (offline)
│   ├── text_processor.py   # Structuring + keyword extraction (YAKE) with post-filtering
│   ├── summarizer.py       # TextRank extractive (offline) + abstractive (optional, online first-run)
│   └── exporter.py         # .txt / .docx / .pdf / .md export (offline)
├── requirements.txt        # lightweight — cloud-friendly, extractive summary only
├── requirements-full.txt   # adds transformers+torch for abstractive summarization
└── README.md
```

---

## Cross-platform notes

| Aspect | Linux | Windows |
|---|---|---|
| Python venv activate | `source venv/bin/activate` | `venv\Scripts\activate` |
| ffmpeg (system) | `sudo apt install ffmpeg` | `winget install ffmpeg` or manual PATH |
| ffmpeg (fallback) | `imageio-ffmpeg` portable binary (auto) | Same |
| GPU acceleration | Not used — this app runs faster-whisper CPU-only by design | Same |
| soundfile / libsndfile | May need `sudo apt install libsndfile1` | Bundled in pip wheel |

---

## Troubleshooting

- **`[WinError 2]` (Windows)** — ffmpeg not on PATH. Restart your terminal after installing, or rely on the `imageio-ffmpeg` fallback.
- **`ffmpeg: command not found` (Linux)** — install with `sudo apt install ffmpeg`.
- **Transcript in the wrong script (e.g. Hindi shown in Latin letters)** — happens most often under "Auto-detect", since native-script seeding currently only activates when Hindi or Assamese is explicitly selected in the sidebar. Manually selecting the language, and using at least the `small` model, gives much more reliable results for these languages.
- **Transcription slow or stuck** — switch to "Fast" decoding, try a smaller model (`base`/`tiny`), pick a specific language instead of "Auto-detect", or disable VAD.
- **"Abstractive" option missing from the sidebar** — `transformers`/`torch` aren't installed. Install with `pip install -r requirements-full.txt` instead of `requirements.txt`.
- **Abstractive summary unavailable offline** — the BART model must be downloaded once while online. After that it's cached locally. If offline and not cached, the app automatically falls back to extractive summary with a notice.
- **Keywords contain generic words** — the app post-filters YAKE keywords against a ~300-word blacklist. Increase the "Number of keywords" slider for more candidates.
- **`pip install` fails on Linux** — install build tools: `sudo apt install build-essential python3-dev libsndfile1`.
- **`pip install` fails with NVIDIA/CUDA errors (installing `requirements-full.txt`)** — that file is already pinned to the CPU-only PyTorch index; if you still hit CUDA-related errors, clear pip's cache (`pip cache purge`) and retry. For actual GPU acceleration, install the CUDA-enabled wheel first from pytorch.org before running `pip install -r requirements-full.txt`.