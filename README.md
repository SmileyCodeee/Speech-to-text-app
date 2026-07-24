# Speech-to-Text Note Taking Application

A multilingual, privacy-first note-taking app that converts spoken audio (live recording or uploaded files) into clean, structured, exportable notes — with automatic summarization, keyword extraction, and optional read-aloud.

**Hybrid architecture** — most features work fully offline, with optional online enhancements. Works on Windows, Linux, and macOS.

---

## Pipeline — Hybrid Architecture

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
       Local Processing          Online Processing
          (OFFLINE)                 (OPTIONAL)
      • Formatting              • Abstractive Summary
      • Keywords (YAKE)           (Hugging Face BART)
      • Extractive Summary      • 🔊 Read Aloud (gTTS)
              │                         │
              └────────────┬────────────┘
                           ▼
                    Structured Notes
                           │
              ┌────────────┴────────────┐
              ▼                         ▼
         Export Notes             🔊 Read Notes Aloud
       TXT / DOCX / PDF / MD             │
            (OFFLINE)                    ▼
                                        gTTS
                                      (ONLINE)
```

| Stage | Technique / Library | Offline? |
|---|---|---|
| Audio capture | Streamlit `st.audio_input` (browser mic) / file upload — both offer a direct download button for the captured/uploaded audio | ✓ |
| Audio normalization | ffmpeg re-encode to 16kHz mono WAV, decoded via `soundfile` — avoids silent 0-sample decode failures on browser-recorded audio. Falls back to `imageio-ffmpeg` portable binary if system ffmpeg isn't found. | ✓ |
| Speech recognition | `faster-whisper` (CTranslate2) — same accuracy as OpenAI Whisper, 2–4× faster on CPU via int8 quantization. ~100 languages, auto-detect, adjustable beam size and VAD. No API keys needed. | ✓ |
| Text structuring | Pause-based paragraph segmentation using Whisper timestamps + regex sentence splitting with word-window fallback | ✓ |
| Keyword extraction | YAKE — unsupervised, multilingual, with comprehensive post-filtering to remove generic/useless words | ✓ |
| Extractive summary | TextRank graph-based with TF-IDF weighting + position bias — ranks sentences by centrality to the transcript's themes, not raw word frequency | ✓ |
| Abstractive summary | Hugging Face `facebook/bart-large-cnn` via `AutoModelForSeq2SeqLM` — needs internet the first time to download, then cached locally. Falls back to extractive if unavailable. | ✗ (first run) → ✓ (cached) |
| Read aloud | gTTS — Google Text-to-Speech converts notes/summary into spoken audio. Requires internet. | ✗ |
| Export | `python-docx` (.docx), `reportlab` (.pdf), plain `.txt`, `.md` | ✓ |
| Interface | Streamlit | ✓ |

### What works offline vs. online

| Feature | Offline | Online |
|---|---|---|
| Record / upload audio | ✓ | ✓ |
| Transcribe (faster-whisper) | ✓ (after model download) | ✓ (model download) |
| Auto-detect language | ✓ | ✓ |
| Translate to English | ✓ | ✓ |
| Structure into paragraphs | ✓ | ✓ |
| Extract keywords | ✓ | ✓ |
| Extractive summary | ✓ | ✓ |
| Abstractive summary | ✗ (must download model once) | ✓ |
| Read notes aloud (gTTS) | ✗ | ✓ |
| Export to TXT/DOCX/PDF/MD | ✓ | ✓ |
| Edit notes before export | ✓ | ✓ |

> **Bottom line:** After the Whisper model downloads once (requires internet), all core features work fully offline. Only abstractive summarization (first run) and read-aloud need internet.

---

## Features

- 🎙️ Record live from your browser microphone, or upload a pre-recorded file (wav/mp3/m4a/ogg/flac/webm) — both let you download the captured audio directly.
- 🔒 Offline ASR — `faster-whisper` runs locally. No audio or text is sent anywhere. No API keys needed.
- ⚡ Speed vs. accuracy control — choose "Fast" (greedy) or "Accurate" (beam search) decoding, toggle VAD to skip silence.
- 🌍 Multilingual — auto-detect the spoken language or pick from 20+ languages (English, Hindi, Assamese, Bengali, Spanish, French, German, Chinese, Japanese, Korean, Arabic, Russian, and more).
- 🔄 Optional translation of any language directly to English.
- 📝 Automatically restructures raw speech into readable paragraphs/bullet points based on natural pauses.
- ✨ One-click summary — extractive (TextRank + TF-IDF, offline, any language) or abstractive (BART, needs internet first run, falls back to extractive if offline).
- 🔑 Keyword extraction with comprehensive post-filtering to remove generic words.
- ✏️ Edit the generated notes before exporting.
- 📤 Export to `.txt`, `.docx`, `.pdf`, or `.md` — fully offline.
- 🔊 Read Notes Aloud — gTTS converts your notes or summary into spoken audio you can play in the browser and download as MP3. Requires internet.

---

## Setup

### 1. Install ffmpeg

The app needs ffmpeg to re-encode audio before transcription. If it isn't on your system PATH, the app automatically falls back to the portable binary bundled with `imageio-ffmpeg` (already in `requirements.txt`), so a system-wide install isn't strictly required — but recommended for reliability.

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

**Windows (PowerShell):**
```powershell
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

**Linux / macOS:**
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

If `pip install` fails on Linux with a compiler error:
```bash
sudo apt install build-essential python3-dev libsndfile1
```

Key packages: `streamlit`, `faster-whisper` (ASR), `soundfile`, `numpy`, `imageio-ffmpeg` (ffmpeg fallback), `yake` (keywords), `python-docx`, `reportlab` (export), `transformers` + `torch` (abstractive summary), `gTTS` (read aloud).

> `requirements.txt` lists only direct dependencies with loose version pins. Transitive deps (CUDA libs, triton, etc.) are resolved automatically by pip — this avoids cross-platform conflicts on Windows.

### 3. Run the app

```bash
streamlit run app.py
```

The first time you transcribe audio, `faster-whisper` downloads the chosen model (e.g. `small` ≈ 500 MB) — this requires internet once; after that, transcription is fully offline.

### Choosing a Whisper model size

| Size | Speed | Accuracy | Good for |
|---|---|---|---|
| tiny | fastest | lowest | quick drafts, low-resource machines |
| base | fast | ok | short notes |
| small | balanced | good | recommended default |
| medium | slower | very good | important recordings |
| large-v3 | slowest | best | maximum accuracy, needs GPU or patience |

> `faster-whisper` uses int8 on CPU (float16 on GPU), giving a 2–4× speedup over the original `openai-whisper`. `device="auto"` picks GPU if available.

---

## Read Notes Aloud (gTTS)

The 🔊 Read Notes Aloud feature uses gTTS (Google Text-to-Speech) to convert your notes, summary, or both into spoken audio:

1. Choose what to read: **Notes**, **Summary**, or **Notes + Summary**
2. Click **🔊 Generate Audio**
3. The audio plays in the browser and can be downloaded as `.mp3`

gTTS auto-detects the language from the transcription result and uses the appropriate voice. If the language isn't supported by gTTS (e.g. Assamese), it falls back to English.

> gTTS requires internet — it sends the text to Google's TTS servers. All other features (ASR, structuring, keywords, extractive summary, export) work offline without internet.

---

## Project structure

```
speech_notes_app/
├── app.py                  # Streamlit UI — entry point, gTTS read-aloud
├── modules/
│   ├── asr_engine.py       # faster-whisper ASR (offline) + gTTS lang mapping
│   ├── text_processor.py   # Structuring + keyword extraction (YAKE) with post-filtering
│   ├── summarizer.py       # TextRank extractive (offline) + abstractive (online, cached)
│   └── exporter.py         # .txt / .docx / .pdf / .md export (offline)
├── requirements.txt
└── README.md
```

---

## Cross-platform notes

| Aspect | Linux | Windows |
|---|---|---|
| Python venv activate | `source venv/bin/activate` | `venv\Scripts\activate` |
| ffmpeg (system) | `sudo apt install ffmpeg` | `winget install ffmpeg` or manual PATH |
| ffmpeg (fallback) | `imageio-ffmpeg` portable binary (auto) | Same |
| GPU acceleration | CUDA toolkit + PyTorch CUDA wheel | PyTorch CUDA wheel from pytorch.org |
| triton | Auto-installed with CUDA torch | Not on Windows — not needed; int8 works |
| soundfile / libsndfile | May need `sudo apt install libsndfile1` | Bundled in pip wheel |
| gTTS (read aloud) | Needs internet | Needs internet |

---

## Troubleshooting

- **`[WinError 2]` (Windows)** — ffmpeg not on PATH. Restart your terminal after installing, or rely on the `imageio-ffmpeg` fallback.
- **`ffmpeg: command not found` (Linux)** — install with `sudo apt install ffmpeg`.
- **`cannot reshape tensor of 0 elements...`** — browser-recorded audio with incomplete metadata. The app normalizes audio before Whisper; if it persists, re-record.
- **Transcription slow or stuck** — switch to "Fast" decoding, try a smaller model (`base`/`tiny`), pick a specific language instead of "Auto-detect", or disable VAD.
- **Abstractive summary unavailable offline** — the BART model must be downloaded once while online. After that it's cached locally. If offline and not cached, the app automatically falls back to extractive summary with a notice.
- **"Could not generate audio" (Read Aloud)** — gTTS needs internet. Check your connection. All other features work offline.
- **gTTS not installed** — run `pip install gTTS`.
- **Keywords contain generic words** — the app post-filters YAKE keywords against a ~300-word blacklist. Increase the "Number of keywords" slider for more candidates.
- **`pip install` fails on Linux** — install build tools: `sudo apt install build-essential python3-dev libsndfile1`.
- **`pip install` fails with NVIDIA/CUDA errors on Windows** — use the cleaned `requirements.txt` (only direct deps). For GPU, install the PyTorch CUDA wheel first from pytorch.org.