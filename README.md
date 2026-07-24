# Speech-to-Text Note Taking Application

A multilingual, privacy-first note-taking app that converts spoken audio
(live recording or uploaded files) into clean, structured, exportable notes —
with automatic summarization and keyword extraction.

Built to match the project brief: real-time ASR, automatic note structuring,
multilingual support, multiple export formats, and both offline and online
processing. Works on **Windows** and **Linux** (and macOS).

## Pipeline

```
Audio (mic / file) ──▶ Normalize (ffmpeg) ──▶ ASR (faster-whisper / Google) ──▶ Text Structuring ──▶ Summarization
                                                                                       │                    │
                                                                                       ▼                    ▼
                                                                                 Keyword Extraction    Export (.txt/.docx/.pdf/.md)
```

| Stage              | Technique / Library                              |
|--------------------|---------------------------------------------------|
| Audio capture      | Streamlit `st.audio_input` (browser mic) / file upload — both offer a direct download button for the captured/uploaded audio |
| Audio normalization| ffmpeg re-encode to 16kHz mono WAV, decoded via `soundfile` — avoids silent 0-sample decode failures on browser-recorded audio |
| Speech recognition | **Offline**: [faster-whisper](https://github.com/SYSTRAN/faster-whisper) (CTranslate2 reimplementation of OpenAI Whisper) — same accuracy as the original model, 2-4x faster on CPU via int8 quantization, ~100 languages, auto language detection, adjustable beam size and voice-activity-detection (VAD) filtering. **Online**: Google's free Web Speech API via `SpeechRecognition` — no model download, but needs internet and a specific language |
| Text structuring   | Pause-based paragraph segmentation (offline mode) + regex sentence splitting with a word-window fallback for unpunctuated speech |
| Summarization      | **Extractive**: graph-based TextRank (sentences ranked by similarity to the rest of the transcript, not just raw word frequency — more accurate at surfacing genuinely central sentences and ignoring filler). Offline, any language. **Abstractive** (optional): Hugging Face BART, loaded directly via `AutoModelForSeq2SeqLM` with beam search for more coherent output |
| Keyword extraction | [YAKE](https://github.com/LIAAD/yake) — unsupervised, multilingual |
| Export             | `python-docx` (.docx), `reportlab` (.pdf), plain `.txt`, `.md` |
| Interface          | [Streamlit](https://streamlit.io/) |

## Features

- 🎙️ Record live from your browser microphone, or upload a pre-recorded file (wav/mp3/m4a/ogg/flac/webm) — both let you download the captured audio directly.
- 🌐 **Offline or online transcription**: run faster-whisper fully locally (private, no internet needed after the model is downloaded once), or switch to Google's free Web Speech API for a lighter, no-download option when you have internet.
- ⚡ **Speed vs. accuracy control**: choose "Fast" (greedy decoding) for quicker offline transcription on CPU, or "Accurate" (beam search) for better quality when time isn't critical. A voice-activity-detection (VAD) toggle can also skip silent stretches to speed things up further.
- 🌍 **Multilingual**: auto-detect the spoken language or pick from 19+ languages (English, Hindi, Assamese, Bengali, Spanish, French, German, Chinese, Japanese, Korean, Arabic, Russian, and more). Online mode requires picking a specific language (no auto-detect).
- 🔄 Optional translation of any language directly to English (offline mode).
- 📝 Automatically restructures raw speech into readable paragraphs/bullet points based on natural pauses (offline mode) or sentence-count windows (online mode, since Google's free API doesn't return timestamps).
- ✨ One-click summary (TextRank extractive or Hugging Face abstractive) and keyword extraction for quick review.
- ✏️ Edit the generated notes before exporting.
- 📤 Export to `.txt`, `.docx`, `.pdf`, or `.md`.
- 🔒 Offline mode keeps audio and text entirely on your machine; online mode sends audio to Google's servers for that transcription only.

## Setup

### 1. Install ffmpeg

Whisper/faster-whisper and audio normalization both rely on ffmpeg. If it
isn't found on your system PATH, the app automatically falls back to the
portable binary bundled with the `imageio-ffmpeg` pip package (already in
`requirements.txt`), so a system-wide install isn't strictly required —
but installing it system-wide is still recommended for reliability.

**Windows:**
```powershell
winget install ffmpeg
```
After installing, **fully close and reopen your terminal / IDE** — PATH
changes only apply to newly started processes, not ones already running.
Verify it worked with:
```powershell
ffmpeg -version
```
If that fails to run after a restart, find the install location with:
```powershell
Get-Command ffmpeg | Select-Object -ExpandProperty Source
```
and manually add that folder to your PATH via
*Edit the system environment variables → Environment Variables → Path → New*.

**Linux (Debian/Ubuntu):**
```bash
sudo apt update && sudo apt install ffmpeg
```
**Linux (Fedora):**
```bash
sudo dnf install ffmpeg
```
**Linux (Arch):**
```bash
sudo pacman -S ffmpeg
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

If `pip install` fails on any package with a compiler error on Linux, you
may need build essentials first:
```bash
sudo apt install build-essential python3-dev
```

Key packages installed: `streamlit`, `faster-whisper` (offline ASR),
`soundfile`, `numpy`, `av` (audio decoding), `SpeechRecognition` (online
mode), `imageio-ffmpeg` (ffmpeg fallback), `yake`, `python-docx`,
`reportlab`, and `transformers` + `torch` (only used if you select
"abstractive" summarization).

### 3. Run the app

**Windows:**
```powershell
streamlit run app.py
```

**Linux / macOS:**
```bash
streamlit run app.py
```

In **Offline mode**, the first time you transcribe audio, faster-whisper
will download the chosen model (e.g. `small`) — this requires an internet
connection once; after that, transcription runs fully offline.

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
| large-v3| slowest       | best              | maximum accuracy, needs a strong GPU or patience on CPU |

Online mode has no model size setting — Google's API handles this server-side.

Within offline mode, the **"Offline decoding"** setting further trades speed
for quality:
- **Fast** — greedy decoding (`beam_size=1`), quickest on CPU-only machines.
- **Accurate** — beam search (`beam_size=5`), slower but generally more precise.

The **"Skip silent stretches (VAD)"** toggle uses voice-activity detection to
skip over silence in the recording, which can noticeably speed up
transcription of audio with pauses — at the cost of a small extra model
download the first time it's used.

## Project structure

```
speech_notes_app/
├── app.py                  # Streamlit UI - entry point
├── modules/
│   ├── asr_engine.py        # faster-whisper (offline) + Google Web Speech API (online) ASR, with ffmpeg-based audio normalization
│   ├── text_processor.py    # Structuring + keyword extraction (YAKE), with a word-window fallback for unpunctuated speech
│   ├── summarizer.py         # TextRank extractive + optional abstractive summarization
│   └── exporter.py           # .txt / .docx / .pdf / .md export
├── requirements.txt
└── README.md
```

## Troubleshooting

- **"Transcription failed: [WinError 2] The system cannot find the file specified"** (Windows) — ffmpeg isn't visible on PATH from the process running Streamlit. Restart your terminal/IDE after installing ffmpeg, or rely on the automatic `imageio-ffmpeg` fallback.
- **"ffmpeg: command not found"** (Linux) — ffmpeg isn't installed system-wide and the `imageio-ffmpeg` fallback wasn't able to run either; install it with your distro's package manager (see Setup step 1) or check `pip show imageio-ffmpeg` installed correctly.
- **"cannot reshape tensor of 0 elements..."** — usually caused by browser-recorded audio that plays fine but has incomplete container metadata. The app normalizes and decodes audio itself before handing it to the ASR engine, which should prevent this; if it still occurs, try re-recording or re-uploading the file.
- **Online mode fails with a connection error** — check your internet connection, or switch back to Offline mode from the sidebar.
- **Offline transcription feels slow or stuck** — switch "Offline decoding" to "Fast", try a smaller model size (e.g. `base` or `tiny`), pick a specific spoken language instead of "Auto-detect", or toggle VAD off to rule out the VAD model download as the cause.
- **Abstractive summary fails with an "Unknown task" or pipeline-related error** — this is handled automatically: the app loads the BART model directly (bypassing `pipeline("summarization")`), and falls back to the extractive summary with an explanatory message if abstractive summarization still fails for any reason (e.g. no internet on first run).
- **Summary looks identical to (or very close to) the transcript** — very short recordings (a few sentences) are summarized proportionally rather than cut down to a fixed count; for genuinely short notes, the "summary" may reasonably include most of the content.
- **`pip install` fails on Linux with a compiler/build error** — install build tools first: `sudo apt install build-essential python3-dev` (Debian/Ubuntu) or the equivalent for your distro.

## Notes & possible extensions

- The "Paste Transcript" tab lets you test structuring/summarization/export
  without any audio at all — handy for quickly trying the app out.
- Extractive summarization uses TextRank (graph-based sentence ranking) —
  more robust than plain word-frequency scoring, but for genuinely fluent,
  rewritten summaries, use "Abstractive" mode instead.
- Abstractive summarization defaults to `facebook/bart-large-cnn`, which is
  English-only; for multilingual abstractive summaries, swap in a model like
  `csebuetnlp/mT5_multilingual_XLSum` in `summarizer.py`.
- Online mode uses Google's *free*, unofficial Web Speech endpoint, which
  has no uptime or rate-limit guarantees — fine for personal/demo use, but
  not a substitute for a paid API in production.
- Could be extended with speaker diarization, live streaming transcription,
  or collaborative/cloud sync for a future version.
