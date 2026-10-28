# Universal Video-to-Text Transcription & Speaker Diarization Pipeline

A production-ready pipeline for transcribing video or audio files into structured text using **FFmpeg**, **Qwen3-ASR-1.7B** (`QwenLM/Qwen3-ASR`), **Qwen3-ForcedAligner-0.6B** for word-level timestamps, and **PyAnnote Speaker Diarization**.

---

## Repository Layout

```
speech-to-text_interview/
├── README.md                          # Project documentation & usage guide
├── pyproject.toml                     # Python project configuration (uv / pip)
├── requirements.txt                   # Locked dependencies specification
├── uv.lock                            # Deterministic package resolution lockfile
├── .gitignore                         # Git exclusion rules (ignores media, venv, caches)
│
├── scripts/                           # Runnable shell and Python scripts
│   ├── setup_env.sh                   # One-click environment bootstrap script
│   ├── transcribe_pipeline.py         # Standalone transcription pipeline runner
│   └── download_weights.py            # Standalone weights pre-download runner
│
├── src/                               # Modular Python source package
│   └── video_transcription/
│       ├── __init__.py                # Package exports
│       ├── pipeline.py                # Core transcription & diarization orchestrator
│       ├── audio.py                   # Universal FFmpeg media extraction
│       ├── export.py                  # Formatters: TXT, SRT, VTT, JSON
│       └── weights.py                 # Weight pre-caching utilities
│
├── tests/                             # Automated test suite
│   └── test_pipeline.py               # Unit tests for formatters, alignment, diarization
│
├── data/
│   ├── input/                         # Local media inputs (ignored by git)
│   └── output/                        # Default output destination (ignored by git)
│
└── examples/                          # Sample outputs from verified run
    ├── Carl Sagan's 1994 "Lost" Lecture...txt
    ├── Carl Sagan's 1994 "Lost" Lecture...srt
    ├── Carl Sagan's 1994 "Lost" Lecture...vtt
    └── Carl Sagan's 1994 "Lost" Lecture...json
```

---

## Architecture & Features

1. **Qwen3-ASR-1.7B Speech Recognition**:
   - High-accuracy speech-to-text across 52 languages and dialects.
   - Accelerated natively on Apple Silicon Mac (`mps` via Metal Performance Shaders) with `torch.float16`, on CUDA with `torch.bfloat16`, and on CPU with `torch.float32`.

2. **Qwen3-ForcedAligner-0.6B Timestamp Alignment**:
   - Non-autoregressive forced alignment engine providing word- and token-level start and end timestamps.

3. **PyAnnote Speaker Diarization**:
   - Cascaded speaker clustering (`pyannote/speaker-diarization-community-1`) assigns speaker turn labels (`SPEAKER_00`, `SPEAKER_01`, etc.) to aligned words and dialogue segments.

4. **Robust Multi-Format Subtitle & Transcript Exporter**:
   - Exports `.txt`, `.srt`, `.vtt`, and `.json`.
   - Includes millisecond roll-over protection (prevents 4-digit milliseconds like `00:00:01,1000`).
   - Supports safe handling of missing timestamps and filenames containing dots.

---

## Quick Start

### 1. Environment Setup

Run the bootstrap script from anywhere:
```bash
chmod +x scripts/setup_env.sh
./scripts/setup_env.sh
```

Or manually using `uv`:
```bash
uv venv --python 3.12 .venv
uv pip install -r requirements.txt
uv pip install -e .
```

### 2. Activating the Virtual Environment

Activate the `.venv` environment in your shell:

**macOS / Linux (Bash/Zsh):**
```bash
source .venv/bin/activate
```

**Windows (PowerShell):**
```powershell
.venv\Scripts\Activate.ps1
```

**Windows (Command Prompt):**
```cmd
.venv\Scripts\activate.bat
```

> **Tip**: Once activated, your shell prompt will show `(.venv)`. You can then run any command directly (e.g. `transcribe`, `download-weights`, or `pytest`) without prefixing with `uv run`.
> To exit the environment at any time:
> ```bash
> deactivate
> ```

### 3. Pre-download Weights (Standard Cache)

Pre-load `Qwen/Qwen3-ASR-1.7B` and `Qwen/Qwen3-ForcedAligner-0.6B` directly into `~/.cache/huggingface/hub/`:
```bash
uv run download-weights
# or: python scripts/download_weights.py
```

To pre-cache PyAnnote Speaker Diarization models (requires Hugging Face token):
```bash
uv run download-weights --hf-token "your_hf_token_here"
```

---

## Running the Transcription Pipeline

### Basic Usage (Full Video)
```bash
# Using the installed CLI:
transcribe "data/input/video.webm"

# Or using uv run:
uv run transcribe "data/input/video.webm"

# Or using the script directly:
python scripts/transcribe_pipeline.py "data/input/video.webm"
```

### Quick Test Snippet (e.g., first 30 seconds)
```bash
transcribe "data/input/video.webm" --duration 30
```

### Specifying Output Directory
```bash
transcribe "data/input/video.webm" --output-dir "data/output"
```

### With Speaker Diarization
To identify and label speakers (`SPEAKER_00`, `SPEAKER_01`, etc.):
```bash
export HF_TOKEN="hf_xxxxxxxxxxxxxxxxxxxx"
transcribe "data/input/video.webm" --diarize
```
Or pass the token directly:
```bash
transcribe "data/input/video.webm" --diarize --hf-token "hf_xxxxxxxxxxxxxxxxxxxx"
```

> **Note for Diarization**: Ensure you have accepted user conditions on:
> - [pyannote/speaker-diarization-community-1](https://huggingface.co/pyannote/speaker-diarization-community-1)
> - [pyannote/segmentation-3.0](https://huggingface.co/pyannote/segmentation-3.0)

### Universal Format Support
Accepts any video container and codec:
- Video: `.webm`, `.mp4`, `.mkv`, `.mov`, `.avi`, `.flv`, `.wmv`, `.m4v`, `.ts`
- Audio: `.mp3`, `.m4a`, `.aac`, `.flac`, `.wav`, `.ogg`, `.opus`

FFmpeg extracts audio stream into a normalized 16kHz 16-bit mono PCM WAV for maximum ASR accuracy.

---

## Generated Outputs

For an input file named `lecture.webm`, the following files are produced:

| Output File | Description |
|---|---|
| `lecture.txt` | Clean plain-text transcript with timestamps and speaker labels. |
| `lecture.srt` | Standard SubRip subtitle file for video players (VLC, YouTube, etc.). |
| `lecture.vtt` | WebVTT caption file formatted with speaker voice tags (`<v SPEAKER_00>`). |
| `lecture.json` | Complete structured JSON containing word-level timestamps, confidence scores, and speaker IDs. |
| `lecture.extracted.wav` | Extracted 16kHz mono audio (pass `--cleanup-audio` to remove). |

---

## Running Automated Tests

Run the test suite using `pytest`:
```bash
uv run pytest tests/
```

---

## CLI Options

| Argument | Description | Default |
|---|---|---|
| `media_file` | Path to video or audio file | *(Required)* |
| `-o`, `--output-dir` | Target output directory | Input file directory |
| `--model` | Qwen ASR model repository ID | `Qwen/Qwen3-ASR-1.7B` |
| `--aligner-model` | Qwen Forced Aligner model repository ID | `Qwen/Qwen3-ForcedAligner-0.6B` |
| `-l`, `--language` | Language (e.g. `English`, `Chinese`, or code `en`, `zh`). Auto-detected if omitted. | Auto-detect |
| `--device` | Inference device (`mps`, `cuda`, `cpu`) | Auto-detect (`mps` on Mac) |
| `--no-align` | Skip word-level forced alignment | Alignment enabled |
| `--diarize` | Enable PyAnnote Speaker Diarization | Disabled |
| `--hf-token` | Hugging Face token for diarization | `$HF_TOKEN` env var |
| `--min-speakers` | Minimum number of speakers | Auto |
| `--max-speakers` | Maximum number of speakers | Auto |
| `--start-time` | Start offset in seconds or `HH:MM:SS` | `0` |
| `--duration` | Process only first N seconds | Full file |
| `--cleanup-audio` | Delete intermediate WAV after transcription | Keep WAV |
