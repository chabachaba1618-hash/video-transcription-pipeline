# Universal Video-to-Text Transcription & Speaker Diarization Pipeline

A production-ready pipeline for transcribing video or audio files into structured text using **FFmpeg**, **OpenAI Whisper large-v3** (via WhisperX / CTranslate2), **Wav2Vec2** phoneme alignment, and **PyAnnote Speaker Diarization**.

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

## Deliverables Included

1. **UV Environment with all packages loaded**:
   - Python 3.12 virtual environment managed via `uv`.
   - Pre-installed wheels for `whisperx`, `torch`, `torchaudio`, `ctranslate2`, `pyannote-audio`, and `tqdm`.
   - Registered CLI commands `transcribe` and `download-weights`.

2. **Pre-cached Weights in Standard Directories**:
   - Whisper `large-v3` cached in `~/.cache/huggingface/hub/models--Systran--faster-whisper-large-v3`.
   - Wav2Vec2 alignment models cached in `~/.cache/torch/hub/checkpoints/`.
   - Script `scripts/download_weights.py` to pre-download any model offline.

3. **Python Pipeline Script**:
   - `scripts/transcribe_pipeline.py` (or `transcribe` CLI): Universal media conversion, `tqdm` progress tracking, word alignment, optional speaker diarization, and multi-format output generation (`.txt`, `.srt`, `.vtt`, `.json`).

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

### 2. Pre-download Weights (Standard Cache)

Pre-load `large-v3` and alignment weights directly into `~/.cache/huggingface/hub/`:
```bash
uv run download-weights --model large-v3
# or: python scripts/download_weights.py --model large-v3
```

To pre-cache PyAnnote Speaker Diarization models (requires Hugging Face token):
```bash
uv run download-weights --model large-v3 --hf-token "your_hf_token_here"
```

---

## 3. Running the Transcription Pipeline

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
Or pass token directly:
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

## CLI Options

| Argument | Description | Default |
|---|---|---|
| `media_file` | Path to video or audio file | *(Required)* |
| `-o`, `--output-dir` | Target output directory | Input file directory |
| `--model` | Whisper model size (`tiny`, `base`, `small`, `medium`, `large-v2`, `large-v3`) | `large-v3` |
| `-l`, `--language` | Language code (e.g. `en`, `fr`, `es`). Auto-detected if omitted. | Auto-detect |
| `--device` | Inference device (`cpu` or `cuda`) | Auto-detect (`cpu` on Mac) |
| `--compute-type` | Precision (`int8`, `float32`, `float16`) | `int8` on CPU |
| `--batch-size` | Batch size for transcription | `8` |
| `--no-align` | Skip word-level alignment | Alignment enabled |
| `--diarize` | Enable PyAnnote Speaker Diarization | Disabled |
| `--hf-token` | Hugging Face token for diarization | `$HF_TOKEN` env var |
| `--min-speakers` | Minimum number of speakers | Auto |
| `--max-speakers` | Maximum number of speakers | Auto |
| `--start-time` | Start offset in seconds or `HH:MM:SS` | `0` |
| `--duration` | Process only first N seconds | Full file |
| `--cleanup-audio` | Delete intermediate WAV after transcription | Keep WAV |
