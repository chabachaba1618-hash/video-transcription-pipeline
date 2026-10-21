"""
Universal Video-to-Text Transcription & Speaker Diarization Pipeline.
Powered by FFmpeg, WhisperX (large-v3), and PyAnnote.
"""

from .pipeline import run_pipeline, main as transcribe_main
from .audio import extract_audio_ffmpeg, find_ffmpeg
from .export import save_transcription_outputs
from .weights import download_whisper_weights, download_alignment_weights, download_diarization_weights

__version__ = "0.1.0"
__all__ = [
    "run_pipeline",
    "transcribe_main",
    "extract_audio_ffmpeg",
    "find_ffmpeg",
    "save_transcription_outputs",
    "download_whisper_weights",
    "download_alignment_weights",
    "download_diarization_weights",
]
