"""Transcript formatting and export utilities for TXT, SRT, VTT, and JSON."""

import json
from pathlib import Path
from typing import Dict, Any, List


def format_timestamp_srt(seconds: float) -> str:
    """Format seconds into SRT timestamp: HH:MM:SS,mmm"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int(round((seconds - int(seconds)) * 1000))
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def format_timestamp_vtt(seconds: float) -> str:
    """Format seconds into WebVTT timestamp: HH:MM:SS.mmm"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int(round((seconds - int(seconds)) * 1000))
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"


def save_transcription_outputs(
    result: Dict[str, Any],
    output_stem: Path,
    include_speakers: bool = True,
) -> Dict[str, Path]:
    """Save transcripts in TXT, SRT, VTT, and JSON formats."""
    segments: List[Dict[str, Any]] = result.get("segments", [])

    # 1. Plain Text (.txt) with timestamps and optional speaker markers
    txt_path = output_stem.with_suffix(".txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        for seg in segments:
            speaker = seg.get("speaker") if include_speakers else None
            start_fmt = format_timestamp_vtt(seg.get("start", 0.0))
            end_fmt = format_timestamp_vtt(seg.get("end", 0.0))
            text = seg.get("text", "").strip()

            if speaker:
                f.write(f"[{start_fmt} - {end_fmt}] {speaker}: {text}\n\n")
            else:
                f.write(f"[{start_fmt} - {end_fmt}] {text}\n\n")

    # 2. SubRip Subtitles (.srt)
    srt_path = output_stem.with_suffix(".srt")
    with open(srt_path, "w", encoding="utf-8") as f:
        for i, seg in enumerate(segments, start=1):
            start = seg.get("start", 0.0)
            end = seg.get("end", 0.0)
            text = seg.get("text", "").strip()
            speaker = seg.get("speaker") if include_speakers else None

            f.write(f"{i}\n")
            f.write(f"{format_timestamp_srt(start)} --> {format_timestamp_srt(end)}\n")
            if speaker:
                f.write(f"[{speaker}] {text}\n\n")
            else:
                f.write(f"{text}\n\n")

    # 3. WebVTT (.vtt)
    vtt_path = output_stem.with_suffix(".vtt")
    with open(vtt_path, "w", encoding="utf-8") as f:
        f.write("WEBVTT\n\n")
        for i, seg in enumerate(segments, start=1):
            start = seg.get("start", 0.0)
            end = seg.get("end", 0.0)
            text = seg.get("text", "").strip()
            speaker = seg.get("speaker") if include_speakers else None

            f.write(f"{i}\n")
            f.write(f"{format_timestamp_vtt(start)} --> {format_timestamp_vtt(end)}\n")
            if speaker:
                f.write(f"<v {speaker}>{text}</v>\n\n")
            else:
                f.write(f"{text}\n\n")

    # 4. Structured JSON (.json)
    json_path = output_stem.with_suffix(".json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    return {
        "txt": txt_path,
        "srt": srt_path,
        "vtt": vtt_path,
        "json": json_path,
    }
