"""Automated unit tests for Qwen3-ASR pipeline components and export utilities."""

import json
from pathlib import Path
import pytest

from video_transcription.export import (
    format_timestamp_srt,
    format_timestamp_vtt,
    save_transcription_outputs,
)
from video_transcription.pipeline import (
    assign_speakers_to_words,
    group_words_into_segments,
    smart_join_tokens,
)


def test_format_timestamp_srt():
    assert format_timestamp_srt(0.0) == "00:00:00,000"
    assert format_timestamp_srt(12.345) == "00:00:12,345"
    # Millisecond overflow roll-over protection
    assert format_timestamp_srt(1.9999) == "00:00:02,000"
    assert format_timestamp_srt(59.9996) == "00:01:00,000"
    assert format_timestamp_srt(3599.9999) == "01:00:00,000"


def test_format_timestamp_vtt():
    assert format_timestamp_vtt(0.0) == "00:00:00.000"
    assert format_timestamp_vtt(12.345) == "00:00:12.345"
    assert format_timestamp_vtt(1.9999) == "00:00:02.000"
    assert format_timestamp_vtt(59.9996) == "00:01:00.000"


def test_smart_join_tokens():
    assert smart_join_tokens(["Hello", "world"]) == "Hello world"
    assert smart_join_tokens(["你", "好", "世", "界"]) == "你好世界"
    assert smart_join_tokens(["Hello", "世界"]) == "Hello 世界"
    assert smart_join_tokens([]) == ""


def test_assign_speakers_to_words():
    words = [
        {"text": "Good", "start": 0.1, "end": 0.5, "speaker": None},
        {"text": "morning", "start": 0.6, "end": 1.0, "speaker": None},
        {"text": "Thank", "start": 2.1, "end": 2.5, "speaker": None},
        {"text": "you", "start": 2.6, "end": 2.9, "speaker": None},
    ]
    turns = [
        (0.0, 1.5, "SPEAKER_00"),
        (2.0, 3.5, "SPEAKER_01"),
    ]
    assigned = assign_speakers_to_words(words, turns)
    assert assigned[0]["speaker"] == "SPEAKER_00"
    assert assigned[1]["speaker"] == "SPEAKER_00"
    assert assigned[2]["speaker"] == "SPEAKER_01"
    assert assigned[3]["speaker"] == "SPEAKER_01"


def test_group_words_into_segments():
    words = [
        {"text": "Hello", "start": 0.0, "end": 0.4, "speaker": "SPEAKER_00"},
        {"text": "world.", "start": 0.5, "end": 0.9, "speaker": "SPEAKER_00"},
        {"text": "How", "start": 2.0, "end": 2.4, "speaker": "SPEAKER_01"},
        {"text": "are", "start": 2.5, "end": 2.8, "speaker": "SPEAKER_01"},
        {"text": "you?", "start": 2.9, "end": 3.2, "speaker": "SPEAKER_01"},
    ]
    segs = group_words_into_segments(words, max_duration=10.0, pause_threshold=1.0)
    assert len(segs) == 2
    assert segs[0]["speaker"] == "SPEAKER_00"
    assert segs[0]["text"] == "Hello world."
    assert segs[0]["start"] == 0.0
    assert segs[0]["end"] == 0.9

    assert segs[1]["speaker"] == "SPEAKER_01"
    assert segs[1]["text"] == "How are you?"
    assert segs[1]["start"] == 2.0
    assert segs[1]["end"] == 3.2


def test_save_transcription_outputs(tmp_path: Path):
    result = {
        "text": "Hello world.",
        "language": "English",
        "segments": [
            {
                "start": None,  # Test None timestamp protection
                "end": 1.5,
                "text": "Hello world.",
                "speaker": "SPEAKER_00",
            }
        ],
    }

    # Test filename with dots
    output_stem = tmp_path / "interview.2026.10.file"
    outputs = save_transcription_outputs(result, output_stem, include_speakers=True)

    # Check paths exist and preserve full stem
    assert outputs["txt"].name == "interview.2026.10.file.txt"
    assert outputs["srt"].name == "interview.2026.10.file.srt"
    assert outputs["vtt"].name == "interview.2026.10.file.vtt"
    assert outputs["json"].name == "interview.2026.10.file.json"

    assert outputs["txt"].exists()
    assert outputs["srt"].exists()
    assert outputs["vtt"].exists()
    assert outputs["json"].exists()

    # Verify content
    vtt_content = outputs["vtt"].read_text(encoding="utf-8")
    assert "WEBVTT" in vtt_content
    assert "<v SPEAKER_00>Hello world.</v>" in vtt_content

    srt_content = outputs["srt"].read_text(encoding="utf-8")
    assert "00:00:00,000 --> 00:00:01,500" in srt_content
    assert "[SPEAKER_00] Hello world." in srt_content

    json_content = json.loads(outputs["json"].read_text(encoding="utf-8"))
    assert json_content["language"] == "English"
