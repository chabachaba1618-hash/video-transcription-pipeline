"""Main transcription and diarization pipeline execution using Qwen3-ASR 1.7B and PyAnnote."""

import argparse
import os
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from tqdm import tqdm

# Suppress harmless third-party runtime warnings
warnings.filterwarnings("ignore", category=UserWarning, module=r"pyannote.*")
warnings.filterwarnings("ignore", category=SyntaxWarning, module=r"nagisa.*")
warnings.filterwarnings("ignore", message=r".*torchcodec.*")
warnings.filterwarnings("ignore", message=r".*upgrade_checkpoint.*")

from .audio import extract_audio_ffmpeg
from .export import save_transcription_outputs


def is_cjk(char: str) -> bool:
    """Check if character is within Chinese/Japanese/Korean unicode ranges."""
    code = ord(char)
    return (
        0x4E00 <= code <= 0x9FFF
        or 0x3400 <= code <= 0x4DBF
        or 0x3040 <= code <= 0x309F
        or 0x30A0 <= code <= 0x30FF
        or 0xAC00 <= code <= 0xD7AF
    )


def smart_join_tokens(tokens: List[str]) -> str:
    """Join tokens intelligently: insert spaces between non-CJK words, no spaces between CJK chars."""
    if not tokens:
        return ""
    result = []
    for i, token in enumerate(tokens):
        if i == 0:
            result.append(token)
            continue
        prev = tokens[i - 1]
        prev_is_cjk = prev and is_cjk(prev[-1])
        curr_is_cjk = token and is_cjk(token[0])
        # Only omit space when transitioning between CJK characters
        if prev_is_cjk and curr_is_cjk:
            result.append(token)
        else:
            result.append(" " + token)
    return "".join(result).strip()


def assign_speakers_to_words(
    words: List[Dict[str, Any]],
    speaker_turns: List[Tuple[float, float, str]],
) -> List[Dict[str, Any]]:
    """Assign speaker labels to words based on maximum temporal overlap with diarization intervals."""
    if not speaker_turns:
        return words

    for w in words:
        w_start = float(w["start"])
        w_end = float(w["end"])
        best_overlap = 0.0
        best_speaker = None
        min_dist = float("inf")
        closest_speaker = None

        w_mid = (w_start + w_end) / 2.0

        for t_start, t_end, speaker in speaker_turns:
            overlap = max(0.0, min(w_end, t_end) - max(w_start, t_start))
            if overlap > best_overlap:
                best_overlap = overlap
                best_speaker = speaker

            dist = 0.0 if (t_start <= w_mid <= t_end) else min(abs(w_mid - t_start), abs(w_mid - t_end))
            if dist < min_dist:
                min_dist = dist
                closest_speaker = speaker

        w["speaker"] = best_speaker if best_speaker is not None else closest_speaker

    return words


def group_words_into_segments(
    words: List[Dict[str, Any]],
    max_duration: float = 7.0,
    pause_threshold: float = 1.0,
) -> List[Dict[str, Any]]:
    """Group aligned words into dialogue segments breaking on speaker change, pauses, or sentence punctuation."""
    if not words:
        return []

    segments: List[Dict[str, Any]] = []
    current_words: List[Dict[str, Any]] = []
    current_speaker = words[0].get("speaker")

    for w in words:
        speaker = w.get("speaker")
        speaker_changed = (speaker != current_speaker) if speaker is not None else False
        pause = (float(w["start"]) - float(current_words[-1]["end"])) if current_words else 0.0
        dur = (float(w["end"]) - float(current_words[0]["start"])) if current_words else 0.0
        ends_sentence = (
            current_words and current_words[-1]["text"].rstrip()[-1:] in ".?!。？！"
        )

        if current_words and (speaker_changed or pause > pause_threshold or dur > max_duration or ends_sentence):
            token_texts = [x["text"] for x in current_words]
            seg_text = smart_join_tokens(token_texts)
            segments.append({
                "start": float(current_words[0]["start"]),
                "end": float(current_words[-1]["end"]),
                "text": seg_text,
                "speaker": current_speaker,
                "words": current_words,
            })
            current_words = []
            current_speaker = speaker

        current_words.append(w)

    if current_words:
        token_texts = [x["text"] for x in current_words]
        seg_text = smart_join_tokens(token_texts)
        segments.append({
            "start": float(current_words[0]["start"]),
            "end": float(current_words[-1]["end"]),
            "text": seg_text,
            "speaker": current_speaker,
            "words": current_words,
        })

    return segments


def run_pipeline(
    media_file: str,
    output_dir: Optional[str] = None,
    asr_model_name: str = "Qwen/Qwen3-ASR-1.7B",
    aligner_model_name: str = "Qwen/Qwen3-ForcedAligner-0.6B",
    language: Optional[str] = None,
    device: Optional[str] = None,
    do_align: bool = True,
    do_diarize: bool = False,
    hf_token: Optional[str] = None,
    min_speakers: Optional[int] = None,
    max_speakers: Optional[int] = None,
    start_time: Optional[str] = None,
    duration: Optional[float] = None,
    cleanup_audio: bool = False,
):
    media_path = Path(media_file).expanduser().resolve()
    if not media_path.exists():
        raise FileNotFoundError(f"Input media file not found: {media_path}")

    # Set up output directory and base stem preserving internal dots
    out_dir = Path(output_dir).expanduser().resolve() if output_dir else media_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    base_name = media_path.stem
    if base_name.endswith(".extracted"):
        base_name = base_name[:-len(".extracted")]
    output_stem = out_dir / base_name

    # Determine hardware device and dtype
    import torch
    if device is None:
        if torch.cuda.is_available():
            chosen_device = "cuda"
            chosen_dtype = torch.bfloat16
        elif torch.backends.mps.is_available():
            chosen_device = "mps"
            chosen_dtype = torch.float16
        else:
            chosen_device = "cpu"
            chosen_dtype = torch.float32
    else:
        chosen_device = device
        if chosen_device.startswith("cuda"):
            chosen_dtype = torch.bfloat16
        elif chosen_device == "mps":
            chosen_dtype = torch.float16
        else:
            chosen_dtype = torch.float32

    print("======================================================================")
    print("           VIDEO-TO-TEXT TRANSCRIPTION & DIARIZATION PIPELINE          ")
    print("======================================================================")
    print(f"Input file:     {media_path}")
    print(f"Output dir:     {out_dir}")
    print(f"ASR Model:      {asr_model_name} (Device: {chosen_device.upper()}, Dtype: {chosen_dtype})")
    print(f"Forced Aligner: {aligner_model_name if do_align else 'Disabled'}")
    print(f"Diarization:    {do_diarize} (PyAnnote)")
    if duration:
        print(f"Segment Limit:  start={start_time or 0}s, duration={duration}s")
    print("======================================================================\n")

    # Stage 1: Audio Extraction (FFmpeg)
    extracted_wav = out_dir / f"{base_name}.extracted.wav"
    if media_path.resolve() == extracted_wav.resolve() and start_time is None and duration is None:
        print("[1/5] Input is already extracted audio. Using directly.")
    else:
        if media_path.resolve() == extracted_wav.resolve():
            temp_wav = out_dir / f"{base_name}.tmp.wav"
            extract_audio_ffmpeg(
                media_path=media_path,
                output_wav_path=temp_wav,
                start_time=start_time,
                duration=duration,
            )
            temp_wav.replace(extracted_wav)
        else:
            extract_audio_ffmpeg(
                media_path=media_path,
                output_wav_path=extracted_wav,
                start_time=start_time,
                duration=duration,
            )

    # Stage 2: Qwen3-ASR Model Loading & Transcription
    print(f"\n[2/5] Loading Qwen3-ASR ('{asr_model_name}')...")
    from qwen_asr import Qwen3ASRModel

    # Standardize language code/name for Qwen3 if provided (e.g. 'en' -> 'English')
    lang_map = {
        "en": "English", "zh": "Chinese", "es": "Spanish", "fr": "French",
        "de": "German", "ja": "Japanese", "ko": "Korean", "ru": "Russian",
        "it": "Italian", "pt": "Portuguese", "ar": "Arabic", "hi": "Hindi"
    }
    qwen_lang = lang_map.get(language.lower(), language) if language else None

    aligner_kwargs = dict(device_map=chosen_device, dtype=chosen_dtype) if do_align else None

    with tqdm(total=1, desc="Loading Qwen3-ASR Model", unit="model") as pbar:
        model = Qwen3ASRModel.from_pretrained(
            asr_model_name,
            dtype=chosen_dtype,
            device_map=chosen_device,
            forced_aligner=aligner_model_name if do_align else None,
            forced_aligner_kwargs=aligner_kwargs,
        )
        pbar.update(1)

    print(f"Transcribing audio with Qwen3-ASR (Forced Alignment: {do_align})...")
    transcriptions = model.transcribe(
        audio=str(extracted_wav),
        language=qwen_lang,
        return_time_stamps=do_align,
    )
    res = transcriptions[0]
    detected_lang = res.language or (language or "unknown")
    full_text = res.text or ""
    print(f"✓ Transcription complete. Detected language: '{detected_lang}'")

    # Stage 3: Word Alignment Extraction
    words: List[Dict[str, Any]] = []
    if do_align and res.time_stamps and hasattr(res.time_stamps, "items"):
        print(f"\n[3/5] Processing word-level timestamps ({len(res.time_stamps.items)} tokens)...")
        for item in res.time_stamps.items:
            words.append({
                "text": item.text,
                "start": float(item.start_time),
                "end": float(item.end_time),
                "speaker": None,
            })
        print(f"✓ Extracted {len(words)} aligned timestamped units.")
    else:
        print("\n[3/5] Word-level alignment skipped or not produced.")

    # Stage 4: Speaker Diarization (PyAnnote)
    speaker_labeled = False
    speaker_turns: List[Tuple[float, float, str]] = []
    if do_diarize:
        print(f"\n[4/5] Running Speaker Diarization on {chosen_device.upper()} (PyAnnote)...")
        import huggingface_hub
        token = hf_token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN") or huggingface_hub.get_token()

        if not token:
            print("⚠ Hugging Face token missing!")
            print("  Speaker Diarization requires an access token with access to 'pyannote/speaker-diarization-community-1'.")
            print("  Provide --hf-token or export HF_TOKEN=\"hf_...\"")
            print("  Continuing export without speaker diarization labels.")
        else:
            try:
                import torchaudio
                from pyannote.audio import Pipeline

                with tqdm(total=1, desc="Loading PyAnnote Diarization", unit="model") as pbar:
                    diarize_pipeline = Pipeline.from_pretrained(
                        "pyannote/speaker-diarization-community-1",
                        token=token,
                    )
                    diarize_pipeline.to(torch.device(chosen_device))
                    pbar.update(1)

                with tqdm(total=1, desc="Diarizing speaker turns", unit="diarize") as pbar:
                    waveform, sample_rate = torchaudio.load(str(extracted_wav))
                    audio_input = {"waveform": waveform, "sample_rate": sample_rate}

                    diarize_kwargs = {}
                    if min_speakers is not None:
                        diarize_kwargs["min_speakers"] = min_speakers
                    if max_speakers is not None:
                        diarize_kwargs["max_speakers"] = max_speakers

                    diarization_output = diarize_pipeline(audio_input, **diarize_kwargs)
                    annotation = getattr(diarization_output, "speaker_diarization", diarization_output)
                    for turn, _, speaker in annotation.itertracks(yield_label=True):
                        speaker_turns.append((turn.start, turn.end, speaker))
                    pbar.update(1)

                if words and speaker_turns:
                    assign_speakers_to_words(words, speaker_turns)
                    speaker_labeled = True
                    print(f"✓ Speaker diarization complete ({len(speaker_turns)} turns merged).")
                elif speaker_turns:
                    speaker_labeled = True
                    print(f"✓ Speaker diarization completed ({len(speaker_turns)} turns).")
            except Exception as e:
                print(f"⚠ Warning: Speaker diarization skipped: {e}")
                print("  Note: Ensure your token has access to https://huggingface.co/pyannote/speaker-diarization-community-1")
                print("  Continuing with standard transcription export.")
    else:
        print("\n[4/5] Speaker diarization skipped (use --diarize to enable).")

    # Build Structured Output Segments
    if words:
        segments = group_words_into_segments(words)
    else:
        # Fallback if no timestamps were produced
        segments = [{
            "start": 0.0,
            "end": 0.0,
            "text": full_text,
            "speaker": "SPEAKER_00" if speaker_labeled else None,
            "words": [],
        }]

    pipeline_result: Dict[str, Any] = {
        "text": full_text,
        "language": detected_lang,
        "segments": segments,
        "words": words,
        "speaker_turns": [
            {"start": t[0], "end": t[1], "speaker": t[2]} for t in speaker_turns
        ],
    }

    # Stage 5: Exporting outputs
    print("\n[5/5] Exporting formatted transcript files...")
    with tqdm(total=4, desc="Writing outputs", unit="file") as pbar:
        saved_files = save_transcription_outputs(
            result=pipeline_result,
            output_stem=output_stem,
            include_speakers=speaker_labeled,
        )
        pbar.update(4)

    # Optional Audio Cleanup
    if cleanup_audio and extracted_wav.exists():
        extracted_wav.unlink()
        print("✓ Intermediate extracted WAV cleaned up.")
    else:
        print(f"✓ Intermediate audio preserved at: {extracted_wav}")

    print("\n======================================================================")
    print("                    PIPELINE COMPLETED SUCCESSFULLY                   ")
    print("======================================================================")
    print(f"TXT Transcript:  {saved_files['txt']}")
    print(f"SRT Subtitles:   {saved_files['srt']}")
    print(f"VTT Subtitles:   {saved_files['vtt']}")
    print(f"JSON Structure:  {saved_files['json']}")
    print("======================================================================\n")
    return saved_files


def main():
    parser = argparse.ArgumentParser(
        description="Universal Video-to-Text Pipeline using FFmpeg and Qwen3-ASR 1.7B with PyAnnote Speaker Diarization."
    )
    parser.add_argument("media_file", type=str, help="Path to input video or audio file.")
    parser.add_argument(
        "--output-dir",
        "-o",
        type=str,
        default=None,
        help="Directory to save generated transcripts and audio (default: next to media file).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen3-ASR-1.7B",
        help="Qwen ASR model repository ID (default: Qwen/Qwen3-ASR-1.7B).",
    )
    parser.add_argument(
        "--aligner-model",
        type=str,
        default="Qwen/Qwen3-ForcedAligner-0.6B",
        help="Qwen Forced Aligner model (default: Qwen/Qwen3-ForcedAligner-0.6B).",
    )
    parser.add_argument(
        "--language",
        "-l",
        type=str,
        default=None,
        help="Language (e.g. English, Chinese, or code en, zh). Auto-detected if omitted.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to run on ('cpu', 'cuda', 'mps'). Auto-detected if omitted.",
    )
    parser.add_argument(
        "--no-align",
        dest="do_align",
        action="store_false",
        help="Disable word-level forced alignment step.",
    )
    parser.set_defaults(do_align=True)

    parser.add_argument(
        "--diarize",
        dest="do_diarize",
        action="store_true",
        help="Enable Speaker Diarization (PyAnnote).",
    )
    parser.add_argument(
        "--no-diarize",
        dest="do_diarize",
        action="store_false",
        help="Disable Speaker Diarization.",
    )
    parser.set_defaults(do_diarize=False)

    parser.add_argument(
        "--hf-token",
        type=str,
        default=None,
        help="Hugging Face access token for PyAnnote speaker diarization.",
    )
    parser.add_argument(
        "--min-speakers",
        type=int,
        default=None,
        help="Minimum number of speakers (optional).",
    )
    parser.add_argument(
        "--max-speakers",
        type=int,
        default=None,
        help="Maximum number of speakers (optional).",
    )
    parser.add_argument(
        "--start-time",
        type=str,
        default=None,
        help="Start offset for media processing (e.g. '0', '00:01:30').",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=None,
        help="Duration in seconds to process (useful for quick testing, e.g. 10).",
    )
    parser.add_argument(
        "--cleanup-audio",
        action="store_true",
        help="Delete intermediate 16kHz WAV file after completion.",
    )

    args = parser.parse_args()

    # Normalize user model flag (e.g. "1.7b" -> "Qwen/Qwen3-ASR-1.7B")
    model_name = args.model
    if model_name.lower() in ["1.7b", "1.7", "qwen3-asr-1.7b"]:
        model_name = "Qwen/Qwen3-ASR-1.7B"

    # If --hf-token is provided or HF_TOKEN is in environment, enable diarization by default unless explicitly --no-diarize
    if (args.hf_token or os.environ.get("HF_TOKEN")) and "--no-diarize" not in sys.argv:
        args.do_diarize = True

    run_pipeline(
        media_file=args.media_file,
        output_dir=args.output_dir,
        asr_model_name=model_name,
        aligner_model_name=args.aligner_model,
        language=args.language,
        device=args.device,
        do_align=args.do_align,
        do_diarize=args.do_diarize,
        hf_token=args.hf_token,
        min_speakers=args.min_speakers,
        max_speakers=args.max_speakers,
        start_time=args.start_time,
        duration=args.duration,
        cleanup_audio=args.cleanup_audio,
    )


if __name__ == "__main__":
    main()
