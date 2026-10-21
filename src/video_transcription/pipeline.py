"""Main transcription and diarization pipeline execution."""

import os
import sys
import argparse
import warnings
from pathlib import Path
from typing import Optional
from tqdm import tqdm

# Suppress harmless third-party runtime warnings (e.g. torchcodec fallback)
warnings.filterwarnings("ignore", category=UserWarning, module=r"pyannote.*")
warnings.filterwarnings("ignore", message=r".*torchcodec.*")
warnings.filterwarnings("ignore", message=r".*upgrade_checkpoint.*")

from .audio import extract_audio_ffmpeg
from .export import save_transcription_outputs


def run_pipeline(
    media_file: str,
    output_dir: Optional[str] = None,
    model_name: str = "large-v3",
    language: Optional[str] = None,
    device: Optional[str] = None,
    compute_type: Optional[str] = None,
    batch_size: int = 8,
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

    # Set up output directory and base stem
    out_dir = Path(output_dir).expanduser().resolve() if output_dir else media_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    base_name = media_path.stem
    if base_name.endswith(".extracted"):
        base_name = base_name[:-len(".extracted")]
    output_stem = out_dir / base_name

    # Determine hardware device and compute type
    import torch
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if compute_type is None:
        # On CPU (including Apple Silicon ARM64), int8 is faster and memory-efficient
        compute_type = "float16" if device == "cuda" else "int8"

    print("======================================================================")
    print("           VIDEO-TO-TEXT TRANSCRIPTION & DIARIZATION PIPELINE          ")
    print("======================================================================")
    print(f"Input file:     {media_path}")
    print(f"Output dir:     {out_dir}")
    print(f"Model:          Whisper {model_name}")
    print(f"Device:         {device}")
    print(f"Compute Type:   {compute_type}")
    print(f"Batch Size:     {batch_size}")
    print(f"Word Alignment: {do_align}")
    print(f"Diarization:    {do_diarize}")
    if duration:
        print(f"Segment Limit:  start={start_time or 0}s, duration={duration}s")
    print("======================================================================\n")

    # Stage 1: Audio Extraction
    extracted_wav = output_stem.with_suffix(".extracted.wav")
    if media_path.resolve() == extracted_wav.resolve() and start_time is None and duration is None:
        print("[1/5] Input is already the extracted audio file. Using directly.")
    else:
        # If input and output are the same path but need clipping, extract via temp file
        if media_path.resolve() == extracted_wav.resolve():
            temp_wav = output_stem.with_suffix(".tmp.wav")
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

    import whisperx

    # Stage 2: WhisperX Transcription
    print(f"\n[2/5] Loading Whisper '{model_name}' and transcribing...")
    with tqdm(total=1, desc="Loading Whisper Model", unit="model") as pbar:
        model = whisperx.load_model(
            model_name,
            device=device,
            compute_type=compute_type,
            language=language,
        )
        pbar.update(1)

    print("Loading audio waveform into memory...")
    audio = whisperx.load_audio(str(extracted_wav))

    print("Transcribing with Voice Activity Detection (VAD)...")
    transcribe_result = model.transcribe(
        audio,
        batch_size=batch_size,
        language=language,
        print_progress=True,
    )
    detected_lang = transcribe_result.get("language", language or "en")
    print(f"✓ Transcription complete. Detected language: '{detected_lang}'")

    # Stage 3: Word Alignment
    if do_align:
        print(f"\n[3/5] Performing word-level alignment (Language: {detected_lang})...")
        try:
            with tqdm(total=1, desc="Loading Alignment Model", unit="model") as pbar:
                model_a, metadata = whisperx.load_align_model(
                    language_code=detected_lang,
                    device=device,
                )
                pbar.update(1)

            with tqdm(total=1, desc="Aligning transcript segments", unit="align") as pbar:
                transcribe_result = whisperx.align(
                    transcribe_result["segments"],
                    model_a,
                    metadata,
                    audio,
                    device,
                    return_char_alignments=False,
                )
                pbar.update(1)
            print("✓ Word-level alignment finished successfully.")
        except Exception as e:
            print(f"⚠ Warning: Alignment skipped due to error: {e}")
    else:
        print("\n[3/5] Word-level alignment skipped (--no-align).")

    # Stage 4: Speaker Diarization
    speaker_labeled = False
    if do_diarize:
        print("\n[4/5] Running Speaker Diarization (PyAnnote)...")
        import huggingface_hub
        token = hf_token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN") or huggingface_hub.get_token()
        if not token:
            print("⚠ Hugging Face token missing!")
            print("  Speaker Diarization requires an access token with access to 'pyannote/speaker-diarization-community-1'.")
            print("  Provide --hf-token or export HF_TOKEN=\"hf_...\"")
            print("  Continuing export without speaker diarization labels.")
        else:
            try:
                from whisperx.diarize import DiarizationPipeline
                with tqdm(total=1, desc="Loading PyAnnote Diarization", unit="model") as pbar:
                    diarize_model = DiarizationPipeline(
                        token=token,
                        device=device,
                    )
                    pbar.update(1)

                with tqdm(total=1, desc="Diarizing speaker turns", unit="diarize") as pbar:
                    diarize_kwargs = {}
                    if min_speakers is not None:
                        diarize_kwargs["min_speakers"] = min_speakers
                    if max_speakers is not None:
                        diarize_kwargs["max_speakers"] = max_speakers

                    diarize_segments = diarize_model(str(extracted_wav), **diarize_kwargs)
                    transcribe_result = whisperx.assign_word_speakers(diarize_segments, transcribe_result)
                    pbar.update(1)
                speaker_labeled = True
                print("✓ Speaker diarization complete and merged with transcript.")
            except Exception as e:
                print(f"⚠ Warning: Speaker diarization skipped: {e}")
                print("  Note: Make sure your token has access to https://huggingface.co/pyannote/speaker-diarization-community-1")
                print("  Continuing with standard transcription export.")
    else:
        print("\n[4/5] Speaker diarization skipped (use --diarize to enable).")

    # Stage 5: Exporting outputs
    print("\n[5/5] Exporting formatted transcript files...")
    with tqdm(total=4, desc="Writing outputs", unit="file") as pbar:
        saved_files = save_transcription_outputs(
            result=transcribe_result,
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


def main():
    parser = argparse.ArgumentParser(
        description="Universal Video-to-Text Pipeline using FFmpeg and WhisperX large-v3 with Speaker Diarization."
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
        default="large-v3",
        help="Whisper model size (default: large-v3). Options: tiny, base, small, medium, large-v2, large-v3",
    )
    parser.add_argument(
        "--language",
        "-l",
        type=str,
        default=None,
        help="Language code (e.g. en, fr, de, es). If omitted, language is auto-detected.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to run on ('cpu' or 'cuda'). Auto-detected if omitted.",
    )
    parser.add_argument(
        "--compute-type",
        type=str,
        default=None,
        help="Compute type ('int8', 'float32', 'float16'). Defaults to 'int8' on CPU.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size for transcription (default: 8).",
    )
    parser.add_argument(
        "--no-align",
        dest="do_align",
        action="store_false",
        help="Disable word-level alignment step.",
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
        help="Duration in seconds to process (useful for quick testing, e.g. 30).",
    )
    parser.add_argument(
        "--cleanup-audio",
        action="store_true",
        help="Delete intermediate 16kHz WAV file after completion.",
    )

    args = parser.parse_args()

    # If --hf-token is provided or HF_TOKEN is in environment, enable diarization by default unless explicitly --no-diarize
    if (args.hf_token or os.environ.get("HF_TOKEN")) and "--no-diarize" not in sys.argv:
        args.do_diarize = True

    run_pipeline(
        media_file=args.media_file,
        output_dir=args.output_dir,
        model_name=args.model,
        language=args.language,
        device=args.device,
        compute_type=args.compute_type,
        batch_size=args.batch_size,
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
