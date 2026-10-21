"""Utilities to download and pre-cache model weights in standard system directories."""

import os
import sys
import argparse
import warnings
from pathlib import Path
from tqdm import tqdm

# Filter out non-fatal torchcodec / lightning warnings from terminal output
warnings.filterwarnings("ignore", category=UserWarning, module=r"pyannote.*")
warnings.filterwarnings("ignore", message=r".*torchcodec.*")
warnings.filterwarnings("ignore", message=r".*upgrade_checkpoint.*")


def get_default_hf_cache_dir() -> Path:
    hf_home = os.environ.get("HF_HOME")
    if hf_home:
        return Path(hf_home)
    return Path.home() / ".cache" / "huggingface" / "hub"


def download_whisper_weights(model_name: str, device: str = "cpu", compute_type: str = "int8"):
    print(f"\n[1/3] Downloading Whisper '{model_name}' weights to standard Hugging Face cache...")
    import whisperx

    with tqdm(total=1, desc=f"Loading/Caching Whisper ({model_name})", unit="model") as pbar:
        model = whisperx.load_model(
            model_name,
            device=device,
            compute_type=compute_type,
            language="en",
        )
        pbar.update(1)
    print(f"✓ Successfully cached Whisper '{model_name}' model.")
    return model


def download_alignment_weights(language_code: str = "en", device: str = "cpu"):
    print(f"\n[2/3] Downloading Wav2Vec2 alignment model ({language_code}) to standard cache...")
    import whisperx

    with tqdm(total=1, desc=f"Loading/Caching Alignment ({language_code})", unit="model") as pbar:
        model_a, metadata = whisperx.load_align_model(
            language_code=language_code,
            device=device,
        )
        pbar.update(1)
    print(f"✓ Successfully cached alignment model for '{language_code}'.")
    return model_a, metadata


def download_diarization_weights(
    hf_token: str | None = None,
    device: str = "cpu",
    model_name: str = "pyannote/speaker-diarization-community-1",
):
    print("\n[3/3] Checking Speaker Diarization model weights...")
    import huggingface_hub
    token = hf_token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN") or huggingface_hub.get_token()

    if not token:
        print("ℹ No Hugging Face token provided or stored.")
        print("  To pre-download PyAnnote speaker diarization models, run:")
        print("    download-weights --hf-token <YOUR_HF_TOKEN>")
        print("  or export HF_TOKEN=\"<YOUR_HF_TOKEN>\"")
        print("  Ensure you accepted user agreements on:")
        print("    https://huggingface.co/pyannote/speaker-diarization-community-1")
        print("    https://huggingface.co/pyannote/segmentation-3.0")
        return None

    try:
        from whisperx.diarize import DiarizationPipeline
        with tqdm(total=1, desc="Loading/Caching PyAnnote Diarization", unit="model") as pbar:
            diarize_model = DiarizationPipeline(
                model_name=model_name,
                token=token,
                device=device,
            )
            pbar.update(1)
        print("✓ Successfully cached PyAnnote Speaker Diarization pipeline.")
        return diarize_model
    except Exception as e:
        print(f"⚠ Note on Diarization weights: {e}")
        print("  To enable diarization, visit:")
        print("    https://huggingface.co/pyannote/speaker-diarization-community-1")
        print("  to accept the community conditions with your Hugging Face account.")
        return None


def main():
    parser = argparse.ArgumentParser(
        description="Pre-download WhisperX and PyAnnote model weights into standard cache directories."
    )
    parser.add_argument(
        "--model",
        type=str,
        default="large-v3",
        help="Whisper model size to download (default: large-v3). Options: tiny, base, small, medium, large-v2, large-v3",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device to load model on (cpu or cuda, default: cpu).",
    )
    parser.add_argument(
        "--compute-type",
        type=str,
        default="int8",
        help="Model compute type (int8 or float32, default: int8).",
    )
    parser.add_argument(
        "--language",
        type=str,
        default="en",
        help="Language code for alignment model (default: en).",
    )
    parser.add_argument(
        "--hf-token",
        type=str,
        default=None,
        help="Hugging Face access token for pyannote diarization models.",
    )
    args = parser.parse_args()

    cache_dir = get_default_hf_cache_dir()
    print("==================================================")
    print(f"Target Cache Directory: {cache_dir}")
    print(f"Model Selection:        {args.model}")
    print(f"Device:                 {args.device}")
    print(f"Compute Type:           {args.compute_type}")
    print("==================================================")

    download_whisper_weights(args.model, device=args.device, compute_type=args.compute_type)
    download_alignment_weights(args.language, device=args.device)
    download_diarization_weights(args.hf_token, device=args.device)

    print("\n==================================================")
    print("✓ All requested weights are downloaded and ready!")
    print(f"Standard cache directory location: {cache_dir}")
    print("==================================================")


if __name__ == "__main__":
    main()
