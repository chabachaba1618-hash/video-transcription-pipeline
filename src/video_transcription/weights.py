"""Utilities to download and pre-cache Qwen3-ASR and PyAnnote weights in standard cache directories."""

import argparse
import os
import warnings
from pathlib import Path
from tqdm import tqdm

# Suppress harmless third-party runtime warnings
warnings.filterwarnings("ignore", category=UserWarning, module=r"pyannote.*")
warnings.filterwarnings("ignore", message=r".*torchcodec.*")
warnings.filterwarnings("ignore", message=r".*upgrade_checkpoint.*")


def get_default_hf_cache_dir() -> Path:
    hf_home = os.environ.get("HF_HOME")
    if hf_home:
        return Path(hf_home)
    return Path.home() / ".cache" / "huggingface" / "hub"


def download_qwen_weights(
    asr_model: str = "Qwen/Qwen3-ASR-1.7B",
    aligner_model: str = "Qwen/Qwen3-ForcedAligner-0.6B",
):
    """Download Qwen3-ASR and Qwen3-ForcedAligner models to standard Hugging Face cache."""
    from huggingface_hub import snapshot_download

    print(f"\n[1/3] Downloading Qwen3 ASR model '{asr_model}' to Hugging Face cache...")
    with tqdm(total=1, desc=f"Caching {asr_model}", unit="model") as pbar:
        snapshot_download(repo_id=asr_model)
        pbar.update(1)
    print(f"✓ Successfully cached ASR model '{asr_model}'.")

    if aligner_model:
        print(f"\n[2/3] Downloading Qwen3 Forced Aligner '{aligner_model}' to cache...")
        with tqdm(total=1, desc=f"Caching {aligner_model}", unit="model") as pbar:
            snapshot_download(repo_id=aligner_model)
            pbar.update(1)
        print(f"✓ Successfully cached Forced Aligner '{aligner_model}'.")


def download_diarization_weights(
    hf_token: str | None = None,
    model_name: str = "pyannote/speaker-diarization-community-1",
):
    """Download PyAnnote speaker diarization model to Hugging Face cache."""
    print("\n[3/3] Checking Speaker Diarization model weights...")
    import huggingface_hub
    from huggingface_hub import snapshot_download

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
        with tqdm(total=1, desc=f"Caching {model_name}", unit="model") as pbar:
            snapshot_download(repo_id=model_name, token=token)
            pbar.update(1)
        print(f"✓ Successfully cached PyAnnote Speaker Diarization model '{model_name}'.")
        return model_name
    except Exception as e:
        print(f"⚠ Note on Diarization weights: {e}")
        print("  To enable diarization, visit:")
        print("    https://huggingface.co/pyannote/speaker-diarization-community-1")
        print("  to accept the community conditions with your Hugging Face account.")
        return None


def main():
    parser = argparse.ArgumentParser(
        description="Pre-download Qwen3-ASR (1.7B) and PyAnnote model weights into standard cache directories."
    )
    parser.add_argument(
        "--asr-model",
        type=str,
        default="Qwen/Qwen3-ASR-1.7B",
        help="Qwen ASR model repository ID (default: Qwen/Qwen3-ASR-1.7B).",
    )
    parser.add_argument(
        "--aligner-model",
        type=str,
        default="Qwen/Qwen3-ForcedAligner-0.6B",
        help="Qwen Forced Aligner model repository ID (default: Qwen/Qwen3-ForcedAligner-0.6B).",
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
    print(f"ASR Model:              {args.asr_model}")
    print(f"Aligner Model:          {args.aligner_model}")
    print("==================================================")

    download_qwen_weights(asr_model=args.asr_model, aligner_model=args.aligner_model)
    download_diarization_weights(hf_token=args.hf_token)

    print("\n==================================================")
    print("✓ All requested weights are downloaded and ready!")
    print(f"Standard cache directory location: {cache_dir}")
    print("==================================================")


if __name__ == "__main__":
    main()
