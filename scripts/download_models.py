#!/usr/bin/env python3
"""
Model Download Script for Argentic

Downloads model checkpoints to ./models directory.
Usage:
    python scripts/download_models.py --model gemma-3n-4b
    python scripts/download_models.py --model gemma-3n-2b
    python scripts/download_models.py --list
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

MODELS_DIR = project_root / "models"
MODELS_DIR.mkdir(exist_ok=True)

AVAILABLE_MODELS = {
    "gemma-3n-4b": {
        "name": "Gemma 3n E4B (4B parameters)",
        "source": "huggingface",
        "hf_model_id": "google/gemma-3n-E4B-it",
        "description": "Multimodal (vision + audio) 4B parameter model",
    },
    "gemma-3-12b": {
        "name": "Gemma 3 12B (text-only)",
        "source": "huggingface",
        "hf_model_id": "google/gemma-3-12b-it",
        "description": "Text-only 12B parameter model",
    },
}


def download_from_kaggle(model_id: str, target_dir: Path):
    """Download model from Kaggle using kagglehub."""
    try:
        import kagglehub
    except ImportError:
        print("❌ Error: kagglehub is not installed.")
        print("Install it with: pip install kagglehub")
        print("Then configure Kaggle API credentials:")
        print("  1. Get API key from https://www.kaggle.com/settings")
        print("  2. Place kaggle.json in ~/.kaggle/")
        return False

    model_info = AVAILABLE_MODELS[model_id]
    print(f"\n📥 Downloading {model_info['name']}...")
    print(f"   Source: Kaggle ({model_info['kaggle_handle']})")
    print(f"   Target: {target_dir}")
    print()

    try:
        # Download using kagglehub
        downloaded_path = kagglehub.model_download(model_info["kaggle_handle"])
        print(f"✅ Downloaded to: {downloaded_path}")

        # Create symlink in models directory
        symlink_path = target_dir / model_id
        if symlink_path.exists():
            symlink_path.unlink()
        symlink_path.symlink_to(downloaded_path)

        print(f"✅ Symlinked to: {symlink_path}")
        print()
        print("To use this model, update your config:")
        print(f'  gemma_checkpoint_path: "{symlink_path}"')
        return True

    except Exception as e:
        print(f"❌ Error downloading model: {e}")
        return False


def download_from_huggingface(model_id: str, target_dir: Path):
    """Download model from Hugging Face."""
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("❌ Error: huggingface_hub is not installed.")
        print("Install it with: pip install huggingface-hub")
        return False

    model_info = AVAILABLE_MODELS[model_id]
    hf_model_id = model_info["hf_model_id"]

    print(f"\n📥 Downloading {model_info['name']}...")
    print(f"   Source: Hugging Face ({hf_model_id})")
    print(f"   Target: {target_dir}")
    print()

    try:
        # Download using huggingface_hub
        downloaded_path = snapshot_download(
            repo_id=hf_model_id,
            cache_dir=MODELS_DIR / ".cache",
            local_dir=target_dir / model_id,
            local_dir_use_symlinks=False,
        )
        print(f"✅ Downloaded to: {downloaded_path}")
        print()
        print("To use this model, update your config:")
        print(f'  gemma_model_id: "{hf_model_id}"')
        print(f'  gemma_model_path: "{target_dir / model_id}"')
        return True

    except Exception as e:
        print(f"❌ Error downloading model: {e}")
        return False


def list_models():
    """List all available models."""
    print("\n" + "=" * 70)
    print("  Available Models for Download")
    print("=" * 70)
    print()

    for model_id, info in AVAILABLE_MODELS.items():
        status = "✓ Downloaded" if (MODELS_DIR / model_id).exists() else "○ Not downloaded"
        print(f"{status}  {model_id}")
        print(f"     Name: {info['name']}")
        print(f"     Description: {info['description']}")
        print(f"     Source: {info['source']}")
        print()

    print("Download with: python scripts/download_models.py --model <model_id>")
    print()


def check_environment():
    """Check if environment is properly configured."""
    print("\n🔍 Checking environment...")

    # Check Kaggle credentials
    kaggle_config = Path.home() / ".kaggle" / "kaggle.json"
    if kaggle_config.exists():
        print("✅ Kaggle credentials found")
    else:
        print("⚠️  Kaggle credentials not found")
        print("   Get your API key from: https://www.kaggle.com/settings")
        print(f"   Place kaggle.json in: {kaggle_config.parent}/")

    # Check kagglehub installation
    try:
        import kagglehub

        print(f"✅ kagglehub installed (version: {kagglehub.__version__})")
    except ImportError:
        print("❌ kagglehub not installed")
        print("   Install with: pip install kagglehub")

    print()


def main():
    parser = argparse.ArgumentParser(
        description="Download model checkpoints for Argentic",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # List available models
  python scripts/download_models.py --list
  
  # Download Gemma 3n 4B model
  python scripts/download_models.py --model gemma-3n-4b
  
  # Check environment
  python scripts/download_models.py --check
        """,
    )

    parser.add_argument(
        "--model",
        choices=list(AVAILABLE_MODELS.keys()),
        help="Model to download",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List all available models",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Check environment configuration",
    )

    args = parser.parse_args()

    if args.list:
        list_models()
        return 0

    if args.check:
        check_environment()
        return 0

    if not args.model:
        parser.print_help()
        return 1

    # Ensure models directory exists
    MODELS_DIR.mkdir(exist_ok=True)
    print(f"\n📁 Models directory: {MODELS_DIR}")

    # Download the model
    target_dir = MODELS_DIR / args.model
    target_dir.mkdir(exist_ok=True)

    model_info = AVAILABLE_MODELS[args.model]
    if model_info["source"] == "kaggle":
        success = download_from_kaggle(args.model, MODELS_DIR)
    elif model_info["source"] == "huggingface":
        success = download_from_huggingface(args.model, MODELS_DIR)
    else:
        print(f"❌ Unknown source: {model_info['source']}")
        return 1

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
