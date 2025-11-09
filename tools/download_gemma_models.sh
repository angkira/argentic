#!/bin/bash
# Gemma Model Downloader
# Downloads Gemma models to ~/.argentic/models/

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "======================================================================"
echo "  Gemma Model Downloader"
echo "======================================================================"
echo ""

# Setup directories
MODELS_DIR="${HOME}/.argentic/models"
mkdir -p "$MODELS_DIR"

echo "📁 Models directory: $MODELS_DIR"
echo ""

# Source environment if exists
if [ -f "${HOME}/.argentic_models.env" ]; then
    source "${HOME}/.argentic_models.env"
    echo "✅ Loaded model configuration"
else
    echo "⚠️  Model configuration not found. Creating..."
    cat > "${HOME}/.argentic_models.env" << 'EOF'
# Argentic Model Configuration
export ARGENTIC_MODELS_DIR="${HOME}/.argentic/models"
export GEMMA3_4B_IT_PATH="${ARGENTIC_MODELS_DIR}/gemma-3n-4b-it"
export GEMMA3_2B_IT_PATH="${ARGENTIC_MODELS_DIR}/gemma-3n-2b-it"
export GOOGLE_GEMINI_API_KEY=""
EOF
    echo "✅ Created ${HOME}/.argentic_models.env"
    echo ""
    echo "Add to your shell RC file (~/.bashrc or ~/.zshrc):"
    echo "  source ~/.argentic_models.env"
fi

echo ""
echo "======================================================================"
echo "  Download Options"
echo "======================================================================"
echo ""
echo "Gemma 3n models are available from:"
echo ""
echo "1. 📦 Direct Download (GitHub Releases)"
echo "   https://github.com/google-deepmind/gemma/releases"
echo ""
echo "2. 🤗 Hugging Face (when available)"
echo "   https://huggingface.co/google"
echo ""
echo "3. 🔑 Kaggle (requires API key)"
echo "   https://www.kaggle.com/models/google/gemma"
echo ""
echo "======================================================================"
echo ""

read -p "Choose download method (1/2/3) or 'skip': " choice

case $choice in
    1)
        echo ""
        echo "📥 Opening GitHub releases page..."
        echo "Download the model checkpoint and extract to:"
        echo "  ${MODELS_DIR}/gemma-3n-4b-it/"
        echo ""
        if command -v xdg-open &> /dev/null; then
            xdg-open "https://github.com/google-deepmind/gemma/releases"
        elif command -v open &> /dev/null; then
            open "https://github.com/google-deepmind/gemma/releases"
        else
            echo "Visit: https://github.com/google-deepmind/gemma/releases"
        fi
        ;;
    2)
        echo ""
        echo "🤗 Hugging Face download (requires huggingface_hub)..."
        python3 << 'PYTHON'
try:
    from huggingface_hub import snapshot_download
    import os
    
    model_id = "google/gemma-2-9b-it"  # Using Gemma 2 as Gemma 3n not yet available
    print(f"Downloading {model_id}...")
    path = snapshot_download(
        repo_id=model_id,
        cache_dir=os.path.expanduser("~/.argentic/models/huggingface_cache")
    )
    print(f"✅ Downloaded to: {path}")
except ImportError:
    print("❌ huggingface_hub not installed. Install with:")
    print("   pip install huggingface-hub")
except Exception as e:
    print(f"❌ Error: {e}")
PYTHON
        ;;
    3)
        echo ""
        echo "🔑 Kaggle download (requires kagglehub)..."
        python3 << 'PYTHON'
try:
    import kagglehub
    
    print("Note: Kaggle download requires authentication.")
    print("Set KAGGLE_USERNAME and KAGGLE_KEY environment variables")
    print("or create ~/.kaggle/kaggle.json")
    print("")
    
    # This will prompt for kaggle credentials if not configured
    path = kagglehub.model_download("google/gemma/flax/2b-it")
    print(f"✅ Downloaded to: {path}")
except ImportError:
    print("❌ kagglehub not installed. Install with:")
    print("   pip install kagglehub")
except Exception as e:
    print(f"❌ Error: {e}")
PYTHON
        ;;
    skip|*)
        echo ""
        echo "⏭️  Skipping model download"
        echo ""
        echo "💡 TIP: Use Google Gemini API instead (no download needed):"
        echo "   1. Get API key: https://makersuite.google.com/app/apikey"
        echo "   2. Set environment: export GOOGLE_GEMINI_API_KEY='your-key'"
        echo "   3. Run: python examples/visual_agent_gemini_test.py"
        ;;
esac

echo ""
echo "======================================================================"
echo "  Setup Complete"
echo "======================================================================"
echo ""
echo "✅ Model directory: $MODELS_DIR"
echo "✅ Environment file: ${HOME}/.argentic_models.env"
echo ""
echo "To use models, add to your shell RC file:"
echo "  source ~/.argentic_models.env"
echo ""
echo "For quick testing without model download:"
echo "  python examples/visual_agent_gemini_test.py"
echo ""

