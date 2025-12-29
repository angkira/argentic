# Argentic Scripts

Utility scripts for Argentic framework management.

## Model Management

### download_models.py

Download and manage model checkpoints for local inference.

```bash
# List available models
python scripts/download_models.py --list

# Check environment setup
python scripts/download_models.py --check

# Download a specific model
python scripts/download_models.py --model gemma-3n-4b
```

#### Prerequisites

1. **Install kagglehub:**
   ```bash
   pip install kagglehub
   ```

2. **Set up Kaggle API credentials:**
   - Go to https://www.kaggle.com/settings
   - Click "Create New Token" in API section
   - Save `kaggle.json` to `~/.kaggle/kaggle.json`
   - Set permissions: `chmod 600 ~/.kaggle/kaggle.json`

3. **Accept model terms:**
   - Visit the model page on Kaggle
   - Accept the terms and conditions

#### Available Models

- **gemma-3n-4b**: Gemma 3n E4B (4B parameters) - Multimodal vision + audio + text
- **gemma-3n-2b**: Gemma 3n E2B (2B parameters) - Smaller multimodal model

#### Configuration

After downloading, models are stored in `./models/` directory and can be referenced in your config:

```yaml
llm:
  provider: gemma
  gemma_checkpoint_path: "./models/gemma-3n-4b"
  gemma_model_id: "gemma-3n-e4b-it"
```

Or via environment variable in `.env`:
```bash
GEMMA_CHECKPOINT_PATH="./models/gemma-3n-4b"
```

## Notes

- Models are NOT included in git (see `.gitignore`)
- Downloaded models are symlinked from Kaggle cache to `./models/`
- Model files are typically 4-10 GB each

