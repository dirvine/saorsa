# AI Models Directory

This directory contains the AI models used by Saorsa:

## Whisper Models
- `whisper-base.pt` - Base Whisper model for speech recognition
- `whisper-small.pt` - Small Whisper model (faster inference)
- `whisper-medium.pt` - Medium Whisper model (better accuracy)

## Pi-Zero Models
- `pi-zero-base/` - Physical Intelligence π0 foundation model

## Installation

Run the model downloader script:
```bash
python scripts/download_models.py
```

## Model Sizes

| Model | Size | Speed | Accuracy |
|-------|------|-------|----------|
| Whisper Base | ~140MB | Fast | Good |
| Whisper Small | ~480MB | Medium | Better |
| Whisper Medium | ~1.4GB | Slow | Best |
| Pi-Zero Base | ~2.5GB | Medium | High |

## Storage Requirements

- Minimum: 3GB (Base models only)
- Recommended: 5GB (All models)
- Maximum: 8GB (All models + cache)