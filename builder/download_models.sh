#!/bin/bash
set -e  # Exit immediately if a command exits with a non-zero status

# Define cache and model directories
CACHE_DIR="/cache/models"
MODELS_DIR="/models"

# Ensure necessary directories exist
mkdir -p /root/.cache/torch/hub/checkpoints

# Download function with caching
download() {
  local file_url="$1"
  local destination_path="$2"
  local cache_path="${CACHE_DIR}/${destination_path##*/}"

  mkdir -p "$(dirname "$cache_path")"
  mkdir -p "$(dirname "$destination_path")"

  if [ ! -e "$cache_path" ]; then
    echo "Downloading $file_url to cache..."
    wget -O "$cache_path" "$file_url"
  else
    echo "Using cached version of ${cache_path##*/}"
  fi

  cp "$cache_path" "$destination_path"
}

# ===================================
# Python block: Hugging Face downloads using secret
# ===================================
python3 -c "
import os
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    print('WARNING: python-dotenv not installed, skipping .env loading')

from huggingface_hub import snapshot_download

# Try to read HF token from BuildKit secret file.
hf_token = None
try:
    with open('/run/secrets/hf_token', 'r') as f:
        hf_token = f.read().strip()
except Exception as e:
    print('No secret file found, falling back to environment variable:', e)
    hf_token = os.environ.get('HF_TOKEN')

# Download Faster Whisper large-v3 model
snapshot_download(repo_id='Systran/faster-whisper-large-v3', local_dir='/models/faster-whisper-large-v3')
print('Faster Whisper model downloaded.')

# Download SpeechBrain speaker recognition model
snapshot_download(repo_id='speechbrain/spkrec-ecapa-voxceleb')

# Optionally download PyAnnote models if HF_TOKEN is set
if hf_token:
    snapshot_download(repo_id='pyannote/embedding', use_auth_token=hf_token)
    snapshot_download(repo_id='pyannote/speaker-diarization-2.1', use_auth_token=hf_token)
else:
    print('WARNING: HF_TOKEN not set, skipping pyannote models download')
"
echo "All models downloaded successfully."