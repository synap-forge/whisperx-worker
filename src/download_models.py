import os
import logging
from huggingface_hub import snapshot_download

logger = logging.getLogger(__name__)

VOLUME_MODELS_DIR = "/runpod-volume/models"
FASTER_WHISPER_DIR = os.path.join(VOLUME_MODELS_DIR, "faster-whisper-large-v3")


def ensure_models_downloaded():
    hf_token = os.environ.get("HF_TOKEN", "").strip()

    os.makedirs(VOLUME_MODELS_DIR, exist_ok=True)

    # Faster Whisper large-v3
    if not os.path.exists(os.path.join(FASTER_WHISPER_DIR, "model.bin")):
        logger.info("Downloading faster-whisper-large-v3 to network volume...")
        snapshot_download(
            repo_id="Systran/faster-whisper-large-v3",
            local_dir=FASTER_WHISPER_DIR,
            token=hf_token or None,
        )
        logger.info("faster-whisper-large-v3 downloaded.")
    else:
        logger.info("faster-whisper-large-v3 already on volume.")

    # SpeechBrain ECAPA (cached via HF_HOME=/runpod-volume/hf_cache)
    try:
        from speechbrain.pretrained import EncoderClassifier
        if not _hf_cache_exists("speechbrain/spkrec-ecapa-voxceleb"):
            logger.info("Downloading speechbrain/spkrec-ecapa-voxceleb...")
            snapshot_download(repo_id="speechbrain/spkrec-ecapa-voxceleb", token=hf_token or None)
            logger.info("speechbrain/spkrec-ecapa-voxceleb downloaded.")
    except Exception as e:
        logger.warning(f"SpeechBrain model download skipped: {e}")

    # PyAnnote models (gated — require HF_TOKEN with accepted terms)
    if hf_token:
        for repo_id in ["pyannote/embedding", "pyannote/speaker-diarization-2.1"]:
            if not _hf_cache_exists(repo_id):
                logger.info(f"Downloading {repo_id}...")
                try:
                    snapshot_download(repo_id=repo_id, token=hf_token)
                    logger.info(f"{repo_id} downloaded.")
                except Exception as e:
                    logger.warning(f"Failed to download {repo_id}: {e}")
    else:
        logger.warning("HF_TOKEN not set — skipping pyannote model downloads.")


def _hf_cache_exists(repo_id: str) -> bool:
    hf_home = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
    hub_dir = os.path.join(hf_home, "hub")
    repo_folder = "models--" + repo_id.replace("/", "--")
    return os.path.isdir(os.path.join(hub_dir, repo_folder))
