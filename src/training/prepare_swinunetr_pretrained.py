
import sys
import urllib.request
from pathlib import Path

SSL_CKPT_URL = (
    "https://github.com/Project-MONAI/MONAI-extra-test-data"
    "/releases/download/0.8.1/model_swinvit.pt"
)

OUT_DIR = Path(__file__).resolve().parents[2] / "pretrained_weights"
OUT_DIR.mkdir(exist_ok=True)
OUT_PATH = OUT_DIR / "swinunetr_v1_ssl.pt"

if OUT_PATH.exists():
    print(f"Checkpoint already exists: {OUT_PATH}")
    print(f"\nReady to submit:\n  sbatch swinunetr_v1_pretrained_job.job")
    sys.exit(0)

print("Downloading SSL pretrained SwinUNETR encoder weights from MONAI...")
print(f"URL: {SSL_CKPT_URL}\n")

def _progress(block_num, block_size, total_size):
    downloaded = block_num * block_size
    pct = min(100.0, 100.0 * downloaded / total_size) if total_size > 0 else 0.0
    print(f"\r  {pct:5.1f}%  ({downloaded / 1e6:.1f} MB)", end="", flush=True)

try:
    urllib.request.urlretrieve(SSL_CKPT_URL, OUT_PATH, reporthook=_progress)
    print(f"\n\nSaved to: {OUT_PATH}")
except Exception as e:
    OUT_PATH.unlink(missing_ok=True)
    print(f"\nERROR: Download failed: {e}")
    print("Check your internet connection or download manually:")
    print(f"  wget -O {OUT_PATH} '{SSL_CKPT_URL}'")
    sys.exit(1)

import torch
try:
    ckpt = torch.load(str(OUT_PATH), map_location="cpu")
    if isinstance(ckpt, dict):
        keys = list(ckpt.keys())[:5]
    else:
        keys = ["<non-dict object>"]
    print(f"Checkpoint looks valid. Top-level keys: {keys}")
except Exception as e:
    print(f"WARNING: Could not verify checkpoint: {e}")

print(f"\nReady to submit:")
print(f"  sbatch swinunetr_v1_pretrained_job.job")
