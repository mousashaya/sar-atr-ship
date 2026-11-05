#!/usr/bin/env python3
"""Upload SAR ship Gulf dataset to HuggingFace."""

import shutil
import sys
from pathlib import Path
from huggingface_hub import HfApi

REPO_ID = "mshaya/sar-ship-gulf"
CHIP_DIR = Path("/Volumes/Machine Learning/sar_ship_chips_gulf")
LABELS_FILE = Path("/Users/mousashaya/Projects/defense/data/resolved_labels.json")
CARD_FILE = Path("/Users/mousashaya/Projects/defense/hf_dataset/README.md")

api = HfApi()

# Create repo (no-op if it already exists)
print(f"Creating dataset repo {REPO_ID} ...")
try:
    api.create_repo(repo_id=REPO_ID, repo_type="dataset", exist_ok=True)
    print("  repo ready")
except Exception as e:
    print(f"  create_repo error: {e}", file=sys.stderr)
    sys.exit(1)

# Upload dataset card
print("Uploading README.md (dataset card) ...")
api.upload_file(
    path_or_fileobj=str(CARD_FILE),
    path_in_repo="README.md",
    repo_id=REPO_ID,
    repo_type="dataset",
)
print("  done")

# Upload resolved_labels.json
print("Uploading resolved_labels.json ...")
api.upload_file(
    path_or_fileobj=str(LABELS_FILE),
    path_in_repo="resolved_labels.json",
    repo_id=REPO_ID,
    repo_type="dataset",
)
print("  done")

# Upload all NPZ chips (folder upload — HF handles chunking)
print(f"Uploading chips from {CHIP_DIR} ...")
print("  This will take a while (12 GB). Safe to interrupt — re-run to resume.")
api.upload_folder(
    folder_path=str(CHIP_DIR),
    repo_id=REPO_ID,
    repo_type="dataset",
    path_in_repo="chips",       # nest chips under chips/ subfolder
    ignore_patterns=["*.log"],
)
print("Upload complete!")
print(f"Dataset URL: https://huggingface.co/datasets/{REPO_ID}")
