"""Download a model from Hugging Face into a local directory.  python scripts/fetch_model.py REPO OUT_DIR"""
import sys
from huggingface_hub import snapshot_download
d = snapshot_download(sys.argv[1], local_dir=sys.argv[2], ignore_patterns=["*.md", "*.png", "*.jpg", "*.git*"])
print("into", d)
