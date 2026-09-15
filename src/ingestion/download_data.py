"""
Downloads the IU X-ray (Indiana University Chest X-ray / Open-I) dataset from Kaggle.

Setup (one-time):
1. Create a free Kaggle account at kaggle.com
2. Go to Settings -> API -> "Create New Token" -> downloads kaggle.json
3. Place kaggle.json at ~/.kaggle/kaggle.json (Linux/Mac) or
   C:\\Users\\<you>\\.kaggle\\kaggle.json (Windows)
4. pip install kaggle
5. Run: python src/ingestion/download_data.py
"""

import os
import zipfile
from pathlib import Path

DATASET = "raddar/chest-xrays-indiana-university"
DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"


def download_dataset():
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except ImportError:
        raise SystemExit(
            "Kaggle package not installed. Run: pip install kaggle"
        )

    api = KaggleApi()
    api.authenticate()  # reads ~/.kaggle/kaggle.json

    print(f"Downloading {DATASET} into {DATA_DIR} ...")
    api.dataset_download_files(DATASET, path=str(DATA_DIR), unzip=False)

    # unzip whatever got downloaded
    for zip_path in DATA_DIR.glob("*.zip"):
        print(f"Extracting {zip_path.name} ...")
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(DATA_DIR)
        zip_path.unlink()  # remove zip after extracting

    print("Done. Contents of data/raw:")
    for item in sorted(DATA_DIR.iterdir()):
        print(" -", item.name)


if __name__ == "__main__":
    download_dataset()
