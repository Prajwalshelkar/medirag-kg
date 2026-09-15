"""
Parses the IU X-ray (Indiana University) dataset as packaged on Kaggle by
raddar: https://www.kaggle.com/datasets/raddar/chest-xrays-indiana-university

Actual folder layout after download_data.py runs:
    data/raw/images/                     (PNG images, possibly nested one level
                                           deeper in an images_normalized/ folder)
    data/raw/indiana_projections.csv     (uid, filename, projection [Frontal/Lateral])
    data/raw/indiana_reports.csv         (uid, indication, findings, impression, ...)

Each patient (uid) can have multiple images (frontal + lateral) and exactly
one report. This script merges the two CSVs on `uid` and resolves each
image filename to an actual file on disk.
"""

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"
REPORTS_CSV = RAW_DIR / "indiana_reports.csv"
PROJECTIONS_CSV = RAW_DIR / "indiana_projections.csv"


def _find_images_dir() -> Path:
    """Images may be directly in data/raw/images/ or nested one level deeper."""
    candidates = [
        RAW_DIR / "images" / "images_normalized",
        RAW_DIR / "images",
    ]
    for c in candidates:
        if c.exists() and any(c.iterdir()):
            return c
    raise FileNotFoundError(
        f"Could not find image files under {RAW_DIR / 'images'}. "
        "Check the folder structure with: dir data\\raw\\images (Windows) "
        "or ls data/raw/images (Mac/Linux)."
    )


@dataclass
class Report:
    uid: str
    indication: str = ""
    findings: str = ""
    impression: str = ""
    image_paths: list = field(default_factory=list)

    def full_text(self) -> str:
        parts = [self.indication, self.findings, self.impression]
        return " ".join(p for p in parts if isinstance(p, str) and p.strip()).strip()


def load_all_reports(limit: int = None) -> list:
    """Merge reports + projections CSVs and resolve image paths."""
    if not REPORTS_CSV.exists() or not PROJECTIONS_CSV.exists():
        raise FileNotFoundError(
            f"Expected CSVs not found in {RAW_DIR}. Did you run download_data.py first?"
        )

    reports_df = pd.read_csv(REPORTS_CSV)
    projections_df = pd.read_csv(PROJECTIONS_CSV)

    images_dir = _find_images_dir()

    # normalize expected column names defensively (some releases vary slightly)
    reports_df.columns = [c.strip().lower() for c in reports_df.columns]
    projections_df.columns = [c.strip().lower() for c in projections_df.columns]

    reports = []
    skipped = 0

    uids = reports_df["uid"].unique()
    if limit:
        uids = uids[:limit]

    for uid in uids:
        row = reports_df[reports_df["uid"] == uid].iloc[0]

        report = Report(
            uid=str(uid),
            indication=str(row.get("indication", "") or ""),
            findings=str(row.get("findings", "") or ""),
            impression=str(row.get("impression", "") or ""),
        )

        matching_images = projections_df[projections_df["uid"] == uid]
        for _, img_row in matching_images.iterrows():
            filename = img_row.get("filename")
            if isinstance(filename, str):
                img_path = images_dir / filename
                if img_path.exists():
                    report.image_paths.append(img_path)

        if report.full_text() and report.image_paths:
            reports.append(report)
        else:
            skipped += 1

    print(f"Loaded {len(reports)} usable reports (skipped {skipped} incomplete ones).")
    return reports


if __name__ == "__main__":
    reports = load_all_reports(limit=5)
    for r in reports:
        print("=" * 60)
        print("UID:", r.uid)
        print("Images:", [p.name for p in r.image_paths])
        print("Findings:", r.findings[:200])
        print("Impression:", r.impression[:200])
