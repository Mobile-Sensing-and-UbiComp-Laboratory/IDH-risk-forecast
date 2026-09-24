"""Local paths and core input dimensions; set environment variables before import."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = os.environ.get("IDH_DATA_ROOT", str(ROOT / "data"))
RAW_ROOT = os.environ.get("IDH_RAW_ROOT", str(ROOT / "raw_data"))
FIGURE_ROOT = os.environ.get("IDH_FIGURE_ROOT", str(ROOT / "figures"))
SAMPLE_EVERY = int(os.environ.get("IDH_SAMPLE_EVERY", "1"))
SEQ_LENGTH = {1: 256, 3: 128, 5: 64, 10: 32, 15: 24}[SAMPLE_EVERY]
