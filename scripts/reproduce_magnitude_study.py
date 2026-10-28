"""Reproduce fixed magnitude validation and conditional saved-power results."""
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.tutoring_statistics.magnitude_reproduction import main

if __name__ == "__main__":
    main()
