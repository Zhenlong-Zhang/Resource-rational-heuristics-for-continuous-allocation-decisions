"""Prepare existing saved coverage inputs or diagnose their original intervals."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.tutoring_statistics.saved_coverage import main

if __name__ == "__main__":
    main()
