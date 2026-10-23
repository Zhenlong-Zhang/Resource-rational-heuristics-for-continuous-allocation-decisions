#!/usr/bin/env python3
"""Portable baseline, saved-fit diagnostic and fresh CR2 comparison routes."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from tutoring_statistics.comparison_runner import main

if __name__ == "__main__":
    main()
