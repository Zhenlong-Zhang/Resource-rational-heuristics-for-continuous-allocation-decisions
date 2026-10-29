#!/usr/bin/env python3
"""Explicit portable entry for joint and interval comparisons."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.tutoring_statistics.joint_interval_reproduction import main

if __name__ == "__main__":
    main()
