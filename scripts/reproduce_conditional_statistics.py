"""Run portable conditional mixed-model reproduction from shared source."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from tutoring_statistics.reproduction import main

if __name__ == "__main__":
    main()
