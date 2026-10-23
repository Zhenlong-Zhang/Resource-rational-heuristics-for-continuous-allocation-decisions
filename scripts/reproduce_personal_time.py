"""Portable entry for finite personal-time expected endpoints and EO examples."""
from pathlib import Path
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from design_validation.personal_time_reproduction import main


if __name__ == '__main__':
    print(json.dumps(main(), sort_keys=True))
