"""Portable six-price joint REML mapping and independent profile check."""
from pathlib import Path
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from tutoring_statistics.proxy_h10_mapping import main

if __name__ == '__main__':
    print(json.dumps(main(), sort_keys=True))
