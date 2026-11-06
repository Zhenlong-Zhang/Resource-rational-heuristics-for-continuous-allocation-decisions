"""Compare common-t with within-IUT2 on saved coverage inputs."""
from pathlib import Path
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from tutoring_statistics.common_t_reproduction import main

if __name__ == '__main__':
    print(json.dumps(main(), sort_keys=True))
