#!/usr/bin/env python3
"""Export a byte-checked fixed-policy candidate, without private envelopes.

The output is not authenticated until independently pinned to the accepted
server export; its input checksum alone is not an output acceptance proof.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.design_validation.c5_public_inputs import export_fixed_inputs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--accepted-input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    data = export_fixed_inputs(args.accepted_input)
    encoded = (json.dumps(data, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()
    with args.output.open("xb") as out:
        out.write(encoded)


if __name__ == "__main__":
    main()
