#!/usr/bin/env python3
"""Portable fixed-batch Gamma attribution and independent scientific readback."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    produce = commands.add_parser("produce", help="Generate original fixed-shape selected batches")
    produce.add_argument("--batches", nargs="+", type=int, choices=(0, 39), required=True,
                         help="0 or 39 gives250 cases; 0 39 gives the complete fixed500")
    read = commands.add_parser("read", help="Independently check a public or original packet")
    read.add_argument("--input", type=Path, required=True)
    read.add_argument("--require-full", action="store_true",
                      help="Reject selected250 packets rather than report partial scope")
    for command in (produce, read):
        command.add_argument("--config", type=Path, default=ROOT / "configs/tutoring_calibration_v01.json")
        command.add_argument("--output", type=Path, required=True, help="New directory; never overwritten")
    args = parser.parse_args(argv)
    try:
        if args.command == "produce":
            from src.tutoring_analysis.paired_diagnostic import produce as run
            result = run(args.config, args.output, args.batches)
        else:
            from src.tutoring_analysis.paired_readback import read_packet
            result = read_packet(args.config, args.input, args.output, args.require_full)
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    print(json.dumps({key: result[key] for key in
                      ("schema", "cases", "selectedComplete", "fixed500Complete")}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
