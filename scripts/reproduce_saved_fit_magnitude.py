"""Bind an existing target fit, compute eight contrasts, or independently verify them."""
import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare", help="Bind existing saved artifacts; never generate or fit")
    source = prepare.add_mutually_exclusive_group(required=True)
    source.add_argument("--reproduction-root", type=Path)
    source.add_argument("--historical-request", type=Path)
    prepare.add_argument("--locations", type=Path, help="Optional original-path to current-path JSON map")
    prepare.add_argument("--output", type=Path, required=True)
    for name in ("compute", "verify"):
        command = commands.add_parser(name)
        command.add_argument("--input", type=Path, required=True)
        command.add_argument("--output", type=Path, required=True)
        command.add_argument("--rscript", default="Rscript")
        if name == "verify":
            command.add_argument("--producer-result", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        from src.tutoring_statistics.saved_fit_inputs import load, prepare_input
        value = prepare_input(args.output, reproduction_root=args.reproduction_root,
            historical_request=args.historical_request, locations=load(args.locations) if args.locations else None)
        print(json.dumps(dict(schema=value["schema"], kind=value["kind"], output=str(args.output))))
    elif args.command == "compute":
        from src.tutoring_statistics.saved_fit_reproduction import run
        value = run(args.input, args.output, args.rscript)
        print(json.dumps(dict(status=value["status"], output=str(args.output))))
    else:
        from src.tutoring_statistics.independent_saved_fit import verify
        value = verify(args.input, args.producer_result, args.output, args.rscript)
        print(json.dumps(dict(passed=value["passed"], output=str(args.output))))


if __name__ == "__main__":
    main()
