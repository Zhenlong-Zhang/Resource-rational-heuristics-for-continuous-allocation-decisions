"""Analyze checked proxy allocations and target-key fixed model outcomes."""
import argparse
import json
from pathlib import Path
import sys


def main(argv=None):
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True, help='Checked six-price output directory')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--config', type=Path, default=root / 'configs/personal_time_diagnostics.json')
    parser.add_argument('--figures', action='store_true', help='Plot complete condition/price summaries')
    args = parser.parse_args(argv)
    sys.path.insert(0, str(root / 'src'))
    from design_validation.personal_time_diagnostics import analyze
    result = analyze(args.input, args.output, args.config)
    if args.figures:
        from design_validation.personal_time_plots import plot
        plot(args.output)
    return result


if __name__ == '__main__':
    print(json.dumps(main(), sort_keys=True))
