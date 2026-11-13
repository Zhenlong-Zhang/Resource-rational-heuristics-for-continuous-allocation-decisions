"""Analyze true-need coefficients and saved allocation rules without rerunning policies."""
import argparse
import json
from pathlib import Path
import sys


def main(argv=None):
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True,
                        help='Independently checked six-price policy output directory')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--config', type=Path, default=root / 'configs/true_need_diagnostics.json')
    parser.add_argument('--full', action='store_true', help='Require all 2700 original endpoints')
    args = parser.parse_args(argv)
    sys.path.insert(0, str(root / 'src'))
    from design_validation.true_need_reproduction import analyze
    return analyze(args.input, args.output, args.config, full=args.full)


if __name__ == '__main__':
    print(json.dumps(main(), sort_keys=True))
