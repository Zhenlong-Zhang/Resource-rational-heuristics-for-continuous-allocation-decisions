"""Conditional C5 precision planning from checked designs, never a policy rerun."""
import argparse
import json
from pathlib import Path
import sys


def main(argv=None):
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('all', 'check'))
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--input-kind', choices=('public-preparation', 'saved-designs'),
                        default='public-preparation',
                        help='Checked public C5 directory, or explicit mathematical operand JSON (not execution evidence)')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--configs', type=Path, default=root/'configs')
    args = parser.parse_args(argv)
    sys.path.insert(0, str(root/'src'))
    from design_validation.c5_precision_reproduction import run
    return run(args.input, args.output, args.configs,
               mathematical_operands=args.input_kind == 'saved-designs', checking=args.phase == 'check')


if __name__ == '__main__':
    print(json.dumps(main(), sort_keys=True))
