"""Reduce independently checked operation-cost outputs into a finite-menu comparison."""
import argparse
import json
from pathlib import Path
import sys


def main(argv=None):
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('all', 'check', 'export'))
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--configs', type=Path, default=root / 'configs')
    parser.add_argument('--conditions', default='C3,C5')
    parser.add_argument('--input-kind', choices=('checked-products', 'saved-operands'),
                        default='checked-products', help='Saved operands prove arithmetic, not original execution')
    args = parser.parse_args(argv)
    sys.path.insert(0, str(root / 'src'))
    from design_validation.operation_menu_reproduction import run
    return run(args.input, args.output, args.configs,
               conditions=tuple(args.conditions.split(',')), checking=args.phase == 'check',
               input_kind=args.input_kind, exporting=args.phase == 'export')


if __name__ == '__main__':
    print(json.dumps(main(), sort_keys=True))
