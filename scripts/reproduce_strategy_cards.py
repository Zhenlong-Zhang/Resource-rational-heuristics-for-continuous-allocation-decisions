"""Reproduce a fixed-design comparison of five fixed rules and one adaptive rule."""
import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('prepare', 'produce', 'check', 'aggregate'))
    parser.add_argument('--config', type=Path, default=ROOT / 'configs/strategy_cards.json')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--node', default='node')
    scope = parser.add_mutually_exclusive_group()
    scope.add_argument('--full', action='store_true')
    scope.add_argument('--only', help='Original coordinates, for example C3:0:1,C5:0:1')
    args = parser.parse_args()
    if args.phase != 'aggregate' and not (args.full or args.only):
        parser.error('Choose --full or --only; no implicit policy computation')
    if args.phase == 'aggregate' and args.only:
        parser.error('Aggregate reads all checked outputs; use --full to require 450 roots')
    for variable in ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS'):
        os.environ[variable] = '1'
    sys.path.insert(0, str(ROOT))
    from threadpoolctl import threadpool_limits
    from src.design_validation.strategy_cards import inputs as I, reproduction as R
    with threadpool_limits(limits=1):
        if args.phase == 'prepare':
            selected = I.I.select_roots(full=args.full, only=args.only)
            I.prepare(args.output, args.config, selected, node=args.node)
            return {'phase': 'prepare', 'preparedRoots': len(selected), 'newAdaptiveTrajectories': 0}
        if args.phase == 'aggregate':
            return R.aggregate(args.output, args.config, full=args.full)
        return R.process(args.output, args.config, phase=args.phase, full=args.full,
                         only=args.only, node=args.node)


if __name__ == '__main__':
    print(json.dumps(main(), sort_keys=True))
