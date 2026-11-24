"""Reproduce all four finite-allocation paper figures or resume checked numerical work."""
import argparse
import json
from pathlib import Path
import sys


def main(argv=None):
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('all', 'produce', 'check', 'reduce', 'redraw', 'redraw-original'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--input', type=Path, help='Policy directory (check/reduce), sealed bundle (redraw), or original compact summaries (redraw-original)')
    parser.add_argument('--config', type=Path, default=root / 'configs/paper_figures.json')
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--full', action='store_true', help='All 450 roots, six prices and four paper figures')
    group.add_argument('--only', help='Small explicit original-root selection, e.g. C1:0:1; no full figures')
    parser.add_argument('--node', default='node')
    parser.add_argument('--workers', type=int, default=1, help='Independent root processes; use allocated CPUs only')
    args = parser.parse_args(argv)
    if args.workers < 1:
        parser.error('--workers must be positive')
    redraw = args.mode in ('redraw', 'redraw-original')
    if redraw and (not args.input or args.full or args.only):
        parser.error('redraw modes require --input and no root selection')
    if not redraw and not (args.full or args.only):
        parser.error('all/produce/check/reduce requires --full or --only')
    if ((args.mode in ('all', 'produce') and args.input)
            or (args.mode in ('check', 'reduce') and not args.input)):
        parser.error('--input is for check/reduce/redraw modes, not all/produce')
    sys.path.insert(0, str(root / 'src'))
    from design_validation import paper_reproduction as study
    from design_validation.personal_time_inputs import select_roots
    selected = None if redraw else select_roots(full=args.full, only=args.only)
    return study.run(args.mode, args.output, args.config, selected=selected,
                     source=args.input, node=args.node, workers=args.workers)


if __name__ == '__main__':
    print(json.dumps(main(), sort_keys=True))
