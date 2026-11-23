"""Reproduce saved coefficient diagnosis and original C5 finite-law precision preparation."""
import argparse
import json
from pathlib import Path
import sys


def main(argv=None):
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('fresh', 'diagnose', 'check-diagnostics', 'c5', 'check-c5',
                                        'import-original', 'diagnose-original', 'check-original'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--input', type=Path, help='Public bundle, original compact input directory, or explicit original import; depends on mode')
    parser.add_argument('--policy-input', type=Path, help='Compatible checked public six-price output (C5 only)')
    parser.add_argument('--configs', type=Path, default=root / 'configs')
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--full', action='store_true')
    group.add_argument('--only', help='Explicit C5 coordinates, e.g. C5:0:1; not a full-design result')
    parser.add_argument('--node', default='node')
    parser.add_argument('--workers', type=int, default=1, help='Fresh endpoint processes; C5 is serial')
    args = parser.parse_args(argv)
    if args.workers < 1:
        parser.error('--workers must be positive')
    if args.mode == 'fresh':
        if not args.full or args.input or args.policy_input or args.only:
            parser.error('fresh requires explicit --full and no input; expensive original population route')
    elif args.mode in ('diagnose', 'check-diagnostics', 'import-original', 'diagnose-original', 'check-original'):
        if not args.input or args.policy_input or args.full or args.only or args.workers != 1:
            parser.error('input-admission/diagnosis modes require only --input and --output, no root selection')
    elif args.input or not (args.full or args.only) or args.workers != 1:
        parser.error('C5 requires --full or --only, optional --policy-input, and serial execution')
    sys.path.insert(0, str(root / 'src'))
    from design_validation import coefficient_reproduction as D
    from design_validation import c5_reproduction as C5
    if args.mode == 'fresh':
        return D.fresh(args.output, args.configs, node=args.node, workers=args.workers)
    if args.mode in ('diagnose', 'check-diagnostics'):
        return D.diagnose(args.input, args.output, args.configs, checking=args.mode == 'check-diagnostics')
    if args.mode == 'import-original':
        return D.O.prepare(args.input, args.output, args.configs)
    if args.mode in ('diagnose-original', 'check-original'):
        return D.diagnose_original(args.input, args.output, args.configs, checking=args.mode == 'check-original')
    chosen = C5.selection(full=args.full, only=args.only)
    recipe = args.configs / 'personal_time_sensitivity.json'
    if args.mode == 'check-c5':
        return C5.check(args.output, recipe, chosen, policy=args.policy_input)
    produced = C5.produce(args.output, recipe, chosen, policy=args.policy_input, node=args.node)
    checked = C5.check(args.output, recipe, chosen, policy=args.policy_input)
    return dict(producer=produced, independent=checked)


if __name__ == '__main__':
    print(json.dumps(main(), sort_keys=True))
