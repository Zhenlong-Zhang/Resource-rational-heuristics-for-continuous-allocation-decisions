"""Reproduce the six-price finite personal-time comparison from native inputs."""

import argparse
import json
from pathlib import Path
import sys


def main(argv=None):
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('prepare', 'produce', 'check', 'all'))
    parser.add_argument('--config', type=Path,
                        default=root / 'configs/personal_time_sensitivity.json')
    parser.add_argument('--output', type=Path, required=True)
    scope = parser.add_mutually_exclusive_group(required=True)
    scope.add_argument('--full', action='store_true',
                       help='All 450 original roots at each of the six fixed prices')
    scope.add_argument('--only', help='Original coordinates, e.g. C3:0:1,C5:0:1')
    scope.add_argument('--selection', type=Path, help='JSON list of original coordinates')
    parser.add_argument('--node', default='node', help='Native source generator executable')
    args = parser.parse_args(argv)
    output = args.output.resolve()
    if root.is_relative_to(output) or any(
        output.is_relative_to(root / name)
        for name in ('src', 'scripts', 'configs', 'tests', 'notebooks')
    ):
        parser.error('Use a generated-output directory, not a source directory.')

    sys.path.insert(0, str(root / 'src'))
    from design_validation import personal_time_sensitivity as study

    selected = study.select_roots(
        full=args.full, only=args.only,
        selected=json.loads(args.selection.read_text()) if args.selection else None,
    )
    context = study.load_context(args.config)
    if args.mode in ('prepare', 'all'):
        study.prepare(context, selected, output, node=args.node)
    if args.mode in ('produce', 'all'):
        study.produce(context, selected, output)
    if args.mode in ('check', 'all'):
        return study.check(context, selected, output)
    return {'mode': args.mode, 'selectedRoots': len(selected), 'output': str(output)}


if __name__ == '__main__':
    print(json.dumps(main(), sort_keys=True))
