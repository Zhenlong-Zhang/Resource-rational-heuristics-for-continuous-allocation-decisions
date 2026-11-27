#!/usr/bin/env python3
"""Prepare or run a prospective C5 population, not a historical replay.

Run only from an isolated read-only source checkout and isolated inputs.
Fresh outputs are unverified and do not authorize statistical inference.
"""
import argparse
import json
from pathlib import Path
import runpy
import sys

ROOT = Path(__file__).resolve().parents[1]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    pins = sub.add_parser('inspect-pins', help='describe current input/source bytes, not authenticate them')
    pins.add_argument('--manifest', required=True)
    pins.add_argument('--manifest-sha256', required=True)
    prepare = sub.add_parser('prepare', help='prepare 90 inputs without historical prefix or outcomes')
    for name in ('manifest', 'manifest-sha256', 'pins', 'pins-sha256', 'output'):
        prepare.add_argument('--' + name, required=True)
    for command in ('sample', 'verify', 'audit'):
        stage = sub.add_parser(command, help='independently replay existing fresh members')
        for name in ('manifest', 'manifest-sha256', 'output', 'run-id', 'replicates'):
            stage.add_argument('--' + name, required=True)
    args = parser.parse_args(argv)
    bootstrap = runpy.run_path(str(ROOT / 'src/design_validation/c5_runtime_profile.py'))
    runtime = bootstrap['validate_runtime'](mode='new')
    sys.path.insert(0, str(ROOT))
    from src.design_validation import c5_fresh_prepare as fresh
    from src.design_validation import c5_public_reproduction as common
    if args.command == 'inspect-pins':
        result = fresh.inspect_population_pins(args.manifest, args.manifest_sha256, ROOT)
    elif args.command == 'prepare':
        pins_raw = Path(args.pins).read_bytes()
        if common._sha(pins_raw) != args.pins_sha256:
            raise ValueError('Reviewed pins byte digest mismatch')
        pins = common._strict_json(pins_raw)
        if type(pins) is not dict or set(pins) != {'inputs', 'sources', 'entrySHA256'}:
            raise ValueError('Pins must contain inputs, sources and entry identity')
        result = fresh.prepare_population(
            args.manifest, args.manifest_sha256, args.output, ROOT,
            expected_inputs=pins['inputs'], expected_sources=pins['sources'],
            expected_entry_sha256=pins['entrySHA256'], runtime=runtime,
            expected_runtime_sha256=common._sha(common.canonical_bytes(common.sanitize_runtime(runtime))))
        result = {'status': 'fresh-input-unverified', 'rootCount': len(result['roots']),
                  'manifest': str(Path(args.output) / 'input-manifest.json'),
                  'manifestSHA256': common._sha(Path(args.output, 'input-manifest.json').read_bytes())}
    else:
        from src.design_validation import c5_fresh_lazy
        pieces = args.replicates.split(':')
        if len(pieces) != 2:
            raise ValueError('Replicates must be inclusive START:END')
        start, end = map(int, pieces)
        if not 1 <= start <= end <= 1840:
            raise ValueError('Replicates must lie within 1:1840')
        if args.command == 'audit':
            from src.design_validation import c5_fresh_audit
            result = c5_fresh_audit.audit_prepared_manifest(
                args.manifest, args.manifest_sha256, args.output, ROOT, runtime=runtime,
                domain='fresh', run_id=args.run_id, replicates=range(start, end + 1))
        else:
            result = c5_fresh_lazy.run_lazy(
                args.manifest, args.manifest_sha256, args.output, ROOT, runtime=runtime,
                domain='fresh', run_id=args.run_id, replicates=range(start, end + 1),
                verify_only=args.command == 'verify')
    print(json.dumps(result, sort_keys=True, indent=2))


if __name__ == '__main__':
    main()
