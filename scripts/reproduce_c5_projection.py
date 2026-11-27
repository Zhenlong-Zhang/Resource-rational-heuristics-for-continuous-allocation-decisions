#!/usr/bin/env python3
"""Source-checkout C5 exploration; unverified profiles cannot certify final results.

Long-run numerical drift is not authenticated, so canonical execution remains
fail-closed. Population jobs require an externally isolated immutable staged
checkout and inputs; read-only file modes and boundary hashes cannot guarantee
immutability. Checkout-local checkpoint files must be git-ignored.
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
    prepare = sub.add_parser('prepare', help='check supplied prefix and lookup bytes and shape, not authority')
    for arg in ('prefix', 'prefix-sha256', 'lookup', 'lookup-sha256'):
        prepare.add_argument('--' + arg, required=True)
    for command in ('sample', 'verify'):
        stage = sub.add_parser(command, help='one-root new-unverified trajectories and independent replay')
        for arg in ('spec', 'spec-sha256', 'prefix', 'prefix-sha256', 'output', 'replicates'):
            stage.add_argument('--' + arg, required=True)
        stage.add_argument('--mode', choices=['new-unverified'], required=True)
    for command in ('population-sample', 'population-verify', 'audit', 'final'):
        stage = sub.add_parser(command, help=('90-root new-unverified sampling/replay' if command != 'audit'
                                              else 'complete provisional independent population audit and reduction'))
        for arg in ('manifest', 'manifest-sha256', 'prefix', 'prefix-sha256'):
            stage.add_argument('--' + arg, required=True)
        stage.add_argument('--mode', choices=['new-unverified'], required=True)
        stage.add_argument('--input-mode', choices=['accepted-prefix', 'caller-prefix'], required=True,
                           help='accepted fixed byte pins or caller-hashed provisional inputs')
        if command in ('population-sample', 'population-verify'):
            stage.add_argument('--output', required=True)
            stage.add_argument('--replicates', required=True, help='inclusive future range START:END')
        else:
            locations = stage.add_mutually_exclusive_group(required=True)
            locations.add_argument('--output', help='single checkpoint directory')
            locations.add_argument('--shard-dir', action='append', dest='shard_dirs',
                                   help='explicit checkpoint shard directory; repeat for each shard')
            stage.add_argument('--temp-dir', help='existing directory for temporary audit databases')
            if command == 'final':
                for arg in ('high-output', 'high-config', 'adopted-lookup', 'latent-config',
                            'geometry-summary', 'geometry-certificate'):
                    stage.add_argument('--' + arg, required=True)
                stage.add_argument('--policy', help='optional policy used for the high-price preparation')
    redraw = sub.add_parser('redraw', help='redraw caller-supplied conditional JSON without authenticating it')
    redraw.add_argument('--conditional-json', required=True)
    redraw.add_argument('--output-dir', required=True)
    historical = sub.add_parser('historical-redraw', help='redraw pinned external saved summary, not a fresh simulation')
    historical.add_argument('--summary-json', required=True, help='external exact saved summary supplied by a researcher')
    historical.add_argument('--summary-sha256', required=True, help='published pinned saved-summary byte SHA256')
    historical.add_argument('--output-dir', required=True)
    args = parser.parse_args(argv)
    try:
        runtime = None
        if args.command in ('sample', 'verify', 'population-sample', 'population-verify', 'audit', 'final'):
            # Load the bootstrap directly: importing src first imports NumPy.
            bootstrap = runpy.run_path(str(ROOT / 'src/design_validation/c5_runtime_profile.py'))
            runtime = bootstrap['validate_runtime'](mode='new')
        sys.path.insert(0, str(ROOT))
        from src.design_validation import c5_public_reproduction as reproduction
        if args.command == 'prepare':
            result = reproduction.prepare(args.prefix, args.prefix_sha256, args.lookup,
                                          args.lookup_sha256, ROOT)
            print(json.dumps(result, sort_keys=True, indent=2))
        elif args.command == 'redraw':
            from src.design_validation import c5_projection_plot
            import matplotlib.pyplot as plt
            supplied = reproduction._strict_json(Path(args.conditional_json).read_bytes())
            if isinstance(supplied, dict) and set(supplied) == {'status', 'result'}:
                if supplied['status'] != 'conditional-new-unverified':
                    raise ValueError('Unexpected conditional CLI envelope status')
                supplied = supplied['result']
            directory = Path(args.output_dir)
            if directory.is_symlink():
                raise ValueError('Redraw output directory symlink refused')
            directory = directory.resolve()
            target = directory / 'c5_conditional_projection.pdf'
            reproduction._require_ignored(directory / '.c5-redraw-check', ROOT)
            reproduction._require_ignored(target, ROOT)
            directory.mkdir(parents=True, exist_ok=True)
            figure, _ = c5_projection_plot.plot_conditional_projection(supplied)
            try:
                with target.open('xb') as handle:
                    figure.savefig(handle, format='pdf')
            finally:
                plt.close(figure)
            print(json.dumps({'status': 'conditional-redraw-unverified', 'output': str(target),
                              'historicalExactResultEstablished': False,
                              'inputAuthenticated': False}, sort_keys=True, indent=2))
        elif args.command == 'historical-redraw':
            from src.design_validation import c5_projection_plot
            import matplotlib.pyplot as plt
            if args.summary_sha256 != c5_projection_plot.ACCEPTED_PLOT_SUMMARY_SHA256:
                raise ValueError('Supplied summary SHA256 must equal the fixed accepted byte pin')
            directory = Path(args.output_dir)
            if directory.is_symlink():
                raise ValueError('Redraw output directory symlink refused')
            directory = directory.resolve()
            target = directory / 'c5_historical_saved_projection.pdf'
            reproduction._require_ignored(directory / '.c5-redraw-check', ROOT)
            reproduction._require_ignored(target, ROOT)
            directory.mkdir(parents=True, exist_ok=True)
            figure, _ = c5_projection_plot.plot_verified_saved_summary(args.summary_json)
            try:
                with target.open('xb') as handle:
                    figure.savefig(handle, format='pdf')
            finally:
                plt.close(figure)
            print(json.dumps({'status': 'pinned-saved-result-redraw', 'output': str(target),
                              'savedSummarySHA256': c5_projection_plot.ACCEPTED_PLOT_SUMMARY_SHA256,
                              'freshSimulation': False, 'runtimeEquivalenceEstablished': False},
                             sort_keys=True, indent=2))
        elif args.command in ('sample', 'verify'):
            pieces = args.replicates.split(':')
            if len(pieces) != 2:
                raise ValueError('replicates must be inclusive START:END')
            start, end = map(int, pieces)
            if not 2 <= start <= end <= 1840:
                raise ValueError('replicates must lie within 2:1840')
            context = reproduction.load_context(args.spec, args.spec_sha256, args.prefix,
                                                args.prefix_sha256, ROOT, runtime=runtime)
            result = reproduction.run_members(context, args.output, range(start, end + 1),
                                               verify_only=args.command == 'verify')
            result = {'runtimeProfile': context['identity']['runtime'], 'result': result}
            print(json.dumps(result, sort_keys=True, indent=2))
        elif args.command in ('population-sample', 'population-verify', 'audit', 'final'):
            if args.command in ('population-sample', 'population-verify'):
                pieces = args.replicates.split(':')
                if len(pieces) != 2:
                    raise ValueError('replicates must be inclusive START:END')
                start, end = map(int, pieces)
                if not 2 <= start <= end <= 1840:
                    raise ValueError('replicates must lie within 2:1840')
            population = reproduction.load_population(args.manifest, args.manifest_sha256,
                                                       args.prefix, args.prefix_sha256, ROOT,
                                                       runtime=runtime, input_mode=args.input_mode)
            if args.command == 'audit':
                result = reproduction.audit_population(population, args.output, shard_dirs=args.shard_dirs,
                                                       temp_dir=args.temp_dir)
            elif args.command == 'final':
                from src.design_validation import c5_conditional_final
                shards = reproduction._shard_directories(args.output, args.shard_dirs)
                high_output = Path(args.high_output).resolve(strict=True)
                if not high_output.is_dir() or any(high_output.is_relative_to(p) or p.is_relative_to(high_output)
                                                   for p in shards):
                    raise ValueError('High-price output must be an isolated directory outside checkpoint shards')
                if args.temp_dir:
                    temp = Path(args.temp_dir).resolve(strict=True)
                    if (not temp.is_dir() or temp.is_relative_to(ROOT) or
                            any(temp.is_relative_to(p) for p in [high_output, *shards])):
                        raise ValueError('Temporary audit directory must be outside checkout and input directories')
                summary = reproduction._strict_json(Path(args.geometry_summary).read_bytes())
                certificate = reproduction._strict_json(Path(args.geometry_certificate).read_bytes())
                result = c5_conditional_final.conditional_final(
                    population, high_output=high_output, high_config=args.high_config,
                    adopted_lookup=args.adopted_lookup, latent_config=args.latent_config,
                    geometry_summary=summary, geometry_certificate=certificate, policy=args.policy,
                    output=args.output, shard_dirs=args.shard_dirs, temp_dir=args.temp_dir)
                result = {'status': 'conditional-new-unverified', 'result': result}
            else:
                result = reproduction.run_population(population, args.output, range(start, end + 1),
                    verify_only=args.command == 'population-verify')
            print(json.dumps(result, sort_keys=True, indent=2, allow_nan=False))
        else:
            reproduction.unavailable(args.command)
    except (ValueError, TypeError, KeyError, RuntimeError, OSError) as exc:
        print(f'c5: {exc}', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
