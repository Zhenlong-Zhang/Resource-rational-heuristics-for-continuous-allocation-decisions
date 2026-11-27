"""Generate portable C5 population inputs without C5 policy outcome draws.

The native root generator draws synthetic latent needs, reports and baseline
outcomes as discarded ancestry; these are not C5 policy draws or historical
outcomes. Only the root inputs are retained. A matching report product is not a claim of
historical runtime equivalence. Output must be a new directory outside the
checkout. Failed preparations cannot be resumed: use a different new directory.

For population loading, stage the executing source checkout, generated inputs,
and externally supplied prefix as read-only files in an independently isolated
snapshot. Keep writable results outside that snapshot. File modes alone do not
guarantee isolation. The diagnostic summary is not a completion signal; only
the atomically published manifest signals successful assembly and completion.
"""
import argparse
import errno
import hashlib
import json
import os
from pathlib import Path
import stat
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.design_validation import canonical_report, canonical_report_reference
from src.design_validation import c5_input_assembly, c5_public_inputs
from src.design_validation import personal_time_inputs, personal_time_sensitivity
from scripts.reproduce_canonical_laws import ACCEPTED_TABLE_SHA

MAX_REPORT_BYTES = 16 * 1024 * 1024


def sha(data):
    return hashlib.sha256(data).hexdigest()


def json_bytes(value):
    return (json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False) + '\n').encode()


def supplied_report(width, directory):
    """Read one supplied scientific product, admitting only accepted bytes and table."""
    name = hashlib.sha256(width.encode('ascii')).hexdigest() + '.json'
    path = Path(directory) / name
    before = path.lstat()
    if not stat.S_ISREG(before.st_mode):
        raise ValueError('Supplied report must be a regular file, not a symlink or special file')
    if not hasattr(os, 'O_NONBLOCK'):
        raise OSError('Nonblocking open is required for supplied reports')
    flags = os.O_RDONLY | getattr(os, 'O_NOFOLLOW', 0) | os.O_NONBLOCK
    try:
        fd = os.open(path, flags)
    except OSError as exc:
        if exc.errno == errno.ELOOP:
            raise ValueError('Supplied report must not be a symlink') from exc
        raise
    with os.fdopen(fd, 'rb') as stream:
        opened = os.fstat(stream.fileno())
        after = path.lstat()
        if (not stat.S_ISREG(opened.st_mode) or not stat.S_ISREG(after.st_mode)
                or (before.st_dev, before.st_ino) != (opened.st_dev, opened.st_ino)
                or (after.st_dev, after.st_ino) != (opened.st_dev, opened.st_ino)):
            raise ValueError('Supplied report must remain the same regular file')
        if opened.st_size > MAX_REPORT_BYTES:
            raise ValueError('Supplied report exceeds the 16 MiB limit')
        data = stream.read(MAX_REPORT_BYTES + 1)
        if len(data) > MAX_REPORT_BYTES:
            raise ValueError('Supplied report exceeds the 16 MiB limit')
    expected = ACCEPTED_TABLE_SHA.get(width)
    if expected is None or sha(data) != expected:
        raise ValueError('Supplied report product byte identity mismatch')
    canonical_report_reference.verify_report_table(json.loads(data), width)
    return data


def prepare(output, *, node='node', report_dir=None):
    """Create a new input bundle and return its explicit verification summary."""
    output = Path(output).absolute()
    personal_time_sensitivity.safe_output(output)
    if output.resolve().is_relative_to(ROOT.resolve()):
        raise ValueError('Output must be outside the source checkout')
    output.mkdir(parents=True, exist_ok=False)
    config_dir = output / 'configs'
    config_dir.mkdir()
    recipe = ROOT / 'configs' / 'personal_time_sensitivity.json'
    context = personal_time_sensitivity.load_context(recipe)
    names = ['personal_time_sensitivity.json', context['recipe']['reference']]
    names.extend(context['reference']['recipe'][key] for key in personal_time_inputs.INPUT_SHA256)
    names.extend(['canonical_c5_latent.json', 'c5_fixed_initial_policy.json'])
    for name in names:
        if Path(name).name != name:
            raise ValueError('Configuration names must be local file names')
        (config_dir / name).write_bytes((ROOT / 'configs' / name).read_bytes())
    lookup = config_dir / 'c5_fixed_initial_policy.json'
    latent = config_dir / 'canonical_c5_latent.json'
    lookup_sha = sha(lookup.read_bytes())
    if lookup_sha != c5_input_assembly.LOOKUP_BYTES_SHA256 or sha(latent.read_bytes()) != c5_input_assembly.LATENT_BYTES_SHA256:
        raise ValueError('Fixed scientific configuration byte identity mismatch')
    fixed = c5_public_inputs.load_fixed_inputs(lookup, lookup_sha)
    selected = personal_time_inputs.selection(
        [dict(zip(personal_time_inputs.FIELDS, key)) for key in c5_public_inputs.ROSTER])
    roots = personal_time_inputs.prepare(context['reference'], selected, output / 'native', node=node)
    widths = sorted({float(row['reportWidth']).hex() for row in fixed['roots']})
    (output / 'reports').mkdir()
    reports, checks, paths = {}, {}, {}
    for index, width in enumerate(widths):
        if report_dir is None:
            table = canonical_report.generate_report_table(width)
            canonical_report_reference.verify_report_table(table, width)
            data = json_bytes(table)
        else:
            data = supplied_report(width, report_dir)
        name = f'reports/report_{index}.json'
        (output / name).write_bytes(data)
        digest = sha(data)
        expected = ACCEPTED_TABLE_SHA.get(width)
        reports[width] = data
        paths[width] = name
        checks[width] = {'sha256': digest, 'acceptedSHA256': expected,
                         'acceptedProductByteMatch': expected is not None and digest == expected,
                         'source': 'freshly_generated' if report_dir is None else 'supplied_accepted_product',
                         'identity': 'accepted_finite_table_bytes' if expected is not None and digest == expected
                         else 'new_unverified_table_bytes'}
    manifest = {'recipePath': 'configs/personal_time_sensitivity.json',
                'rootsPath': 'native/inputs/roots.json',
                'lookupPath': 'configs/c5_fixed_initial_policy.json',
                'lookupSHA256': lookup_sha, 'latentPath': 'configs/canonical_c5_latent.json',
                'reportPaths': paths}
    manifest_bytes = json_bytes(manifest)
    summary = {'manifestSHA256': sha(manifest_bytes), 'reportProducts': checks,
               'acceptedProductBytesMatch': all(x['acceptedProductByteMatch'] for x in checks.values()),
               'summaryScope': 'diagnostic-only-manifest-is-sole-completion-signal', 'c5PolicyDraws': 0,
               'nativeSyntheticAncestry': 'generated and discarded synthetic needs, reports and baseline outcomes; not C5 policy draws',
               'historicalRuntimeEquivalenceEstablished': False}
    summary_path = output / 'preparation_summary.json'
    if not summary['acceptedProductBytesMatch']:
        summary_path.write_bytes(json_bytes(summary))
        raise ValueError('Report product mismatch: new unverified identity; see preparation_summary.json')
    try:
        c5_input_assembly.assemble(lookup, lookup_sha, latent.read_bytes(), roots, context, reports)
    except Exception:
        summary_path.write_bytes(json_bytes(summary))
        raise
    summary_path.write_bytes(json_bytes(summary))
    # The manifest is the final completion signal and must never be replaced.
    personal_time_inputs.write_bytes(output / 'population_manifest.json', manifest_bytes)
    return {**summary, 'assemblyPassed': True}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--output', type=Path, required=True,
                        help='new directory outside the checkout; failed preparations cannot be resumed')
    parser.add_argument('--node', default='node')
    parser.add_argument('--report-dir', type=Path,
                        help='optional directory of supplied canonical report products named by SHA256 of width hex; accepted bytes and tables are independently checked')
    args = parser.parse_args()
    print(json.dumps(prepare(args.output, node=args.node, report_dir=args.report_dir), sort_keys=True))


if __name__ == '__main__':
    main()
