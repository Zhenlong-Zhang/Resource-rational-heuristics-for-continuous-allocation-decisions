"""Transparent import of the original accepted compact coefficient inputs.

This is NOT a paper bundle, a policy-output epoch, or reconstruction of original
execution provenance. Exact scientific byte identities select one historical
dataset. A new import records current validation; original source-status flags
remain unchanged. The public fresh route needs none of these optional inputs.
"""
import csv
import hashlib
import io
from pathlib import Path

from . import personal_time_inputs as I
from . import personal_time_sensitivity as S
from . import true_need_reproduction as T
from . import coefficient_diagnostics as D
from . import coefficient_io as IO

# Scientific content identities only. No private receipt or actor is consumed.
INPUTS = {
    'admitted_rows.json': 'ca594cefae06305e8e6166b4a61137be96c3e09082310ef06cf2e762d1f3b3f1',
    'public_summary.json': 'e68d3c19fcc25ceaae32e3196b059aca2b90553b92cd152822f965b282ff612e',
    'true_need_schedules.csv': 'ec15975a68e2640573dcf70ecd2ba02a5b330a57395196f0cb37b8b00e5e4fc9',
}
SOURCE_FILES = {
    'true_need_diagnostics.py': 'b7a1efd627ad8547c1018e76527bfc50bcd38f3551939bb353a6a8145bba537a',
    'true_need_reproduction.py': 'fd2eeca8e8b59416d063e41ba28bcb0ae92f320843f50154d1bcc0b9b9171e4a',
    'personal_time_diagnostics.py': '3f8213aa53aa5379c60560760e5f66a01784199484bc23240b53cd9a880afa4c',
}
CONFIG_FILES = {
    'true_need_diagnostics.json': 'b97f1840b24a4d82391f92ed65ec1a0f5def7f8eb59088a1b4058b9a420e048f',
    'personal_time_sensitivity.json': 'a836a25f273f3dfa4b99367e527d8828a926ab0f8c508e6679c19c2ecf9745ca',
    'personal_time_reference.json': '0f28da0863dc3ad19a866cdcf674c2e0d0287150fdf4d79992aa21e3800d7d12',
}
CSV_FIELDS = ('condition', 'price', 'schedule', 'available_rows', 'missing', 'ties',
              'eligible', 'gap_slope', 'lower_need_partial', 'partial_rank')
BOUNDARY = {
    'inputKind': 'accepted_original_compact_scientific_inputs',
    'currentWriterRole': 'portable_import_not_original_execution',
    'historicalAcceptance': 'established_by_prior_review_not_reexecuted_by_import',
    'historicalExecutionReconstructed': False,
    'historicalAcceptanceRevalidated': False,
    'newPublicPolicyEpoch': False,
    'paperBundleCreated': False,
    'originalSummaryFlagsChanged': False,
    'newDraws': 0,
    'newPolicyRuns': 0,
    'newFitInvocations': 0,
}


def scientific_identity(configs):
    configs, here = Path(configs), Path(__file__).resolve().parent
    for name, expected in SOURCE_FILES.items():
        I.need(S.file_sha(here / name) == expected, 'Original scientific source changed: ' + name)
    for name, expected in CONFIG_FILES.items():
        I.need(S.file_sha(configs / name) == expected, 'Original scientific recipe changed: ' + name)
    context = S.load_context(configs / 'personal_time_sensitivity.json')
    I.need(I.read(configs / 'true_need_diagnostics.json') == T.RECIPE, 'Exact true-need recipe')
    return dict(referencePublicCommit='290ef18893c17c0e25add5f4ab450096f8a8a66d',
                originalScientificSources=dict(SOURCE_FILES), recipeFiles=dict(CONFIG_FILES),
                nativeInputFiles=context['inputFileSHA256'],
                proxyRecipe=context['recipe'], trueNeedRecipe=T.RECIPE,
                scope=dict(endpoints=2700, schedules=270, cells=30, ties=66, eligible=2634),
                seedScope='declared_original_recipe_not_new_latent_draw_verification')


def admit_compact(rows, summary, csv_text):
    """Full scientific structure/CSV linkage; no fit, policy, or provenance claim."""
    data = D.admit(rows, summary)
    for name in ('rowSourceAuthenticated', 'historicalEventClaimed', 'policyRerunByAnalysis',
                 'monteCarloPrecisionEstablished', 'humanInferenceAdopted'):
        I.need(summary.get(name) is False, 'Original summary boundary flag changed: ' + name)
    reader = csv.DictReader(io.StringIO(csv_text))
    I.need(reader.fieldnames == list(CSV_FIELDS), 'Exact original schedule CSV columns')
    csv_rows = list(reader)
    I.need(all(set(row) == set(CSV_FIELDS) and all(value is not None for value in row.values())
               for row in csv_rows), 'Malformed schedule CSV row')
    converted = [{**row, 'schedule': int(row['schedule'])} for row in csv_rows]
    original = D.indexed(converted, D.schedule_key, D.schedule_roster(), 'Original CSV')
    for key, saved in data['schedules'].items():
        for name in ('available_rows', 'missing', 'ties', 'eligible', 'partial_rank'):
            I.need(int(original[key][name]) == saved[name], 'Original CSV count/rank mismatch')
        for name in D.METRICS:
            value = float(original[key][name])
            I.need(D.finite(value) and abs(value-saved[name]) <= 1e-10,
                   'Original CSV coefficient representation mismatch')
    return dict(verifiedEndpointRows=2700, verifiedScheduleRows=270, verifiedCells=30,
                exactInputScientificRoster=True, csvSummaryAgreement=True,
                comparison='saved coefficient literals within original 1e-10 representation tolerance',
                independentGeometryChecked=False, historicalExecutionChecked=False)


def _read_inputs(source):
    source = Path(source)
    I.need(source.is_dir() and not source.is_symlink(), 'Ordinary compact-input directory required')
    raw = {}
    for name, expected in INPUTS.items():
        path = source / name
        I.need(path.is_file() and not path.is_symlink(), 'Missing ordinary compact input: ' + name)
        value = path.read_bytes()
        I.need(hashlib.sha256(value).hexdigest() == expected, 'Changed original scientific bytes: ' + name)
        raw[name] = value
    rows = I.parse(raw['admitted_rows.json'])
    summary = I.parse(raw['public_summary.json'])
    check = admit_compact(rows, summary, raw['true_need_schedules.csv'].decode('utf-8'))
    return raw, rows, summary, check


def _binding(identity):
    return dict(schema='coefficient-original-import-v1', inputSHA256=dict(INPUTS),
                scientificIdentity=identity, importSources=IO.source_identity(),
                boundary=dict(BOUNDARY))


def prepare(source, output, configs):
    """Import exact compact bytes into a new, explicitly non-policy artifact."""
    source, output = IO.separate(source, output)
    identity = scientific_identity(configs)
    raw, _, _, check = _read_inputs(source)
    binding = _binding(identity)
    # Establish identity before writing inputs, so a foreign completed import
    # cannot be extended or silently relabeled as this one.
    stored_manifest, new = IO.stage(output / 'original_import.json', binding, lambda: check)
    I.need(stored_manifest == check, 'Changed import scientific check')
    for name, value in raw.items():
        I.write_bytes(output / 'inputs' / name, value)
    # Re-admit stored bytes; do not infer completeness from a manifest alone.
    I.need({p.name for p in (output / 'inputs').iterdir()} == set(INPUTS), 'Exact imported file roster')
    _, _, _, stored_check = _read_inputs(output / 'inputs')
    I.need(stored_check == check, 'Imported scientific data changed')
    return dict(output=str(output), newImport=new, inputSHA256=dict(INPUTS),
                validationScope='exact_original_compact_bytes_and_scientific_crosschecks',
                **BOUNDARY, **check)


def load(source, configs):
    """Revalidate the explicit original import; never accept an arbitrary seal."""
    source = Path(source)
    binding = _binding(scientific_identity(configs))
    check, _ = IO.stage(source / 'original_import.json', binding, None, readonly=True)
    inputs = source / 'inputs'
    I.need({p.name for p in inputs.iterdir()} == set(INPUTS), 'Exact compact imported file roster')
    _, rows, summary, actual = _read_inputs(inputs)
    I.need(actual == check, 'Import scientific check mismatch')
    return rows, summary, binding, S.file_sha(source / 'original_import.json')
