"""Check fixed latent laws and generate no-draw report tables.

Numerical output has a new execution identity. Matching scientific table
payloads does not establish general historical or cross-platform equivalence.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.design_validation import canonical_latent_reference as LR
from src.design_validation import canonical_report as P
from src.design_validation import canonical_report_reference as R

WIDTHS = ('0x1.9333333333334p+1', '0x1.6ccccccccccccp+1', '0x1.8000000000000p+1')
ADOPTED_CONFIG_SHA = 'ae28df66e90981b68cba0b38e7fedaa5303330fa2da875e770e6ba41e629ee5c'
ACCEPTED_TABLE_SHA = {
    WIDTHS[0]: 'ba563fd341874d99c0a74df66a2dee4c7aea19d7eb52ce64b7569dccfebba85c',
    WIDTHS[1]: 'f85a6c3b83cca15bc83b8b35c57754a3a43beeb59f31cb0fd4128d2d4453d95c',
    WIDTHS[2]: 'e3a4632ba2924afae5df9be893c9c2e3b209e8250d31c9c386d6c3e922728bb6',
}
HISTORICAL_SHA = {
    WIDTHS[0]: 'b3d109993d1ebd9f3ff5c62baa72fd23544f2aa29d18e936a6a379d895786e31',
    WIDTHS[1]: '5c06cf40b96d2cf8c5e7bb9f4a3a6b30df4347025c7d7552660ea07a64444318',
    WIDTHS[2]: '883bb8426cb2e00c20ba4c1a276bbe0f389db1ea41679864e3abbb3ff47e5134',
}


def classify_product(width, digest):
    matched = digest == ACCEPTED_TABLE_SHA[width]
    return matched, 'accepted_finite_table_bytes' if matched else 'new_unverified_table_bytes'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT/'configs/canonical_c5_latent.json')
    parser.add_argument('--digest', type=Path, default=ROOT/'configs/canonical_c5_latent.sha256')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--allow-unverified-config', action='store_true',
                        help='Allow a non-adopted fixed input, explicitly marked unverified')
    parser.add_argument('--historical-tables', type=Path,
                        help='Optional supplied historical table directory; payload parity only')
    args = parser.parse_args()
    data = args.config.read_bytes()
    expected = args.digest.read_text().strip()
    if len(expected) != 64 or hashlib.sha256(data).hexdigest() != expected:
        raise ValueError('Fixed scientific configuration digest mismatch')
    pin_path = ROOT/'configs/canonical_c5_latent.sha256'
    if not pin_path.is_file() or pin_path.read_text().strip() != ADOPTED_CONFIG_SHA:
        raise ValueError('Adopted scientific configuration pin mismatch')
    adopted = expected == ADOPTED_CONFIG_SHA
    if not adopted and not args.allow_unverified_config:
        raise ValueError('Configuration is not the adopted public scientific input')
    config = json.loads(data)
    if (set(config) != {'schema', 'denominator', 'recipientOrder', 'lastCategoryIncludesFallback', 'identity', 'roots'}
            or config['schema'] != 'fixed-c5-latent-53bit-v1'
            or config['denominator'] != 2**53 or config['recipientOrder'] != ['A', 'B']
            or config['lastCategoryIncludesFallback'] is not True
            or config['identity'] != 'adopted-fixed-integer-scientific-input-not-fresh-exp-log'
            or [x['rootKey'] for x in config['roots']] != [['C5', v, m] for v in range(9) for m in range(1, 11)]):
        raise ValueError('Invalid fixed scientific configuration')
    for root in config['roots']:
        if set(root) != {'rootKey', 'recipientLaws'} or len(root['recipientLaws']) != 2:
            raise ValueError('Exactly two recipient laws')
        for law in root['recipientLaws']:
            if set(law) != {'counts', 'cumulativeHex'}:
                raise ValueError('Invalid law fields')
            LR.verify_historical_cdf(law['counts'], law['cumulativeHex'])
    args.output.mkdir(parents=True, exist_ok=False)
    table_checks = []
    for width in WIDTHS:
        table = P.generate_report_table(width)
        R.verify_report_table(table, width)
        path = args.output/(table['key'] + '.json')
        path.write_text(json.dumps(table, sort_keys=True, separators=(',', ':'))+'\n')
        product_sha = hashlib.sha256(path.read_bytes()).hexdigest()
        product_match, product_identity = classify_product(width, product_sha)
        parity = None
        if args.historical_tables is not None:
            raw = (args.historical_tables/path.name).read_bytes()
            if hashlib.sha256(raw).hexdigest() != HISTORICAL_SHA[width]:
                raise ValueError('Historical table identity mismatch')
            old = json.loads(raw)
            for field in ('reportWidthHex', 'key', 'needs', 'reports', 'samplingRule', 'rows'):
                if old[field] != table[field]:
                    raise ValueError('Historical scientific table payload differs: ' + field)
            parity = True
        table_checks.append({'width': width, 'filename': path.name, 'historicalPayloadParity': parity,
                             'productSHA256': product_sha,
                             'acceptedProductByteMatch': product_match,
                             'productIdentity': product_identity})
    result = {'fixedConfigSHA256': expected, 'fixedInputAdopted': adopted,
              'fixedInputScope': 'adopted' if adopted else 'unverified_external',
              'fixedRootCount': 90, 'fixedRecipientLaws': 180,
              'tables': table_checks, 'newPolicyDraws': 0,
              'scope': 'no_draw_fixed_inputs_and_new_table_identity',
              'crossCPUorIFUNCEquivalenceEstablished': False,
              'historicalPolicyEquivalenceEstablished': False}
    (args.output/'summary.json').write_text(json.dumps(result, sort_keys=True, indent=2)+'\n')
    print(json.dumps(result, sort_keys=True))


if __name__ == '__main__':
    main()
