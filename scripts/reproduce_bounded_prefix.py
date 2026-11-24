"""Certify bounded-prefix planning from explicit saved design geometry."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from design_validation import c5_precision_planning as P
from design_validation import c5_prefix_reference as V
from design_validation import c5_precision_reproduction as R
from design_validation import c5_precision_reference as OLD


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True, help='Checked public preparation directory or saved operands')
    parser.add_argument('--input-kind', choices=('public-preparation', 'saved-designs'), default='public-preparation')
    parser.add_argument('--configs', type=Path, default=ROOT / 'configs')
    parser.add_argument('--output', type=Path, required=True, help='New output JSON')
    args = parser.parse_args()
    if args.input_kind == 'saved-designs':
        designs, summary, evidence = R.load_operands(args.input)
    else:
        designs, summary, evidence = R.load_preparation(args.input, args.configs)
    plan = P.calculations(designs, summary)
    OLD.verify_plan(plan, designs, summary)
    if [row['model'] for row in plan['rows']] != list(P.MODELS):
        raise ValueError('Exactly two ordered targets required')
    results, geometries = [], []
    for row in plan['rows']:
        radii = [P.fraction(x) for x in row['centeredRootRadii']]
        variance = P.fraction(row['centeredVarianceSumUpper'])
        maximum = P.fraction(row['centeredMaxRange'])
        geometries.append((radii, variance, maximum))
        certificate = P.bounded_prefix_certificate(radii, variance, maximum)
        proof = V.verify(radii, variance, maximum, certificate['totalReplicates'])
        results.append({'model': row['model'], 'certificate': certificate, 'independent': proof})
    family = max(r['certificate']['totalReplicates'] for r in results)
    family_checks = []
    for row, geometry in zip(results, geometries):
        current, previous, previous_upper = V.family_margins(*geometry, family)
        previous_status = ('pass' if previous_upper is not None and previous_upper >= 0 else
                           'fail' if previous is None or previous < 0 else 'unresolved')
        family_checks.append({'model': row['model'], 'atNPasses': current is not None and current >= 0,
                              'atPreviousStatus': previous_status})
    if not all(x['atNPasses'] for x in family_checks) or not any(x['atPreviousStatus'] == 'fail' for x in family_checks):
        raise ValueError('Independent family certificate failed')
    payload = {'targets': results, 'familyN': family, 'historicalPrefixReplicates': 1,
               'futureCanonicalReplicates': family-1, 'familyCertificates': family_checks,
               'sourceScope': evidence['scope'], 'sourceEvidence': evidence,
               'historicalSamePolicyReference': '268_for_the_accepted_geometry_not_a_claim_about_other_inputs',
               'interpretation': 'conditional_design_only_not_achieved_precision',
               'futureIndependence': 'root_by_replicate_conditional_on_fixed_prefix_and_policy',
               'policyDraws': 0}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as output:
        json.dump(payload, output, indent=2, sort_keys=True)
        output.write('\n')
    print(json.dumps({'familyN': family, 'output': str(args.output)}))


if __name__ == '__main__':
    main()
