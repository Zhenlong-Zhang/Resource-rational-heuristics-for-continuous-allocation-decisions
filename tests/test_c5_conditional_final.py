"""Software seam fixtures only; no fixture is accepted scientific evidence."""
import copy
import hashlib
import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from src.design_validation import c5_conditional_final as final


def fixture():
    pin = final.c5_public_authority.ADOPTED_INITIAL_SHA256
    width = 3.0
    report = final.c5_public_streams.ACCEPTED_REPORT_BYTES[width.hex()]
    inverse = [['0', '1'], ['1', '1']]
    design = {name: {'inverse': [[], inverse]} for name in ('gap', 'partial_low')}
    roots, records, fixed, latent_roots = [], [], [], []
    for index in range(90):
        key = ['C5', index // 10, index % 10 + 1]
        mapping = ['A', 'B']
        counts = [[1, 2], [1, 2]]
        fixed.append({'rootKey': key, 'rootSHA256': 'root', 'configSHA256': 'cfg',
                      'logicalToDisplayed': mapping, 'reportWidth': width,
                      'config': {'task': {'total_millihours': 1000}}})
        roots.append({'rootKey': key, 'rootSHA256': 'root', 'configSHA256': 'cfg',
                      'logicalToDisplayed': mapping, 'reportWidth': width, 'budget': 1000,
                      'latentConfigSHA256': 'latent', 'latentCounts': counts,
                      'reportProductSHA256': report,
                      'design': {name: inverse for name in ('gap', 'partial_low')}})
        records.append({'rootKey': key, 'logicalToDisplayed': mapping, 'budget': 1000,
                        'laws': [{'counts': x} for x in counts]})
        latent_roots.append(tuple((tuple(x), ()) for x in counts))
    audit = {'population': 165600, 'lookupSHA256': pin, 'rootBindings': roots}
    proof = {'adoptedLookupSHA256': pin, 'checkedLawRecords': records,
             'rootsSHA256': final.personal_time_inputs.digest(records)}
    return audit, proof, {'roots': fixed}, SimpleNamespace(input_sha256='latent', roots=latent_roots), design


class BindingTests(unittest.TestCase):
    def check(self, audit, proof, lookup, latent, design):
        with patch.object(final.c5_reference, 'design_reference', return_value=design), \
             patch.object(final.c5_moments, 'design_payload', return_value=design), \
             patch.object(final.c5_final_reference, 'check_supplied_geometry') as cert:
            final._bind(audit, proof, lookup, latent, {}, {})
            cert.assert_called_once()

    def test_rejects_modified_laws_even_if_proof_digest_is_rehashed(self):
        audit, proof, lookup, latent, design = fixture()
        self.check(audit, proof, lookup, latent, design)
        proof['checkedLawRecords'][0]['laws'][0]['counts'] = [2, 1]
        proof['rootsSHA256'] = final.personal_time_inputs.digest(proof['checkedLawRecords'])
        with self.assertRaisesRegex(ValueError, 'mapping mismatch'):
            self.check(audit, proof, lookup, latent, design)

    def test_rejects_changed_design_and_fixed_mapping(self):
        for field in ('design', 'logicalToDisplayed', 'reportProductSHA256', 'latentConfigSHA256'):
            audit, proof, lookup, latent, design = fixture()
            root = audit['rootBindings'][0]
            root[field] = {'gap': [], 'partial_low': []} if field == 'design' else 'changed'
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.check(audit, proof, lookup, latent, design)

    def test_independent_design_disagreement_is_refused(self):
        audit, proof, lookup, latent, design = fixture()
        with patch.object(final.c5_reference, 'design_reference', return_value=design), \
             patch.object(final.c5_moments, 'design_payload', return_value=copy.deepcopy({})):
            with self.assertRaisesRegex(ValueError, 'design reconstruction'):
                final._bind(audit, proof, lookup, latent, {}, {})

    def test_incomplete_roster_and_unpinned_lookup_refused(self):
        for mutation in ('roster', 'lookup'):
            audit, proof, lookup, latent, design = fixture()
            if mutation == 'roster':
                audit['rootBindings'].pop()
            else:
                audit['lookupSHA256'] = 'arbitrary'
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                self.check(audit, proof, lookup, latent, design)

    def test_conditional_intervals_and_reflections_use_exact_fractions(self):
        audit = {'arithmetic': {'results': {'gap': {'lowPriceEstimate': ['1', '4']},
                                             'partial_low': {'lowPriceEstimate': ['-1', '5']}}},
                 'orderedPopulationRowsSHA256': 'rows', 'prefixContentSHA256': 'prefix',
                 'lookupSHA256': 'lookup'}
        proof = {'highPricePopulationSlopes': {'gap': ['0', '1'], 'partial_low': ['0', '1']},
                 'rootsSHA256': 'laws'}
        raw = b'{}'
        raw_sha = hashlib.sha256(raw).hexdigest()
        population = {'prefix_sha256': 'prefix-bytes', 'contexts': [{'identity': {
            'inputs': {'lookup': final.c5_public_authority.ADOPTED_INITIAL_SHA256,
                       'latent': raw_sha, 'prefix': 'prefix-bytes'},
            'sources': {}, 'runtime': {'observation': {'python': 'original'}}}}]}
        def mutated_runtime(*args, **kwargs):
            population['contexts'][0]['identity']['runtime']['observation']['python'] = '/private/secret-token'
            return proof
        with tempfile.NamedTemporaryFile() as file:
            file.write(raw)
            file.flush()
            with patch.object(final.c5_input_assembly, 'LATENT_BYTES_SHA256', raw_sha), \
                 patch.object(final.c5_public_streams, 'ValidatedLatentConfig'), \
                 patch.object(final.c5_public_inputs, 'load_fixed_inputs'), \
                 patch.object(final.c5_public_reproduction, 'audit_population', return_value=audit), \
                 patch.object(final.c5_high_price_proof, 'prove', side_effect=mutated_runtime), \
                 patch.object(final, '_bind'):
                result = final.conditional_final(population, high_output='fixture', high_config='fixture',
                                                 adopted_lookup='fixture', latent_config=file.name,
                                                 geometry_summary={}, geometry_certificate={})
        self.assertEqual(result['targets']['gap']['lowPriceInterval'], [['3', '20'], ['7', '20']])
        self.assertEqual(result['targets']['gap']['highMinusLowInterval'], [['-7', '20'], ['-3', '20']])
        self.assertEqual(result['targets']['partial_low']['highMinusLowInterval'], [['1', '10'], ['3', '10']])
        self.assertFalse(result['historicalExactResultEstablished'])
        self.assertFalse(result['runtimeEquivalenceEstablished'])
        self.assertFalse(result['runtimeMetadataVerified'])
        self.assertNotIn('callerSuppliedRuntimeMetadata', result)
        self.assertNotIn('runtimeIdentity', result)
        self.assertNotIn('/private/secret-token', json.dumps(result))
        population['contexts'][0]['identity']['runtime']['observation']['python'] = 'later mutation'
        self.assertNotIn('later mutation', json.dumps(result))


@unittest.skipUnless(os.environ.get('C5_CONDITIONAL_ACTUAL_REPLAY') == '1',
                     'explicit scheduled full-population replay opt-in required')
class ActualConditionalReplayTests(unittest.TestCase):
    """Uses existing complete outputs; does not generate or rerun a campaign.

    Run only in an appropriately registered scheduled job. The source/inputs
    must be read-only, the 165510 future checkpoints already present, and the
    complete 90-root high-price producer and independent stages available.
    Environment variables below supply the actual paths and input byte pins.
    This full independent replay is computationally substantial.
    """
    def test_real_complete_conditional_path(self):
        env = os.environ
        repository = Path(env['C5_REPLAY_REPOSITORY']).resolve(strict=True)
        def read_json(name):
            return json.loads(Path(env[name]).read_text(encoding='utf-8'))
        population = final.c5_public_reproduction.load_population(
            env['C5_REPLAY_MANIFEST'], env['C5_REPLAY_MANIFEST_SHA256'],
            env['C5_REPLAY_PREFIX'], env['C5_REPLAY_PREFIX_SHA256'], repository,
            runtime=read_json('C5_REPLAY_RUNTIME_JSON'), input_mode=env['C5_REPLAY_INPUT_MODE'])
        shards = json.loads(env['C5_REPLAY_SHARD_DIRS_JSON'])
        self.assertIsInstance(shards, list)
        self.assertTrue(shards)
        result = final.conditional_final(
            population, high_output=env['C5_REPLAY_HIGH_OUTPUT'],
            high_config=env['C5_REPLAY_HIGH_CONFIG'], adopted_lookup=env['C5_REPLAY_LOOKUP'],
            latent_config=env['C5_REPLAY_LATENT'], geometry_summary=read_json('C5_REPLAY_GEOMETRY_SUMMARY'),
            geometry_certificate=read_json('C5_REPLAY_GEOMETRY_CERTIFICATE'),
            policy=env.get('C5_REPLAY_HIGH_POLICY') or None, shard_dirs=shards,
            temp_dir=env.get('C5_REPLAY_TEMP_DIR') or None)
        self.assertFalse(result['historicalExactResultEstablished'])
        self.assertFalse(result['runtimeEquivalenceEstablished'])
        self.assertFalse(result['runtimeMetadataVerified'])
        self.assertEqual(result['halfWidth'], ['1', '10'])
        for target in ('gap', 'partial_low'):
            row = result['targets'][target]
            lo, hi = map(final.c5_final.rational, row['lowPriceInterval'])
            self.assertEqual(row['highPricePopulationSlope'], ['0', '1'])
            self.assertEqual(row['highMinusLowInterval'],
                             [final.c5_final.encode(-hi), final.c5_final.encode(-lo)])


if __name__ == '__main__':
    unittest.main()
