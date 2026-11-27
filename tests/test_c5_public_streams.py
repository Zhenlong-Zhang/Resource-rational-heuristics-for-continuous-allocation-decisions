import copy
import json
import math
from pathlib import Path
import unittest
from unittest.mock import patch

from src.design_validation import c5_public_streams as streams
from src.design_validation import canonical_report
from src.design_validation import personal_time_sensitivity_streams as original


class C5StreamsTest(unittest.TestCase):
    def test_private_key_exact_fields_and_domain(self):
        key = streams.private_key(8, 10, 1840, 'B', 'report', 4)
        self.assertEqual(key, {
            'streamVersion': original.STREAM, 'populationId': streams.POPULATION,
            'seed': 20261019, 'randomizationGroup': streams.GROUP, 'conditionId': 'C5',
            'scheduleVariant': 8, 'replicateId': 1840, 'mainIndex': 10,
            'logicalRecipient': 'B', 'streamKind': 'report', 'reportIndex': 4})
        for args in [(0, 1, 0, 'A', 'latent', 0), (0, 1, 1841, 'A', 'latent', 0),
                     (0, 1, 1, 'A', 'latent', 1), (True, 1, 1, 'A', 'latent', 0)]:
            with self.assertRaises(ValueError):
                streams.private_key(*args)

    def test_integer_partition_and_grid_refusals(self):
        law = {'denominator': streams.GRID, 'counts': [2, 0] + [0] * 97 + [streams.GRID - 2]}
        self.assertEqual(streams.accepted_index(law, 1), 1)
        self.assertEqual(streams.accepted_index(law, 2), 100)
        self.assertEqual(streams.accepted_uniform(law, 2 / streams.GRID), 100)
        for value in [-1, streams.GRID, True]:
            with self.assertRaises(ValueError):
                streams.accepted_index(law, value)
        for value in [1.0, -0.1, 2.0 ** -54]:
            with self.assertRaises(ValueError):
                streams.accepted_uniform(law, value)
        bad = copy.deepcopy(law)
        bad['counts'][0] = -1
        with self.assertRaises(ValueError):
            streams.accepted_index(bad, 0)

    def test_terminal_strict_comparison_and_fallback(self):
        self.assertEqual(streams.choose_terminal([0.25, 0.75], 0.25), 1)
        tail = math.nextafter(1.0, 0.0)
        self.assertEqual(streams.choose_terminal([0.5, tail - 0.5, 0.0], tail), 2)
        for probabilities, uniform in [([], 0.0), ([0.0], 0.0), ([float('nan')], 0.0),
                                       ([-0.1, 1.0], 0.0), ([1.1], 0.0),
                                       ([0.25, 0.25], 0.9),
                                       ([1.0], float('nan')), ([1.0], 1.0),
                                       ([1.0], -0.1), ([True], 0.0)]:
            with self.assertRaises(ValueError):
                streams.choose_terminal(probabilities, uniform)

    def test_private_key_rejects_modified_identity(self):
        key = streams.private_key(0, 1, 1, 'A', 'latent', 0)
        key['seed'] = 1
        with self.assertRaises(ValueError):
            streams.private_draw(key, {})

    def test_key_type_identity_rejected(self):
        for field in ('seed', 'scheduleVariant', 'replicateId', 'mainIndex', 'reportIndex'):
            key = streams.private_key(0, 1, 1, 'A', 'latent', 0)
            key[field] = float(key[field])
            with self.assertRaises(ValueError):
                streams.private_draw(key, {})

    @patch.object(streams, 'GROUP', 'c5-software-test-private-v1')
    def test_public_fixed_config_and_coordinate_binding(self):
        path = Path(__file__).resolve().parents[1] / 'configs/canonical_c5_latent.json'
        config = json.loads(path.read_text())
        key = streams.private_key(8, 10, 1, 'B', 'latent', 0)
        validated = streams.ValidatedLatentConfig(config)
        selected = streams.latent_partition(validated, key)
        self.assertEqual(selected['counts'], tuple(config['roots'][89]['recipientLaws'][1]['counts']))
        config['roots'][89]['recipientLaws'][1]['counts'][0] = -1
        with patch.object(streams.canonical_latent_reference, 'verify_historical_cdf',
                          side_effect=AssertionError('unexpected repeat validation')):
            event = streams.private_draw(key, validated)
            self.assertEqual(streams.latent_partition(validated, key), selected)
        self.assertEqual(event['schema'], 'r14-c5-fixed-latent-partition-draw-v1')
        self.assertNotIn('logHex', event)
        self.assertNotIn('probabilityHex', event)
        self.assertNotIn('rawMassHex', event)
        for mutate in ('roster', 'other_law', 'denominator'):
            bad = json.loads(path.read_text())
            if mutate == 'roster':
                bad['roots'][0]['rootKey'][1] = 0.0
            elif mutate == 'other_law':
                bad['roots'][0]['recipientLaws'][0]['counts'][0] = -1
            else:
                bad['denominator'] = float(streams.GRID)
            with self.assertRaises(ValueError):
                streams.ValidatedLatentConfig(bad)
        for bad_key in (dict(key, seed=20261019.0), dict(key, logicalRecipient='C'),
                        dict(key, streamKind='report'), dict(key, unexpected=1)):
            with self.assertRaises(ValueError):
                streams.latent_partition(validated, bad_key)
        with self.assertRaises((AttributeError, TypeError)):
            validated.roots = ()
        with self.assertRaises(TypeError):
            selected['counts'][0] = 0
        with self.assertRaises(ValueError):
            streams.private_draw(key, config)

    def test_report_draw_explicitly_unavailable(self):
        key = streams.private_key(0, 1, 1, 'A', 'report', 0)
        with self.assertRaisesRegex(ValueError, 'Report draws unavailable'):
            streams.private_draw(key, {})

    @patch.object(streams, 'GROUP', 'c5-software-test-private-v1')
    def test_verified_report_table_and_latent_binding(self):
        path = Path(__file__).resolve().parents[1] / 'configs/canonical_c5_latent.json'
        config = streams.ValidatedLatentConfig(json.loads(path.read_text()))
        width = '0x1.8000000000000p+1'
        product = canonical_report.generate_report_table(width)
        raw = (json.dumps(product, sort_keys=True, separators=(',', ':')) + '\n').encode()
        table = streams.ValidatedReportTable(raw, width)
        self.assertEqual(table.accepted_product_bytes,
                         table.product_sha256 == streams.ACCEPTED_REPORT_BYTES[width])
        alternative = streams.ValidatedReportTable(json.dumps(product, indent=2).encode(), width)
        self.assertFalse(alternative.accepted_product_bytes)
        self.assertEqual(alternative.rows, table.rows)
        latent_key = streams.private_key(2, 3, 4, 'A', 'latent', 0)
        latent = streams.private_draw(latent_key, config)
        report_key = streams.private_key(2, 3, 4, 'A', 'report', 1)
        with patch.object(streams.canonical_report_reference, 'verify_report_table',
                          side_effect=AssertionError('unexpected repeat validation')):
            event = streams.report_draw(report_key, table, width, latent, config)
        self.assertEqual(event['latentCategory'], latent['category'])
        self.assertEqual(event['tableProductSHA256'], table.product_sha256)
        self.assertEqual(event['acceptedProductByteMatch'], table.accepted_product_bytes)
        self.assertFalse(event['historicalEventEstablished'])
        for bad_key in (streams.private_key(2, 3, 4, 'B', 'report', 1),
                        streams.private_key(2, 4, 4, 'A', 'report', 1),
                        streams.private_key(2, 3, 5, 'A', 'report', 1)):
            with self.assertRaises(ValueError):
                streams.report_draw(bad_key, table, width, latent, config)
        altered = dict(latent, category=1.0)
        with self.assertRaises(ValueError):
            streams.report_draw(report_key, table, width, altered, config)
        with self.assertRaises(ValueError):
            streams.report_draw(report_key, table, '0x1.0000000000000p+0', latent, config)
        product['rows'][0]['counts'][0] += 1
        with self.assertRaises(ValueError):
            streams.ValidatedReportTable(json.dumps(product).encode(), width)
        with self.assertRaises((AttributeError, TypeError)):
            table.rows = ()

    def test_terminal_fixed_identity(self):
        snapshot = {'candidateId': 'model', 'task': {'condition_id': 'C5'}}
        identity = {'seed': 20261019, 'modelId': 'model', 'personId': 'p', 'trialId': 't',
                    'conditionId': 'C5', 'variant': 0, 'replicateId': 1, 'mainIndex': 1}
        with patch.object(original.A, 'validate'):
            streams.terminal_key(snapshot, identity)
            for field, value in [('seed', 1), ('conditionId', 'C4'), ('replicateId', 0),
                                 ('replicateId', 1841), ('variant', 9)]:
                bad = dict(identity, **{field: value})
                with self.assertRaises(ValueError):
                    streams.terminal_key(snapshot, bad)
