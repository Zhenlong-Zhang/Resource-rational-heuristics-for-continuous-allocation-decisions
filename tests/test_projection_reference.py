"""Test purpose: isolate reference dispatch and reject ineffective settings.

These interface fixtures do not claim historical numerical parity; that is
checked on the full fixed predictor roster in the reference environment.
"""
import json
import os
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from src.design_validation import projection_reference as P


class ProjectionReferenceTests(unittest.TestCase):
    def tearDown(self):
        P.frozen_plans.cache_clear()

    def test_reference_is_isolated_and_cached(self):
        before = dict(os.environ)
        payload = dict(plans=[None], runtime=dict(mklRequested="AVX", mklEffective="avx"))
        with patch.object(P.subprocess, "run", return_value=SimpleNamespace(stdout=json.dumps(payload))) as run:
            self.assertEqual(P.frozen_plans(((0.,) * 10,)), payload)
            self.assertEqual(P.frozen_plans(((0.,) * 10,)), payload)
            self.assertEqual(run.call_count, 1)
            options = run.call_args.kwargs
            self.assertEqual(options['env']['MKL_CBWR'], 'AVX')
            self.assertEqual(options['env']['MKL_NUM_THREADS'], '1')
            self.assertEqual(json.loads(options['input']), [[0.] * 10])
            self.assertTrue(options['check'])
        self.assertEqual(dict(os.environ), before)

    def test_ineffective_reference_is_rejected(self):
        payload = dict(plans=[], runtime=dict(mklRequested="AVX", mklEffective="auto"))
        with patch.object(P.subprocess, "run", return_value=SimpleNamespace(stdout=json.dumps(payload))):
            with self.assertRaises(ValueError):
                P.frozen_plans(())
