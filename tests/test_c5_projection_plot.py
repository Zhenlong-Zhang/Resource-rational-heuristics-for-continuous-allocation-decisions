import csv
import copy
import hashlib
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.design_validation import c5_conditional_final as producer
from src.design_validation.c5_projection_plot import (
    conditional_rows, load_display_csv, plot_conditional_projection, plot_projection,
    load_verified_saved_summary, plot_verified_saved_summary,
)


FIELDS = ("target", "view", "estimate", "band_lower", "band_upper", "band_kind")


def example_rows():
    # Synthetic display values exercise the interface, not the accepted result.
    return [
        ("gap", "low_price_0.001", "-0.5", "-0.6", "-0.4", "simultaneous_95_percent"),
        ("gap", "high_price_0.01", "0", "", "", "analytic_exact"),
        ("gap", "high_minus_low", "0.5", "0.4", "0.6", "simultaneous_95_percent"),
        ("partial_low", "low_price_0.001", "0.3", "0.2", "0.4", "simultaneous_95_percent"),
        ("partial_low", "high_price_0.01", "0", "", "", "analytic_exact"),
        ("partial_low", "high_minus_low", "-0.3", "-0.4", "-0.2", "simultaneous_95_percent"),
    ]


def conditional_example():
    # Synthetic producer-shaped data, not a historical result.
    def target(low, interval, contrast, reflected):
        return {"lowPriceEstimate": low, "lowPriceInterval": interval,
                "highPricePopulationSlope": ["0", "1"],
                "highMinusLowEstimate": contrast, "highMinusLowInterval": reflected}
    return {
        "schema": "c5-conditional-final-v1",
        "targets": {
            "gap": target(["-1", "2"], [["-3", "5"], ["-2", "5"]],
                          ["1", "2"], [["2", "5"], ["3", "5"]]),
            "partial_low": target(["3", "10"], [["1", "5"], ["2", "5"]],
                                  ["-3", "10"], [["-2", "5"], ["-1", "5"]]),
        },
        "halfWidth": ["1", "10"], "familyAlpha": ["1", "20"], "inferenceLooks": 1,
        "orderedPopulationRowsSHA256": "a" * 64, "prefixContentSHA256": "b" * 64,
        "lookupSHA256": "c" * 64, "highLawRecordsSHA256": "d" * 64,
        "latentBytesSHA256": "e" * 64, "inputByteSHA256": {"lookup": "c" * 64},
        "sourceByteSHA256": {"source": "f" * 64}, "runtimeMetadataVerified": False,
        "historicalExactResultEstablished": False, "runtimeEquivalenceEstablished": False,
        "scope": "conditional_on_supplied_prefix_future_runtime_fixed_model_and_independence",
        "assumptions": [
            "The supplied prefix is bounded but not authenticated as historical.",
            "Future outcomes follow the fixed model in their actual execution environments.",
            "Root-by-replicate random variables are independent; key checks do not prove independence.",
            "High-price recipient draws are independent and follow the checked integer laws.",
        ],
    }


def saved_example():
    """Synthetic schema fixture, not the accepted historical summary."""
    low = {"gap": ["-1", "2"], "partial_low": ["3", "10"]}
    bands = {"gap": [["-3", "5"], ["-2", "5"]],
             "partial_low": [["1", "5"], ["2", "5"]]}
    return {"schema": "c5-fixed-repetition-final-plot-summary-v1", "sourceSHA256": "a" * 64,
            "scope": "supplemental finite-model population projection; fixed canonical target; retained bounded historical prefix; one final look",
            "simultaneousConfidence": "95% for the two precommitted low-price targets",
            "halfWidth": ["1", "10"], "conditionalAssumptions": [
                "recorded canonical numerical implementation", "full-support share semantics",
                "future root-by-repetition independence"],
            "originalExactFields": {"fixedReplicates": 1840, "roots": 165600,
                "nonTie": 165599, "exactTies": 1, "actualUsedEvents": 1,
                "estimates": low, "bands": bands,
                "highPrice": {"gap": ["0", "1"], "partial_low": ["0", "1"]},
                "highMinusLow": {"estimates": {"gap": ["1", "2"], "partial_low": ["-3", "10"]},
                    "bands": {"gap": [["2", "5"], ["3", "5"]],
                              "partial_low": [["-2", "5"], ["-1", "5"]]}},
                "keyDigest": "b" * 64, "seedDigest": "c" * 64},
            "extraction": "Original JSON strings and fields copied; no local coefficient, interval, or decimal calculation."}


class VerifiedSavedSummaryTests(unittest.TestCase):
    def load_fixture(self, value):
        raw = json.dumps(value).encode()
        with patch("src.design_validation.c5_projection_plot.Path.read_bytes", return_value=raw), \
             patch("src.design_validation.c5_projection_plot.ACCEPTED_PLOT_SUMMARY_SHA256",
                   hashlib.sha256(raw).hexdigest()):
            return load_verified_saved_summary("external.json")

    def test_refuses_changed_bytes(self):
        with patch("src.design_validation.c5_projection_plot.Path.read_bytes", return_value=b"{}"):
            with self.assertRaises(ValueError):
                load_verified_saved_summary("external.json")

    def test_schema_and_exact_arithmetic_refusals(self):
        for case in ("extra", "target", "fraction", "band", "contrast", "high", "scope", "counts"):
            value = saved_example()
            fields = value["originalExactFields"]
            if case == "extra": value["extra"] = True
            elif case == "target": del fields["estimates"]["gap"]
            elif case == "fraction": fields["estimates"]["gap"] = ["-2", "4"]
            elif case == "band": fields["bands"]["gap"].reverse()
            elif case == "contrast": fields["highMinusLow"]["estimates"]["gap"] = ["1", "3"]
            elif case == "high": fields["highPrice"]["gap"] = ["1", "1"]
            elif case == "scope": value["conditionalAssumptions"] = []
            else: fields["exactTies"] = 2
            with self.subTest(case=case), self.assertRaises(ValueError):
                self.load_fixture(value)

    def test_fixture_and_figure_labels(self):
        value = saved_example()
        raw = json.dumps(value).encode()
        with patch("src.design_validation.c5_projection_plot.Path.read_bytes", return_value=raw), \
             patch("src.design_validation.c5_projection_plot.ACCEPTED_PLOT_SUMMARY_SHA256",
                   hashlib.sha256(raw).hexdigest()):
            self.assertEqual(len(load_verified_saved_summary("external.json")), 6)
            fig, ax = plot_verified_saved_summary("external.json")
        self.addCleanup(__import__("matplotlib.pyplot", fromlist=["close"]).close, fig)
        self.assertIn("saved-result redraw", ax.get_title())
        self.assertTrue(any("not a fresh simulation" in t.get_text() for t in fig.texts))
        self.assertTrue(any("conditional" in t.get_text().lower() for t in fig.texts))
        labels = [t.get_text() for t in ax.get_yticklabels()]
        self.assertIn("Need gap: Low price (0.001)", labels)
        self.assertIn("Partial lower need: High minus low", labels)
        note = " ".join(t.get_text() for t in fig.texts)
        self.assertIn("share of remaining tutoring to higher true need", note)
        self.assertIn("predictors scaled by initial budget", note)
        self.assertIn("Finite-model population projection, not historical OLS or schedule SD", note)

    @unittest.skipUnless(os.environ.get("C5_ACCEPTED_PLOT_SUMMARY"), "external accepted summary not supplied")
    def test_external_accepted_summary(self):
        rows = load_verified_saved_summary(os.environ["C5_ACCEPTED_PLOT_SUMMARY"])
        self.assertEqual(len(rows), 6)
        self.assertEqual(sum(row.lower is not None for row in rows), 4)
        fig, ax = plot_verified_saved_summary(os.environ["C5_ACCEPTED_PLOT_SUMMARY"])
        self.addCleanup(__import__("matplotlib.pyplot", fromlist=["close"]).close, fig)
        self.assertEqual(len(ax.get_yticklabels()), 6)


class ProjectionPlotTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / "display.csv"

    def write_rows(self, rows):
        with self.path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(FIELDS)
            writer.writerows(rows)

    def test_six_rows_four_bands_and_draft_label(self):
        self.write_rows(example_rows())
        rows = load_display_csv(self.path)
        self.assertEqual(len(rows), 6)
        self.assertEqual(sum(row.lower is not None for row in rows), 4)
        fig, ax = plot_projection(self.path)
        self.addCleanup(__import__("matplotlib.pyplot", fromlist=["close"]).close, fig)
        self.assertIn("UNVERIFIED DRAFT", ax.get_title())
        self.assertEqual(len(ax.collections), 4)
        self.assertEqual(len(ax.get_yticklabels()), 6)

    def test_duplicate_missing_and_invalid_bands_rejected(self):
        rows = example_rows()
        for bad in (rows[:-1], rows + [rows[0]],
                    rows[:1] + [("gap", "high_price_0.01", "0", "0", "0", "analytic_exact")] + rows[2:],
                    [("gap", "low_price_0.001", "nan", "-1", "0", "simultaneous_95_percent")] + rows[1:]):
            with self.subTest(bad=bad):
                self.write_rows(bad)
                with self.assertRaises(ValueError):
                    load_display_csv(self.path)

    def test_self_declared_receipt_does_not_authenticate(self):
        self.write_rows(example_rows())
        fig, ax = plot_projection(self.path)
        self.addCleanup(__import__("matplotlib.pyplot", fromlist=["close"]).close, fig)
        self.assertIn("UNVERIFIED", ax.get_title())

    def test_conditional_exact_intervals_and_analytic_zero(self):
        result = conditional_example()
        rows = conditional_rows(result)
        self.assertEqual(len(rows), 6)
        self.assertEqual(sum(row.lower is not None for row in rows), 4)
        self.assertTrue(all(row.lower is None and row.upper is None
                            for row in rows if row.view == "high_price_0.01"))
        fig, ax = plot_conditional_projection(result)
        self.addCleanup(__import__("matplotlib.pyplot", fromlist=["close"]).close, fig)
        self.assertEqual(len(ax.collections), 4)
        self.assertIn("CONDITIONAL", ax.get_title())
        self.assertIn("NOT HISTORICAL EXACT", ax.get_title())
        self.assertTrue(any("UNVERIFIED INPUT" in text.get_text() for text in ax.texts))
        self.assertTrue(any("[-3/5, -2/5]" in text.get_text() for text in ax.texts))
        self.assertTrue(any("not authenticated" in text.get_text() for text in fig.texts))
        self.assertEqual(sum("analytic; no CI" in text.get_text() for text in ax.texts), 2)

    def test_actual_producer_contract_bridges_to_redraw(self):
        raw = b'{}'
        latent_sha = hashlib.sha256(raw).hexdigest()
        lookup_sha = producer.c5_public_authority.ADOPTED_INITIAL_SHA256
        prefix_sha = 'b' * 64
        audit = {'arithmetic': {'results': {
            'gap': {'lowPriceEstimate': ['-1', '2']},
            'partial_low': {'lowPriceEstimate': ['3', '10']}}},
            'orderedPopulationRowsSHA256': 'a' * 64, 'prefixContentSHA256': prefix_sha,
            'lookupSHA256': lookup_sha}
        proof = {'highPricePopulationSlopes': {'gap': ['0', '1'], 'partial_low': ['0', '1']},
                 'rootsSHA256': 'd' * 64}
        population = {'prefix_sha256': prefix_sha, 'contexts': [{'identity': {
            'inputs': {'lookup': lookup_sha, 'latent': latent_sha, 'prefix': prefix_sha},
            'sources': {'source': 'f' * 64}}}]}
        with tempfile.NamedTemporaryFile() as file:
            file.write(raw)
            file.flush()
            with patch.object(producer.c5_input_assembly, 'LATENT_BYTES_SHA256', latent_sha), \
                 patch.object(producer.c5_public_streams, 'ValidatedLatentConfig'), \
                 patch.object(producer.c5_public_inputs, 'load_fixed_inputs'), \
                 patch.object(producer.c5_public_reproduction, 'audit_population', return_value=audit), \
                 patch.object(producer.c5_high_price_proof, 'prove', return_value=proof), \
                 patch.object(producer, '_bind'):
                result = producer.conditional_final(
                    population, high_output='fixture', high_config='fixture', adopted_lookup='fixture',
                    latent_config=file.name, geometry_summary={}, geometry_certificate={})
        fig, ax = plot_conditional_projection(result)
        self.addCleanup(__import__('matplotlib.pyplot', fromlist=['close']).close, fig)
        self.assertEqual(len(ax.collections), 4)
        self.assertTrue(any('UNVERIFIED INPUT' in text.get_text() for text in ax.texts))
        self.assertTrue(any('actual execution environments' in text.get_text() for text in fig.texts))

    def test_conditional_invalid_contracts_rejected(self):
        cases = []
        for change in ("extra", "target", "zero", "interval", "contrast", "rational",
                       "historical", "runtime_metadata", "alpha", "digest", "assumptions"):
            bad = copy.deepcopy(conditional_example())
            if change == "extra":
                bad["extra"] = True
            elif change == "target":
                del bad["targets"]["gap"]
            elif change == "zero":
                bad["targets"]["gap"]["highPricePopulationSlope"] = ["1", "10"]
            elif change == "interval":
                bad["targets"]["gap"]["lowPriceInterval"][0] = ["-1", "2"]
            elif change == "contrast":
                bad["targets"]["gap"]["highMinusLowInterval"].reverse()
            elif change == "rational":
                bad["targets"]["gap"]["lowPriceEstimate"] = ["-2", "4"]
            elif change == "historical":
                bad["historicalExactResultEstablished"] = True
            elif change == "runtime_metadata":
                bad["runtimeMetadataVerified"] = True
            elif change == "alpha":
                bad["familyAlpha"] = ["1", "10"]
            elif change == "digest":
                bad["prefixContentSHA256"] = "not-a-hash"
            else:
                bad["assumptions"] = []
            cases.append((change, bad))
        for name, bad in cases:
            with self.subTest(case=name), self.assertRaises(ValueError):
                conditional_rows(bad)


if __name__ == "__main__":
    unittest.main()
