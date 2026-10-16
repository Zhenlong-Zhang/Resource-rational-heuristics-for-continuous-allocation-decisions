"""Prespecified one-candidate controls; no new task mathematics or simulation seeds."""

from __future__ import annotations

import copy
from dataclasses import asdict
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"src"))
from design_validation.bounded_task import expand_templates, file_sha256, load_design
from design_validation.revision import candidate_gate, load_revision, prior_comparison, run_revision, stage_spec

BASE = ROOT/"configs/tutoring_conditions_baseline.json"
PATCH = ROOT/"configs/tutoring_design_revision_v02.json"


class RevisionTests(unittest.TestCase):
    def test_exact_scientific_change_and_unchanged_base(self):
        before = file_sha256(BASE)
        base, revised, _ = load_revision(BASE, PATCH)
        for old, new in zip(expand_templates(base), expand_templates(revised)):
            a, b = asdict(old), asdict(new)
            if old.condition_id == "C3":
                self.assertEqual(b.pop("prior_centres"), (50*old.scale, 70*old.scale))
                a.pop("prior_centres")
            self.assertEqual(a, b)
        revised["conditions"][2]["prior_kernel_centres"][0] = 999
        self.assertEqual(base["conditions"][2]["prior_kernel_centres"], [60, 60])
        self.assertEqual(before, file_sha256(BASE))

    def test_reject_patch_drift_base_and_engine_hashes(self):
        original = json.loads(PATCH.read_text())
        for field, value in (("base_sha256", "bad"), ("engine_sha256", "bad"),
                             ("change", {**original["change"], "new_value": [45, 75]})):
            changed = copy.deepcopy(original)
            changed[field] = value
            with tempfile.TemporaryDirectory(prefix="revision-patch-") as directory:
                path = Path(directory)/"patch.json"
                path.write_text(json.dumps(changed))
                with self.assertRaises(ValueError):
                    load_revision(BASE, path)

    def test_resolved_loader_and_metadata(self):
        _, revised, _ = load_revision(BASE, PATCH)
        self.assertIn("pending", revised["status"])
        self.assertEqual(revised["conditions"][2]["initial_information"], "differentiated_broad_direct_priors")
        with tempfile.TemporaryDirectory(prefix="revision-resolved-") as directory:
            path = Path(directory)/"resolved.json"
            path.write_text(json.dumps(revised))
            self.assertEqual(load_design(path), revised)

    def test_fixed_stage_schedule(self):
        self.assertEqual(stage_spec("screening"), (30, 2026091901))
        self.assertEqual(stage_spec("confirmation"), (1200, 2026091902))
        for name in ("precision", "confirmation_retry", ""):
            with self.assertRaises(ValueError):
                stage_spec(name)

    def test_gate_boundaries_and_screening_not_selection(self):
        summary = [{"condition_id": "C3", "target_match": {"wilson_95_interval": [.8, 1]},
                    "EO_vs_split_diagnostic": {"wilson_95_interval": [.25, .5]},
                    "cap_binding": {"wilson_95_interval": [0, .009]},
                    "gain_vs_immediate_optimum": {"normal_approx_95_interval": [.001, 1]}}]
        diagnostics = [{"config": {"condition_id": "C3", "template_id": f"C3_s{i}_g0"},
                        "decision": {"gain_A": 1, "gain_B": .5}} for i in range(3)]
        self.assertTrue(candidate_gate(summary, diagnostics, "confirmation")["selected_under_prospective_rule"])
        self.assertIsNone(candidate_gate(summary, diagnostics, "screening")["selected_under_prospective_rule"])
        for field, interval, value in (("target_match", "wilson_95_interval", [.799999, 1]),
                                      ("EO_vs_split_diagnostic", "wilson_95_interval", [.249999, .5]),
                                      ("cap_binding", "wilson_95_interval", [0, .01]),
                                      ("gain_vs_immediate_optimum", "normal_approx_95_interval", [0, 1])):
            changed = copy.deepcopy(summary)
            changed[0][field][interval] = value
            self.assertFalse(candidate_gate(changed, diagnostics, "confirmation")["selected_under_prospective_rule"])
        diagnostics[0]["decision"] = {"gain_A": 1e-10, "gain_B": 0}
        self.assertFalse(candidate_gate(summary, diagnostics, "confirmation")["selected_under_prospective_rule"])

    def test_finite_prior_moments_are_not_kernel_centres(self):
        base, revised, _ = load_revision(BASE, PATCH)
        comparisons = prior_comparison(base, revised)
        self.assertEqual(len(comparisons), 3)
        for row in comparisons:
            old, new = row["v0.1"], row["v0.2"]
            self.assertAlmostEqual(sum(old["kernel_centres"]), sum(new["kernel_centres"]))
            self.assertNotAlmostEqual(old["actual_mean_sum"], new["actual_mean_sum"], places=3)
            self.assertNotAlmostEqual(old["actual_variance_sum"], new["actual_variance_sum"], places=3)
            self.assertLess(new["initial_EO_ES_gap"], .08)

    def test_import_and_cli_isolation(self):
        code = ("import sys;sys.path.insert(0,"+repr(str(ROOT/"src"))+");"
                "import design_validation.revision;assert 'src' not in sys.modules;"
                "assert not any(k.startswith(('src.experiments','src.mdp','src.policies')) for k in sys.modules)")
        subprocess.run([sys.executable, "-c", code], check=True, capture_output=True)
        result = subprocess.run([sys.executable, str(ROOT/"scripts/validate_tutoring_design_revision.py"),
                                 "--help"], check=True, capture_output=True, text=True)
        self.assertIn("--stage {screening,confirmation}", result.stdout)

    def test_no_overwrite_or_wrong_source_root(self):
        environment = {k: "1" for k in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
                                       "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS")}
        with tempfile.TemporaryDirectory(prefix="revision-no-overwrite-") as directory, patch.dict(os.environ, environment):
            with self.assertRaises(FileExistsError):
                run_revision(BASE, PATCH, Path(directory), "screening", ROOT)
            with self.assertRaises(ValueError):
                run_revision(BASE, PATCH, Path(directory), "screening", Path(directory))


if __name__ == "__main__":
    unittest.main()
