"""Check capped reproduction scope without running formal simulation recipes."""

import ast
import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from design_validation.bounded_task import expand_templates, load_design


class CappedPublicationTests(unittest.TestCase):
    def test_frozen_designs_use_twenty_queries(self):
        for name in ("tutoring_conditions.json", "tutoring_conditions_baseline.json"):
            design = load_design(ROOT / "configs" / name)
            self.assertEqual(design["common"]["maximum_online_assessments"], 20)
            self.assertTrue(all(t.max_assessments == 20 for t in expand_templates(design)))
            invalid = copy.deepcopy(design)
            invalid["common"]["maximum_online_assessments"] = 21
            with tempfile.TemporaryDirectory() as folder:
                path = Path(folder) / "invalid.json"
                path.write_text(json.dumps(invalid))
                with self.assertRaises(ValueError):
                    load_design(path)

    def test_frozen_revision_dependencies_unchanged(self):
        patch = json.loads((ROOT / "configs/tutoring_design_revision_v02.json").read_text())
        for path, field in (("configs/tutoring_conditions_baseline.json", "base_sha256"),
                            ("src/design_validation/bounded_task.py", "engine_sha256")):
            self.assertEqual(hashlib.sha256((ROOT / path).read_bytes()).hexdigest(), patch[field])

    def test_notebooks_are_scoped_install_first_and_default_off(self):
        expected = {"finite_tutoring/design_validation.ipynb",
                    "statistical_methods/calibration_and_power.ipynb"}
        self.assertEqual({str(p.relative_to(ROOT / "notebooks"))
                          for p in (ROOT / "notebooks").rglob("*.ipynb")}, expected)
        for relative in expected:
            notebook = json.loads((ROOT / "notebooks" / relative).read_text())
            cells = [c for c in notebook["cells"] if c["cell_type"] == "code"]
            first = "".join(cells[0]["source"])
            self.assertIn("pip", first)
            self.assertIn("install", first)
            assignments = []
            for cell in cells:
                self.assertEqual(cell.get("outputs"), [])
                self.assertIsNone(cell.get("execution_count"))
                tree = ast.parse("".join(cell["source"]))
                assignments.extend(node for node in ast.walk(tree)
                                   if isinstance(node, ast.Assign)
                                   and any(isinstance(t, ast.Name) and t.id == "RUN"
                                           for t in node.targets))
            self.assertEqual(len(assignments), 1)
            self.assertIs(ast.literal_eval(assignments[0].value), False)

    def test_no_incompatible_workflow_entrypoints(self):
        for relative in ("src/experiments", "src/tutoring_statistics",
                         "src/design_validation/budget_horizon.py",
                         "src/design_validation/cost_aware_policy.py",
                         "src/design_validation/personal_time_policy.py",
                         "src/tutoring_analysis/native_calibration.py",
                         "src/tutoring_analysis/nominal_inference.py",
                         "src/tutoring_analysis/paired_recipe.py"):
            self.assertFalse((ROOT / relative).exists(), relative)


if __name__ == "__main__":
    unittest.main()
