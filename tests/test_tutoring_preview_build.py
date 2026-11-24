"""Check portable standalone packaging, immutable inputs and safe inline source."""

import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("preview_builder", ROOT / "scripts/build_tutoring_preview.py")
BUILDER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BUILDER)


class PreviewBuildTests(unittest.TestCase):
    def test_actual_cli_from_another_directory(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "preview"
            subprocess.run([sys.executable, str(ROOT / "scripts/build_tutoring_preview.py"),
                            "--output", str(output)], cwd=temp, check=True, capture_output=True)
            manifest = json.loads((output / "build_manifest.json").read_text())
            self.assertEqual(set(p.name for p in output.iterdir()),
                             {"index.html", "resolved_design.json", "build_manifest.json"})
            self.assertEqual(manifest["formalPolicyDraws"], 0)
            self.assertFalse(manifest["humanData"])
            self.assertFalse(manifest["qualtricsKit"])
            for name, digest in manifest["outputHashes"].items():
                self.assertEqual(BUILDER.sha256((output / name).read_bytes()), digest)
            html = (output / "index.html").read_text()
            self.assertIn('window.TUTORING_LOCAL_STRATEGY=true;', html)
            self.assertNotIn('Qualtrics.SurveyEngine', html)
            self.assertNotIn('<script src=', html)
            with self.assertRaises(FileExistsError):
                BUILDER.build(output)

    def _copy_inputs(self, root):
        for name in ("configs/tutoring_conditions.json", "configs/tutoring_preview.json",
                     "scripts/build_tutoring_preview.py"):
            target = root / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / name, target)
        shutil.copytree(ROOT / "src/tutoring_task", root / "src/tutoring_task")

    def test_changed_scientific_input_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self._copy_inputs(root)
            path = root / "configs/tutoring_conditions.json"
            path.write_text(path.read_text() + "\n")
            with self.assertRaisesRegex(ValueError, "Scientific input hash"):
                BUILDER.build(root / "out", root)
            self.assertFalse((root / "out").exists())

    def test_unsupported_profile_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self._copy_inputs(root)
            path = root / "configs/tutoring_preview.json"
            config = json.loads(path.read_text())
            config["profile_id"] = "legacy-4p30m-v1"
            path.write_text(json.dumps(config))
            with self.assertRaisesRegex(ValueError, "Unsupported standalone"):
                BUILDER.resolve_design(root)

    def test_script_and_style_terminators_rejected(self):
        for name, suffix in (("engine.js", "</script>"), ("ui.css", "</style>")):
            with self.subTest(name=name), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                self._copy_inputs(root)
                path = root / "src/tutoring_task" / name
                path.write_text(path.read_text() + suffix)
                with self.assertRaisesRegex(ValueError, "Unsafe inline"):
                    BUILDER.build(root / "out", root)
                self.assertFalse((root / "out").exists())


if __name__ == "__main__":
    unittest.main()
