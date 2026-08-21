"""Smoke checks for the public scheduler-free scarcity runner."""

from __future__ import annotations

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUNNER = PROJECT_ROOT / "scripts" / "run_scarcity_public.py"


class PublicScarcityRunnerTests(unittest.TestCase):
    def test_help_is_available_from_repository_root(self) -> None:
        result = subprocess.run(
            [sys.executable, str(RUNNER), "--help"],
            cwd=PROJECT_ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("Portable R6 scarcity runner", result.stdout)

    def test_tiny_smoke_writes_public_stage_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output_dir = Path(temporary) / "scarcity"
            result = subprocess.run(
                [
                    sys.executable,
                    str(RUNNER),
                    "--mode",
                    "smoke",
                    "--smoke-episodes",
                    "1",
                    "--smoke-object-descriptors",
                    "1",
                    "--oracle-grid-size",
                    "51",
                    "--output-dir",
                    str(output_dir),
                ],
                cwd=PROJECT_ROOT,
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            for name in (
                "scarcity_oracle_rows.csv",
                "scarcity_oracle_summary.csv",
                "scarcity_object_gate.json",
                "scarcity_selected_anchors.json",
                "scarcity_development_summary.csv",
                "scarcity_confirmation_summary.csv",
                "scarcity_run_metadata.json",
            ):
                self.assertTrue((output_dir / name).is_file(), name)
