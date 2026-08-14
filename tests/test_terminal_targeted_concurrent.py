"""Tests for the targeted concurrent terminal-validation entrypoint."""

from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUNNER = PROJECT_ROOT / "scripts" / "run_terminal_targeted_concurrent.py"
SUBMITTER = PROJECT_ROOT / "scripts" / "submit_hoffman2_terminal_targeted_concurrent.sh"
AUDITOR = PROJECT_ROOT / "scripts" / "audit_terminal_targeted_concurrent.py"


class TerminalTargetedConcurrentEntrypointTests(unittest.TestCase):
    def test_help_works_outside_project_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            completed = subprocess.run(
                [sys.executable, str(RUNNER), "--help"],
                cwd=directory,
                check=False,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True,
            )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("target", completed.stdout)

    def test_submitter_and_auditor_share_the_24_hour_scheduler_contract(self):
        submitter = SUBMITTER.read_text(encoding="utf-8")
        auditor = AUDITOR.read_text(encoding="utf-8")
        self.assertIn("#$ -l h_rt=24:00:00", submitter)
        self.assertIn('"#$ -q campus2.q", "#$ -l h_rt=24:00:00"', auditor)
        self.assertNotIn("h_rt=06:00:00", submitter)
        self.assertNotIn("h_rt=06:00:00", auditor)


if __name__ == "__main__":
    unittest.main()
