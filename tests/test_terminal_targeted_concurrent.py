"""Tests for the targeted concurrent terminal-validation entrypoint."""

from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUNNER = PROJECT_ROOT / "scripts" / "run_terminal_targeted_concurrent.py"


class TerminalTargetedConcurrentEntrypointTests(unittest.TestCase):
    def test_help_works_outside_project_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            completed = subprocess.run(
                [sys.executable, str(RUNNER), "--help"],
                cwd=directory,
                check=False,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("target", completed.stdout)


if __name__ == "__main__":
    unittest.main()
