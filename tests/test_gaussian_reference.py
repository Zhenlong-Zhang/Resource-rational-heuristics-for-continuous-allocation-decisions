"""Purpose: preserve historical Gaussian recipes and isolate their source safely."""

import io
from pathlib import Path
import tarfile
import tempfile
import unittest
from unittest.mock import patch

from scripts.reproduce_gaussian_reference import (
    REFERENCE_COMMIT, RECIPES, command, unpack_source, resolve_python,
)


class GaussianReferenceTests(unittest.TestCase):
    def test_full_recipes_preserve_scientific_settings(self):
        self.assertEqual(len(REFERENCE_COMMIT), 40)
        self.assertEqual(RECIPES, {
            "near_50_50": (120, 64), "equal_outcome": (120, 64),
            "equal_outcome_distinct_serious": (120, 32),
            "near_50_50_focused": (1200, 36), "equal_outcome_focused": (1200, 48),
            "equal_outcome_distinct_focused": (1200, 64),
        })
        for recipe, (episodes, chunks) in RECIPES.items():
            args = command(recipe, Path("source"), Path("output"), "python", 2)
            for flag, value in (("--episodes", episodes),
                                ("--regime-grid-chunks", chunks),
                                ("--voi-samples", 500),
                                ("--observations-per-person", 200)):
                self.assertEqual(args[args.index(flag) + 1], str(value))
            self.assertNotIn("--terminal-integration", args)

    def test_relative_interpreter_keeps_virtual_environment_path(self):
        with patch("shutil.which", return_value=".venv/bin/python"):
            self.assertEqual(resolve_python(".venv/bin/python"),
                             str(Path.cwd() / ".venv/bin/python"))
        with patch("shutil.which", return_value=None):
            with self.assertRaises(ValueError):
                resolve_python("missing-interpreter")

    def test_smoke_is_explicitly_reduced(self):
        args = command("equal_outcome_distinct_serious", Path("source"),
                       Path("output"), "python", 1, smoke=True)
        self.assertEqual(args[args.index("--episodes") + 1], "1")
        self.assertEqual(args[args.index("--regime-grid") + 1],
                         "equal_outcome_distinct_focused")
        self.assertIn("--max-regime-grid-points", args)
        with self.assertRaises(ValueError):
            command("near_50_50", Path("s"), Path("o"), "python", 0)

    @staticmethod
    def archive(name, kind=tarfile.REGTYPE):
        data = io.BytesIO()
        with tarfile.open(fileobj=data, mode="w") as handle:
            member = tarfile.TarInfo(name)
            member.type = kind
            member.linkname = "/outside" if kind == tarfile.SYMTYPE else ""
            handle.addfile(member)
        return data.getvalue()

    def test_regular_source_and_fresh_destination(self):
        with tempfile.TemporaryDirectory() as temporary:
            destination = Path(temporary) / "source"
            unpack_source(self.archive("src/__init__.py"), destination)
            self.assertTrue((destination / "src/__init__.py").is_file())
            with self.assertRaises(FileExistsError):
                unpack_source(self.archive("src/__init__.py"), destination)

    def test_rejects_links_escape_and_unexpected_files(self):
        for name, kind in (("../escape", tarfile.REGTYPE),
                           ("/absolute", tarfile.REGTYPE),
                           ("src/link", tarfile.SYMTYPE),
                           ("private.md", tarfile.REGTYPE)):
            with tempfile.TemporaryDirectory() as temporary:
                destination = Path(temporary) / "source"
                with self.assertRaises(ValueError):
                    unpack_source(self.archive(name, kind), destination)
                self.assertFalse(destination.exists())


if __name__ == "__main__":
    unittest.main()
