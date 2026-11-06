"""Purpose: verify saved-input admission, source rebasing and safe CLI defaults."""
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from tutoring_statistics import common_t_reproduction as run, reproduction as io


class CommonTReproductionTests(unittest.TestCase):
    def test_original_roster_selection(self):
        self.assertEqual(run.selected([0, 10000]), [0, 10000])
        self.assertEqual(len(run.selected(all_ids=True)), 12129)
        for ids in ([0, 0], [10000, 0], [-1], [12129], [True], []):
            with self.subTest(ids=ids), self.assertRaises(ValueError):
                run.selected(ids)
        with self.assertRaises(ValueError):
            run.selected([0], True)

    def test_source_archive_is_an_explicit_byte_equivalent_location(self):
        with tempfile.TemporaryDirectory() as root:
            archive = Path(root) / 'preserved'
            sources = []
            for name in ('within_projection.py', 'within_reproduction.py'):
                path = archive / 'src/tutoring_statistics' / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text('# saved source\n')
                sources.append(dict(io.pin(path), path='/old/checkout/src/tutoring_statistics/' + name))
            bound = run.source_bindings(sources, archive)
            self.assertEqual([r['original'] for r in bound], sources)
            self.assertTrue(all(r['observed']['path'].startswith(str(archive)) for r in bound))
            (archive / 'src/tutoring_statistics/within_projection.py').write_text('# changed\n')
            with self.assertRaises(ValueError):
                run.source_bindings(sources, archive)

    def test_source_roster_rejects_duplicates_and_mixed_roots(self):
        with tempfile.TemporaryDirectory() as root:
            paths = [Path(root) / 'src/tutoring_statistics' / n
                     for n in ('within_projection.py', 'within_reproduction.py')]
            for path in paths:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text('# immutable\n')
            sources = [io.pin(path) for path in paths]
            with self.assertRaises(ValueError):
                run.source_bindings(sources + sources[:1])
            mixed = [sources[0], dict(sources[1], path='/foreign/src/tutoring_statistics/within_reproduction.py')]
            with self.assertRaises(ValueError):
                run.source_bindings(mixed, root)
            escaped = [dict(sources[0], path='/old/src/tutoring_statistics/../within_projection.py')]
            with self.assertRaises(ValueError):
                run.source_bindings(escaped, root)

    def test_tampered_artifact_is_not_admitted(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / 'saved.json'
            pin = io.write(path, {'ordinal': 0})
            self.assertEqual(run.read_pin(pin), {'ordinal': 0})
            path.write_text('{"ordinal": 1}\n')
            with self.assertRaises(ValueError):
                run.read_pin(pin)

    def test_cli_plan_and_help_do_not_generate_inputs(self):
        script = Path(__file__).resolve().parents[1] / 'scripts/reproduce_common_t.py'
        with tempfile.TemporaryDirectory() as root:
            out = Path(root) / 'must_not_exist'
            for args in (['--help'], ['plan', '--output', str(out)]):
                result = subprocess.run([sys.executable, str(script), *args], capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertFalse(out.exists())
            result = subprocess.run([sys.executable, str(script), 'produce', '--output', str(out), '--ids', '0'],
                                    capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertFalse(out.exists())

    def test_output_cannot_overlap_coverage(self):
        with tempfile.TemporaryDirectory() as root:
            with self.assertRaises(ValueError):
                run.prepare(root, Path(root) / 'new')


if __name__ == '__main__':
    unittest.main()
