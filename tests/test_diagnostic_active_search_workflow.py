"""Test purpose: validate manifest, shard, provenance, and strict collection behavior for active-search diagnostics."""

from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from scripts.diagnostic_active_search_workflow import (
    COMPLETE_MARKER,
    FAILURE_FILENAME,
    PROVENANCE_FILENAME,
    REQUIRED_OUTPUTS,
    build_manifest,
    canonical_shard_name,
    combine_validated_outputs,
    completed_shard_provenance,
    progress_payload,
    run_manifest_shard,
    scientific_config,
    record_completed_shard,
    shard_provenance,
    validate_shards,
    write_json,
)


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


class ActiveSearchWorkflowTest(unittest.TestCase):
    def config(self, chunks: int, max_grid_points: int | None = None) -> dict[str, object]:
        grid_size = max_grid_points if max_grid_points is not None else 972
        return scientific_config(
            grid="active_search_benchmark",
            grid_size=grid_size,
            chunk_count=chunks,
            preset="server",
            episodes=1200,
            voi_samples=500,
            common_observations="on",
            observations_per_person=500,
            manual_active_samples_per_person=3,
            max_grid_points=max_grid_points,
        )

    def manifest(self, run_dir: Path, chunks: int = 2) -> dict[str, object]:
        return build_manifest(
            run_dir=run_dir,
            git_commit="test-scheduling-commit",
            baseline_commit="e92d64d",
            throttle=160,
            config=self.config(chunks, max_grid_points=chunks if chunks < 486 else None),
        )

    def populate_shard(self, manifest: dict[str, object], chunk_index: int) -> None:
        task = manifest["tasks"][chunk_index]
        output_dir = Path(task["output_dir"])
        write_csv(
            output_dir / REQUIRED_OUTPUTS[0],
            ["environment", "policy", "mean_utility"],
            [{"environment": f"env_{chunk_index}", "policy": "myopic_voi", "mean_utility": 1.0}],
        )
        write_csv(
            output_dir / REQUIRED_OUTPUTS[1],
            ["environment", "regime_grid", "grid_index"],
            [
                {
                    "environment": f"env_{grid_index}",
                    "regime_grid": "active_search_benchmark",
                    "grid_index": grid_index,
                }
                for grid_index in range(
                    chunk_index,
                    int(manifest["scientific_config"]["grid_size"]),
                    int(manifest["scientific_config"]["chunk_count"]),
                )
            ],
        )
        write_csv(output_dir / REQUIRED_OUTPUTS[2], ["environment", "candidate_type"], [])
        record_completed_shard(manifest, chunk_index)

    def test_full_manifest_has_unique_canonical_486_task_mapping(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            manifest = self.manifest(Path(temporary_dir), chunks=486)
        tasks = manifest["tasks"]
        self.assertEqual(len(tasks), 486)
        self.assertEqual(len({task["chunk_index"] for task in tasks}), 486)
        self.assertEqual(len({task["output_dir"] for task in tasks}), 486)
        expected = {
            0: "active_search_benchmark_chunk00_of486",
            9: "active_search_benchmark_chunk09_of486",
            99: "active_search_benchmark_chunk99_of486",
            100: "active_search_benchmark_chunk100_of486",
            485: "active_search_benchmark_chunk485_of486",
        }
        for index, shard in expected.items():
            self.assertEqual(tasks[index]["shard"], shard)
            self.assertEqual(canonical_shard_name("active_search_benchmark", index, 486), shard)

    def test_full_scientific_configuration_is_preserved(self) -> None:
        config = self.config(486)
        self.assertEqual(
            config,
            {
                "preset": "server",
                "section": "active_search_diagnostic",
                "regime_grid": "active_search_benchmark",
                "grid_size": 972,
                "chunk_count": 486,
                "episodes": 1200,
                "voi_samples": 500,
                "common_observations": "on",
                "observations_per_person": 500,
                "manual_active_samples_per_person": 3,
                "max_grid_points": None,
            },
        )

    def test_full_task_command_preserves_common_randomness_settings(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            manifest = self.manifest(Path(temporary_dir), chunks=486)
        command = manifest["tasks"][123]["scientific_command"]
        self.assertEqual(command[0], "scripts/generate_results.py")
        self.assertIn("--common-observations", command)
        self.assertEqual(command[command.index("--common-observations") + 1], "on")
        self.assertEqual(command[command.index("--episodes") + 1], "1200")
        self.assertEqual(command[command.index("--voi-samples") + 1], "500")
        self.assertEqual(command[command.index("--observations-per-person") + 1], "500")
        self.assertEqual(command[command.index("--regime-grid-chunk-index") + 1], "123")
        self.assertEqual(command[command.index("--regime-grid-chunks") + 1], "486")

    def test_full_modulo_shard_requires_both_expected_grid_indices(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            manifest = self.manifest(Path(temporary_dir), chunks=486)
            self.populate_shard(manifest, 100)
            result = validate_shards(manifest)
            self.assertIn(100, result.complete)

            summary_path = (
                Path(manifest["tasks"][100]["output_dir"])
                / "active_search_diagnostic_environment_summary.csv"
            )
            write_csv(
                summary_path,
                ["environment", "regime_grid", "grid_index"],
                [
                    {
                        "environment": "env_100",
                        "regime_grid": "active_search_benchmark",
                        "grid_index": 100,
                    }
                ],
            )
            result = validate_shards(manifest)
        self.assertIn(100, result.invalid)

    def test_submission_script_uses_throttled_one_slot_array(self) -> None:
        script = (
            Path(__file__).resolve().parents[1]
            / "scripts"
            / "submit_hoffman2_diagnostic_active_search.sh"
        ).read_text(encoding="utf-8")
        self.assertIn("#$ -t 1-${GRID_TASKS}", script)
        self.assertIn("#$ -tc ${MAX_CONCURRENT_TASKS}", script)
        self.assertNotIn("#$ -pe", script)
        self.assertIn("s/^Your job-array ([0-9]+).*/\\1/p", script)
        self.assertIn("s/^Your job ([0-9]+).*/\\1/p", script)
        self.assertNotIn("awk '/Your job-array/", script)
        self.assertIn("record-jobs \\\n  --manifest", script)
        self.assertNotIn("record-jobs \\\\\n", script)

    def test_missing_shard_fails_strict_validation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            manifest = self.manifest(Path(temporary_dir))
            self.populate_shard(manifest, 0)
            result = validate_shards(manifest)
            self.assertFalse(result.ok)
            self.assertEqual(result.complete, [0])
            self.assertEqual(result.missing, [1])

    def test_stale_provenance_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            manifest = self.manifest(Path(temporary_dir), chunks=1)
            self.populate_shard(manifest, 0)
            provenance_path = Path(manifest["tasks"][0]["output_dir"]) / PROVENANCE_FILENAME
            provenance = completed_shard_provenance(manifest, 0)
            provenance["git_commit"] = "stale-commit"
            write_json(provenance_path, provenance)
            result = validate_shards(manifest)
            self.assertFalse(result.ok)
            self.assertIn(0, result.invalid)

    def test_tampered_manifest_task_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            manifest = self.manifest(Path(temporary_dir))
            command = manifest["tasks"][0]["scientific_command"]
            command[command.index("--episodes") + 1] = "999"
            result = validate_shards(manifest)
        self.assertFalse(result.ok)
        self.assertIn("manifest task mapping or command mismatch", result.invalid[-1])

    def test_unaccepted_scientific_baseline_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            manifest = self.manifest(Path(temporary_dir))
            manifest["scientific_baseline_commit"] = "wrong-baseline"
            result = validate_shards(manifest)
        self.assertFalse(result.ok)
        self.assertIn("scientific baseline commit mismatch", result.invalid[-1])

    def test_strict_collection_combines_without_running_simulations(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            run_dir = Path(temporary_dir)
            manifest = self.manifest(run_dir)
            self.populate_shard(manifest, 0)
            self.populate_shard(manifest, 1)
            with patch("scripts.diagnostic_active_search_workflow.subprocess.run") as run_mock, patch(
                "scripts.generate_results.write_figures"
            ), patch(
                "scripts.diagnostic_active_search_workflow.current_git_commit",
                return_value="test-scheduling-commit",
            ):
                combine_validated_outputs(manifest)
            run_mock.assert_not_called()
            self.assertTrue((run_dir / COMPLETE_MARKER).is_file())
            with (run_dir / REQUIRED_OUTPUTS[1]).open(encoding="utf-8", newline="") as handle:
                self.assertEqual(len(list(csv.DictReader(handle))), 2)

    def test_retry_provenance_and_command_fingerprints_are_deterministic(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            first = self.manifest(Path(temporary_dir), chunks=2)
            second = self.manifest(Path(temporary_dir), chunks=2)
        self.assertEqual(first["scientific_config_fingerprint"], second["scientific_config_fingerprint"])
        self.assertEqual(
            first["tasks"][1]["scientific_command_fingerprint"],
            second["tasks"][1]["scientific_command_fingerprint"],
        )
        self.assertEqual(shard_provenance(first, 1), shard_provenance(second, 1))

    def test_valid_completed_shard_is_not_recomputed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            manifest = self.manifest(Path(temporary_dir), chunks=1)
            self.populate_shard(manifest, 0)
            with patch("scripts.diagnostic_active_search_workflow.subprocess.run") as run_mock, patch(
                "scripts.diagnostic_active_search_workflow.current_git_commit",
                return_value="test-scheduling-commit",
            ):
                returncode = run_manifest_shard(manifest, 0)
        self.assertEqual(returncode, 0)
        run_mock.assert_not_called()

    def test_nonzero_simulation_exit_records_failed_shard(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            manifest = self.manifest(Path(temporary_dir), chunks=1)
            with patch(
                "scripts.diagnostic_active_search_workflow.current_git_commit",
                return_value="test-scheduling-commit",
            ), patch(
                "scripts.diagnostic_active_search_workflow.subprocess.run",
                return_value=SimpleNamespace(returncode=3),
            ):
                returncode = run_manifest_shard(manifest, 0)
            failure_path = Path(manifest["tasks"][0]["output_dir"]) / FAILURE_FILENAME
            failure_exists = failure_path.is_file()
            result = validate_shards(manifest)
        self.assertEqual(returncode, 3)
        self.assertTrue(failure_exists)
        self.assertEqual(result.failed, [0])

    def test_zero_exit_with_missing_outputs_records_failed_shard(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            manifest = self.manifest(Path(temporary_dir), chunks=1)
            with patch(
                "scripts.diagnostic_active_search_workflow.current_git_commit",
                return_value="test-scheduling-commit",
            ), patch(
                "scripts.diagnostic_active_search_workflow.subprocess.run",
                return_value=SimpleNamespace(returncode=0),
            ):
                returncode = run_manifest_shard(manifest, 0)
            failure_path = Path(manifest["tasks"][0]["output_dir"]) / FAILURE_FILENAME
            failure_exists = failure_path.is_file()
            result = validate_shards(manifest)
        self.assertEqual(returncode, 70)
        self.assertTrue(failure_exists)
        self.assertEqual(result.failed, [0])

    def test_execution_checkout_must_match_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            manifest = self.manifest(Path(temporary_dir), chunks=1)
            with patch(
                "scripts.diagnostic_active_search_workflow.current_git_commit",
                return_value="different-commit",
            ), patch("scripts.diagnostic_active_search_workflow.subprocess.run") as run_mock:
                with self.assertRaisesRegex(RuntimeError, "does not match manifest"):
                    run_manifest_shard(manifest, 0)
        run_mock.assert_not_called()

    def test_progress_uses_validated_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            manifest = self.manifest(Path(temporary_dir))
            self.populate_shard(manifest, 0)
            with patch("scripts.diagnostic_active_search_workflow.query_scheduler_states", return_value={"status": "test"}):
                payload = progress_payload(manifest)
        self.assertEqual(payload["expected_shards"], 2)
        self.assertEqual(payload["complete_shards"], 1)
        self.assertEqual(payload["missing_shards"], [1])
        self.assertEqual(payload["percent_complete"], 50.0)


if __name__ == "__main__":
    unittest.main()
