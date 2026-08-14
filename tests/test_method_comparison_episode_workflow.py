"""Test purpose: validate paired episode tasks, locking, recovery, and collection for method comparisons."""

from __future__ import annotations

import csv
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path
from unittest import mock

from scripts.method_comparison_episode_workflow import (
    COMPLETION_MARKER,
    EPISODE_FILENAME,
    ExclusiveLock,
    MANIFEST_FILENAME,
    METADATA_FILENAME,
    SUMMARY_FILENAME,
    attempt_provenance,
    build_stage_evidence,
    build_staged_complete_view,
    certify_smoke,
    current_git_commit,
    file_sha256,
    fingerprint,
    freeze_checkpoint,
    load_manifest,
    progress_payload,
    promote_staged_view,
    recover_promotion,
    reclaim_stale_lock,
    record_scheduler_smoke_evidence,
    require_execution_checkout,
    run_missing_shard_negative_check,
    run_manifest_task,
    scientific_row,
    set_tree_writable,
    tree_fingerprint,
    validate_task_metadata,
    expected_identity,
    validate_shards,
    write_csv_atomic,
    write_json_atomic,
    verify_smoke_gate,
)
from scripts.run_method_comparison_task import (
    EPISODE_FIELDNAMES,
    policy_from_task_metadata,
    run_method_episode_row,
    run_single_episode_from_task_metadata,
    write_summary_from_task_metadata,
)
from src.experiments.randomization import (
    build_evaluation_episode,
    build_evaluation_episodes,
)
from src.mdp.meta_mdp import EnvironmentConfig


def string_row(row: dict[str, object]) -> dict[str, str]:
    return {field: str(row.get(field, "")) for field in EPISODE_FIELDNAMES}


class MethodComparisonEpisodeWorkflowTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.canonical = self.root / "canonical"
        self.workflow = self.root / "workflow"
        self.task_manifest = self.root / "tasks.tsv"
        self.writer_evidence = self.root / "writer_evidence.json"
        self.scheduler_evidence = self.root / "scheduler_evidence.json"
        self.environment = "fixture_environment"
        self.policy_label = "myopic_voi_samples2"
        self.config = EnvironmentConfig(
            mu_need=8.0,
            sigma_need=2.0,
            sigma_sample=1.5,
            total_time=5.0,
            allocation_grid_size=7,
            expected_utility_draws=7,
            random_seed=23,
        )
        self.metadata = {
            "script": "run_method_comparison_task.py",
            "metadata_version": 1,
            "preset": "fixture",
            "environment": self.environment,
            "policy_arg": "myopic_voi",
            "policy": {
                "name": self.policy_label,
                "class": "MyopicValueOfInformationPolicy",
                "observation_draws": 2,
                "horizon": "",
                "max_samples": "",
                "mean_grid_size": "",
                "observation_branches": "",
            },
            "settings": {
                "n_episodes": 3,
                "rr_observation_draws": 2,
                "blinkered_observation_draws": 2,
                "blinkered_horizon": 2,
                "use_common_observation_streams": True,
                "observations_per_person": 10,
            },
            "environment_config": asdict(self.config),
        }
        self._write_fixture(completed_indices=(0, 1))

    def tearDown(self) -> None:
        if self.root.exists():
            for snapshot in self.root.rglob("snapshot/canonical"):
                set_tree_writable(snapshot)
        self.temporary.cleanup()

    def _task_dir(self, root: Path | None = None) -> Path:
        base = self.canonical if root is None else root
        return base / "tasks" / "methods" / self.environment / self.policy_label

    def _write_fixture(self, completed_indices: tuple[int, ...]) -> None:
        task_dir = self._task_dir()
        task_dir.mkdir(parents=True, exist_ok=True)
        write_json_atomic(task_dir / METADATA_FILENAME, self.metadata)
        rows = [
            string_row(run_single_episode_from_task_metadata(self.metadata, index))
            for index in completed_indices
        ]
        write_csv_atomic(task_dir / EPISODE_FILENAME, EPISODE_FIELDNAMES, rows)
        write_summary_from_task_metadata(task_dir / SUMMARY_FILENAME, self.metadata, rows)
        self.task_manifest.write_text(
            f"{self.environment}\t{self.metadata['policy_arg']}\t"
            f"{self.metadata['policy']['name']}\t\t\t\n",
            encoding="utf-8",
        )
        self._write_writer_evidence()

    def _file_record(self, path: Path) -> dict[str, object]:
        return {
            "path": str(path.resolve()),
            "size": path.stat().st_size,
            "sha256": file_sha256(path),
        }

    def _write_writer_evidence(self, final_qstat_text: str = "header\nheader\n") -> None:
        evidence_dir = self.root / "writer_scheduler"
        evidence_dir.mkdir(exist_ok=True)
        before = evidence_dir / "qstat_before.txt"
        after = evidence_dir / "qstat_after.txt"
        final = evidence_dir / "qstat_final.txt"
        before.write_text("header\nheader\n", encoding="utf-8")
        after.write_text("header\nheader\n", encoding="utf-8")
        final.write_text(final_qstat_text, encoding="utf-8")
        records = []
        for job_id in ("old-array", "old-checker"):
            qacct = evidence_dir / f"qacct_{job_id}.txt"
            qacct.write_text(f"jobnumber {job_id}\nexit_status 0\n", encoding="utf-8")
            records.append(
                {
                    "job_id": job_id,
                    "can_write": False,
                    "scheduler_disposition": "absent_from_final_qstat_and_present_in_qacct",
                    "qacct": self._file_record(qacct),
                }
            )
        payload = {
            "writers_quiescent": True,
            "accounted_job_ids": ["old-array", "old-checker"],
            "writer_job_name_pattern": "rr_method_comparison_m",
            "qstat_before": self._file_record(before),
            "qstat_after": self._file_record(after),
            "qstat_final": self._file_record(final),
            "jobs": records,
            "successors": [],
        }
        payload["evidence_fingerprint"] = fingerprint(payload)
        write_json_atomic(self.writer_evidence, payload)

    def _write_scheduler_evidence(self, manifest: dict[str, object]) -> None:
        qstat = self.root / "smoke_qstat.txt"
        qstat.write_text("header\nheader\n", encoding="utf-8")
        qacct_specs = []
        lane_job_ids = ["101", "102"]
        for task_id, lane_job_id in zip((1, 2), lane_job_ids):
            qacct = self.root / f"smoke_qacct_{task_id}.txt"
            qacct.write_text(
                f"jobnumber {lane_job_id}\ntaskid 1\n"
                "slots 1\nexit_status 0\ngranted_pe NONE\n",
                encoding="utf-8",
            )
            qacct_specs.append(f"{task_id}:{lane_job_id}.1:{qacct}")
        collector_qacct = self.root / "smoke_collector_qacct.txt"
        collector_qacct.write_text(
            "jobnumber 103\ntaskid undefined\n"
            "slots 1\nexit_status 0\ngranted_pe NONE\n",
            encoding="utf-8",
        )
        record_scheduler_smoke_evidence(
            manifest,
            qstat,
            qacct_specs,
            lane_job_ids,
            "103",
            collector_qacct,
            self.scheduler_evidence,
        )

    def test_scheduler_evidence_rejects_forged_qacct_identity(self) -> None:
        manifest = self._freeze()
        qstat = self.root / "forged_qstat.txt"
        qstat.write_text("header\nheader\n", encoding="utf-8")
        forged_lane = self.root / "forged_lane_qacct.txt"
        forged_lane.write_text(
            "jobnumber 999999\ntaskid 777\n"
            "slots 1\nexit_status 0\ngranted_pe NONE\n",
            encoding="utf-8",
        )
        collector = self.root / "collector_qacct.txt"
        collector.write_text(
            "jobnumber 102\ntaskid undefined\n"
            "slots 1\nexit_status 0\ngranted_pe NONE\n",
            encoding="utf-8",
        )
        with self.assertRaisesRegex(RuntimeError, "jobnumber/taskid"):
            record_scheduler_smoke_evidence(
                manifest,
                qstat,
                [f"1:101.1:{forged_lane}", f"2:101.2:{forged_lane}"],
                ["101"],
                "102",
                collector,
                self.scheduler_evidence,
            )

        valid_specs = []
        for task_id in (1,):
            lane = self.root / f"valid_lane_{task_id}.txt"
            lane.write_text(
                f"jobnumber 101\ntaskid {task_id}\n"
                "slots 1\nexit_status 0\ngranted_pe NONE\n",
                encoding="utf-8",
            )
            valid_specs.append(f"{task_id}:101.{task_id}:{lane}")
        forged_collector = self.root / "forged_collector_qacct.txt"
        forged_collector.write_text(
            "jobnumber 999999\ntaskid undefined\n"
            "slots 1\nexit_status 0\ngranted_pe NONE\n",
            encoding="utf-8",
        )
        with self.assertRaisesRegex(RuntimeError, "Collector qacct jobnumber"):
            record_scheduler_smoke_evidence(
                manifest,
                qstat,
                valid_specs,
                ["101"],
                "102",
                forged_collector,
                self.scheduler_evidence,
            )

    def test_scheduler_evidence_rejects_cross_lane_task_substitution(self) -> None:
        shutil.rmtree(self.canonical)
        self._write_empty_current_fixture()
        manifest = self._freeze(lanes=2)
        qstat = self.root / "lane_swap_qstat.txt"
        qstat.write_text("header\nheader\n", encoding="utf-8")
        specs = []
        for manifest_task_id, lane_job_id in ((2, "101"), (1, "102")):
            qacct = self.root / f"lane_swap_{lane_job_id}.txt"
            qacct.write_text(
                f"jobnumber {lane_job_id}\ntaskid 1\n"
                "slots 1\nexit_status 0\ngranted_pe NONE\n",
                encoding="utf-8",
            )
            specs.append(f"{manifest_task_id}:{lane_job_id}.1:{qacct}")
        collector = self.root / "lane_swap_collector.txt"
        collector.write_text(
            "jobnumber 103\ntaskid undefined\n"
            "slots 1\nexit_status 0\ngranted_pe NONE\n",
            encoding="utf-8",
        )
        with self.assertRaisesRegex(RuntimeError, "slot does not map"):
            record_scheduler_smoke_evidence(
                manifest,
                qstat,
                specs,
                ["101", "102"],
                "103",
                collector,
                self.scheduler_evidence,
            )

    def _freeze(
        self,
        *,
        lanes: int = 2,
        throttle: int = 80,
        run_mode: str = "test",
        reviewed_commit: str = "",
        enforce_clean_checkout: bool = False,
        serial_reference: Path | None = None,
    ) -> dict[str, object]:
        return freeze_checkpoint(
            canonical_run_dir=self.canonical,
            workflow_run_dir=self.workflow,
            task_manifest_path=self.task_manifest,
            writer_evidence_path=self.writer_evidence,
            required_writer_job_ids=["old-array", "old-checker"],
            git_commit=current_git_commit(),
            array_lanes=lanes,
            lane_throttle=throttle,
            expected_task_count=1,
            run_mode=run_mode,
            reviewed_commit=reviewed_commit,
            enforce_clean_checkout=enforce_clean_checkout,
            serial_reference_path=serial_reference,
        )

    def _run_all_tasks(self, manifest: dict[str, object]) -> None:
        for task in manifest["tasks"]:
            run_manifest_task(manifest, int(task["task_id"]), f"attempt_{task['task_id']}")

    def _synthetic_episode_row(
        self,
        metadata: dict[str, object],
        episode_index: int,
        *,
        elapsed_seconds: str = "0.0",
    ) -> dict[str, str]:
        identity = expected_identity(metadata, episode_index)
        row = {field: "" for field in EPISODE_FIELDNAMES}
        row.update(
            {
                "environment": str(metadata["environment"]),
                "episode_index": str(episode_index),
                "policy": str(metadata["policy"]["name"]),
                "policy_observation_draws": "250",
                "policy_horizon": "2",
                "true_need_1": str(identity["true_need_1"]),
                "true_need_2": str(identity["true_need_2"]),
                "observation_stream_hash_1": str(identity["observation_stream_hash_1"]),
                "observation_stream_hash_2": str(identity["observation_stream_hash_2"]),
                "episode_fingerprint": str(identity["episode_fingerprint"]),
                "realized_utility": "0.0",
                "sample_count": "0",
                "sample_1_count": "0",
                "sample_2_count": "0",
                "terminated": "1.0",
                "action_sequence": "terminate",
                "allocation_to_person1": "0.5",
                "elapsed_seconds": elapsed_seconds,
            }
        )
        return row

    def _write_synthetic_attempt(
        self, manifest: dict[str, object], task_id: int, attempt_id: str
    ) -> Path:
        task = next(item for item in manifest["tasks"] if int(item["task_id"]) == task_id)
        metadata = json.loads(Path(task["metadata_path"]).read_text(encoding="utf-8"))
        row = self._synthetic_episode_row(metadata, int(task["episode_index"]))
        attempt = Path(task["attempts_dir"]) / attempt_id
        attempt.mkdir(parents=True)
        write_csv_atomic(attempt / EPISODE_FILENAME, EPISODE_FIELDNAMES, [row])
        provenance = attempt_provenance(manifest, task, attempt_id, row)
        provenance["episode_csv_sha256"] = file_sha256(attempt / EPISODE_FILENAME)
        write_json_atomic(attempt / "attempt_provenance.json", provenance)
        return attempt

    def _write_empty_current_fixture(self) -> None:
        task_dir = self._task_dir()
        task_dir.mkdir(parents=True, exist_ok=True)
        write_json_atomic(task_dir / METADATA_FILENAME, self.metadata)
        write_csv_atomic(task_dir / EPISODE_FILENAME, EPISODE_FIELDNAMES, [])
        (task_dir / SUMMARY_FILENAME).write_text("placeholder\n", encoding="utf-8")
        self.task_manifest.write_text(
            f"{self.environment}\t{self.metadata['policy_arg']}\t"
            f"{self.metadata['policy']['name']}\t\t\t\n",
            encoding="utf-8",
        )
        self._write_writer_evidence()

    def _configure_smoke_metadata(self) -> None:
        self.policy_label = "blinkered_samples250"
        self.metadata["policy_arg"] = "blinkered"
        self.metadata["policy"] = {
            "name": self.policy_label,
            "class": "BlinkeredPolicy",
            "observation_draws": 250,
            "horizon": 2,
            "max_samples": "",
            "mean_grid_size": "",
            "observation_branches": "",
        }
        self.metadata["settings"].update(
            {
                "rr_observation_draws": 500,
                "blinkered_observation_draws": 250,
                "blinkered_horizon": 2,
                "use_common_observation_streams": True,
                "observations_per_person": 500,
            }
        )

    def _write_pre_freeze_serial_reference(
        self,
        path: Path,
        updates: dict[int, dict[str, str]] | None = None,
    ) -> list[dict[str, str]]:
        rows = []
        for episode_index in range(int(self.metadata["settings"]["n_episodes"])):
            row = self._synthetic_episode_row(
                self.metadata,
                episode_index,
                elapsed_seconds=f"independent-serial-runtime-{episode_index}",
            )
            row.update((updates or {}).get(episode_index, {}))
            rows.append(row)
        write_csv_atomic(path, EPISODE_FIELDNAMES, rows)
        return rows

    def _prepare_certifiable_smoke(
        self,
        reference_updates: dict[int, dict[str, str]] | None = None,
    ) -> tuple[dict[str, object], Path, str, dict[str, object]]:
        shutil.rmtree(self.canonical)
        self._configure_smoke_metadata()
        self._write_empty_current_fixture()
        serial_reference = self.root / "independent_serial_reference.csv"
        self._write_pre_freeze_serial_reference(serial_reference, reference_updates)
        commit = current_git_commit()
        manifest = self._freeze(
            run_mode="smoke",
            reviewed_commit=commit,
            enforce_clean_checkout=False,
            serial_reference=serial_reference,
        )
        for task in manifest["tasks"]:
            self._write_synthetic_attempt(
                manifest, int(task["task_id"]), f"attempt_{task['task_id']}"
            )
        self._write_synthetic_attempt(manifest, 1, "retry_scheduler")
        build_staged_complete_view(manifest)
        negative = self.root / "negative.json"
        negative_payload = run_missing_shard_negative_check(manifest, 2, negative)
        self._write_scheduler_evidence(manifest)
        return manifest, negative, commit, negative_payload

    def test_indexed_episode_is_identical_to_batch_episode(self) -> None:
        batch = build_evaluation_episodes(
            self.config,
            n_episodes=3,
            include_observation_streams=True,
            observations_per_person=10,
        )
        indexed = build_evaluation_episode(
            self.config,
            episode_index=2,
            include_observation_streams=True,
            observations_per_person=10,
        )
        self.assertEqual(batch[2], indexed)

        batch_row = run_method_episode_row(
            environment=self.environment,
            config=self.config,
            policy=policy_from_task_metadata(self.metadata),
            episode=batch[2],
            use_common_observation_streams=True,
        )
        indexed_row = run_single_episode_from_task_metadata(self.metadata, 2)
        self.assertEqual(scientific_row(string_row(batch_row)), scientific_row(string_row(indexed_row)))

    def test_freeze_builds_exact_disjoint_lane_manifest(self) -> None:
        manifest = self._freeze(lanes=4, throttle=100)
        self.assertEqual(len(manifest["tasks"]), 1)
        self.assertEqual(manifest["tasks"][0]["episode_index"], 2)
        lane_ids = [task_id for lane in manifest["lanes"] for task_id in lane["task_ids"]]
        self.assertEqual(lane_ids, [1])
        self.assertEqual(len(lane_ids), len(set(lane_ids)))
        self.assertEqual(manifest["lane_throttle"], 100)
        self.assertEqual(load_manifest(self.workflow / MANIFEST_FILENAME), manifest)

    def test_complete_checkpoint_is_validated_no_op(self) -> None:
        shutil.rmtree(self.canonical)
        self._write_fixture(completed_indices=(0, 1, 2))
        manifest = self._freeze()
        self.assertEqual(manifest["tasks"], [])
        self.assertEqual(progress_payload(manifest)["validated_percent_complete"], 100.0)

    def test_freeze_rejects_unaccounted_writer_and_duplicate_checkpoint(self) -> None:
        evidence = json.loads(self.writer_evidence.read_text(encoding="utf-8"))
        evidence["accounted_job_ids"] = ["old-array"]
        evidence.pop("evidence_fingerprint")
        evidence["evidence_fingerprint"] = fingerprint(evidence)
        write_json_atomic(self.writer_evidence, evidence)
        with self.assertRaisesRegex(RuntimeError, "does not account"):
            self._freeze()

        shutil.rmtree(self.workflow, ignore_errors=True)
        self._write_writer_evidence()
        episode_path = self._task_dir() / EPISODE_FILENAME
        with episode_path.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        with episode_path.open("a", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=EPISODE_FIELDNAMES)
            writer.writerow(rows[0])
        with self.assertRaisesRegex(RuntimeError, "Duplicate episode index"):
            self._freeze()

    def test_retries_are_isolated_and_conflicts_fail(self) -> None:
        manifest = self._freeze()
        first = run_manifest_task(manifest, 1, "scheduler_A")
        second = run_manifest_task(manifest, 1, "scheduler_B")
        validation = validate_shards(manifest)
        self.assertTrue(validation.ok)
        self.assertEqual(validation.selected_attempts[1], "scheduler_A")

        fields, rows = self._read_episode(second / EPISODE_FILENAME)
        rows[0]["realized_utility"] = str(float(rows[0]["realized_utility"]) + 1.0)
        write_csv_atomic(second / EPISODE_FILENAME, fields, rows)
        task = manifest["tasks"][0]
        provenance = attempt_provenance(manifest, task, "scheduler_B", rows[0])
        provenance["episode_csv_sha256"] = file_sha256(second / EPISODE_FILENAME)
        write_json_atomic(second / "attempt_provenance.json", provenance)
        validation = validate_shards(manifest)
        self.assertFalse(validation.ok)
        self.assertEqual(validation.duplicates[1], ["scheduler_A", "scheduler_B"])
        self.assertNotEqual(first, second)

    @staticmethod
    def _read_episode(path: Path) -> tuple[list[str], list[dict[str, str]]]:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            return list(reader.fieldnames or []), list(reader)

    def test_strict_stage_preserves_frozen_rows_and_runs_global_combiner_on_stage(self) -> None:
        manifest = self._freeze()
        original_hash = tree_fingerprint(self.canonical)
        self._run_all_tasks(manifest)
        frozen_fields, frozen_rows = self._read_episode(
            Path(manifest["snapshot_root"])
            / manifest["affected_tasks"][0]["relative_dir"]
            / EPISODE_FILENAME
        )
        self.assertEqual(frozen_fields, EPISODE_FIELDNAMES)
        with mock.patch(
            "scripts.method_comparison_episode_workflow.run_single_episode_from_task_metadata",
            side_effect=AssertionError("collector must not simulate"),
        ):
            stage = build_staged_complete_view(manifest)
        self.assertEqual(tree_fingerprint(self.canonical), original_hash)
        _, staged_rows = self._read_episode(self._task_dir(stage) / EPISODE_FILENAME)
        self.assertEqual(staged_rows[:2], frozen_rows)
        status = json.loads((self.workflow / "stage_validation.json").read_text(encoding="utf-8"))
        command = status["global_combiner_command"]
        combined_input = Path(command[command.index("--input-dir") + 1])
        combined_output = Path(command[command.index("--output-dir") + 1])
        self.assertEqual(combined_input, combined_output)
        self.assertTrue(combined_input.resolve().is_relative_to(self.workflow.resolve()))
        self.assertNotEqual(combined_input.resolve(), self.canonical.resolve())

    def test_stage_evidence_fingerprint_survives_double_digit_task_ids(self) -> None:
        inventory = [{"path": "result.csv", "size": 12, "sha256": "abc"}]
        evidence = build_stage_evidence(
            "manifest-fingerprint",
            ["combine", "--require-complete"],
            inventory,
            {task_id: f"attempt-{task_id}" for task_id in range(1, 12)},
        )
        path = self.root / "stage_evidence.json"
        write_json_atomic(path, evidence)
        persisted = json.loads(path.read_text(encoding="utf-8"))
        payload = {
            key: value
            for key, value in persisted.items()
            if key != "evidence_fingerprint"
        }

        self.assertEqual(persisted["evidence_fingerprint"], fingerprint(payload))
        self.assertEqual(persisted["selected_attempts"]["10"], "attempt-10")

    def test_promotion_crash_recovers_original_and_retry_is_idempotent(self) -> None:
        manifest = self._freeze()
        original_hash = tree_fingerprint(self.canonical)
        self._run_all_tasks(manifest)
        for boundary in (
            "prepared",
            "original_renamed",
            "original_moved",
            "stage_renamed",
            "stage_promoted",
            "marker_written",
        ):
            with self.subTest(boundary=boundary):
                build_staged_complete_view(manifest)
                with self.assertRaisesRegex(RuntimeError, "Injected failure"):
                    promote_staged_view(manifest, fail_after=boundary)
                self.assertEqual(recover_promotion(manifest), "rolled_back")
                self.assertEqual(tree_fingerprint(self.canonical), original_hash)
                self.assertFalse((self.canonical / COMPLETION_MARKER).exists())

        build_staged_complete_view(manifest)
        marker = promote_staged_view(manifest)
        self.assertTrue(marker.is_file())
        self.assertEqual(promote_staged_view(manifest), marker)

    def test_smoke_gate_requires_two_shards_retry_stage_and_negative_evidence(self) -> None:
        manifest, negative, commit, negative_payload = self._prepare_certifiable_smoke()
        self.assertTrue(negative_payload["collector_failed_nonzero"])
        self.assertTrue(negative_payload["canonical_hash_unchanged"])
        self.assertTrue(negative_payload["isolated_shard_restored"])
        serial_reference = Path(manifest["serial_reference"]["path"])
        _, serial_rows = self._read_episode(serial_reference)
        self.assertEqual(manifest["serial_reference"]["size"], serial_reference.stat().st_size)
        self.assertEqual(manifest["serial_reference"]["sha256"], file_sha256(serial_reference))
        self.assertEqual(len(manifest["serial_reference"]["target_rows"]), 3)
        self.assertEqual(
            manifest["serial_reference"]["target_rows_fingerprint"],
            fingerprint(manifest["serial_reference"]["target_rows"]),
        )
        gate = self.root / "smoke_gate.json"
        payload = certify_smoke(manifest, negative, self.scheduler_evidence, gate)
        self.assertTrue(payload["smoke_passed"])
        self.assertTrue(payload["retry_determinism_passed"])
        self.assertTrue(payload["negative_case_passed"])
        self.assertTrue(payload["serial_reference_parity_passed"])
        self.assertEqual(payload["serial_reference_path"], str(serial_reference.resolve()))
        self.assertEqual(
            payload["serial_reference_parity_evidence"]["matched_shard_count"], 3
        )
        self.assertEqual(
            payload["serial_reference_parity_evidence"]["compared_fields"],
            [field for field in EPISODE_FIELDNAMES if field != "elapsed_seconds"],
        )
        self.assertTrue(gate.is_file())
        self.assertEqual(
            verify_smoke_gate(gate, commit)["manifest_fingerprint"],
            manifest["manifest_fingerprint"],
        )
        forged = self.root / "forged_gate.json"
        write_json_atomic(
            forged,
            {
                "smoke_passed": True,
                "retry_determinism_passed": True,
                "negative_case_passed": True,
            },
        )
        with self.assertRaisesRegex(RuntimeError, "fingerprint"):
            verify_smoke_gate(forged, commit)
        forged_parity = json.loads(gate.read_text(encoding="utf-8"))
        parity = forged_parity["serial_reference_parity_evidence"]
        parity["matched_shard_count"] = 2
        parity.pop("evidence_fingerprint")
        parity["evidence_fingerprint"] = fingerprint(parity)
        forged_parity.pop("gate_fingerprint")
        forged_parity["gate_fingerprint"] = fingerprint(forged_parity)
        forged_parity_path = self.root / "forged_parity_gate.json"
        write_json_atomic(forged_parity_path, forged_parity)
        with self.assertRaisesRegex(RuntimeError, "parity evidence mismatch"):
            verify_smoke_gate(forged_parity_path, commit)
        serial_rows[0]["realized_utility"] = "tampered-after-certification"
        write_csv_atomic(serial_reference, EPISODE_FIELDNAMES, serial_rows)
        with self.assertRaisesRegex(RuntimeError, "serial reference changed"):
            verify_smoke_gate(gate, commit)
        self._write_pre_freeze_serial_reference(serial_reference)
        verify_smoke_gate(gate, commit)
        scheduler_payload = json.loads(self.scheduler_evidence.read_text(encoding="utf-8"))
        qacct_path = Path(scheduler_payload["qacct_records"][0]["file"]["path"])
        qacct_path.write_text("slots 8\nexit_status 0\n", encoding="utf-8")
        with self.assertRaisesRegex(RuntimeError, "qacct"):
            verify_smoke_gate(gate, commit)

    def test_synthetic_smoke_rows_cannot_certify_against_different_serial_reference(
        self,
    ) -> None:
        manifest, negative, _, _ = self._prepare_certifiable_smoke(
            reference_updates={0: {"realized_utility": "not-the-synthetic-result"}}
        )
        with self.assertRaisesRegex(RuntimeError, "Serial reference mismatch.*realized_utility"):
            certify_smoke(
                manifest,
                negative,
                self.scheduler_evidence,
                self.root / "must_not_certify.json",
            )
        self.assertFalse((self.root / "must_not_certify.json").exists())

    def test_smoke_freeze_requires_external_serial_reference(self) -> None:
        shutil.rmtree(self.canonical)
        self._configure_smoke_metadata()
        self._write_empty_current_fixture()
        commit = current_git_commit()
        with self.assertRaisesRegex(RuntimeError, "requires a serial-reference CSV"):
            self._freeze(
                run_mode="smoke",
                reviewed_commit=commit,
                enforce_clean_checkout=False,
            )
        self.assertFalse(self.workflow.exists())

    def test_smoke_freeze_rejects_in_tree_serial_references_after_symlink_resolution(
        self,
    ) -> None:
        shutil.rmtree(self.canonical)
        self._configure_smoke_metadata()
        self._write_empty_current_fixture()
        commit = current_git_commit()
        canonical_reference = self.canonical / "serial_reference.csv"
        self._write_pre_freeze_serial_reference(canonical_reference)
        symlink_reference = self.root / "serial_reference_link.csv"
        symlink_reference.symlink_to(canonical_reference)
        with self.assertRaisesRegex(RuntimeError, "outside canonical and workflow"):
            self._freeze(
                run_mode="smoke",
                reviewed_commit=commit,
                enforce_clean_checkout=False,
                serial_reference=symlink_reference,
            )

        workflow_reference = self.workflow / "serial_reference.csv"
        self._write_pre_freeze_serial_reference(workflow_reference)
        with self.assertRaisesRegex(RuntimeError, "outside canonical and workflow"):
            self._freeze(
                run_mode="smoke",
                reviewed_commit=commit,
                enforce_clean_checkout=False,
                serial_reference=workflow_reference,
            )

    def test_smoke_manifest_rejects_post_freeze_serial_reference_tampering(self) -> None:
        shutil.rmtree(self.canonical)
        self._configure_smoke_metadata()
        self._write_empty_current_fixture()
        serial_reference = self.root / "serial_reference_before_freeze.csv"
        rows = self._write_pre_freeze_serial_reference(serial_reference)
        commit = current_git_commit()
        self._freeze(
            run_mode="smoke",
            reviewed_commit=commit,
            enforce_clean_checkout=False,
            serial_reference=serial_reference,
        )
        rows[0]["realized_utility"] = "tampered-after-freeze"
        write_csv_atomic(serial_reference, EPISODE_FIELDNAMES, rows)
        with self.assertRaisesRegex(RuntimeError, "serial reference changed"):
            load_manifest(self.workflow / MANIFEST_FILENAME)

    def test_missing_or_tampered_shard_cannot_stage_or_promote(self) -> None:
        manifest = self._freeze()
        original_hash = tree_fingerprint(self.canonical)
        with self.assertRaisesRegex(RuntimeError, "Strict episode collection refused"):
            build_staged_complete_view(manifest)
        self.assertEqual(tree_fingerprint(self.canonical), original_hash)
        self.assertFalse((self.canonical / COMPLETION_MARKER).exists())

        attempt = run_manifest_task(manifest, 1, "tampered")
        provenance = json.loads((attempt / "attempt_provenance.json").read_text(encoding="utf-8"))
        provenance["git_commit"] = "wrong"
        write_json_atomic(attempt / "attempt_provenance.json", provenance)
        self.assertIn(1, validate_shards(manifest).invalid)

    def test_snapshot_tamper_and_second_collector_are_rejected(self) -> None:
        manifest = self._freeze()
        lock_path = self.workflow / "collector.lock"
        with ExclusiveLock(lock_path):
            with self.assertRaisesRegex(RuntimeError, "already held"):
                with ExclusiveLock(lock_path):
                    pass

        snapshot_episode = (
            Path(manifest["snapshot_root"])
            / manifest["affected_tasks"][0]["relative_dir"]
            / EPISODE_FILENAME
        )
        snapshot_episode.chmod(snapshot_episode.stat().st_mode | 0o200)
        with snapshot_episode.open("a", encoding="utf-8") as handle:
            handle.write("tamper\n")
        self._run_all_tasks(manifest)
        with self.assertRaisesRegex(RuntimeError, "snapshot inventory mismatch"):
            build_staged_complete_view(manifest)

    def test_committed_promotion_revalidates_marker(self) -> None:
        manifest = self._freeze()
        self._run_all_tasks(manifest)
        build_staged_complete_view(manifest)
        marker = promote_staged_view(manifest)
        marker.write_text("tampered\n", encoding="utf-8")
        with self.assertRaisesRegex(RuntimeError, "marker is missing or invalid"):
            promote_staged_view(manifest)

    def test_lane_limits_are_enforced(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "array_lanes"):
            self._freeze(lanes=5, throttle=80)
        with self.assertRaisesRegex(RuntimeError, "lane_throttle"):
            self._freeze(lanes=1, throttle=101)

    def test_production_settings_and_dirty_checkout_are_rejected(self) -> None:
        metadata = json.loads(json.dumps(self.metadata))
        metadata["settings"].update(
            {
                "n_episodes": 1200,
                "rr_observation_draws": 500,
                "blinkered_observation_draws": 250,
                "blinkered_horizon": 2,
                "use_common_observation_streams": True,
                "observations_per_person": 500,
            }
        )
        metadata["policy"]["observation_draws"] = 500
        manifest_row = {
            "environment": self.environment,
            "policy_arg": "myopic_voi",
            "policy": self.policy_label,
        }
        validate_task_metadata(metadata, manifest_row, run_mode="production")
        drifts = {
            "n_episodes": 1199,
            "rr_observation_draws": 499,
            "blinkered_observation_draws": 249,
            "blinkered_horizon": 3,
            "use_common_observation_streams": False,
            "observations_per_person": 499,
        }
        for name, value in drifts.items():
            with self.subTest(setting=name):
                drifted = json.loads(json.dumps(metadata))
                drifted["settings"][name] = value
                with self.assertRaisesRegex(RuntimeError, name):
                    validate_task_metadata(drifted, manifest_row, run_mode="production")
        blinkered = json.loads(json.dumps(metadata))
        blinkered["policy_arg"] = "blinkered"
        blinkered["policy"].update(
            {
                "name": "blinkered_samples250",
                "class": "BlinkeredPolicy",
                "observation_draws": 249,
                "horizon": 2,
            }
        )
        blinkered_row = dict(manifest_row)
        blinkered_row.update(
            {"policy_arg": "blinkered", "policy": "blinkered_samples250"}
        )
        with self.assertRaisesRegex(RuntimeError, "250 observation"):
            validate_task_metadata(blinkered, blinkered_row, run_mode="production")

        manifest = self._freeze()
        dirty_manifest = dict(manifest)
        dirty_manifest["enforce_clean_checkout"] = True
        with mock.patch(
            "scripts.method_comparison_episode_workflow.tracked_checkout_is_clean",
            return_value=False,
        ):
            with self.assertRaisesRegex(RuntimeError, "tracked modifications"):
                require_execution_checkout(dirty_manifest)

    def test_final_qstat_and_qacct_hashes_are_revalidated(self) -> None:
        self._write_writer_evidence(
            "header\nheader\nold-array user rr_method_comparison_m r queue\n"
        )
        with self.assertRaisesRegex(RuntimeError, "Final qstat still contains"):
            self._freeze()

        self._write_writer_evidence()
        manifest = self._freeze()
        frozen_evidence = json.loads(
            Path(manifest["writer_evidence_path"]).read_text(encoding="utf-8")
        )
        qacct = Path(frozen_evidence["jobs"][0]["qacct"]["path"])
        qacct.write_text("tampered\n", encoding="utf-8")
        payload = progress_payload(manifest)
        self.assertFalse(payload["status_valid"])
        self.assertTrue(any("qacct" in value for value in payload["status_errors"]))

    def test_hard_exit_lock_reclamation_and_concurrent_recovery_refusal(self) -> None:
        lock_path = self.root / "hard_exit.lock"
        child = subprocess.run(
            [
                sys.executable,
                "-c",
                (
                    "import os; from pathlib import Path; "
                    "from scripts.method_comparison_episode_workflow import ExclusiveLock; "
                    f"lock=ExclusiveLock(Path({str(lock_path)!r}), 'manifest-x'); "
                    "lock.__enter__(); os._exit(0)"
                ),
            ],
            cwd=Path(__file__).parents[1],
            check=False,
        )
        self.assertEqual(child.returncode, 0)
        self.assertTrue(lock_path.is_dir())
        owner = reclaim_stale_lock(lock_path, "manifest-x")
        self.assertEqual(owner["manifest_fingerprint"], "manifest-x")
        with ExclusiveLock(lock_path, "manifest-x"):
            self.assertTrue(lock_path.is_dir())

        manifest = self._freeze()
        collector_lock = self.workflow / "collector.lock"
        with ExclusiveLock(collector_lock, str(manifest["manifest_fingerprint"])):
            with self.assertRaisesRegex(RuntimeError, "may still be alive"):
                recover_promotion(manifest)

    def test_status_revalidates_stage_and_committed_marker(self) -> None:
        manifest = self._freeze()
        self._run_all_tasks(manifest)
        stage = build_staged_complete_view(manifest)
        stage_episode = self._task_dir(stage) / EPISODE_FILENAME
        with stage_episode.open("a", encoding="utf-8") as handle:
            handle.write("tamper\n")
        payload = progress_payload(manifest)
        self.assertFalse(payload["status_valid"])
        self.assertFalse(payload["stage_validated"])

        build_staged_complete_view(manifest)
        marker = promote_staged_view(manifest)
        marker.write_text("tampered\n", encoding="utf-8")
        payload = progress_payload(manifest)
        self.assertFalse(payload["complete"])
        self.assertFalse(payload["status_valid"])
        command = subprocess.run(
            [
                sys.executable,
                "scripts/method_comparison_episode_workflow.py",
                "status",
                "--manifest",
                str(self.workflow / MANIFEST_FILENAME),
            ],
            cwd=Path(__file__).parents[1],
            capture_output=True,
            text=True,
        )
        self.assertNotEqual(command.returncode, 0)

    def _write_fake_scheduler(self, fail_call: int = 0) -> tuple[Path, Path, Path, Path]:
        fake_dir = self.root / "fake_scheduler"
        fake_dir.mkdir(exist_ok=True)
        counter = fake_dir / "counter.txt"
        qdel_log = fake_dir / "qdel.log"
        qsub = fake_dir / "qsub"
        qdel = fake_dir / "qdel"
        qstat = fake_dir / "qstat"
        qsub.write_text(
            "#!/bin/bash\n"
            "set -euo pipefail\n"
            f"counter={str(counter)!r}\n"
            "count=0\n"
            "[[ -f \"$counter\" ]] && count=$(cat \"$counter\")\n"
            "count=$((count + 1))\n"
            "printf '%s\\n' \"$count\" > \"$counter\"\n"
            f"if [[ {fail_call} -gt 0 && \"$count\" -eq {fail_call} ]]; then exit 19; fi\n"
            "if printf '%s\\n' \"$@\" | grep -q -- '-t'; then\n"
            "  printf '%s.1-3:1\\n' \"$((100 + count))\"\n"
            "else\n"
            "  printf '%s\\n' \"$((200 + count))\"\n"
            "fi\n",
            encoding="utf-8",
        )
        qdel.write_text(
            "#!/bin/bash\n"
            f"printf '%s\\n' \"$@\" >> {str(qdel_log)!r}\n",
            encoding="utf-8",
        )
        qstat.write_text(
            "#!/bin/bash\nprintf 'header\\nheader\\n'\nexit 0\n",
            encoding="utf-8",
        )
        qsub.chmod(0o755)
        qdel.chmod(0o755)
        qstat.chmod(0o755)
        return qsub, qdel, qstat, qdel_log

    def _run_submit_script(self, manifest: dict[str, object], fail_call: int = 0):
        qsub, qdel, qstat, qdel_log = self._write_fake_scheduler(fail_call)
        job_dir = self.root / "jobs"
        env = {
            **os.environ,
            "PROJECT_ROOT": str(Path(__file__).parents[1]),
            "PYTHON_BIN": sys.executable,
            "WORKFLOW_RUN_DIR": str(self.workflow),
            "MANIFEST": str(self.workflow / MANIFEST_FILENAME),
            "JOB_DIR": str(job_dir),
            "LOG_DIR": str(self.root / "logs"),
            "QSUB_BIN": str(qsub),
            "QDEL_BIN": str(qdel),
            "QSTAT_BIN": str(qstat),
            "ROLLBACK_SLEEP_SECONDS": "0",
        }
        completed = subprocess.run(
            ["bash", "scripts/submit_hoffman2_method_comparison_episode_array.sh"],
            cwd=Path(__file__).parents[1],
            env=env,
            capture_output=True,
            text=True,
        )
        return completed, job_dir, qdel_log

    def test_fake_qsub_generates_and_executes_deferred_lane_job(self) -> None:
        manifest = self._freeze()
        completed, job_dir, _ = self._run_submit_script(manifest)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        lane_job = job_dir / "method_comparison_episode_array_lane_1.job"
        text = lane_job.read_text(encoding="utf-8")
        self.assertIn('${SGE_TASK_ID}', text)
        self.assertIn('${JOB_ID}', text)
        env = {**os.environ, "JOB_ID": "901", "SGE_TASK_ID": "1", "TASK_ID": "1"}
        executed = subprocess.run(
            ["bash", str(lane_job)],
            cwd=Path(__file__).parents[1],
            env=env,
            capture_output=True,
            text=True,
        )
        self.assertEqual(executed.returncode, 0, executed.stderr)

    def test_lane_submission_failure_rolls_back_recorded_job(self) -> None:
        shutil.rmtree(self.canonical)
        self._write_empty_current_fixture()
        manifest = self._freeze(lanes=2)
        completed, _, qdel_log = self._run_submit_script(manifest, fail_call=2)
        self.assertNotEqual(completed.returncode, 0)
        journal = json.loads(
            (self.workflow / "method_comparison_episode_array_submission.json").read_text(encoding="utf-8")
        )
        self.assertEqual(journal["state"], "rolled_back")
        self.assertEqual(set(journal["lane_jobs"]), {"1"})
        self.assertIn(journal["lane_jobs"]["1"], qdel_log.read_text(encoding="utf-8"))

    def test_collector_submission_failure_rolls_back_all_lanes(self) -> None:
        shutil.rmtree(self.canonical)
        self._write_empty_current_fixture()
        manifest = self._freeze(lanes=2)
        completed, _, qdel_log = self._run_submit_script(manifest, fail_call=3)
        self.assertNotEqual(completed.returncode, 0)
        journal = json.loads(
            (self.workflow / "method_comparison_episode_array_submission.json").read_text(encoding="utf-8")
        )
        self.assertEqual(journal["state"], "rolled_back")
        self.assertEqual(set(journal["lane_jobs"]), {"1", "2"})
        deleted = qdel_log.read_text(encoding="utf-8")
        for job_id in journal["lane_jobs"].values():
            self.assertIn(job_id, deleted)

    def test_submission_scripts_use_disjoint_one_slot_lanes(self) -> None:
        submit = (Path(__file__).parents[1] / "scripts" / "submit_hoffman2_method_comparison_episode_array.sh").read_text(
            encoding="utf-8"
        )
        cutover = (Path(__file__).parents[1] / "scripts" / "prepare_hoffman2_method_comparison_cutover.sh").read_text(
            encoding="utf-8"
        )
        self.assertNotIn("#$ -pe", submit)
        self.assertIn('-tc "${lane_throttle}"', submit)
        self.assertIn('-hold_jid "${hold_ids}"', submit)
        self.assertIn('ARRAY_LANES="${ARRAY_LANES:-2}"', cutover)
        self.assertIn('LANE_THROTTLE="${LANE_THROTTLE:-80}"', cutover)
        self.assertIn("SMOKE_GATE", cutover)
        self.assertIn("qacct", cutover.lower())
        self.assertNotIn("for (index", cutover)


if __name__ == "__main__":
    unittest.main()
