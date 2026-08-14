"""Test purpose: validate frozen strategy-mapping manifests, shards, and atomic collection."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import unittest
from collections import Counter
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from unittest import mock

from scripts import strategy_mapping_workflow as workflow
from src.mdp.meta_mdp import EnvironmentConfig


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FROZEN_CONFIG = PROJECT_ROOT / "configs/strategy_mapping_environments.json"
SCHEMA_CONTRACT = PROJECT_ROOT / "configs/strategy_mapping_output_schemas.json"


def frozen_inputs(seed_offset: int):
    payload = json.loads(FROZEN_CONFIG.read_text(encoding="utf-8"))
    rows = []
    configs = []
    for entry in payload["environments"]:
        config = EnvironmentConfig(**entry["config"])
        rows.append({"grid_index": entry["source_grid_index"]})
        configs.append(
            (
                entry["environment"],
                replace(config, random_seed=(config.random_seed or 0) + seed_offset),
            )
        )
    return rows, configs


def make_manifest(
    output_dir: Path,
    *,
    episodes: int = 2,
    episodes_per_task: int = 2,
    draws: int = 2,
    oracle_grid_size: int = 101,
    seed_offset: int = 80_000_000,
):
    rows, configs = frozen_inputs(seed_offset)
    return workflow.build_manifest(
        output_dir=output_dir,
        selected_rows=rows,
        selected_configs=configs,
        diagnostic_active_search_dir=output_dir / "diagnostic_active_search",
        diagnostic_active_search_hashes=workflow.DIAGNOSTIC_ACTIVE_SEARCH_EXPECTED_HASHES,
        frozen_config_path=FROZEN_CONFIG,
        schema_contract_path=SCHEMA_CONTRACT,
        episodes=episodes,
        episodes_per_task=episodes_per_task,
        observation_draws=draws,
        oracle_grid_size=oracle_grid_size,
        observations_per_person=10,
        sigma_need_values=workflow.SERIOUS_SIGMA_NEED_VALUES,
        seed_namespace_offset=seed_offset,
        analysis_mode="test",
    )


def write_manifest(output_dir: Path, manifest) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / workflow.MANIFEST_FILENAME
    workflow.atomic_write_json(path, manifest)
    return path


def update_task_hash(manifest_path: Path, task_index: int) -> None:
    rows_path, status_path, _ = workflow.task_paths(manifest_path, task_index)
    status = json.loads(status_path.read_text(encoding="utf-8"))
    status["rows_sha256"] = workflow.sha256_file(rows_path)
    workflow.atomic_write_json(status_path, status)


def completed_result_dir(manifest_path: Path) -> Path:
    completion = json.loads(
        (manifest_path.parent / workflow.COMPLETION_FILENAME).read_text(encoding="utf-8")
    )
    return manifest_path.parent / completion["result_directory"]


def mutate_persisted_row(
    manifest_path: Path,
    task,
    policy: str,
    field: str,
    value,
) -> None:
    rows_path, _, _ = workflow.task_paths(manifest_path, int(task["task_index"]))
    rows = workflow.read_csv(rows_path)
    target = next(row for row in rows if str(row["policy"]) == policy)
    target[field] = value
    manifest = workflow.load_manifest(manifest_path)
    workflow.atomic_write_csv(
        rows_path,
        rows,
        manifest["expected_output_schemas"][workflow.EPISODE_OUTPUTS[str(task["analysis"])]],
    )
    update_task_hash(manifest_path, int(task["task_index"]))


class StrategyMappingWorkflowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._temporary = tempfile.TemporaryDirectory()
        cls.reference_dir = Path(cls._temporary.name) / "complete_reference"
        cls.reference_manifest = make_manifest(cls.reference_dir)
        cls.reference_manifest_path = write_manifest(
            cls.reference_dir, cls.reference_manifest
        )
        for task in cls.reference_manifest["tasks"]:
            workflow.run_task(cls.reference_manifest_path, int(task["task_index"]))
        workflow.collect(cls.reference_manifest_path)

    @classmethod
    def tearDownClass(cls):
        cls._temporary.cleanup()

    def copy_reference(self, name: str) -> Path:
        destination = Path(self._temporary.name) / name
        shutil.copytree(self.reference_dir, destination)
        return destination / workflow.MANIFEST_FILENAME

    def test_serious_task_map_has_exact_counts_and_ranges(self):
        rows, configs = frozen_inputs(workflow.SERIOUS_SEED_NAMESPACE_OFFSET)
        with mock.patch.object(workflow, "git_is_clean", return_value=True):
            manifest = workflow.build_manifest(
                output_dir=Path(self._temporary.name) / "serious_preview",
                selected_rows=rows,
                selected_configs=configs,
                diagnostic_active_search_dir=Path(self._temporary.name) / "diagnostic_active_search",
                diagnostic_active_search_hashes=workflow.DIAGNOSTIC_ACTIVE_SEARCH_EXPECTED_HASHES,
                frozen_config_path=FROZEN_CONFIG,
                schema_contract_path=SCHEMA_CONTRACT,
                episodes=1200,
                episodes_per_task=10,
                observation_draws=500,
                oracle_grid_size=4001,
                observations_per_person=500,
                sigma_need_values=(10.0, 60.0, 100.0),
                seed_namespace_offset=60_000_000,
                analysis_mode="serious",
            )
        self.assertEqual(len(manifest["tasks"]), 1320)
        self.assertEqual(
            Counter(task["analysis"] for task in manifest["tasks"]),
            {"four_way": 360, "sigma_need": 360, "fixed_total_need": 600},
        )
        self.assertEqual(
            manifest["expected_row_counts"],
            {
                "strategy_mapping_four_way_episodes.csv": 14400,
                "strategy_mapping_four_way_policy_summary.csv": 108,
                "strategy_mapping_four_way_paired_comparisons.csv": 189,
                "strategy_mapping_sigma_need_episodes.csv": 3600,
                "strategy_mapping_sigma_need_environment_summary.csv": 3,
                "strategy_mapping_sigma_need_gap_strata.csv": 54,
                "strategy_mapping_fixed_total_need_episodes.csv": 6000,
                "strategy_mapping_fixed_total_need_summary.csv": 46,
            },
        )
        keys = [
            (
                task["analysis"],
                task["condition_value"],
                task["episode_start"],
                task["episode_end"],
            )
            for task in manifest["tasks"]
        ]
        self.assertEqual(len(keys), len(set(keys)))
        self.assertEqual(manifest["tasks"][0]["episode_start"], 0)
        self.assertEqual(manifest["tasks"][359]["episode_end"], 1199)
        self.assertEqual(manifest["tasks"][-1]["episode_end"], 1199)
        self.assertTrue(all(task["episode_count"] == 10 for task in manifest["tasks"]))
        self.assertTrue(
            all(task["seed_range"]["namespace_offset"] == 60_000_000 for task in manifest["tasks"])
        )

    def test_cli_facing_create_builds_serious_manifest(self):
        output_dir = Path(self._temporary.name) / "cli_manifest"
        rows, configs = frozen_inputs(workflow.SERIOUS_SEED_NAMESPACE_OFFSET)
        args = argparse.Namespace(
            analysis_mode="serious",
            seed_namespace_offset=None,
            output_dir=output_dir,
            diagnostic_active_search_dir=Path(self._temporary.name) / "diagnostic_active_search",
            frozen_config=FROZEN_CONFIG,
            schema_contract=SCHEMA_CONTRACT,
            episodes=1200,
            episodes_per_task=10,
            observation_draws=500,
            oracle_grid_size=4001,
            observations_per_person=500,
            sigma_need_values=[10.0, 60.0, 100.0],
        )
        with mock.patch.object(workflow, "git_is_clean", return_value=True), mock.patch.object(
            workflow,
            "_load_frozen_inputs",
            return_value=(rows, configs, workflow.DIAGNOSTIC_ACTIVE_SEARCH_EXPECTED_HASHES),
        ):
            path = workflow.create_manifest_from_args(args)
            manifest = workflow.load_manifest(path)
        self.assertEqual(manifest["analysis_mode"], "serious")
        self.assertEqual(manifest["settings"]["episodes_per_task"], 10)
        self.assertEqual(len(manifest["tasks"]), 1320)

    def test_parser_defaults_and_scheduled_smoke_are_separate(self):
        parsed = workflow.build_parser().parse_args(["create", "--output-dir", "unused"])
        self.assertEqual(parsed.episodes_per_task, 10)
        rows, configs = frozen_inputs(workflow.SMOKE_SEED_NAMESPACE_OFFSET)
        with mock.patch.object(workflow, "git_is_clean", return_value=True):
            manifest = workflow.build_manifest(
                output_dir=Path(self._temporary.name) / "smoke_preview",
                selected_rows=rows,
                selected_configs=configs,
                diagnostic_active_search_dir=Path(self._temporary.name) / "diagnostic_active_search",
                diagnostic_active_search_hashes=workflow.DIAGNOSTIC_ACTIVE_SEARCH_EXPECTED_HASHES,
                frozen_config_path=FROZEN_CONFIG,
                schema_contract_path=SCHEMA_CONTRACT,
                episodes=2,
                episodes_per_task=2,
                observation_draws=500,
                oracle_grid_size=4001,
                observations_per_person=500,
                sigma_need_values=(10.0, 60.0, 100.0),
                seed_namespace_offset=70_000_000,
                analysis_mode="smoke",
            )
        tags = workflow._analysis_mode_fields(manifest, "sigma_need")
        self.assertEqual(tags["analysis_mode"], "smoke")
        self.assertEqual(tags["scientific_status"], "smoke_only_not_scientific_evidence")
        self.assertFalse(manifest["scientific_completion_allowed"])

    def test_each_task_evaluates_only_its_condition(self):
        manifest = self.reference_manifest
        for analysis, expected_rows in (("four_way", 8), ("sigma_need", 2), ("fixed_total_need", 2)):
            task = next(task for task in manifest["tasks"] if task["analysis"] == analysis)
            rows = workflow.execute_task_rows(manifest, task)
            self.assertEqual(len(rows), expected_rows)
            self.assertEqual(len({workflow._row_key(analysis, row) for row in rows}), expected_rows)
            workflow.validate_episode_rows(manifest, task, rows)
            if analysis == "sigma_need":
                self.assertEqual({float(row["sigma_need"]) for row in rows}, {10.0})
            if analysis == "fixed_total_need":
                self.assertEqual(
                    {float(row["constructed_need_difference"]) for row in rows}, {0.0}
                )

    def test_episode_range_partition_is_scientifically_invariant(self):
        whole = make_manifest(
            Path(self._temporary.name) / "partition_whole",
            episodes=4,
            episodes_per_task=4,
            draws=2,
        )
        split = make_manifest(
            Path(self._temporary.name) / "partition_split",
            episodes=4,
            episodes_per_task=2,
            draws=2,
        )
        condition_by_analysis = {
            "four_way": whole["selected_environments"][0]["environment"],
            "sigma_need": 60.0,
            "fixed_total_need": 20.0,
        }
        for analysis, condition in condition_by_analysis.items():
            def matches(task):
                if task["analysis"] != analysis:
                    return False
                if analysis == "four_way":
                    return task["condition_value"] == condition
                return float(task["condition_value"]) == float(condition)

            whole_task = next(
                task
                for task in whole["tasks"]
                if matches(task)
            )
            split_tasks = [
                task
                for task in split["tasks"]
                if matches(task)
            ]
            whole_rows = workflow.execute_task_rows(whole, whole_task)
            split_rows = []
            for task in split_tasks:
                split_rows.extend(workflow.execute_task_rows(split, task))
            whole_payload = workflow.canonical_json(
                sorted(
                    whole_rows,
                    key=lambda row: (int(row["episode_index"]), str(row["policy"])),
                )
            )
            split_payload = workflow.canonical_json(
                sorted(
                    split_rows,
                    key=lambda row: (int(row["episode_index"]), str(row["policy"])),
                )
            )
            self.assertEqual(whole_payload, split_payload)

    def test_complete_collection_and_retry_are_deterministic(self):
        manifest_path = self.copy_reference("complete_copy")
        result = workflow.validate_final_tree(manifest_path)
        self.assertEqual(result["status"], "valid")
        self.assertFalse(result["scientific_completion"])
        rows_path, _, _ = workflow.task_paths(manifest_path, 0)
        before = workflow.sha256_file(rows_path)
        workflow.run_task(manifest_path, 0)
        self.assertEqual(before, workflow.sha256_file(rows_path))
        completion = json.loads(
            (manifest_path.parent / workflow.COMPLETION_FILENAME).read_text(encoding="utf-8")
        )
        self.assertEqual(completion["status"], "strict_test_validation_complete")

    def test_missing_or_overlapping_shard_prevents_completion(self):
        missing_manifest = self.copy_reference("missing_copy")
        shutil.rmtree(workflow.task_paths(missing_manifest, 0)[0].parent)
        with self.assertRaisesRegex(RuntimeError, "Strict StrategyMapping collection refused"):
            workflow.collect(missing_manifest)
        self.assertFalse((missing_manifest.parent / workflow.COMPLETION_FILENAME).exists())

        overlap_manifest = self.copy_reference("overlap_copy")
        sigma_tasks = [
            task
            for task in workflow.load_manifest(overlap_manifest)["tasks"]
            if task["analysis"] == "sigma_need" and float(task["condition_value"]) == 10.0
        ]
        second = sigma_tasks[0]
        rows_path, _, _ = workflow.task_paths(overlap_manifest, int(second["task_index"]))
        rows = workflow.read_csv(rows_path)
        rows[1]["episode_index"] = rows[0]["episode_index"]
        workflow.atomic_write_csv(
            rows_path,
            rows,
            workflow.load_manifest(overlap_manifest)["expected_output_schemas"][
                "strategy_mapping_sigma_need_episodes.csv"
            ],
        )
        update_task_hash(overlap_manifest, int(second["task_index"]))
        with self.assertRaises(RuntimeError):
            workflow.collect(overlap_manifest)
        self.assertFalse((overlap_manifest.parent / workflow.COMPLETION_FILENAME).exists())

    def test_global_common_randomness_tamper_is_rejected(self):
        manifest_path = self.copy_reference("common_randomness_copy")
        manifest = workflow.load_manifest(manifest_path)
        sigma_100 = next(
            task
            for task in manifest["tasks"]
            if task["analysis"] == "sigma_need" and float(task["condition_value"]) == 100.0
        )
        rows_path, _, _ = workflow.task_paths(manifest_path, int(sigma_100["task_index"]))
        rows = workflow.read_csv(rows_path)
        rows[0]["observation_residual_hash_1"] = "tampered-cross-condition-hash"
        workflow.atomic_write_csv(
            rows_path,
            rows,
            manifest["expected_output_schemas"]["strategy_mapping_sigma_need_episodes.csv"],
        )
        update_task_hash(manifest_path, int(sigma_100["task_index"]))
        with self.assertRaisesRegex(RuntimeError, "Global sigma common-randomness mismatch"):
            workflow.collect(manifest_path)
        self.assertFalse((manifest_path.parent / workflow.COMPLETION_FILENAME).exists())

    def test_manifest_source_completion_and_lock_failures(self):
        manifest_path = self.copy_reference("failure_copy")
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        payload["settings"]["observation_draws"] = 999
        workflow.atomic_write_json(manifest_path, payload)
        with self.assertRaisesRegex(RuntimeError, "manifest hash mismatch"):
            workflow.load_manifest(manifest_path)

        manifest_path = self.copy_reference("stale_completion_copy")
        validation_path = completed_result_dir(manifest_path) / workflow.VALIDATION_FILENAME
        validation_path.write_text("{}\n", encoding="utf-8")
        with self.assertRaisesRegex(RuntimeError, "stale manifest fingerprint"):
            workflow.validate_final_tree(manifest_path)

        manifest_path = self.copy_reference("lock_copy")
        lock_path = manifest_path.parent / workflow.COLLECTION_LOCK_FILENAME
        lock_path.write_text("held\n", encoding="utf-8")
        with self.assertRaisesRegex(RuntimeError, "Another StrategyMapping collector"):
            workflow.collect(manifest_path)
        lock_path.unlink()

        manifest = deepcopy(self.reference_manifest)
        manifest["analysis_mode"] = "serious"
        with mock.patch.object(workflow, "git_is_clean", return_value=False):
            with self.assertRaisesRegex(RuntimeError, "clean worktree"):
                workflow.verify_execution_checkout(manifest)

    def test_recomputed_overlapping_task_map_and_nonfinite_rows_are_rejected(self):
        overlap_dir = Path(self._temporary.name) / "overlap_manifest"
        manifest = make_manifest(overlap_dir, episodes=4, episodes_per_task=2)
        sigma_tasks = [
            task
            for task in manifest["tasks"]
            if task["analysis"] == "sigma_need" and float(task["condition_value"]) == 10.0
        ]
        sigma_tasks[1]["episode_start"] = sigma_tasks[0]["episode_start"]
        sigma_tasks[1]["episode_end"] = sigma_tasks[0]["episode_end"]
        manifest["manifest_hash"] = workflow.digest(
            {key: value for key, value in manifest.items() if key != "manifest_hash"}
        )
        path = write_manifest(overlap_dir, manifest)
        with self.assertRaisesRegex(RuntimeError, "exact frozen Cartesian partition"):
            workflow.load_manifest(path)

        manifest_path = self.copy_reference("nonfinite_copy")
        manifest = workflow.load_manifest(manifest_path)
        task = next(task for task in manifest["tasks"] if task["analysis"] == "four_way")
        rows_path, _, _ = workflow.task_paths(manifest_path, int(task["task_index"]))
        rows = workflow.read_csv(rows_path)
        rows[0]["realized_utility"] = "nan"
        workflow.atomic_write_csv(
            rows_path,
            rows,
            manifest["expected_output_schemas"]["strategy_mapping_four_way_episodes.csv"],
        )
        update_task_hash(manifest_path, int(task["task_index"]))
        with self.assertRaises(RuntimeError):
            workflow.collect(manifest_path)
        self.assertFalse((manifest_path.parent / workflow.COMPLETION_FILENAME).exists())

    def test_exhaustive_numeric_and_domain_faults_are_rejected_at_shard_level(self):
        manifest = self.reference_manifest
        tasks = {
            analysis: next(task for task in manifest["tasks"] if task["analysis"] == analysis)
            for analysis in workflow.ANALYSES
        }
        base_rows = {
            analysis: workflow.execute_task_rows(manifest, task)
            for analysis, task in tasks.items()
        }
        nan_cases = [
            ("four_way", "full_information_oracle", "initial_oracle_allocation"),
            ("four_way", "full_information_oracle", "initial_oracle_utility"),
            ("four_way", "full_information_oracle", "oracle_grid_optimality_violation"),
            ("four_way", "frozen_rr", "raw_utility_regret_to_initial_oracle"),
            ("four_way", "frozen_rr", "utility_regret_to_initial_oracle"),
            *[
                ("four_way", "frozen_rr", field)
                for field in sorted(workflow.RR_TIME_MATCHED_FIELDS)
            ],
            *[
                ("four_way", "manual_active_search_equal_outcome", field)
                for field in sorted(workflow.MANUAL_TIME_MATCHED_FIELDS)
            ],
            *[
                (analysis, "frozen_rr", field)
                for analysis in ("sigma_need", "fixed_total_need")
                for field in (
                    "oracle_allocation",
                    "oracle_utility",
                    "oracle_grid_optimality_violation",
                    "initial_oracle_raw_regret",
                    "initial_oracle_regret",
                    "initial_oracle_optimality_violation",
                )
            ],
        ]
        for analysis, policy, field in nan_cases:
            with self.subTest(analysis=analysis, policy=policy, field=field):
                rows = deepcopy(base_rows[analysis])
                next(row for row in rows if row["policy"] == policy)[field] = "nan"
                with self.assertRaisesRegex(RuntimeError, "Non-finite"):
                    workflow.validate_episode_rows(manifest, tasks[analysis], rows)

        domain_cases = (
            ("four_way", "frozen_rr", {"allocation_to_person1": 1.5}),
            (
                "four_way",
                "manual_active_search_equal_outcome",
                {"sample_count_1": -1},
            ),
            ("four_way", "frozen_rr", {"online_sample_count": 1.5}),
            ("four_way", "frozen_rr", {"online_sample_count": 9}),
            ("four_way", "frozen_rr", {"remaining_time": -1}),
            (
                "four_way",
                "equal_split",
                {
                    "online_sample_count": 1,
                    "sample_count_1": 1,
                    "immediate_termination": 0,
                    "remaining_time": 118.98,
                },
            ),
            (
                "four_way",
                "manual_active_search_equal_outcome",
                {
                    "online_sample_count": 5,
                    "sample_count_2": 2,
                    "remaining_time": 118.9,
                },
            ),
            ("sigma_need", "frozen_rr", {"remaining_time": 999}),
            ("fixed_total_need", "frozen_rr", {"orientation": 0}),
        )
        for analysis, policy, updates in domain_cases:
            with self.subTest(analysis=analysis, policy=policy, updates=updates):
                rows = deepcopy(base_rows[analysis])
                next(row for row in rows if row["policy"] == policy).update(updates)
                with self.assertRaises(RuntimeError):
                    workflow.validate_episode_rows(manifest, tasks[analysis], rows)

    def test_corrupt_domains_are_rejected_by_final_collection(self):
        cases = (
            ("four_way", "full_information_oracle", "initial_oracle_utility", "nan"),
            (
                "four_way",
                "manual_active_search_equal_outcome",
                "sample_count_1",
                -7,
            ),
            ("sigma_need", "frozen_rr", "allocation_to_person1", 1.5),
            ("sigma_need", "frozen_rr", "remaining_time", 999),
            ("fixed_total_need", "frozen_rr", "oracle_utility", "nan"),
        )
        for index, (analysis, policy, field, value) in enumerate(cases):
            with self.subTest(analysis=analysis, field=field):
                manifest_path = self.copy_reference(f"domain_collect_{index}")
                manifest = workflow.load_manifest(manifest_path)
                task = next(task for task in manifest["tasks"] if task["analysis"] == analysis)
                mutate_persisted_row(manifest_path, task, policy, field, value)
                with self.assertRaises(RuntimeError):
                    workflow.collect(manifest_path)
                self.assertFalse(
                    (manifest_path.parent / workflow.COMPLETION_FILENAME).exists()
                )

    def test_concurrent_same_task_retries_are_serialized_and_valid(self):
        output_dir = Path(self._temporary.name) / "concurrent_task"
        manifest = make_manifest(output_dir)
        manifest_path = write_manifest(output_dir, manifest)
        code = """
import sys
import time
from pathlib import Path
from scripts import strategy_mapping_workflow as workflow
original = workflow.execute_task_rows
def delayed(manifest, task):
    time.sleep(0.35)
    return original(manifest, task)
workflow.execute_task_rows = delayed
workflow.run_task(Path(sys.argv[1]), 0)
"""
        environment = os.environ.copy()
        environment["PYTHONPATH"] = str(PROJECT_ROOT)
        first = subprocess.Popen(
            [sys.executable, "-c", code, str(manifest_path)],
            cwd=PROJECT_ROOT,
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        time.sleep(0.05)
        second = subprocess.Popen(
            [sys.executable, "-c", code, str(manifest_path)],
            cwd=PROJECT_ROOT,
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        first_stdout, first_stderr = first.communicate(timeout=20)
        second_stdout, second_stderr = second.communicate(timeout=20)
        self.assertEqual(first.returncode, 0, first_stderr)
        self.assertEqual(second.returncode, 0, second_stderr)
        outputs = first_stdout + second_stdout
        self.assertIn('"status":"ok"', outputs)
        self.assertIn('"status":"already_complete"', outputs)
        task = workflow._task_for_index(workflow.load_manifest(manifest_path), 0)
        state, errors = workflow.validate_task_artifact(
            manifest_path, workflow.load_manifest(manifest_path), task
        )
        self.assertEqual((state, errors), ("complete", []))
        task_dir = workflow.task_paths(manifest_path, 0)[0].parent
        self.assertFalse(any(path.name.endswith(".tmp") for path in task_dir.iterdir()))
        self.assertFalse((task_dir / "failure.json").exists())

    def test_directory_promotion_failures_are_recoverable_and_fail_closed(self):
        scenarios = (
            ("promotion", "_promote_result_directory"),
            ("promoted_readback", "_validate_result_directory"),
            ("final_readback", "validate_final_tree"),
        )
        for name, patched_function in scenarios:
            with self.subTest(stage=name):
                manifest_path = self.copy_reference(f"promotion_failure_{name}")
                with mock.patch.object(
                    workflow,
                    patched_function,
                    side_effect=RuntimeError(f"injected {name} failure"),
                ):
                    with self.assertRaisesRegex(RuntimeError, f"injected {name} failure"):
                        workflow.collect(manifest_path)
                self.assertFalse(
                    (manifest_path.parent / workflow.COMPLETION_FILENAME).exists()
                )
                workflow.collect(manifest_path)
                self.assertEqual(workflow.validate_final_tree(manifest_path)["status"], "valid")
                result_dir = completed_result_dir(manifest_path)
                self.assertTrue(result_dir.is_dir())
                self.assertTrue(all((result_dir / filename).is_file() for filename in workflow.ALL_OUTPUTS))
                self.assertFalse(any((manifest_path.parent / filename).exists() for filename in workflow.ALL_OUTPUTS))

    def test_effective_seed_namespaces_cover_boundaries_without_collision(self):
        rows, configs = frozen_inputs(workflow.SERIOUS_SEED_NAMESPACE_OFFSET)
        base_seed = configs[0][1].random_seed or 0
        for episode_index in (0, 9, 10, 1199):
            true_seed = base_seed + episode_index * 17 + 1
            observation_seed = base_seed + 100_000 + episode_index * 17
            policy_seed = base_seed + 300_000 + episode_index * 17
            fixed_seed = base_seed + 600_000 + episode_index * 17
            self.assertEqual(len({true_seed, observation_seed, policy_seed, fixed_seed}), 4)
        self.assertEqual(rows[0]["grid_index"], 295)

    def test_submit_script_keeps_generated_files_under_ignored_output_dir(self):
        script = PROJECT_ROOT / "scripts/submit_hoffman2_strategy_mapping.sh"
        text = script.read_text(encoding="utf-8")
        self.assertIn('SCHEDULER_DIR="${OUTPUT_DIR_ABS}/scheduler"', text)
        self.assertIn('LOG_DIR="${OUTPUT_DIR_ABS}/logs"', text)
        self.assertIn('source "${PROJECT_ROOT}/scripts/hoffman2_scheduler.sh"', text)
        self.assertIn('strategy_mapping_submit_job "${family}"', text)
        self.assertIn('strategy_mapping_submit_job "collector"', text)
        self.assertNotIn('job_file="jobs/', text)
        self.assertNotIn("#$ -o logs/", text)
        record_jobs = text.index("scripts/strategy_mapping_workflow.py record-jobs")
        clear_err_trap = text.index("trap - ERR")
        self.assertLess(record_jobs, clear_err_trap)
        subprocess.run(["bash", "-n", str(script)], check=True)
        subprocess.run(
            ["bash", "-n", str(PROJECT_ROOT / "scripts" / "hoffman2_scheduler.sh")],
            check=True,
        )


if __name__ == "__main__":
    unittest.main()
