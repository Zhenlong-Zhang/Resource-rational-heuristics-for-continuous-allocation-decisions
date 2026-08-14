"""Test purpose: validate frozen positive-need manifests, task evidence, and strict collection."""

from __future__ import annotations

import argparse
from contextlib import redirect_stdout
from io import StringIO
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from scripts.positive_need_workflow import (
    MANIFEST_NAME,
    ARTIFACT_INDEX_NAME,
    VALIDATION_NAME,
    collect,
    create_confirmation,
    create_development,
    current_sge_task_id,
    diagnose_quadrature_orders,
    diagnose_quadrature_suite,
    find_environment,
    load_manifest,
    load_version_pointer,
    main,
    manifest_hash,
    parse_quadrature_order_pair,
    next_version_path,
    parse_qacct_records,
    progress,
    run_task,
    sha256_file,
    task_directory,
    validate_task,
    verify_local_readback,
    write_version_pointer,
    validate_confirmation_collection,
    validate_scheduler_evidence,
    validate_qacct_evidence,
    read_csv,
    write_csv,
)
from src.experiments.positive_need import (
    CONFIRMATION_EPISODE_SCHEMA,
    build_finite_support_episodes,
    evaluate_serious_environment,
    load_positive_need_spec,
)


def write_scheduler_fixture(run_dir: Path, manifest: dict) -> dict:
    scheduler_dir = run_dir / "scheduler"
    scheduler_dir.mkdir(exist_ok=True)
    array = scheduler_dir / "array.job"
    collector = scheduler_dir / "collector.job"
    array.write_text(
        "#$ -q campus2.q\n#$ -l h_rt=08:00:00\n#$ -l h_data=2G\n"
        f"#$ -t 1-{manifest['task_count']}\n#$ -tc 80\n",
        encoding="utf-8",
    )
    collector.write_text(
        "#$ -q campus2.q\n#$ -l h_rt=08:00:00\n#$ -l h_data=4G\n",
        encoding="utf-8",
    )
    submission_dir = scheduler_dir / "submission_evidence"
    submission_dir.mkdir()
    raw_submission = submission_dir / "array.qsub.stdout"
    raw_submission.write_text("101.1-162:1\n", encoding="utf-8")
    evidence = {
        "array_job_id": "101",
        "collector_job_id": "102",
        "array_job_name": "array-name",
        "collector_job_name": "collector-name",
        "throttle": 80,
        "queue": "campus2.q",
        "task_h_rt": "08:00:00",
        "task_h_data": "2G",
        "collector_h_rt": "08:00:00",
        "collector_h_data": "4G",
        "task_slots": 1,
        "collector_slots": 1,
        "task_count": manifest["task_count"],
        "phase": "development",
        "array_job_path": "scheduler/array.job",
        "collector_job_path": "scheduler/collector.job",
        "array_job_sha256": sha256_file(array),
        "collector_job_sha256": sha256_file(collector),
        "submission_evidence": {
            "scheduler/submission_evidence/array.qsub.stdout": {
                "sha256": sha256_file(raw_submission),
                "bytes": raw_submission.stat().st_size,
            }
        },
    }
    (scheduler_dir / "jobs.json").write_text(json.dumps(evidence), encoding="utf-8")
    return evidence


class PositiveNeedWorkflowTests(unittest.TestCase):
    def test_non_array_sge_sentinel_is_not_treated_as_a_task_id(self) -> None:
        for value in ("", "undefined", "UNDEFINED", "none", "0", "-1"):
            with self.subTest(value=value), patch.dict(
                os.environ, {"SGE_TASK_ID": value}, clear=False
            ):
                self.assertEqual(current_sge_task_id(), "")
        with patch.dict(os.environ, {"SGE_TASK_ID": "73"}, clear=False):
            self.assertEqual(current_sge_task_id(), "73")

    def test_development_manifest_freezes_complete_grid(self) -> None:
        with TemporaryDirectory() as temporary:
            output = Path(temporary) / "development"
            manifest_path = create_development(output, require_clean=False)
            manifest = load_manifest(manifest_path)
            self.assertEqual(manifest["analysis"], "development")
            self.assertEqual(manifest["task_count"], 162)
            self.assertEqual(manifest["environment_task_count"], 72)
            self.assertEqual(manifest["numerical_task_count"], 90)
            self.assertEqual(manifest["expected_rows_per_task"], 2160)
            self.assertEqual(len(manifest["numerical_validation_cases"]), 90)
            self.assertTrue(manifest["git_tree_hash"])
            self.assertTrue(
                {
                    "src/mdp/meta_mdp.py",
                    "src/policies/heuristic.py",
                    "src/experiments/active_search_evaluation.py",
                    "src/experiments/regimes.py",
                    "src/solvers/gauss_hermite.py",
                    "scripts/hoffman2_scheduler.sh",
                }.issubset(manifest["source_hashes"])
            )
            self.assertEqual(len(manifest["episode_schema"]), len(set(manifest["episode_schema"])))
            self.assertEqual(
                [task["task_index"] for task in manifest["tasks"]],
                list(range(162)),
            )
            state = progress(manifest_path)
            self.assertEqual(state["missing_task_count"], 162)
            self.assertFalse(state["complete"])
            with self.assertRaises(RuntimeError):
                collect(manifest_path)

    def test_quadrature_diagnostic_uses_explicit_odd_order_pairs(self) -> None:
        self.assertEqual(parse_quadrature_order_pair("41:81"), (41, 81))
        for value in ("41", "40:81", "41:40", "41:82"):
            with self.subTest(value=value), self.assertRaises(
                argparse.ArgumentTypeError
            ):
                parse_quadrature_order_pair(value)
        with patch(
            "scripts.positive_need_workflow.validate_numerical_case"
        ) as validate:
            validate.side_effect = lambda case, spec: {
                "case_id": case["case_id"],
                "gh_order": spec["numerical_settings"][
                    "matched_voi_gauss_hermite_order"
                ],
                "gh_reference_order": spec["numerical_settings"][
                    "gauss_hermite_reference_order"
                ],
            }
            rows = diagnose_quadrature_orders(0, [(41, 81), (51, 101)])
        self.assertEqual(
            [(row["gh_order"], row["gh_reference_order"]) for row in rows],
            [(41, 81), (51, 101)],
        )
        self.assertTrue(all(row["diagnostic_only"] for row in rows))

    def test_quadrature_suite_reports_all_frozen_cases_and_pair_summaries(self) -> None:
        spec = {
            "numerical_settings": {
                "matched_voi_gauss_hermite_order": 31,
                "gauss_hermite_reference_order": 61,
                "action_value_convergence_tolerance": 1e-4,
                "allocation_convergence_tolerance": 0.0025,
                "action_tie_tolerance": 1e-10,
            }
        }
        cases = [{"case_id": case_id} for case_id in range(90)]

        def fake_validation(case, diagnostic_spec):
            primary = diagnostic_spec["numerical_settings"][
                "matched_voi_gauss_hermite_order"
            ]
            failed = primary == 41 and case["case_id"] == 7
            return {
                "case_id": case["case_id"],
                "gh_order": primary,
                "gh_reference_order": diagnostic_spec["numerical_settings"][
                    "gauss_hermite_reference_order"
                ],
                "gh_max_action_value_error": 2e-4 if failed else 5e-5,
                "terminal_grid_allocation_error": 0.0,
                "terminal_grid_value_error": 0.0,
                "dense_reference_error": 0.0,
                "dense_reference_performed": 1.0 if case["case_id"] < 36 else 0.0,
                "passed": 0.0 if failed else 1.0,
            }

        with patch(
            "scripts.positive_need_workflow.load_positive_need_spec",
            return_value=spec,
        ), patch(
            "scripts.positive_need_workflow.build_numerical_validation_cases",
            return_value=cases,
        ), patch(
            "scripts.positive_need_workflow.validate_numerical_case",
            side_effect=fake_validation,
        ) as validate:
            result = diagnose_quadrature_suite([(41, 81), (51, 101)])

        self.assertEqual(validate.call_count, 180)
        self.assertEqual(result["frozen_case_count"], 90)
        self.assertEqual(result["frozen_dense_reference_case_count"], 36)
        self.assertEqual(len(result["per_case"]), 180)
        self.assertEqual(len(result["aggregate"]), 2)
        first, second = result["aggregate"]
        self.assertEqual(first["failed_case_ids"], [7])
        self.assertFalse(first["valid"])
        self.assertTrue(second["valid"])
        self.assertEqual(first["dense_reference_case_count"], 36)
        self.assertEqual(
            result["frozen_numerical_settings"][
                "action_value_convergence_tolerance"
            ],
            1e-4,
        )
        self.assertEqual(
            spec["numerical_settings"]["matched_voi_gauss_hermite_order"], 31
        )

    def test_quadrature_suite_cli_emits_machine_readable_json(self) -> None:
        diagnostic = {
            "diagnostic_only": True,
            "per_case": [{"case_id": 0}],
            "aggregate": [{"gh_order": 41, "valid": True}],
        }
        output = StringIO()
        with patch(
            "scripts.positive_need_workflow.diagnose_quadrature_suite",
            return_value=diagnostic,
        ) as diagnose, redirect_stdout(output):
            exit_code = main(
                ["diagnose-quadrature-suite", "--order-pair", "41:81"]
            )

        self.assertEqual(exit_code, 0)
        self.assertEqual(json.loads(output.getvalue()), diagnostic)
        diagnose.assert_called_once_with([(41, 81)])

    @patch("scripts.positive_need_workflow.load_version_pointer")
    @patch("scripts.positive_need_workflow.validate_development_for_confirmation")
    def test_confirmation_manifest_has_exact_target_control_tasks(
        self, validate, load_pointer
    ) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            development = root / "development"
            create_development(development, require_clean=False)
            selection = {
                "selection_status": "selected_without_rr_behavior",
                "target_environment": "positive_need_gap=high_sigma=2_sample_cost=0.02",
                "target_environment_hash": "development-test",
                "target_sample_time_cost": 0.02,
                "control_environment": "positive_need_gap=high_sigma=2_sample_cost=8",
                "control_environment_hash": "development-test-control",
                "control_sample_time_cost": 8.0,
                "gap_class": "high",
                "sigma_sample": 2.0,
            }
            validate.return_value = selection
            version = development / "validated_versions" / "version_0001"
            version.mkdir(parents=True)
            selection_path = version / "selected_target_control.json"
            selection_path.write_text(
                json.dumps(selection), encoding="utf-8"
            )
            load_pointer.return_value = (
                version,
                {
                    "version_path": "validated_versions/version_0001",
                    "pointer_hash": "test-pointer",
                },
            )
            smoke = root / "smoke"
            manifest_path = create_confirmation(
                smoke,
                development,
                "smoke",
                require_clean=False,
            )
            manifest = load_manifest(manifest_path)
            self.assertEqual(manifest["task_count"], 36)
            self.assertEqual(manifest["expected_episode_rows"], 144)
            self.assertEqual(
                {task["environment_role"] for task in manifest["tasks"]},
                {"target", "control"},
            )

    def test_confirmation_rejects_unvalidated_selection_file(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            development = root / "development"
            create_development(development, require_clean=False)
            (development / "selected_target_control.json").write_text(
                json.dumps(
                    {
                        "selection_status": "selected_without_rr_behavior",
                        "target_environment": "positive_need_gap=high_sigma=2_sample_cost=0.02",
                        "control_environment": "positive_need_gap=high_sigma=2_sample_cost=8",
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaises(RuntimeError):
                create_confirmation(
                    root / "smoke",
                    development,
                    "smoke",
                    require_clean=False,
                )

    def test_manifest_tampering_is_rejected(self) -> None:
        with TemporaryDirectory() as temporary:
            output = Path(temporary) / "development"
            manifest_path = create_development(output, require_clean=False)
            value = json.loads(manifest_path.read_text(encoding="utf-8"))
            value["task_count"] = 71
            manifest_path.write_text(json.dumps(value), encoding="utf-8")
            with self.assertRaises(RuntimeError):
                load_manifest(manifest_path)

    def test_global_collection_rejects_duplicate_and_mismatched_keys(self) -> None:
        policies = (
            "matched_prior_myopic_rr",
            "manual_active_search_equal_outcome",
            "equal_split",
            "full_information_oracle",
        )
        rows = []
        for environment in ("target", "control"):
            for policy in policies:
                rows.append(
                    {
                        "environment": environment,
                        "episode_index": "0",
                        "policy": policy,
                        "latent_atom_index": "2",
                        "need_1": "100.0",
                        "need_2": "20.0",
                        "total_true_need": "120.0",
                        "realized_true_need_gap": "80.0",
                        "orientation": "1",
                        "episode_fingerprint": "episode-0",
                        "support_hash": "support",
                        "observation_residual_hash_1": "residual-1",
                        "observation_residual_hash_2": "residual-2",
                        "max_observation_reconstruction_error_1": "0",
                        "max_observation_reconstruction_error_2": "0",
                    }
                )
        manifest = {
            "tasks": [
                {"environment": "target", "episode_index": 0},
                {"environment": "control", "episode_index": 0},
            ],
            "selection": {
                "target_environment": "target",
                "control_environment": "control",
            },
            "episodes_per_condition": 1,
        }
        validate_confirmation_collection(manifest, rows)
        with self.assertRaises(RuntimeError):
            validate_confirmation_collection(manifest, rows + [dict(rows[0])])
        corrupted = [dict(row) for row in rows]
        corrupted[-1]["observation_residual_hash_1"] = "wrong"
        with self.assertRaises(RuntimeError):
            validate_confirmation_collection(manifest, corrupted)

    def test_exact_csv_writer_rejects_unexpected_fields(self) -> None:
        with TemporaryDirectory() as temporary:
            path = Path(temporary) / "rows.csv"
            with self.assertRaises(RuntimeError):
                write_csv(path, [{"expected": 1, "unexpected": 2}], ["expected"])

    def test_version_pointer_preserves_previous_immutable_version(self) -> None:
        with TemporaryDirectory() as temporary:
            run_dir = Path(temporary) / "development"
            manifest_path = create_development(run_dir, require_clean=False)
            manifest = load_manifest(manifest_path)
            first = next_version_path(run_dir, "candidate_versions")
            first.mkdir()
            (first / ARTIFACT_INDEX_NAME).write_text("{}\n", encoding="utf-8")
            (first / VALIDATION_NAME).write_text("{}\n", encoding="utf-8")
            write_version_pointer(run_dir, "CANDIDATE.json", first, manifest)
            second = next_version_path(run_dir, "candidate_versions")
            second.mkdir()
            (second / ARTIFACT_INDEX_NAME).write_text("{}\n", encoding="utf-8")
            (second / VALIDATION_NAME).write_text("{}\n", encoding="utf-8")
            write_version_pointer(run_dir, "CANDIDATE.json", second, manifest)
            selected, _ = load_version_pointer(run_dir, "CANDIDATE.json")
            self.assertEqual(selected, second)
            self.assertTrue(first.exists())

    def test_qacct_parser_preserves_exact_task_records(self) -> None:
        output = """
==============================================================
qname campus2.q
hostname n1
jobname r6array
jobnumber 123
taskid 1
slots 1
failed 0
exit_status 0
==============================================================
qname campus2.q
hostname n2
jobname r6array
jobnumber 123
taskid 2
slots 1
failed 0
exit_status 0
"""
        records = parse_qacct_records(output)
        self.assertEqual(len(records), 2)
        self.assertEqual({row["taskid"] for row in records}, {"1", "2"})
        corrupted = output.replace("exit_status 0", "exit_status 1", 1)
        self.assertEqual(parse_qacct_records(corrupted)[0]["exit_status"], "1")

    def test_scheduler_evidence_binds_resources_and_job_files(self) -> None:
        with TemporaryDirectory() as temporary:
            run_dir = Path(temporary) / "development"
            manifest_path = create_development(run_dir, require_clean=False)
            manifest = load_manifest(manifest_path)
            evidence = write_scheduler_fixture(run_dir, manifest)
            jobs = run_dir / "scheduler" / "jobs.json"
            validate_scheduler_evidence(manifest_path)
            evidence["task_h_data"] = "3G"
            jobs.write_text(json.dumps(evidence), encoding="utf-8")
            with self.assertRaises(RuntimeError):
                validate_scheduler_evidence(manifest_path)

    def test_qacct_validation_rejects_raw_evidence_corruption(self) -> None:
        with TemporaryDirectory() as temporary:
            run_dir = Path(temporary) / "development"
            manifest_path = create_development(run_dir, require_clean=False)
            manifest = load_manifest(manifest_path)
            scheduler = write_scheduler_fixture(run_dir, manifest)
            version = run_dir / "candidate"
            raw_dir = version / "qacct_raw"
            raw_dir.mkdir(parents=True)

            def record(job_id: str, job_name: str, task_id: str | None) -> dict:
                value = {
                    "qname": "campus2.q",
                    "hostname": "node",
                    "jobname": job_name,
                    "jobnumber": job_id,
                    "slots": "1",
                    "failed": "0",
                    "exit_status": "0",
                }
                if task_id is not None:
                    value["taskid"] = task_id
                return value

            array_records = [
                record("101", "array-name", str(index))
                for index in range(1, int(manifest["task_count"]) + 1)
            ]
            collector_records = [record("102", "collector-name", None)]

            def render(records: list[dict]) -> str:
                blocks = []
                for item in records:
                    blocks.append(
                        "==============================================================\n"
                        + "\n".join(f"{key} {value}" for key, value in item.items())
                    )
                return "\n".join(blocks) + "\n"

            jobs = []
            for job_id, records in (("101", array_records), ("102", collector_records)):
                raw_path = raw_dir / f"job_{job_id}.txt"
                raw_path.write_text(render(records), encoding="utf-8")
                jobs.append(
                    {
                        "job_id": job_id,
                        "failed_values": ["0"] * len(records),
                        "exit_status_values": ["0"] * len(records),
                        "task_ids": [row["taskid"] for row in records if "taskid" in row],
                        "record_count": len(records),
                        "qacct_records": records,
                        "raw_sha256": sha256_file(raw_path),
                        "raw_path": str(raw_path.relative_to(version)),
                    }
                )
            (version / "qacct_evidence.json").write_text(
                json.dumps({"manifest_hash": manifest["manifest_hash"], "jobs": jobs}),
                encoding="utf-8",
            )
            validate_qacct_evidence(manifest_path, version)
            (raw_dir / "job_101.txt").write_text("corrupted\n", encoding="utf-8")
            with self.assertRaises(RuntimeError):
                validate_qacct_evidence(manifest_path, version)

    @patch("scripts.positive_need_workflow.validate_source_identity")
    @patch("scripts.positive_need_workflow.platform.system", return_value="Linux")
    def test_local_readback_cannot_be_promoted_on_server(
        self, _platform_system, _source_identity
    ) -> None:
        with TemporaryDirectory() as temporary:
            manifest_path = create_development(
                Path(temporary) / "development", require_clean=False
            )
            with self.assertRaisesRegex(RuntimeError, "local Mac"):
                verify_local_readback(manifest_path)

    @patch("scripts.positive_need_workflow.validate_source_identity")
    @patch("scripts.positive_need_workflow.validate_qacct_evidence")
    @patch("scripts.positive_need_workflow.validate_collected_semantics")
    @patch("scripts.positive_need_workflow.validate_scheduler_task_bindings")
    @patch("scripts.positive_need_workflow.validate_scheduler_evidence", return_value={})
    @patch("scripts.positive_need_workflow.progress", return_value={"complete": True})
    @patch("scripts.positive_need_workflow.platform.node", return_value="local-mac")
    @patch("scripts.positive_need_workflow.platform.system", return_value="Darwin")
    def test_confirmation_local_readback_promotes_validated_version(
        self,
        _platform_system,
        _platform_node,
        _progress,
        _scheduler_evidence,
        _scheduler_bindings,
        _collected_semantics,
        _qacct,
        _source_identity,
    ) -> None:
        with TemporaryDirectory() as temporary:
            run_dir = Path(temporary) / "confirmation"
            run_dir.mkdir()
            manifest_path = run_dir / MANIFEST_NAME
            manifest = {
                "analysis": "confirmation",
                "mode": "serious",
                "runtime": {"hostname": "hoffman-node"},
            }
            manifest["manifest_hash"] = manifest_hash(manifest)
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            candidate = run_dir / "candidate_versions" / "version_0001"
            candidate.mkdir(parents=True)
            readiness_path = candidate / "readiness_classification.json"
            readiness_path.write_text(
                json.dumps(
                    {
                        "candidate_readiness_classification": (
                            "ready_for_experiment_design_planning"
                        ),
                        "readiness_classification": "invalid_evidence",
                        "evidence_status": "pending_qacct_and_local_readback",
                    }
                ),
                encoding="utf-8",
            )
            (candidate / VALIDATION_NAME).write_text(
                json.dumps({"valid": True}), encoding="utf-8"
            )
            (candidate / "qacct_evidence.json").write_text("{}\n", encoding="utf-8")
            (candidate / ARTIFACT_INDEX_NAME).write_text(
                json.dumps(
                    {
                        "readiness_classification.json": {
                            "sha256": sha256_file(readiness_path),
                            "bytes": readiness_path.stat().st_size,
                            "rows": None,
                        }
                    }
                ),
                encoding="utf-8",
            )
            with patch(
                "scripts.positive_need_workflow.load_manifest",
                return_value=manifest,
            ), patch(
                "scripts.positive_need_workflow.load_version_pointer",
                return_value=(candidate, {}),
            ):
                verify_local_readback(manifest_path)
            validated, _ = load_version_pointer(run_dir, "CURRENT.json")
            readiness = json.loads(
                (validated / "readiness_classification.json").read_text(
                    encoding="utf-8"
                )
            )
            completion = json.loads(
                (validated / "COMPLETED.json").read_text(encoding="utf-8")
            )
            self.assertEqual(
                readiness["readiness_classification"],
                "ready_for_experiment_design_planning",
            )
            self.assertTrue(completion["scientific_completion"])

    @patch("scripts.positive_need_workflow.validate_source_identity")
    @patch("scripts.positive_need_workflow.validate_numerical_case")
    def test_corrupt_completed_shard_is_recomputed_deterministically(
        self, validate_case, _validate_source
    ) -> None:
        with TemporaryDirectory() as temporary:
            output = Path(temporary) / "development"
            manifest_path = create_development(output, require_clean=False)
            manifest = load_manifest(manifest_path)
            task_index = int(manifest["environment_task_count"])
            case = dict(manifest["numerical_validation_cases"][0])
            row = {
                **case,
                "gh_order": 31,
                "gh_reference_order": 61,
                "primary_action_values": {
                    "terminate": 1.0,
                    "sample_1": 2.0,
                    "sample_2": 1.5,
                },
                "reference_action_values": {
                    "terminate": 1.0,
                    "sample_1": 2.0,
                    "sample_2": 1.5,
                },
                "primary_reference_action_errors": {
                    "terminate": 0.0,
                    "sample_1": 0.0,
                    "sample_2": 0.0,
                },
                "gh_max_action_value_error": 0.0,
                "gh_action": "terminate",
                "gh_reference_action": "terminate",
                "terminal_grid_allocation_error": 0.0,
                "terminal_grid_value_error": 0.0,
                "terminal_reference_action": "terminate",
                "dense_reference_error": 0.0,
                "dense_action_values": {
                    "terminate": 1.0,
                    "sample_1": 2.0,
                    "sample_2": 1.5,
                },
                "primary_dense_action_errors": {
                    "terminate": 0.0,
                    "sample_1": 0.0,
                    "sample_2": 0.0,
                },
                "dense_reference_action": "terminate",
                "dense_reference_performed": 1.0,
                "passed": 1.0,
            }
            validate_case.return_value = row
            run_task(manifest_path, task_index)
            scientific = task_directory(manifest_path, task_index) / "numerical_validation.json"
            expected_hash = sha256_file(scientific)
            self.assertEqual(json.loads(scientific.read_text(encoding="utf-8")), row)
            validate_task(manifest_path, task_index)
            scientific.write_text("{}\n", encoding="utf-8")
            state = progress(manifest_path)
            self.assertEqual(state["invalid_task_count"], 1)
            run_task(manifest_path, task_index)
            validate_task(manifest_path, task_index)
            self.assertEqual(sha256_file(scientific), expected_hash)

            status = task_directory(manifest_path, task_index) / "status.json"
            status.write_text("{\"status\":", encoding="utf-8")
            state = progress(manifest_path)
            self.assertEqual(state["invalid_task_count"], 1)
            run_task(manifest_path, task_index)
            validate_task(manifest_path, task_index)
            self.assertEqual(sha256_file(scientific), expected_hash)

    @patch("scripts.positive_need_workflow.validate_source_identity")
    @patch("scripts.positive_need_workflow.load_version_pointer")
    @patch("scripts.positive_need_workflow.validate_development_for_confirmation")
    def test_confirmation_serial_and_shard_outputs_are_identical(
        self, validate_development, load_pointer, _validate_source
    ) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            development = root / "development"
            create_development(development, require_clean=False)
            selection = {
                "selection_status": "selected_without_rr_behavior",
                "target_environment": "positive_need_gap=high_sigma=2_sample_cost=0.02",
                "target_environment_hash": "development-test-target",
                "target_sample_time_cost": 0.02,
                "control_environment": "positive_need_gap=high_sigma=2_sample_cost=8",
                "control_environment_hash": "development-test-control",
                "control_sample_time_cost": 8.0,
                "gap_class": "high",
                "sigma_sample": 2.0,
            }
            validate_development.return_value = selection
            version = development / "validated_versions" / "version_0001"
            version.mkdir(parents=True)
            (version / "selected_target_control.json").write_text(
                json.dumps(selection), encoding="utf-8"
            )
            load_pointer.return_value = (
                version,
                {
                    "version_path": "validated_versions/version_0001",
                    "pointer_hash": "test-pointer",
                },
            )
            confirmation = root / "confirmation"
            manifest_path = create_confirmation(
                confirmation, development, "smoke", require_clean=False
            )
            manifest = load_manifest(manifest_path)
            run_task(manifest_path, 0)
            shard_path = task_directory(manifest_path, 0) / "episodes.csv"

            spec = load_positive_need_spec()
            task = dict(manifest["tasks"][0])
            environment = find_environment(str(task["environment"]), spec)
            episodes = build_finite_support_episodes(
                environment,
                n_episodes=1,
                episode_start=int(task["episode_index"]),
                stage=str(manifest["mode"]),
                seed_namespace=int(manifest["seed_namespace"]),
                observations_per_person=int(manifest["observations_per_person"]),
                balanced_atoms=True,
            )
            numerical = dict(spec["numerical_settings"])
            development_settings = dict(spec["development"])
            serial_rows = evaluate_serious_environment(
                environment,
                episodes,
                quadrature_order=int(numerical["matched_voi_gauss_hermite_order"]),
                manual_samples_per_person=int(
                    development_settings["confirmation_manual_samples_per_person"]
                ),
                allocation_tolerance=float(numerical["allocation_tolerance"]),
                oracle_grid_size=int(numerical["oracle_grid_size"]),
            )
            serial_path = root / "serial.csv"
            write_csv(
                serial_path,
                serial_rows,
                expected_fields=CONFIRMATION_EPISODE_SCHEMA,
            )
            self.assertEqual(sha256_file(serial_path), sha256_file(shard_path))
            self.assertEqual(read_csv(serial_path), read_csv(shard_path))


if __name__ == "__main__":
    unittest.main()
