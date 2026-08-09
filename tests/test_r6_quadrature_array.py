from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from scripts.r6_prefeedback_workflow import digest, sha256_file
from scripts.r6_quadrature_array import (
    FINAL_RESULT_NAME,
    MANIFEST_NAME,
    ORDER_PAIRS,
    PROVISIONAL_RESULT_NAME,
    QACCT_DIRECTORY_NAME,
    RESULT_NAME,
    SHARD_NAME,
    audit_qacct,
    collect_diagnostic,
    create_diagnostic_manifest,
    finalize_diagnostic,
    load_diagnostic_manifest,
    run_diagnostic_task,
    shard_directory,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SUBMITTER = PROJECT_ROOT / "scripts" / "submit_hoffman2_r6_quadrature_array.sh"


def fake_rows(case: dict, dense_performed: bool) -> list[dict]:
    rows = []
    for primary, reference in ORDER_PAIRS:
        failed = primary == 51 and int(case["case_id"]) == 7
        primary_values = {"terminate": 1.0, "sample_1": 2.0, "sample_2": 1.5}
        reference_values = dict(primary_values)
        reference_values["sample_1"] += 2e-4 if failed else 5e-5
        primary_reference_errors = {
            action: abs(primary_values[action] - reference_values[action])
            for action in primary_values
        }
        dense_values = None
        primary_dense_errors = None
        if dense_performed:
            dense_values = dict(primary_values)
            dense_values["sample_2"] += 5e-5
            primary_dense_errors = {
                action: abs(primary_values[action] - dense_values[action])
                for action in primary_values
            }
        rows.append(
            {
                **case,
                "gh_order": primary,
                "gh_reference_order": reference,
                "primary_action_values": primary_values,
                "reference_action_values": reference_values,
                "primary_reference_action_errors": primary_reference_errors,
                "gh_max_action_value_error": max(primary_reference_errors.values()),
                "gh_action": "sample_1",
                "gh_reference_action": "sample_1",
                "terminal_grid_allocation_error": 0.0,
                "terminal_grid_value_error": 0.0,
                "terminal_reference_action": "sample_1",
                "dense_reference_error": (
                    max(primary_dense_errors.values()) if primary_dense_errors else 0.0
                ),
                "dense_action_values": dense_values,
                "primary_dense_action_errors": primary_dense_errors,
                "dense_reference_action": "sample_1",
                "dense_reference_performed": 1.0 if dense_performed else 0.0,
                "passed": 0.0 if failed else 1.0,
                "diagnostic_only": True,
            }
        )
    return rows


def write_scheduler_evidence(run_dir: Path, manifest: dict) -> None:
    scheduler = run_dir / "scheduler"
    evidence_dir = scheduler / "submission_evidence"
    evidence_dir.mkdir(parents=True)
    array_job = scheduler / "quadrature_array.job"
    collector_job = scheduler / "quadrature_collector.job"
    array_job.write_text("#$ -t 1-90\n#$ -tc 12\n", encoding="utf-8")
    collector_job.write_text("#$ -hold_jid 9001\n", encoding="utf-8")
    files = {
        "array.qsub.stdout": "9001.1-90:1\n",
        "array.qsub.stderr": "",
        "array.qsub.meta": "job_name=r6qda0810123456123\nqsub_exit_status=0\n",
        "collector.qsub.stdout": "9002\n",
        "collector.qsub.stderr": "",
        "collector.qsub.meta": "job_name=r6qdc0810123456123\nqsub_exit_status=0\n",
    }
    for name, content in files.items():
        (evidence_dir / name).write_text(content, encoding="utf-8")
    evidence = {
        "array_job_id": "9001",
        "collector_job_id": "9002",
        "array_job_name": "r6qda0810123456123",
        "collector_job_name": "r6qdc0810123456123",
        "throttle": 12,
        "queue": "campus2.q",
        "task_count": 90,
        "task_slots": 1,
        "collector_slots": 1,
        "array_job_path": str(array_job.relative_to(run_dir)),
        "collector_job_path": str(collector_job.relative_to(run_dir)),
        "array_job_sha256": sha256_file(array_job),
        "collector_job_sha256": sha256_file(collector_job),
        "submission_evidence": {
            str(path.relative_to(run_dir)): {
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
            }
            for path in sorted(evidence_dir.iterdir())
        },
    }
    (scheduler / "jobs.json").write_text(json.dumps(evidence), encoding="utf-8")


def qacct_output(
    job_id: str,
    job_name: str,
    task_ids: list[int] | None,
    *,
    failed: str = "0",
    exit_status: str = "0",
) -> str:
    records = []
    for task_id in task_ids if task_ids is not None else [None]:
        fields = [
            ("qname", "campus2.q@compute-01"),
            ("hostname", "compute-01"),
            ("jobname", job_name),
            ("jobnumber", job_id),
        ]
        if task_id is not None:
            fields.append(("taskid", str(task_id)))
        fields.extend(
            [
                ("slots", "1"),
                ("failed", failed),
                ("exit_status", exit_status),
            ]
        )
        records.append(
            "==============================================================\n"
            + "\n".join(f"{key:<13} {value}" for key, value in fields)
            + "\n"
        )
    return "".join(records)


def rewrite_shard(path: Path, mutate) -> None:
    shard = json.loads(path.read_text(encoding="utf-8"))
    mutate(shard)
    shard["rows_hash"] = digest(shard["rows"])
    payload = dict(shard)
    payload.pop("shard_hash", None)
    shard["shard_hash"] = digest(payload)
    path.write_text(json.dumps(shard), encoding="utf-8")


class R6QuadratureArrayTests(unittest.TestCase):
    def test_manifest_freezes_exact_cases_pairs_and_hashes(self) -> None:
        with TemporaryDirectory() as temporary:
            path = create_diagnostic_manifest(
                Path(temporary) / "quadrature", require_clean=False
            )
            manifest = load_diagnostic_manifest(path)

        self.assertEqual(manifest["task_count"], 90)
        self.assertEqual(manifest["expected_rows_per_task"], 4)
        self.assertEqual(manifest["order_pairs"], [list(pair) for pair in ORDER_PAIRS])
        self.assertEqual(
            [case["case_id"] for case in manifest["numerical_cases"]], list(range(90))
        )
        self.assertEqual(manifest["numerical_cases_hash"], digest(manifest["numerical_cases"]))
        self.assertEqual(len(manifest["dense_reference_case_ids"]), 36)
        self.assertEqual(
            manifest["dense_reference_case_ids_hash"],
            digest(manifest["dense_reference_case_ids"]),
        )
        self.assertTrue(manifest["spec_hash"])
        self.assertTrue(manifest["source_hashes"])
        self.assertTrue(manifest["git_commit"])
        self.assertFalse(manifest["automatic_production_order_freeze"])
        with self.assertRaisesRegex(RuntimeError, "outside the clean checkout"):
            create_diagnostic_manifest(
                PROJECT_ROOT / "results" / "invalid_quadrature_output",
                require_clean=False,
            )

    @patch("scripts.r6_quadrature_array.validate_diagnostic_source_identity")
    @patch("scripts.r6_quadrature_array.diagnose_quadrature_orders")
    def test_collector_requires_90_exact_four_pair_shards(
        self, diagnose, _source_identity
    ) -> None:
        with TemporaryDirectory() as temporary:
            run_dir = Path(temporary) / "quadrature"
            manifest_path = create_diagnostic_manifest(run_dir, require_clean=False)
            manifest = load_diagnostic_manifest(manifest_path)
            dense_case_ids = set(manifest["dense_reference_case_ids"])
            diagnose.side_effect = lambda case_id, _pairs: fake_rows(
                manifest["numerical_cases"][case_id], case_id in dense_case_ids
            )
            for task_index in range(90):
                with patch.dict(
                    os.environ,
                    {"JOB_ID": "9001", "SGE_TASK_ID": str(task_index + 1)},
                    clear=False,
                ):
                    run_diagnostic_task(manifest_path, task_index)
            write_scheduler_evidence(run_dir, manifest)

            missing = shard_directory(manifest_path, 89)
            removed = run_dir / "removed_case_089"
            missing.rename(removed)
            with self.assertRaisesRegex(RuntimeError, "shard is missing"):
                collect_diagnostic(manifest_path)
            removed.rename(missing)

            result_path = collect_diagnostic(manifest_path)
            result = json.loads(result_path.read_text(encoding="utf-8"))

            self.assertEqual(result_path.name, PROVISIONAL_RESULT_NAME)
            self.assertFalse(result["finalized"])
            self.assertEqual(result["evidence_status"], "provisional_pending_qacct")
            self.assertEqual(len(result["per_case"]), 360)
            self.assertEqual(len(result["aggregate"]), 4)
            self.assertEqual(len(result["shard_provenance"]), 90)
            self.assertEqual(result["manifest_runtime"], manifest["runtime"])
            self.assertTrue(result["scheduler_evidence"]["sha256"])
            self.assertTrue(all(row["case_count"] == 90 for row in result["aggregate"]))
            self.assertTrue(
                all(row["dense_reference_case_count"] == 36 for row in result["aggregate"])
            )
            self.assertEqual(result["aggregate"][0]["failed_case_ids"], [7])
            self.assertFalse(result["aggregate"][0]["valid"])
            self.assertTrue(all(row["valid"] for row in result["aggregate"][1:]))
            self.assertFalse(result["automatic_production_order_freeze"])

            with self.assertRaisesRegex(RuntimeError, "qacct evidence is missing"):
                finalize_diagnostic(manifest_path)

            incomplete_array = qacct_output(
                "9001", "r6qda0810123456123", list(range(1, 90))
            )
            collector_output = qacct_output("9002", "r6qdc0810123456123", None)
            with patch(
                "scripts.r6_quadrature_array.subprocess.check_output",
                side_effect=["malformed qacct\n", collector_output],
            ):
                with self.assertRaisesRegex(RuntimeError, "missing required fields"):
                    audit_qacct(manifest_path, ["9001", "9002"])
            self.assertFalse((run_dir / QACCT_DIRECTORY_NAME).exists())

            with patch(
                "scripts.r6_quadrature_array.subprocess.check_output",
                side_effect=[incomplete_array, collector_output],
            ):
                with self.assertRaisesRegex(RuntimeError, "exactly 1..90"):
                    audit_qacct(manifest_path, ["9001", "9002"])
            self.assertFalse((run_dir / QACCT_DIRECTORY_NAME).exists())

            array_output = qacct_output(
                "9001", "r6qda0810123456123", list(range(1, 91))
            )
            with patch(
                "scripts.r6_quadrature_array.subprocess.check_output",
                side_effect=[array_output, collector_output],
            ):
                audit_qacct(manifest_path, ["9001", "9002"])

            raw_array = run_dir / QACCT_DIRECTORY_NAME / "raw" / "array_job_9001.txt"
            raw_array.write_text("malformed\n", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "raw evidence mismatch"):
                finalize_diagnostic(manifest_path)
            raw_array.write_text(array_output, encoding="utf-8")

            final_path = finalize_diagnostic(manifest_path)
            final = json.loads(final_path.read_text(encoding="utf-8"))
            self.assertEqual(final_path.name, FINAL_RESULT_NAME)
            self.assertTrue(final["finalized"])
            self.assertEqual(final["evidence_status"], "finalized_qacct_validated")
            self.assertEqual(len(final["qacct_evidence"]["record"]["jobs"]), 2)
            self.assertTrue(final["qacct_evidence"]["sha256"])

            shard_path = shard_directory(manifest_path, 17) / SHARD_NAME
            original = shard_path.read_text(encoding="utf-8")
            rewrite_shard(
                shard_path,
                lambda shard: shard["rows"][0].pop("reference_action_values"),
            )
            with self.assertRaisesRegex(RuntimeError, "evidence is incomplete"):
                collect_diagnostic(manifest_path)

            shard_path.write_text(original, encoding="utf-8")
            rewrite_shard(
                shard_path,
                lambda shard: shard["runtime"].update({"numpy": "0.invalid"}),
            )
            with self.assertRaisesRegex(RuntimeError, "NumPy version mismatch"):
                collect_diagnostic(manifest_path)

            shard_path.write_text(original, encoding="utf-8")
            rewrite_shard(
                shard_path,
                lambda shard: shard["rows"][0].update({"gh_order": 53}),
            )
            with self.assertRaisesRegex(RuntimeError, "order-pair"):
                collect_diagnostic(manifest_path)

    def test_submitter_has_valid_syntax_and_frozen_one_slot_array(self) -> None:
        result = subprocess.run(
            ["bash", "-n", str(SUBMITTER)],
            cwd=PROJECT_ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        text = SUBMITTER.read_text(encoding="utf-8")
        self.assertIn('${THROTTLE:?', text)
        self.assertIn("#$ -t 1-90", text)
        self.assertIn("#$ -tc ${THROTTLE}", text)
        self.assertIn("r6_submit_job \"array\"", text)
        self.assertIn("r6_submit_job \"collector\"", text)
        self.assertIn("submission_evidence", text)
        self.assertIn("audit-qacct", text)
        self.assertIn("finalize", text)
        self.assertIn("outside the clean checkout", text)


if __name__ == "__main__":
    unittest.main()
