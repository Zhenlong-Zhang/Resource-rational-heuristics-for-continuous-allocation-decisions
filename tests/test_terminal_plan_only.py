"""Test purpose: validate terminal planning diagnostics without executing scientific tasks."""

from __future__ import annotations

import ast
from contextlib import ExitStack
import inspect
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import scripts.terminal_validation_array as terminal_cli
import src.experiments.terminal_evidence_rows as evidence
import src.experiments.terminal_execution as execution
from src.experiments.terminal_plan_diagnostics import (
    PLAN_DIAGNOSTIC_RUN_SCHEMA,
    audit_plan_diagnostic_run,
    validate_qstat_absence_text,
)


HASH = "1" * 64


def row(method: str, tie_status, symmetry_required: bool):
    return SimpleNamespace(
        method=method,
        tie_status=tie_status,
        symmetry_required=symmetry_required,
    )


class TerminalPlanOnlyTests(unittest.TestCase):
    def _diagnostic_fixture(self, root: Path):
        jobs = ({"replicate": "a", "job_id": "101"}, {"replicate": "b", "job_id": "102"})
        metadata = {
            "schema": PLAN_DIAGNOSTIC_RUN_SCHEMA,
            "stage": "smoke",
            "descriptor_count": 16,
            "queue": "campus2.q",
            "requested_slots": 1,
            "requested_memory_bytes": 2 * 1024**3,
            "max_wall_seconds": 300,
            "source_commit": "a" * 40,
            "source_tree": "b" * 40,
            "source_identity_hash": "c" * 64,
            "provider_hash": "d" * 64,
            "jobs": list(jobs),
        }
        (root / "run_metadata.json").write_text(
            json.dumps(metadata, sort_keys=True) + "\n", encoding="utf-8"
        )
        for name in (
            "jobs", "qsub_raw", "qacct", "final_qstat", "logs",
            "replicate_a", "replicate_b",
        ):
            (root / name).mkdir()
        for job in jobs:
            replicate = job["replicate"]
            job_id = job["job_id"]
            script = (
                "#$ -t 1-16\n#$ -tc 16\n#$ -l h_data=2G\n"
                "diagnose-plan --stage smoke --mode plan-only\n"
            )
            (root / "jobs" / f"plan_{replicate}.job").write_text(script, encoding="utf-8")
            (root / "qsub_raw" / f"plan_{replicate}.txt").write_text(
                f"{job_id}.1-16:1\n", encoding="utf-8"
            )
            (root / "final_qstat" / f"plan_{replicate}.raw").write_text(
                "<?xml version='1.0'?><job_info><queue_info></queue_info>"
                "<job_info></job_info></job_info>\n",
                encoding="utf-8",
            )
            (root / "final_qstat" / f"plan_{replicate}.status").write_text(
                "0\n", encoding="utf-8"
            )
            qacct = []
            for task_id in range(1, 17):
                qacct.append(
                    "\n".join((
                        "qname campus2.q@n123",
                        "hostname n123.hoffman2.idre.ucla.edu",
                        f"jobname tvp1_{replicate}",
                        f"jobnumber {job_id}",
                        f"taskid {task_id}",
                        "slots 1",
                        "failed 0",
                        "exit_status 0",
                        "cpu 00:00:01",
                        "ru_wallclock 00:00:02",
                        "maxvmem 100M",
                    ))
                )
                diagnostic = {
                    "schema": execution.PLAN_DIAGNOSTIC_SCHEMA,
                    "mode": "plan_only",
                    "suite_class": "base" if task_id <= 4 else "one_step",
                    "descriptor_index": task_id,
                    "descriptor_hash": f"{task_id:064x}",
                    "source_case_id": task_id,
                    "profile": f"profile_{task_id}",
                    "source_identity_hash": "c" * 64,
                    "provider_hash": "d" * 64,
                    "plan": {
                        "expected_methods": ("production_terminal", "reference_a"),
                        "expected_tie_row_count": task_id % 2,
                        "expected_symmetry_row_count": task_id % 3,
                    },
                    "full_projection": None,
                    "parity_pass": None,
                    "phase_seconds": (
                        ("provider_load", 0.1),
                        ("suite_reconstruction", 0.1),
                        ("source_identity_capture", 0.1),
                        ("source_reconstruction", 0.1),
                        ("plan_plan_computation_total", 0.2),
                        ("plan_canonicalization_serialization", 0.01),
                    ),
                    "diagnostic_hash": "",
                }
                diagnostic["diagnostic_hash"] = execution.logical_hash(
                    execution._without_hash(diagnostic, "diagnostic_hash")
                )
                execution.write_new_json(
                    root / f"replicate_{replicate}" / f"diagnostic_{task_id:03d}.json",
                    diagnostic,
                )
                (root / "logs" / f"plan_{replicate}.{job_id}.{task_id}.log").write_text(
                    "", encoding="utf-8"
                )
            (root / "qacct" / f"plan_{replicate}.raw").write_text(
                "\n==============================================================\n".join(qacct) + "\n",
                encoding="utf-8",
            )
        descriptors = []
        for task_id in range(1, 17):
            descriptors.append({
                "task_id": task_id,
                "suite_class": "base" if task_id <= 4 else "one_step",
                "descriptor_index": task_id,
                "descriptor_hash": f"{task_id:064x}",
                "source_case_id": task_id,
                "profile": f"profile_{task_id}",
                "source_identity_hash": "c" * 64,
                "provider_hash": "d" * 64,
            })
        return {
            "source_commit": "a" * 40,
            "source_tree": "b" * 40,
            "source_identity_hash": "c" * 64,
            "provider_hash": "d" * 64,
            "descriptors": tuple(descriptors),
        }

    def test_projection_is_exact_and_parity_detects_every_field(self):
        rows = (
            row("production_terminal", "structural_symmetry_tie", True),
            row("reference_a", "unique", True),
            row("reference_b", "certified_value_tie", True),
            row("agreement", "unique", True),
        )
        expected = evidence.TerminalEvidencePlan(
            evidence.TERMINAL_METHOD_ORDER,
            2,
            4,
        )
        self.assertEqual(evidence.project_terminal_evidence_plan(rows), expected)
        self.assertEqual(
            evidence.require_terminal_evidence_plan_parity(expected, rows),
            expected,
        )

        changed = (
            evidence.TerminalEvidencePlan(expected.expected_methods[:-1], 2, 3),
            evidence.TerminalEvidencePlan(expected.expected_methods, 1, 4),
            evidence.TerminalEvidencePlan(expected.expected_methods, 2, 3),
        )
        for plan in changed:
            with self.subTest(plan=plan):
                with self.assertRaisesRegex(RuntimeError, "differs from frozen manifest"):
                    evidence.require_terminal_evidence_plan_parity(plan, rows)

        with self.assertRaisesRegex(RuntimeError, "frozen order"):
            evidence.project_terminal_evidence_plan(tuple(reversed(rows)))

    def test_plan_only_succeeds_with_every_full_evidence_boundary_blocked(self):
        symmetry = SimpleNamespace(valid=True)
        production = SimpleNamespace(
            allocation=0.5,
            tie_status="structural_symmetry_tie",
            structural_symmetry=symmetry,
        )
        reference_a = SimpleNamespace(
            evaluation_cap=10,
            tie_status="unique",
            structural_symmetry=symmetry,
        )
        descriptor = SimpleNamespace(suite_class="base")
        mdp = object()
        belief = SimpleNamespace(weights=(1.0,), deliberation_time=0.0)

        def forbidden(*_args, **_kwargs):
            raise AssertionError("full evidence boundary was reached")

        plan_phases = {}
        patchers = (
            patch.object(evidence, "terminal_descriptor_source_failures", return_value=()),
            patch.object(evidence, "optimize_terminal_allocation", return_value=production),
            patch.object(
                evidence,
                "optimal_terminal_results_for_weight_rows",
                return_value=(production,),
            ),
            patch.object(evidence, "validate_structural_symmetry_proof", return_value=True),
            patch.object(evidence, "solve_terminal_reference_a", return_value=reference_a),
            patch.object(
                evidence,
                "source_validate_terminal_reference_record",
                side_effect=forbidden,
            ),
            patch.object(evidence, "validate_production_against_reference_a", side_effect=forbidden),
            patch.object(
                evidence,
                "terminal_reference_b_trigger_reasons",
                return_value=("all_base_beliefs",),
            ),
            patch.object(
                evidence,
                "validate_terminal_reference_agreement",
                side_effect=forbidden,
            ),
            patch.object(evidence, "evaluate_terminal_evidence_descriptor", side_effect=forbidden),
            patch.object(evidence, "TerminalEvidenceBundle", side_effect=forbidden),
            patch.object(evidence, "build_terminal_certificate_sidecar", side_effect=forbidden),
            patch.object(evidence, "optimize_terminal_allocation_with_trace", side_effect=forbidden),
            patch.object(evidence, "solve_terminal_reference_a_with_trace", side_effect=forbidden),
            patch.object(evidence, "solve_terminal_reference_b_with_trace", side_effect=forbidden),
            patch.object(evidence.gzip, "compress", side_effect=forbidden),
            patch.object(evidence, "_logical_hash", side_effect=forbidden),
        )
        with ExitStack() as stack:
            for patcher in patchers:
                stack.enter_context(patcher)
            observed = evidence.evaluate_terminal_evidence_plan(
                descriptor,
                mdp,
                belief,
                phase_seconds=plan_phases,
            )

        self.assertEqual(
            observed,
            evidence.TerminalEvidencePlan(evidence.TERMINAL_METHOD_ORDER, 1, 4),
        )
        self.assertIn("plan_computation_total", plan_phases)
        self.assertNotIn("formal_evidence_generation", plan_phases)

    def test_planning_dependency_guard_excludes_full_evidence_symbols(self):
        forbidden = {
            "evaluate_terminal_evidence_descriptor",
            "TerminalEvidenceBundle",
            "build_terminal_certificate_sidecar",
            "optimize_terminal_allocation_with_trace",
            "solve_terminal_reference_a_with_trace",
            "solve_terminal_reference_b_with_trace",
            "source_validate_terminal_reference_record",
            "validate_production_against_reference_a",
            "validate_terminal_reference_agreement",
        }
        for function in (
            execution._expected_descriptor_plan,
            evidence.evaluate_terminal_evidence_plan,
        ):
            tree = ast.parse(inspect.getsource(function))
            names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
            attributes = {
                node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)
            }
            self.assertFalse(forbidden & (names | attributes))

    def test_scheduled_diagnostic_modes_are_fail_closed_and_profiled(self):
        descriptor = SimpleNamespace(
            suite_class="base",
            descriptor_index=72,
            descriptor_hash="2" * 64,
            source_case_id=72,
            profile="both_predictive_means",
        )
        provider = SimpleNamespace(provider_hash="3" * 64)
        plan = evidence.TerminalEvidencePlan(evidence.TERMINAL_METHOD_ORDER, 2, 0)
        rows = tuple(
            row(method, "certified_value_tie" if index < 2 else "unique", False)
            for index, method in enumerate(evidence.TERMINAL_METHOD_ORDER)
        )
        bundle = SimpleNamespace(rows=rows)

        with (
            patch.object(execution, "reconstruct_terminal_evidence_source", return_value=(object(), object())),
            patch.object(execution, "evaluate_terminal_evidence_plan", return_value=plan),
            patch.object(
                execution,
                "evaluate_terminal_evidence_descriptor",
                side_effect=AssertionError("full evidence must be opt-in"),
            ),
        ):
            diagnostic = execution.create_terminal_plan_diagnostic(
                descriptor,
                provider,
                source_identity_hash=HASH,
                include_full_evidence=False,
            )
        self.assertEqual(diagnostic["mode"], "plan_only")
        self.assertIsNone(diagnostic["full_projection"])
        self.assertIsNone(diagnostic["parity_pass"])

        with (
            patch.object(execution, "reconstruct_terminal_evidence_source", return_value=(object(), object())),
            patch.object(execution, "evaluate_terminal_evidence_plan", return_value=plan),
            patch.object(execution, "evaluate_terminal_evidence_descriptor", return_value=bundle),
        ):
            parity = execution.create_terminal_plan_diagnostic(
                descriptor,
                provider,
                source_identity_hash=HASH,
                include_full_evidence=True,
            )
        self.assertEqual(parity["mode"], "parity")
        self.assertTrue(parity["parity_pass"])
        self.assertEqual(parity["plan"], parity["full_projection"])
        self.assertIn(
            "formal_evidence_generation",
            dict(parity["phase_seconds"]),
        )

    def test_cli_exposes_scheduled_diagnostic_and_phase_profiles(self):
        parser = terminal_cli.build_parser()
        diagnostic = parser.parse_args([
            "diagnose-plan",
            "--stage", "smoke",
            "--descriptor-position", "3",
            "--mode", "plan-only",
            "--output", "diagnostic.json",
        ])
        self.assertEqual(diagnostic.descriptor_position, 3)
        fragment = parser.parse_args([
            "freeze-plan-fragment",
            "--stage", "smoke",
            "--shard-index", "1",
            "--shard-count", "16",
            "--output", "fragment.json",
            "--profile-output", "profile.json",
        ])
        self.assertEqual(str(fragment.profile_output), "profile.json")
        profile = terminal_cli.phase_profile(
            "freeze-plan-fragment",
            {"plan_computation": 1.0},
            stage="smoke",
        )
        self.assertEqual(profile["schema"], terminal_cli.PHASE_PROFILE_SCHEMA)
        execution._validate_self_hash(profile, "profile_hash", "phase profile")

    def test_p1_auditor_accepts_exact_retries_and_rejects_corruption(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            context = self._diagnostic_fixture(root)
            audit = audit_plan_diagnostic_run(root, expected_context=context)
            self.assertTrue(audit["audit_pass"])
            self.assertEqual(audit["descriptor_count"], 16)
            self.assertEqual(len(audit["retry_records"]), 16)

        mutations = (
            ("missing", "output coverage"),
            ("mismatch", "byte-identical"),
            ("timeout", "300-second"),
            ("memory", "memory gate"),
            ("traceback", "forbidden failure marker"),
        )
        for mutation, message in mutations:
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                context = self._diagnostic_fixture(root)
                if mutation == "missing":
                    (root / "replicate_b" / "diagnostic_016.json").unlink()
                elif mutation == "mismatch":
                    path = root / "replicate_b" / "diagnostic_001.json"
                    raw = execution._decode(dict(execution._load_json(path)))
                    raw["plan"] = dict(raw["plan"])
                    raw["plan"]["expected_tie_row_count"] = 9
                    raw["diagnostic_hash"] = execution.logical_hash(
                        execution._without_hash(raw, "diagnostic_hash")
                    )
                    path.unlink()
                    execution.write_new_json(path, raw)
                elif mutation in ("timeout", "memory"):
                    path = root / "qacct" / "plan_a.raw"
                    text = path.read_text(encoding="utf-8")
                    if mutation == "timeout":
                        text = text.replace("ru_wallclock 00:00:02", "ru_wallclock 00:05:01", 1)
                    else:
                        text = text.replace("maxvmem 100M", "maxvmem 1.1G", 1)
                    path.write_text(text, encoding="utf-8")
                else:
                    (root / "logs" / "plan_a.101.1.log").write_text(
                        "Traceback (most recent call last):\n", encoding="utf-8"
                    )
                with self.assertRaisesRegex(RuntimeError, message):
                    audit_plan_diagnostic_run(root, expected_context=context)

    def test_p1_auditor_rejects_descriptor_source_and_qstat_substitution(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            context = self._diagnostic_fixture(root)
            substituted = dict(context)
            descriptors = list(context["descriptors"])
            descriptors[2] = dict(descriptors[2], descriptor_hash="f" * 64)
            substituted["descriptors"] = tuple(descriptors)
            with self.assertRaisesRegex(RuntimeError, "frozen smoke mapping"):
                audit_plan_diagnostic_run(root, expected_context=substituted)

            reordered = dict(context)
            descriptors = list(context["descriptors"])
            descriptors[0], descriptors[1] = descriptors[1], descriptors[0]
            reordered["descriptors"] = tuple(descriptors)
            with self.assertRaisesRegex(RuntimeError, "frozen smoke mapping"):
                audit_plan_diagnostic_run(root, expected_context=reordered)

            mismatches = (
                ("source_commit", "e" * 40),
                ("source_tree", "f" * 40),
                ("source_identity_hash", "e" * 64),
                ("provider_hash", "f" * 64),
            )
            for field, value in mismatches:
                with self.subTest(field=field), self.assertRaisesRegex(RuntimeError, field):
                    audit_plan_diagnostic_run(
                        root, expected_context=dict(context, **{field: value})
                    )

        empty_snapshot = (
            "<?xml version='1.0'?><job_info><queue_info></queue_info>"
            "<job_info></job_info></job_info>\n"
        )
        validate_qstat_absence_text(empty_snapshot, "101", 0)
        invalid_qstat = (
            (empty_snapshot, 1),
            ("unable to contact qmaster\n", 1),
            (
                "<?xml version='1.0'?><job_info><queue_info><job_list>"
                "<JB_job_number>101</JB_job_number></job_list></queue_info>"
                "<job_info></job_info></job_info>\n",
                0,
            ),
            ("Following jobs do not exist or permissions are not sufficient:\n101\n", 1),
        )
        for text, status in invalid_qstat:
            with self.subTest(text=text, status=status), self.assertRaisesRegex(
                RuntimeError, "authoritatively"
            ):
                validate_qstat_absence_text(text, "101", status)
        malformed_structures = (
            "<not_qstat/>",
            "<job_info><queue_info/></job_info>",
            (
                "<job_info><queue_info><job_list><JB_job_number>abc</JB_job_number>"
                "</job_list></queue_info><job_info/></job_info>"
            ),
        )
        for text in malformed_structures:
            with self.subTest(structure=text), self.assertRaisesRegex(
                RuntimeError, "authoritatively|malformed"
            ):
                execution.validate_qstat_snapshot_text(text, 0)
        tagged = (
            "<job_info><queue_info><job_list><JB_job_number>202</JB_job_number>"
            "<JB_name>tvsmoke_0123456789abcdef</JB_name></job_list></queue_info>"
            "<job_info/></job_info>"
        )
        with self.assertRaisesRegex(RuntimeError, "tagged-job"):
            execution.validate_qstat_snapshot_text(
                tagged, 0, absent_run_tag="0123456789abcdef"
            )

    def test_real_submitters_wire_immutable_profiles_and_p1_evidence(self):
        project_root = Path(__file__).resolve().parents[1]
        setup = (project_root / "scripts/submit_hoffman2_terminal_manifest_setup.sh").read_text(
            encoding="utf-8"
        )
        self.assertIn("profiles_a", setup)
        self.assertIn("profiles_b", setup)
        self.assertEqual(setup.count("--profile-output"), 2)
        self.assertIn(".conda/envs/rr-allocation/bin/python", setup)
        self.assertNotIn("git -C", setup)

        p1 = (project_root / "scripts/submit_hoffman2_terminal_plan_diagnostic.sh").read_text(
            encoding="utf-8"
        )
        for token in (
            "Refusing to overwrite",
            ".conda/envs/rr-allocation/bin/python",
            "#$ -t 1-16",
            "--mode plan-only",
            "qsub_raw",
            "QACCT_BIN",
            '-xml -u "${scheduler_user}"',
            "audit_terminal_plan_diagnostics.py",
        ):
            self.assertIn(token, p1)
        self.assertNotIn("git -C", p1)
        formal = (project_root / "scripts/submit_hoffman2_terminal_validation.sh").read_text(
            encoding="utf-8"
        )
        self.assertIn(".conda/envs/rr-allocation/bin/python", formal)
        self.assertNotIn("git -C", formal)


if __name__ == "__main__":
    unittest.main()
