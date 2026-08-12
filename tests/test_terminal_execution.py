from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import src.experiments.terminal_execution as execution
from src.experiments.terminal_validation_suite import (
    AUTHORITATIVE_PROVIDER_KIND,
    DESCRIPTOR_SCHEMA,
    CanonicalBaseProvider,
    TerminalValidationDescriptor,
)
from src.mdp.finite_support import FiniteSupportAtom
from tests.test_terminal_evidence_rows import descriptor_for
from tests.test_terminal_optimizer import one_atom_mdp


HASH = "1" * 64


def source_identity():
    value = {
        "schema": execution.SOURCE_IDENTITY_SCHEMA,
        "commit": "2" * 40,
        "tree": "3" * 40,
        "source_hashes": (("source.py", "4" * 64),),
        "source_hashes_hash": execution.logical_hash((("source.py", "4" * 64),)),
        "identity_hash": "",
    }
    value["identity_hash"] = execution.logical_hash(
        execution._without_hash(value, "identity_hash")
    )
    return value


def provider():
    return CanonicalBaseProvider(
        provider_kind=AUTHORITATIVE_PROVIDER_KIND,
        source_identity_hash="5" * 64,
        diagnostic_only=False,
        records=(),
        records_hash="6" * 64,
        provider_hash="7" * 64,
    )


def descriptor(suite_class, index, owner, *, profile="ordinary", action=(), z=None):
    value = TerminalValidationDescriptor(
        schema=DESCRIPTOR_SCHEMA,
        suite_class=suite_class,
        suite_version=f"{suite_class}_v1",
        descriptor_index=index,
        component_validation_only=suite_class != "base",
        environment_selection_eligible=False,
        legacy_spec_hash="8" * 64,
        legacy_numerical_case_hash="9" * 64,
        scientific_spec_hash="a" * 64,
        numerical_method_config_hash="b" * 64,
        source_case_id=owner if suite_class != "reachable_core" else None,
        environment="test",
        environment_hash="c" * 64,
        support_hash="d" * 64,
        sigma_sample=1.0,
        sample_time_cost=1.0,
        profile=profile,
        orientation="symmetric" if profile == "initial_symmetric" else "+1",
        depth=0,
        deliberation_time=0.0,
        remaining_time_after_termination=10.0,
        action_sequence=tuple(action),
        offset_sequence=(),
        history=(),
        history_hash="e" * 64,
        posterior_weight_hash="f" * 64,
        canonical_belief_hash="0" * 64,
        legacy_belief_hash=None,
        local_legacy_belief_hash=None,
        legacy_reconstruction_matches=None,
        component_index=0 if suite_class == "one_step" else None,
        z_offset=z,
        reference_b_prespecified=True,
        construction_rule="test",
        construction_hash=execution.logical_hash((suite_class, index, owner, profile)),
        descriptor_hash=execution.logical_hash(("descriptor", suite_class, index, owner, profile)),
    )
    return value


def suite(name, descriptors):
    manifest = SimpleNamespace(
        manifest_hash=execution.logical_hash((name, "manifest")),
        ordered_descriptor_hash=execution.logical_hash(tuple(item.descriptor_hash for item in descriptors)),
    )
    return SimpleNamespace(manifest=manifest, descriptors=tuple(descriptors))


def resources():
    return {"queue": "campus", "h_rt_seconds": 3600, "memory_bytes": 2_000_000_000, "throttle": 32}


class TerminalExecutionTests(unittest.TestCase):
    def setUp(self):
        self.provider = provider()
        acceptance = patch.object(
            execution,
            "accepted_canonical_base_provider",
            side_effect=lambda value: value.provider_hash == self.provider.provider_hash,
        )
        self.accepted = acceptance.start()
        self.addCleanup(acceptance.stop)
        plan = patch.object(
            execution,
            "_expected_descriptor_plan",
            side_effect=lambda _descriptor, _provider: (
                execution.TERMINAL_METHOD_ORDER, 1, 1
            ),
        )
        plan.start()
        self.addCleanup(plan.stop)

    def full_suites(self):
        base = [descriptor("base", owner, owner) for owner in range(90)]
        one = [descriptor("one_step", owner, owner, action=("sample_1",), z=0) for owner in range(90)]
        return {
            "base": suite("base", base),
            "one_step": suite("one_step", one),
            "reachable_core": suite("reachable_core", ()),
        }

    def smoke_suites(self):
        base = [descriptor("base", owner, owner) for owner in range(90)]
        one = []
        for owner in range(90):
            one.append(descriptor("one_step", owner * 2, owner, action=("sample_1",), z=0))
            one.append(descriptor("one_step", owner * 2 + 1, owner, action=("sample_2",), z=0))
        profiles = (
            "initial_symmetric", "concentrated_depth_6_-1",
            "concentrated_depth_6_+1", "balanced_late_feasible",
        )
        reachable = [descriptor("reachable_core", index, 0, profile=name) for index, name in enumerate(profiles)]
        return {
            "base": suite("base", base),
            "one_step": suite("one_step", one),
            "reachable_core": suite("reachable_core", reachable),
        }

    def make_manifest(self, stage="smoke", max_size=50):
        suites = self.smoke_suites() if stage == "smoke" else self.full_suites()
        identities = SimpleNamespace(
            scientific_spec_hash="a" * 64,
            numerical_method_config_hash="b" * 64,
        )
        with patch.object(execution, "validate_terminal_validation_suite", return_value=SimpleNamespace(failures=())):
            with patch.object(execution, "canonical_base_provider_failures", return_value=()):
                with patch.object(execution, "load_terminal_validation_identities", return_value=identities):
                    manifest = execution.create_execution_manifest(
                        stage=stage,
                        suites=suites,
                        provider=self.provider,
                        acceptance_validator=self.accepted,
                        source_identity=source_identity(),
                        max_descriptors_per_subshard=max_size,
                        resources=resources(),
                        compute_ceiling_report_hash="1" * 64,
                    )
        return manifest, suites

    def assert_manifest_rejected(self, manifest, suites, pattern="source-reconstructed|mismatch|invalid|subshard|multiple"):
        identities = SimpleNamespace(
            scientific_spec_hash="a" * 64,
            numerical_method_config_hash="b" * 64,
        )
        with patch.object(execution, "validate_terminal_validation_suite", return_value=SimpleNamespace(failures=())):
            with patch.object(execution, "canonical_base_provider_failures", return_value=()):
                with patch.object(execution, "load_terminal_validation_identities", return_value=identities):
                    with self.assertRaisesRegex(RuntimeError, pattern):
                        execution.validate_execution_manifest(
                            manifest, suites, self.provider, self.accepted
                        )

    @staticmethod
    def rehash_manifest(manifest):
        manifest["manifest_hash"] = execution.logical_hash(
            execution._without_hash(manifest, "manifest_hash")
        )

    @staticmethod
    def refresh_owner(manifest, owner):
        refs = [
            ref for task in manifest["tasks"]
            if int(task["logical_case_owner"]) == owner
            for ref in task["descriptors"]
        ]
        owners = [dict(item) for item in manifest["case_owners"]]
        index = next(i for i, item in enumerate(owners) if int(item["logical_case_owner"]) == owner)
        owners[index]["descriptor_count"] = len(refs)
        owners[index]["descriptor_hash"] = execution.logical_hash(
            tuple(item["descriptor_hash"] for item in refs)
        )
        manifest["case_owners"] = tuple(owners)

    def test_manifest_planning_workers_are_scheduler_bounded(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(execution._manifest_planning_worker_count(16), 1)
        with patch.dict(
            os.environ,
            {"TERMINAL_MANIFEST_WORKERS": "8", "NSLOTS": "8"},
            clear=True,
        ):
            self.assertEqual(execution._manifest_planning_worker_count(3), 3)
        with patch.dict(
            os.environ,
            {"TERMINAL_MANIFEST_WORKERS": "9", "NSLOTS": "8"},
            clear=True,
        ):
            with self.assertRaisesRegex(RuntimeError, "scheduler allocation"):
                execution._manifest_planning_worker_count(16)
        with patch.dict(
            os.environ,
            {"TERMINAL_MANIFEST_WORKERS": "invalid"},
            clear=True,
        ):
            with self.assertRaisesRegex(RuntimeError, "positive integer"):
                execution._manifest_planning_worker_count(16)

    def test_parallel_manifest_planning_preserves_descriptor_order(self):
        descriptors = tuple(descriptor("base", index, index) for index in range(3))
        expected = tuple(((str(item.descriptor_index),), 0, 0) for item in descriptors)

        class InlinePool:
            def __init__(self, max_workers, initializer, initargs):
                self.max_workers = max_workers
                self.initializer = initializer
                self.initargs = initargs

            def __enter__(self):
                self.initializer(*self.initargs)
                return self

            def __exit__(self, *_args):
                return False

            def map(self, function, descriptors, chunksize):
                self.chunksize = chunksize
                return tuple(function(item) for item in descriptors)

        with patch.dict(
            os.environ,
            {"TERMINAL_MANIFEST_WORKERS": "3", "NSLOTS": "3"},
            clear=True,
        ):
            with patch.object(execution, "ProcessPoolExecutor", InlinePool):
                with patch.object(
                    execution,
                    "_expected_descriptor_plan",
                    side_effect=lambda item, _provider: (
                        (str(item.descriptor_index),),
                        0,
                        0,
                    ),
                ):
                    observed = execution._expected_descriptor_plans(
                        descriptors, self.provider
                    )
        self.assertEqual(observed, expected)

    def test_fast_suite_build_skips_only_redundant_content_validation(self):
        suites = self.full_suites()
        identities = SimpleNamespace(
            scientific_spec_hash="a" * 64,
            numerical_method_config_hash="b" * 64,
        )
        validation = SimpleNamespace(
            failures=(), authoritative_source_accepted=True
        )
        with patch.object(
            execution, "load_terminal_validation_identities", return_value=identities
        ), patch.object(
            execution, "build_terminal_base_suite", return_value=suites["base"]
        ), patch.object(
            execution, "build_terminal_one_step_suite", return_value=suites["one_step"]
        ), patch.object(
            execution,
            "build_terminal_reachable_core_suite",
            return_value=suites["reachable_core"],
        ), patch.object(
            execution,
            "validate_terminal_validation_suite",
            return_value=validation,
        ) as validate_suite:
            observed = execution.build_terminal_suites(
                self.provider, self.accepted, validate_contents=False
            )
            self.assertEqual(observed, suites)
            validate_suite.assert_not_called()

            execution.build_terminal_suites(self.provider, self.accepted)
            self.assertEqual(validate_suite.call_count, 3)

    def test_distributed_plan_replicates_merge_only_after_exact_agreement(self):
        suites = self.full_suites()
        identities = SimpleNamespace(
            scientific_spec_hash="a" * 64,
            numerical_method_config_hash="b" * 64,
        )
        with patch.object(
            execution,
            "validate_terminal_validation_suite",
            return_value=SimpleNamespace(failures=()),
        ) as validate_suite:
            with patch.object(execution, "canonical_base_provider_failures", return_value=()):
                with patch.object(
                    execution, "load_terminal_validation_identities", return_value=identities
                ):
                    replicate_a = tuple(
                        execution.create_manifest_plan_fragment(
                            stage="full",
                            shard_index=index,
                            shard_count=3,
                            suites=suites,
                            provider=self.provider,
                            acceptance_validator=self.accepted,
                            source_identity=source_identity(),
                        )
                        for index in range(1, 4)
                    )
                    replicate_b = tuple(dict(item) for item in replicate_a)
                    validate_suite.assert_not_called()
                    manifest, assembly = execution.merge_manifest_plan_replicates(
                        stage="full",
                        replicate_a=replicate_a,
                        replicate_b=replicate_b,
                        suites=suites,
                        provider=self.provider,
                        acceptance_validator=self.accepted,
                        source_identity=source_identity(),
                        max_descriptors_per_subshard=50,
                        resources=resources(),
                        compute_ceiling_report_hash="1" * 64,
                    )
                    self.assertEqual(manifest["expected_descriptor_count"], 180)
                    self.assertEqual(assembly["fragment_count_per_replicate"], 3)
                    self.assertTrue(assembly["pairwise_agreement"])
                    self.assertEqual(assembly["manifest_hash"], manifest["manifest_hash"])
                    self.assertEqual(validate_suite.call_count, 3)

                    forged_b = [dict(item) for item in replicate_b]
                    forged_b[0] = dict(forged_b[0])
                    references = [dict(item) for item in forged_b[0]["descriptors"]]
                    references[0]["expected_tie_row_count"] = 0
                    forged_b[0]["descriptors"] = tuple(references)
                    forged_b[0]["fragment_hash"] = execution.logical_hash(
                        execution._without_hash(forged_b[0], "fragment_hash")
                    )
                    with self.assertRaisesRegex(RuntimeError, "replicates disagree"):
                        execution.merge_manifest_plan_replicates(
                            stage="full",
                            replicate_a=replicate_a,
                            replicate_b=tuple(forged_b),
                            suites=suites,
                            provider=self.provider,
                            acceptance_validator=self.accepted,
                            source_identity=source_identity(),
                            max_descriptors_per_subshard=50,
                            resources=resources(),
                            compute_ceiling_report_hash="1" * 64,
                        )
                    self.assertEqual(validate_suite.call_count, 6)

    def test_manifest_rejects_all_self_rehashed_coverage_attacks(self):
        manifest, suites = self.make_manifest("smoke")

        deleted = dict(manifest)
        tasks = [dict(item) for item in deleted["tasks"]]
        refs = list(tasks[0]["descriptors"])
        removed = refs.pop()
        tasks[0]["descriptors"] = tuple(refs)
        tasks[0]["assignment_hash"] = execution._task_hash(tasks[0])
        deleted["tasks"] = tuple(tasks)
        deleted["expected_descriptor_count"] -= 1
        deleted["expected_row_count"] -= len(removed["expected_methods"])
        deleted["expected_sidecar_count"] -= len(removed["expected_methods"])
        deleted["expected_positive_reference_a_count"] -= int("reference_a" in removed["expected_methods"])
        deleted["expected_positive_reference_b_count"] -= int("reference_b" in removed["expected_methods"])
        deleted["expected_tie_path_row_count"] -= removed["expected_tie_row_count"]
        deleted["expected_symmetry_path_row_count"] -= removed["expected_symmetry_row_count"]
        self.refresh_owner(deleted, int(tasks[0]["logical_case_owner"]))
        self.rehash_manifest(deleted)
        self.assert_manifest_rejected(deleted, suites)

        reordered = dict(manifest)
        tasks = [dict(item) for item in reordered["tasks"]]
        tasks[0]["descriptors"] = tuple(reversed(tasks[0]["descriptors"]))
        tasks[0]["assignment_hash"] = execution._task_hash(tasks[0])
        reordered["tasks"] = tuple(tasks)
        self.refresh_owner(reordered, int(tasks[0]["logical_case_owner"]))
        self.rehash_manifest(reordered)
        self.assert_manifest_rejected(reordered, suites)

        inserted = dict(manifest)
        tasks = [dict(item) for item in inserted["tasks"]]
        tasks[0]["descriptors"] = tuple(tasks[0]["descriptors"]) + (tasks[1]["descriptors"][0],)
        tasks[0]["assignment_hash"] = execution._task_hash(tasks[0])
        inserted["tasks"] = tuple(tasks)
        self.rehash_manifest(inserted)
        self.assert_manifest_rejected(inserted, suites)

        reassigned = dict(manifest)
        tasks = [dict(item) for item in reassigned["tasks"]]
        moved = list(tasks[0]["descriptors"]).pop()
        first = list(tasks[0]["descriptors"])
        first.remove(moved)
        tasks[0]["descriptors"] = tuple(first)
        tasks[1]["descriptors"] = (moved,) + tuple(tasks[1]["descriptors"])
        for index in (0, 1):
            tasks[index]["assignment_hash"] = execution._task_hash(tasks[index])
        reassigned["tasks"] = tuple(tasks)
        self.refresh_owner(reassigned, int(tasks[0]["logical_case_owner"]))
        self.refresh_owner(reassigned, int(tasks[1]["logical_case_owner"]))
        self.rehash_manifest(reassigned)
        self.assert_manifest_rejected(reassigned, suites)

        method_removed = dict(manifest)
        tasks = [dict(item) for item in method_removed["tasks"]]
        refs = [dict(item) for item in tasks[0]["descriptors"]]
        methods = tuple(refs[0]["expected_methods"][:-1])
        refs[0]["expected_methods"] = methods
        refs[0]["expected_tie_row_count"] = min(refs[0]["expected_tie_row_count"], len(methods))
        refs[0]["expected_symmetry_row_count"] = min(refs[0]["expected_symmetry_row_count"], len(methods))
        tasks[0]["descriptors"] = tuple(refs)
        tasks[0]["assignment_hash"] = execution._task_hash(tasks[0])
        method_removed["tasks"] = tuple(tasks)
        method_removed["expected_row_count"] -= 1
        method_removed["expected_sidecar_count"] -= 1
        self.rehash_manifest(method_removed)
        self.assert_manifest_rejected(method_removed, suites)

        full, full_suites = self.make_manifest("full")
        shrunken = dict(full)
        shrunken["planned_full_task_strata"] = (full["planned_full_task_strata"][0],)
        shrunken["planned_full_task_count"] = 1
        shrunken["full_stage_strata_counts"] = full["planned_full_task_strata"][0]
        self.rehash_manifest(shrunken)
        self.assert_manifest_rejected(shrunken, full_suites)

    def test_full_manifest_has_exactly_90_logical_owners_and_immutable_subshards(self):
        manifest, suites = self.make_manifest("full", max_size=1)
        self.assertEqual(manifest["case_owner_count"], 90)
        self.assertEqual(tuple(item["logical_case_owner"] for item in manifest["case_owners"]), tuple(range(90)))
        self.assertEqual(manifest["task_count"], 180)
        self.assertTrue(all(task["subshard_count"] == 2 for task in manifest["tasks"]))
        self.assertEqual(tuple(task["task_id"] for task in manifest["tasks"]), tuple(range(1, 181)))
        forged = dict(manifest)
        tasks = [dict(item) for item in forged["tasks"]]
        tasks[0]["subshard_index"] = 1
        tasks[0]["assignment_hash"] = execution._task_hash(tasks[0])
        forged["tasks"] = tuple(tasks)
        forged["manifest_hash"] = execution.logical_hash(execution._without_hash(forged, "manifest_hash"))
        identities = SimpleNamespace(scientific_spec_hash="a" * 64, numerical_method_config_hash="b" * 64)
        with patch.object(execution, "canonical_base_provider_failures", return_value=()):
            with patch.object(execution, "load_terminal_validation_identities", return_value=identities):
                with self.assertRaisesRegex(RuntimeError, "subshard"):
                    execution.validate_execution_manifest(forged, suites, self.provider, self.accepted)

    def test_smoke_selection_is_frozen_to_four_cases_and_bounded_stress(self):
        manifest, _ = self.make_manifest("smoke")
        self.assertEqual(tuple(item["logical_case_owner"] for item in manifest["case_owners"]), execution.SMOKE_CASE_IDS)
        self.assertEqual(manifest["expected_descriptor_count"], 16)
        self.assertEqual(manifest["task_count"], 4)

    def test_manifest_refuses_unaccepted_provider_and_self_rehashed_extra_field(self):
        suites = self.full_suites()
        with self.assertRaisesRegex(RuntimeError, "custom authoritative"):
            execution.create_execution_manifest(
                stage="full", suites=suites, provider=self.provider,
                acceptance_validator=lambda _: False, source_identity=source_identity(),
                max_descriptors_per_subshard=5, resources=resources(),
                compute_ceiling_report_hash=HASH,
            )
        manifest, suites = self.make_manifest("full")
        forged = dict(manifest)
        forged["extra"] = "self-rehashed"
        forged["manifest_hash"] = execution.logical_hash(execution._without_hash(forged, "manifest_hash"))
        identities = SimpleNamespace(scientific_spec_hash="a" * 64, numerical_method_config_hash="b" * 64)
        with patch.object(execution, "load_terminal_validation_identities", return_value=identities):
            with patch.object(execution, "canonical_base_provider_failures", return_value=()):
                with self.assertRaisesRegex(RuntimeError, "exact schema"):
                    execution.validate_execution_manifest(forged, suites, self.provider, self.accepted)

    def scheduler_fixture(self, root, manifest):
        submissions = []
        qacct = {}
        for task in manifest["tasks"]:
            task_id = int(task["task_id"])
            job_id = str(1000 + task_id)
            raw = root / f"qsub_{job_id}.txt"
            raw.write_text(job_id + "\n", encoding="utf-8")
            job = root / f"job_{job_id}.sh"
            job.write_text(
                "#!/bin/sh\n"
                "#$ -cwd\n"
                f"#$ -N smoke_{task_id}\n#$ -q campus\n#$ -j y\n#$ -o /tmp/test.log\n"
                "#$ -l h_rt=01:00:00\n#$ -l h_data=2000000000\n"
                "set -euo pipefail\nexport LANG=C\nexport LC_ALL=C\ncd /tmp\n"
                f"task_id={task_id}\n"
                "python scripts/terminal_validation_array.py run-task "
                "--manifest m --output-root o --task-id ${task_id}\n",
                encoding="utf-8",
            )
            submissions.append({
                "job_id": job_id, "job_name": f"smoke_{task_id}", "queue": "campus",
                "array_job": False, "manifest_task_ids": (task_id,),
                "qsub_raw_path": str(raw), "job_script_path": str(job),
            })
            account = root / f"qacct_{job_id}.txt"
            account.write_text(
                f"jobnumber {job_id}\njobname smoke_{task_id}\nqname campus\n"
                "hostname n1234\nslots 1\nfailed 0\nexit_status 0\n"
                "cpu 00:00:01\nru_wallclock 2\nmaxvmem 100M\n",
                encoding="utf-8",
            )
            qacct[job_id] = account
        return submissions, qacct

    def write_binding_task_artifacts(self, root, manifest, scheduler, qacct):
        submission_by_task = {
            int(task_id): submission
            for submission in scheduler["submissions"]
            for task_id in submission["manifest_task_ids"]
        }
        qacct_by_task = {int(item["task_id"]): item for item in qacct["task_bindings"]}
        for task in manifest["tasks"]:
            task_id = int(task["task_id"])
            target = root / "tasks" / f"task_{task_id:05d}"
            target.mkdir(parents=True)
            (target / "rows.json").write_text("[]\n", encoding="utf-8")
            (target / "metrics.json").write_text("[]\n", encoding="utf-8")
            binding = qacct_by_task[task_id]
            artifact = {
                "schema": execution.TASK_ARTIFACT_SCHEMA,
                "manifest_hash": manifest["manifest_hash"],
                "task_id": task_id,
                "assignment_hash": task["assignment_hash"],
                "logical_case_owner": task["logical_case_owner"],
                "subshard_index": task["subshard_index"],
                "subshard_count": task["subshard_count"],
                "job_id": submission_by_task[task_id]["job_id"],
                "sge_task_id": task_id if manifest["stage"] == "full" else None,
                "slots": 1,
                "hostname": binding["hostname"],
                "source_identity_hash": manifest["source_identity"]["identity_hash"],
                "provider_hash": manifest["provider_hash"],
                "scientific_spec_hash": manifest["scientific_spec_hash"],
                "numerical_method_config_hash": manifest["numerical_method_config_hash"],
                "rows_file_hash": execution.sha256_file(target / "rows.json"),
                "metrics_file_hash": execution.sha256_file(target / "metrics.json"),
                "row_count": 0,
                "sidecar_count": 0,
                "sidecar_index": (),
                "task_cpu_seconds": 0.1,
                "task_wall_seconds": 0.1,
                "logical_record_hash": "",
            }
            artifact["logical_record_hash"] = execution.logical_hash(
                execution._without_hash(artifact, "logical_record_hash")
            )
            execution.write_new_json(target / "task.json", artifact)

    def test_scheduler_rejects_wrong_stage_queue_shape_task_and_duplicates(self):
        manifest, _ = self.make_manifest("smoke")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            submissions, _ = self.scheduler_fixture(root, manifest)
            wrong_queue = [dict(item) for item in submissions]
            wrong_queue[0]["queue"] = "unauthorized"
            with self.assertRaisesRegex(RuntimeError, "queue"):
                execution.create_scheduler_evidence(manifest, wrong_queue, evidence_root=root)
            wrong_shape = [dict(item) for item in submissions]
            wrong_shape[0]["array_job"] = True
            with self.assertRaisesRegex(RuntimeError, "non-array"):
                execution.create_scheduler_evidence(manifest, wrong_shape, evidence_root=root)
            wrong_task = [dict(item) for item in submissions]
            wrong_task[0]["manifest_task_ids"] = (999,)
            with self.assertRaisesRegex(RuntimeError, "unknown"):
                execution.create_scheduler_evidence(manifest, wrong_task, evidence_root=root)
            duplicate = [dict(item) for item in submissions]
            duplicate[1]["manifest_task_ids"] = duplicate[0]["manifest_task_ids"]
            with self.assertRaisesRegex(RuntimeError, "overlap"):
                execution.create_scheduler_evidence(manifest, duplicate, evidence_root=root)
            reordered = list(submissions)
            reordered[0], reordered[1] = reordered[1], reordered[0]
            with self.assertRaisesRegex(RuntimeError, "task order"):
                execution.create_scheduler_evidence(manifest, reordered, evidence_root=root)
            job_path = Path(submissions[0]["job_script_path"])
            original = job_path.read_text(encoding="utf-8")
            job_path.write_text(original.replace("#$ -q campus", "#$ -q unauthorized"), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "queue/name"):
                execution.create_scheduler_evidence(manifest, submissions, evidence_root=root)
            job_path.write_text(original + "#$ -pe shared 2\n", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "parallel-environment"):
                execution.create_scheduler_evidence(manifest, submissions, evidence_root=root)

    def test_full_scheduler_requires_one_exact_one_slot_array(self):
        manifest, _ = self.make_manifest("full")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw = root / "qsub.txt"
            raw.write_text("9001.1-90:1\n", encoding="utf-8")
            job = root / "full.job"
            job.write_text(
                "#!/bin/sh\n#$ -cwd\n#$ -N full_job\n#$ -q campus\n"
                "#$ -j y\n#$ -o /tmp/test.log\n"
                "#$ -l h_rt=01:00:00\n#$ -l h_data=2000000000\n"
                f"#$ -t 1-{manifest['task_count']}\n#$ -tc 32\n"
                "set -euo pipefail\nexport LANG=C\nexport LC_ALL=C\ncd /tmp\n"
                "task_id=${SGE_TASK_ID}\n"
                "python scripts/terminal_validation_array.py run-task "
                "--manifest m --output-root o --task-id ${task_id}\n",
                encoding="utf-8",
            )
            submission = {
                "job_id": "9001", "job_name": "full_job", "queue": "campus",
                "array_job": True,
                "manifest_task_ids": tuple(range(1, manifest["task_count"] + 1)),
                "qsub_raw_path": raw, "job_script_path": job,
            }
            scheduler = execution.create_scheduler_evidence(
                manifest, (submission,), evidence_root=root
            )
            self.assertEqual(len(scheduler["submissions"]), 1)
            forged = dict(submission)
            forged["array_job"] = False
            with self.assertRaisesRegex(RuntimeError, "must be an array"):
                execution.create_scheduler_evidence(manifest, (forged,), evidence_root=root)

    def test_task_artifacts_are_strictly_bound_to_scheduler_qacct_and_source(self):
        manifest, _ = self.make_manifest("smoke")
        fields = {
            "job_id": "999999",
            "hostname": "n9999",
            "source_identity_hash": "f" * 64,
            "scientific_spec_hash": "f" * 64,
            "numerical_method_config_hash": "f" * 64,
            "logical_case_owner": 999,
            "subshard_index": 99,
            "sge_task_id": 1,
            "slots": 2,
        }
        for field, value in fields.items():
            with self.subTest(field=field), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                submissions, qacct_paths = self.scheduler_fixture(root, manifest)
                scheduler = execution.create_scheduler_evidence(manifest, submissions, evidence_root=root)
                qacct = execution.audit_qacct(manifest, scheduler, qacct_paths, evidence_root=root)
                self.write_binding_task_artifacts(root, manifest, scheduler, qacct)
                task_path = root / "tasks" / "task_00001" / "task.json"
                artifact = execution._decode(dict(execution._load_json(task_path)))
                artifact[field] = value
                artifact["logical_record_hash"] = execution.logical_hash(
                    execution._without_hash(artifact, "logical_record_hash")
                )
                task_path.unlink()
                execution.write_new_json(task_path, artifact)
                with self.assertRaisesRegex(RuntimeError, "bound|identity"):
                    execution.validate_task_scheduler_bindings(
                        manifest, task_output_root=root, scheduler=scheduler, qacct=qacct
                    )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            submissions, qacct_paths = self.scheduler_fixture(root, manifest)
            scheduler = execution.create_scheduler_evidence(manifest, submissions, evidence_root=root)
            qacct = execution.audit_qacct(manifest, scheduler, qacct_paths, evidence_root=root)
            self.write_binding_task_artifacts(root, manifest, scheduler, qacct)
            forged = dict(qacct)
            bindings = [dict(item) for item in forged["task_bindings"]]
            bindings[0]["cpu_seconds"] = 0.01
            forged["task_bindings"] = tuple(bindings)
            forged["logical_record_hash"] = execution.logical_hash(
                execution._without_hash(forged, "logical_record_hash")
            )
            with self.assertRaisesRegex(RuntimeError, "bounded by qacct"):
                execution.validate_task_scheduler_bindings(
                    manifest, task_output_root=root, scheduler=scheduler, qacct=forged
                )

    def test_qacct_rejects_unrelated_or_duplicate_records(self):
        manifest, _ = self.make_manifest("smoke")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            submissions, qacct_paths = self.scheduler_fixture(root, manifest)
            scheduler = execution.create_scheduler_evidence(manifest, submissions, evidence_root=root)
            first = next(iter(qacct_paths.values()))
            first.write_text(first.read_text(encoding="utf-8") + "\n" + first.read_text(encoding="utf-8"), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "coverage|differs"):
                execution.audit_qacct(manifest, scheduler, qacct_paths, evidence_root=root)

    def test_scheduler_and_qacct_require_exact_successful_one_slot_coverage(self):
        manifest, _ = self.make_manifest("smoke")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            submissions, qacct = self.scheduler_fixture(root, manifest)
            scheduler = execution.create_scheduler_evidence(
                manifest, submissions, evidence_root=root
            )
            audit = execution.audit_qacct(
                manifest, scheduler, qacct, evidence_root=root
            )
            self.assertTrue(audit["qacct_audit_pass"])
            raw_qsub = Path(submissions[0]["qsub_raw_path"])
            original_qsub = raw_qsub.read_text(encoding="utf-8")
            raw_qsub.write_text("999999\n", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "raw qsub|scheduler raw"):
                execution.audit_qacct(manifest, scheduler, qacct, evidence_root=root)
            raw_qsub.write_text(original_qsub, encoding="utf-8")
            first = next(iter(qacct.values()))
            first.write_text(first.read_text(encoding="utf-8").replace("slots 1", "slots 8"), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "one-slot"):
                execution.audit_qacct(manifest, scheduler, qacct, evidence_root=root)
            missing = submissions[:-1]
            with self.assertRaisesRegex(RuntimeError, "non-array submission|exactly cover"):
                execution.create_scheduler_evidence(manifest, missing, evidence_root=root)

    def test_one_slot_task_and_provisional_collection_use_row_sidecar_source_validation(self):
        mdp = one_atom_mdp(FiniteSupportAtom(80.0, 0.5, -1))
        belief = mdp.initial_belief()
        item = descriptor_for(mdp, belief)
        bundle = execution.evaluate_terminal_evidence_descriptor(item, mdp, belief)
        methods = tuple(row.method for row in bundle.rows)
        task = {
            "task_id": 1, "logical_case_owner": 0, "subshard_index": 0,
            "subshard_count": 1,
            "descriptors": ({
                "suite_class": item.suite_class,
                "descriptor_index": item.descriptor_index,
                "descriptor_hash": item.descriptor_hash,
                "expected_methods": methods,
                "expected_tie_row_count": sum(row.tie_status not in (None, "unique") for row in bundle.rows),
                "expected_symmetry_row_count": sum(row.symmetry_required for row in bundle.rows),
            },),
            "assignment_hash": "",
        }
        task["assignment_hash"] = execution._task_hash(task)
        manifest = {
            "stage": "smoke", "artifact_type": "terminal_smoke",
            "manifest_hash": "1" * 64, "task_count": 1, "tasks": (task,),
            "expected_row_count": len(methods), "expected_sidecar_count": len(methods),
            "expected_positive_reference_a_count": int("reference_a" in methods),
            "expected_positive_reference_b_count": int("reference_b" in methods),
            "expected_tie_path_row_count": sum(row.tie_status not in (None, "unique") for row in bundle.rows),
            "expected_symmetry_path_row_count": sum(row.symmetry_required for row in bundle.rows),
            "scientific_spec_hash": "a" * 64,
            "numerical_method_config_hash": "b" * 64,
            "source_identity": {"identity_hash": "2" * 64},
        }
        suites = {"base": SimpleNamespace(descriptors=(item,))}
        scheduler = {"NSLOTS": "1", "JOB_ID": "1234"}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch.object(execution, "validate_execution_manifest") as validate:
                with patch.object(execution, "reconstruct_terminal_evidence_source", return_value=(mdp, belief)):
                    target = execution.execute_task(
                        manifest=manifest, suites=suites, provider=self.provider,
                        acceptance_validator=self.accepted, output_root=root, task_id=1,
                        scheduler_environment=scheduler,
                    )
                    self.assertFalse(validate.call_args.kwargs["reconstruct_expected"])
                    self.assertTrue((target / "task.json").is_file())
                    with self.assertRaises(FileExistsError):
                        execution.execute_task(
                            manifest=manifest, suites=suites, provider=self.provider,
                            acceptance_validator=self.accepted, output_root=root, task_id=1,
                            scheduler_environment=scheduler,
                        )
                    provisional = execution.collect_provisional(
                        manifest=manifest, suites=suites, provider=self.provider,
                        acceptance_validator=self.accepted, output_root=root,
                        provisional_path=root / "provisional.json",
                    )
            self.assertEqual(provisional["observed_task_count"], 1)
            self.assertEqual(provisional["observed_row_count"], len(methods))
            self.assertTrue(provisional["negative_control_rejection_pass"])
            task_dir = root / "tasks" / "task_00001"
            metrics_path = task_dir / "metrics.json"
            task_path = task_dir / "task.json"
            original_metrics = tuple(
                execution._decode(item)
                for item in json.loads(metrics_path.read_text(encoding="utf-8"))
            )
            original_artifact = execution._decode(dict(execution._load_json(task_path)))

            def install_metrics(metrics, cpu, wall):
                metrics_path.write_text(
                    json.dumps(execution._canonical(metrics), sort_keys=True, separators=(",", ":")) + "\n",
                    encoding="utf-8",
                )
                artifact = dict(original_artifact)
                artifact["metrics_file_hash"] = execution.sha256_file(metrics_path)
                artifact["task_cpu_seconds"] = cpu
                artifact["task_wall_seconds"] = wall
                artifact["logical_record_hash"] = execution.logical_hash(
                    execution._without_hash(artifact, "logical_record_hash")
                )
                task_path.unlink()
                execution.write_new_json(task_path, artifact)

            tiny = tuple(dict(item, cpu_seconds=1e-12, wall_seconds=1e-12) for item in original_metrics)
            install_metrics(tiny, 1e-12, 1e-12)
            with patch.object(execution, "validate_execution_manifest"):
                with patch.object(execution, "reconstruct_terminal_evidence_source", return_value=(mdp, belief)):
                    attacked = execution.recompute_provisional(
                        manifest=manifest, suites=suites, provider=self.provider,
                        acceptance_validator=self.accepted, output_root=root,
                    )
            self.assertFalse(attacked["provisional_gate_pass"])
            self.assertTrue(any("metric" in reason for reason in attacked["failure_reasons"]))

            zero_bytes = tuple(dict(item, row_bytes=0, sidecar_bytes=0) for item in original_metrics)
            install_metrics(
                zero_bytes,
                original_artifact["task_cpu_seconds"],
                original_artifact["task_wall_seconds"],
            )
            with patch.object(execution, "validate_execution_manifest"):
                with patch.object(execution, "reconstruct_terminal_evidence_source", return_value=(mdp, belief)):
                    attacked = execution.recompute_provisional(
                        manifest=manifest, suites=suites, provider=self.provider,
                        acceptance_validator=self.accepted, output_root=root,
                    )
            self.assertFalse(attacked["provisional_gate_pass"])
            self.assertTrue(any("source/file reconstruction" in reason for reason in attacked["failure_reasons"]))

    def make_ceiling(self):
        value = {
            "schema": execution.COMPUTE_CEILING_SCHEMA,
            "captured_at_utc": datetime.now(timezone.utc).isoformat(),
            "max_walltime_seconds": 7200,
            "max_array_tasks": 1000,
            "max_throttle": 100,
            "max_memory_bytes": 4_000_000_000,
            "max_storage_bytes": 10_000_000_000,
            "cpu_hours_quota": None,
            "allowed_queues": ("campus",),
            "raw_evidence_hashes": (("qconf.raw", HASH),),
            "report_hash": "",
        }
        value["report_hash"] = execution.logical_hash(execution._without_hash(value, "report_hash"))
        return value

    def make_provisional(self, manifest):
        metrics = []
        for task_id in range(1, int(manifest["task_count"]) + 1):
            for stratum, _ in manifest["full_stage_strata_counts"]:
                method = stratum.split(":", 1)[0]
                metrics.append({
                    "task_id": task_id, "descriptor_hash": "1" * 64,
                    "method": method, "stratum": stratum,
                    "evaluation_count": 10, "cpu_seconds": 0.1, "wall_seconds": 0.1,
                    "row_bytes": 100, "sidecar_bytes": 200,
                    "timing_scope": "task_conservative_shared",
                })
        value = {
            "schema": execution.PROVISIONAL_SCHEMA,
            "artifact_type": manifest["artifact_type"], "artifact_status": "provisional",
            "stage_complete": False, "manifest_hash": manifest["manifest_hash"],
            "source_hash_match": True, "scientific_spec_hash_match": True,
            "numerical_method_config_hash_match": True, "manifest_hash_match": True,
            "observed_task_count": manifest["task_count"],
            "observed_row_count": manifest["expected_row_count"],
            "observed_sidecar_count": manifest["expected_sidecar_count"],
            "positive_reference_a_count": manifest["expected_positive_reference_a_count"],
            "positive_reference_b_count": manifest["expected_positive_reference_b_count"],
            "reference_a_complete": True, "reference_b_complete": True,
            "tie_path_exercised": True, "tie_path_pass": True,
            "symmetry_path_exercised": True, "symmetry_path_pass": True,
            "scalar_batch_parity_pass": True, "fail_closed_path_exercised": True,
            "negative_control_rejection_pass": True,
            "unexpected_reference_unresolved_count": 0,
            "unexpected_validation_failure_count": 0,
            "missing_duplicate_malformed_nonfinite_stale_invalid_count": 0,
            "coverage_match": True, "failure_reasons": (), "task_artifact_hashes": (),
            "job_hosts": ("n1234",), "metrics": tuple(metrics),
            "qacct_audit_pass": False, "finalization_hash_bind_pass": False,
            "independent_readback_pass": False, "feasibility_gate_pass": False,
            "logical_record_hash": "", "provisional_gate_pass": True,
        }
        value["logical_record_hash"] = execution.logical_hash(execution._without_hash(value, "logical_record_hash"))
        return value

    def test_feasibility_uses_qacct_physical_bytes_fresh_ceiling_and_cap_boundaries(self):
        manifest, _ = self.make_manifest("smoke")
        provisional = self.make_provisional(manifest)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            submissions, qacct_paths = self.scheduler_fixture(root, manifest)
            scheduler = execution.create_scheduler_evidence(manifest, submissions, evidence_root=root)
            qacct = execution.audit_qacct(manifest, scheduler, qacct_paths, evidence_root=root)
            fixed = root / "fixed.bin"
            fixed.write_bytes(b"x" * 4096)
            ceiling = self.make_ceiling()
            bound_manifest = dict(manifest)
            bound_manifest["compute_ceiling_report_hash"] = ceiling["report_hash"]
            result = execution.compute_feasibility(
                bound_manifest, provisional, ceiling, qacct_audit=qacct,
                fixed_artifact_paths=(fixed,), finalization_overhead_seconds=0.25,
                finalization_artifact_bytes=2048,
            )
            self.assertGreaterEqual(result["measured_fixed_artifact_bytes"], 4096)
            self.assertEqual(result["measured_max_memory_bytes"], 100 * 1024**2)
            with self.assertRaisesRegex(RuntimeError, "within 24 hours"):
                execution.compute_feasibility(
                    bound_manifest, provisional, ceiling, qacct_audit=qacct,
                    fixed_artifact_paths=(fixed,), finalization_overhead_seconds=0.25,
                    finalization_artifact_bytes=2048,
                    now=datetime.now(timezone.utc).replace(microsecond=0) + timedelta(hours=25),
                )
            too_small = dict(ceiling)
            too_small["max_memory_bytes"] = resources()["memory_bytes"] - 1
            too_small["report_hash"] = execution.logical_hash(
                execution._without_hash(too_small, "report_hash")
            )
            small_manifest = dict(bound_manifest)
            small_manifest["compute_ceiling_report_hash"] = too_small["report_hash"]
            with self.assertRaisesRegex(RuntimeError, "do not fit"):
                execution.compute_feasibility(
                    small_manifest, provisional, too_small, qacct_audit=qacct,
                    fixed_artifact_paths=(fixed,), finalization_overhead_seconds=0.25,
                    finalization_artifact_bytes=2048,
                )

    def test_feasibility_finalization_readback_and_no_overwrite_are_fail_closed(self):
        manifest, _ = self.make_manifest("smoke")
        ceiling = self.make_ceiling()
        manifest = dict(manifest)
        manifest["compute_ceiling_report_hash"] = ceiling["report_hash"]
        manifest["manifest_hash"] = execution.logical_hash(execution._without_hash(manifest, "manifest_hash"))
        provisional = self.make_provisional(manifest)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            submissions, qacct_paths = self.scheduler_fixture(root, manifest)
            scheduler = execution.create_scheduler_evidence(
                manifest, submissions, evidence_root=root
            )
            qacct = execution.audit_qacct(
                manifest, scheduler, qacct_paths, evidence_root=root
            )
            paths = {name: root / f"{name}.json" for name in ("provisional", "scheduler", "qacct", "ceiling")}
            for name, value in (("provisional", provisional), ("scheduler", scheduler), ("qacct", qacct), ("ceiling", ceiling)):
                execution.write_new_json(paths[name], value)
                with self.assertRaises(FileExistsError):
                    execution.write_new_json(paths[name], value)
            self.assertTrue(execution.compute_feasibility(
                manifest, provisional, ceiling, qacct_audit=qacct,
                fixed_artifact_paths=tuple(paths.values()),
                finalization_overhead_seconds=0.1,
                finalization_artifact_bytes=1000,
            )["feasibility_gate_pass"])
            post = root / "post.json"
            with patch.object(execution, "validate_clean_source_identity"), \
                 patch.object(execution, "validate_execution_manifest"), \
                 patch.object(execution, "validate_task_scheduler_bindings"), \
                 patch.object(execution, "recompute_provisional", return_value=provisional), \
                 patch.object(execution.platform, "node", return_value="n9999"):
                execution.finalize_post_job(
                    manifest=manifest, suites={}, provider=self.provider,
                    acceptance_validator=self.accepted, task_output_root=root,
                    provisional_path=paths["provisional"],
                    scheduler_evidence_path=paths["scheduler"], qacct_audit_path=paths["qacct"],
                    compute_ceiling_path=paths["ceiling"], scheduler_evidence_root=root,
                    output_path=post,
                    project_root=root,
                )
            final = root / "terminal_smoke_final.json"
            with patch.object(execution.platform, "system", return_value="Darwin"), \
                 patch.object(execution, "validate_clean_source_identity"), \
                 patch.object(execution, "validate_execution_manifest"), \
                 patch.object(execution, "validate_task_scheduler_bindings"), \
                 patch.object(execution, "recompute_provisional", return_value=provisional), \
                 patch.object(execution.platform, "node", return_value="local-mac"):
                result = execution.independent_readback(
                    manifest=manifest, suites={}, provider=self.provider,
                    acceptance_validator=self.accepted, task_output_root=root,
                    provisional_path=paths["provisional"],
                    scheduler_evidence_path=paths["scheduler"], qacct_audit_path=paths["qacct"],
                    compute_ceiling_path=paths["ceiling"], scheduler_evidence_root=root,
                    post_job_path=post,
                    final_output_path=final,
                    project_root=root,
                )
            self.assertTrue(result["stage_complete"])
            self.assertTrue(result["independent_readback_pass"])
            original_post = execution._decode(dict(execution._load_json(post)))
            for field, value in (
                ("artifact_status", "forged"),
                ("stage_complete", True),
                ("manifest_hash", "f" * 64),
                ("final_gate_pass", False),
                ("finalization_host", "local-mac"),
            ):
                with self.subTest(post_attack=field):
                    attacked = dict(original_post)
                    attacked[field] = value
                    attacked["logical_record_hash"] = execution.logical_hash(
                        execution._without_hash(attacked, "logical_record_hash")
                    )
                    attacked_path = root / f"post_{field}.json"
                    execution.write_new_json(attacked_path, attacked)
                    with patch.object(execution.platform, "system", return_value="Darwin"), \
                         patch.object(execution, "validate_clean_source_identity"), \
                         patch.object(execution, "validate_execution_manifest"):
                        with self.assertRaisesRegex(RuntimeError, "post-job candidate"):
                            execution.independent_readback(
                                manifest=manifest, suites={}, provider=self.provider,
                                acceptance_validator=self.accepted, task_output_root=root,
                                provisional_path=paths["provisional"],
                                scheduler_evidence_path=paths["scheduler"],
                                qacct_audit_path=paths["qacct"],
                                compute_ceiling_path=paths["ceiling"],
                                scheduler_evidence_root=root,
                                post_job_path=attacked_path,
                                final_output_path=root / f"final_{field}.json",
                                project_root=root,
                            )
            with self.assertRaises(FileExistsError):
                execution.independent_readback(
                    manifest=manifest, suites={}, provider=self.provider,
                    acceptance_validator=self.accepted, task_output_root=root,
                    provisional_path=paths["provisional"],
                    scheduler_evidence_path=paths["scheduler"], qacct_audit_path=paths["qacct"],
                    compute_ceiling_path=paths["ceiling"], scheduler_evidence_root=root,
                    post_job_path=post,
                    final_output_path=final, project_root=root,
                    allow_non_darwin_for_tests=True,
                )

    def test_atomic_no_replace_preserves_preexisting_partial_file(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / "artifact.json"
            target.write_bytes(b"partial")
            with self.assertRaises(FileExistsError):
                execution.write_new_json(target, {"valid": True})
            self.assertEqual(target.read_bytes(), b"partial")
            self.assertEqual(tuple(root.glob(".artifact.json.tmp.*")), ())

    def test_independent_readback_revalidates_clean_local_source_first(self):
        manifest, _ = self.make_manifest("smoke")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch.object(execution.platform, "system", return_value="Darwin"):
                with patch.object(
                    execution, "validate_clean_source_identity",
                    side_effect=RuntimeError("clean source identity differs"),
                ) as validate:
                    with self.assertRaisesRegex(RuntimeError, "clean source identity"):
                        execution.independent_readback(
                            manifest=manifest, suites={}, provider=self.provider,
                            acceptance_validator=self.accepted,
                            task_output_root=root, provisional_path=root / "missing",
                            scheduler_evidence_path=root / "missing2",
                            qacct_audit_path=root / "missing3",
                            compute_ceiling_path=root / "missing4",
                            scheduler_evidence_root=root, post_job_path=root / "missing5",
                            final_output_path=root / "final.json", project_root=root,
                        )
            validate.assert_called_once()

    def test_execution_authorization_is_immutable_and_manifest_specific(self):
        project_root = Path(__file__).resolve().parents[1]
        script_paths = (
            "scripts/submit_hoffman2_terminal_validation.sh",
            "scripts/terminal_validation_array.py",
            "src/experiments/terminal_execution.py",
        )
        records = tuple((path, execution.sha256_file(project_root / path)) for path in script_paths)
        source = {
            "schema": execution.SOURCE_IDENTITY_SCHEMA,
            "commit": "2" * 40,
            "tree": "3" * 40,
            "source_hashes": records,
            "source_hashes_hash": execution.logical_hash(records),
            "identity_hash": "",
        }
        source["identity_hash"] = execution.logical_hash(
            execution._without_hash(source, "identity_hash")
        )
        manifest, _ = self.make_manifest("smoke")
        manifest = dict(manifest)
        manifest["source_identity"] = source
        self.rehash_manifest(manifest)
        authorization = {
            "schema": execution.EXECUTION_AUTHORIZATION_SCHEMA,
            "authorization_status": "reviewer_approved_for_exact_terminal_stage",
            "verdict": "ACCEPT TERMINAL IMPLEMENTATION FOR SCHEDULED SMOKE",
            "manifest_hash": manifest["manifest_hash"],
            "source_identity_hash": source["identity_hash"],
            "source_commit": source["commit"],
            "source_tree": source["tree"],
            "provider_hash": manifest["provider_hash"],
            "provider_source_identity_hash": manifest["provider_source_identity_hash"],
            "scientific_spec_hash": manifest["scientific_spec_hash"],
            "numerical_method_config_hash": manifest["numerical_method_config_hash"],
            "compute_ceiling_report_hash": manifest["compute_ceiling_report_hash"],
            "resources_hash": execution.logical_hash(manifest["resources"]),
            "execution_script_hashes": records,
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "authorization.json"
            execution.write_new_json(path, authorization)
            approved = execution.sha256_file(path)
            with patch.object(execution, "validate_clean_source_identity"):
                execution.validate_execution_authorization(
                    authorization_path=path, approved_file_hash=approved,
                    manifest=manifest, project_root=project_root,
                )
                for field in ("manifest_hash", "compute_ceiling_report_hash", "provider_hash"):
                    with self.subTest(cross_reuse=field):
                        other = dict(manifest)
                        other[field] = "f" * 64
                        with self.assertRaisesRegex(RuntimeError, "exact manifest/source"):
                            execution.validate_execution_authorization(
                                authorization_path=path, approved_file_hash=approved,
                                manifest=other, project_root=project_root,
                            )
                other = dict(manifest)
                other["resources"] = dict(manifest["resources"], throttle=31)
                with self.assertRaisesRegex(RuntimeError, "exact manifest/source"):
                    execution.validate_execution_authorization(
                        authorization_path=path, approved_file_hash=approved,
                        manifest=other, project_root=project_root,
                    )
                other = dict(manifest)
                other_source = dict(source, identity_hash="f" * 64)
                other["source_identity"] = other_source
                with self.assertRaisesRegex(RuntimeError, "exact manifest/source"):
                    execution.validate_execution_authorization(
                        authorization_path=path, approved_file_hash=approved,
                        manifest=other, project_root=project_root,
                    )
                with self.assertRaisesRegex(RuntimeError, "externally approved"):
                    execution.validate_execution_authorization(
                        authorization_path=path, approved_file_hash="f" * 64,
                        manifest=manifest, project_root=project_root,
                    )

    def test_submitter_requires_exact_authorization_before_creating_or_submitting(self):
        script = (
            Path(__file__).resolve().parents[1]
            / "scripts" / "submit_hoffman2_terminal_validation.sh"
        ).read_text(encoding="utf-8")
        self.assertIn('EXECUTION_AUTHORIZATION:?', script)
        self.assertIn('APPROVED_EXECUTION_AUTHORIZATION_HASH:?', script)
        self.assertIn('COMPUTE_CEILING:?', script)
        authorization = script.index("validate-authorization")
        ceiling = script.index("validate-compute-ceiling")
        create_output = script.index('mkdir -p "${OUTPUT_ROOT}/scheduler/qsub_raw"')
        qsub = script.index('"${QSUB_BIN}" -terse')
        self.assertLess(authorization, create_output)
        self.assertLess(ceiling, create_output)
        self.assertLess(create_output, qsub)

    def test_manifest_setup_submitter_uses_dual_segmented_replicates(self):
        script = (
            Path(__file__).resolve().parents[1]
            / "scripts"
            / "submit_hoffman2_terminal_manifest_setup.sh"
        ).read_text(encoding="utf-8")
        self.assertIn('for replicate in a b; do', script)
        self.assertIn('MANIFEST_PLAN_SHARDS:-16', script)
        self.assertIn('MANIFEST_PLAN_SHARDS:-2000', script)
        self.assertIn('MANIFEST_PLAN_SEGMENT_SIZE:-100', script)
        self.assertIn(
            'submit "${role}" "${job_file}" "${job_name}" >/dev/null',
            script,
        )
        self.assertIn('#$ -hold_jid ${hold_ids}', script)
        self.assertIn('merge-plan-fragments', script)
        self.assertNotIn('-pe shared', script)


if __name__ == "__main__":
    unittest.main()
