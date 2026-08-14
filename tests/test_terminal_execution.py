from __future__ import annotations

from dataclasses import replace
from contextlib import ExitStack
from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
import inspect
from unittest.mock import patch
from zoneinfo import ZoneInfo

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
RUN_TAG = "0123456789abcdef"


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
    @staticmethod
    def no_numerical_sentinels():
        stack = ExitStack()
        for target in (
            "src.experiments.terminal_execution.evaluate_terminal_evidence_descriptor",
            "src.experiments.terminal_evidence_rows.optimize_terminal_allocation_with_trace",
            "src.experiments.terminal_evidence_rows.solve_terminal_reference_a_with_trace",
            "src.experiments.terminal_evidence_rows.source_validate_terminal_reference_record",
            "src.experiments.terminal_evidence_rows.solve_terminal_reference_b_with_trace",
            "src.experiments.terminal_evidence_rows.source_validate_terminal_reference_b_record",
            "src.experiments.terminal_evidence_rows.validate_production_against_reference_a",
            "src.experiments.terminal_evidence_rows.validate_terminal_reference_agreement",
            "src.solvers.terminal.optimize_terminal_allocation",
            "src.solvers.terminal.optimize_terminal_allocation_with_trace",
            "src.solvers.terminal_reference.solve_terminal_reference_a",
            "src.solvers.terminal_reference.solve_terminal_reference_a_with_trace",
            "src.solvers.terminal_reference.validate_terminal_reference_record",
            "src.solvers.terminal_reference.validate_production_against_reference_a",
            "src.solvers.terminal_reference_b.solve_terminal_reference_b",
            "src.solvers.terminal_reference_b.solve_terminal_reference_b_with_trace",
            "src.solvers.terminal_reference_b.validate_terminal_reference_b_record",
            "src.solvers.terminal_reference_agreement.validate_terminal_reference_agreement",
            "src.solvers.terminal_reference.source_validate_terminal_reference_record",
            "src.solvers.terminal_reference_b.source_validate_terminal_reference_b_record",
        ):
            stack.enter_context(
                patch(target, side_effect=AssertionError(f"post-task called {target}"))
            )
        return stack

    def test_post_task_acceptance_call_graph_has_no_numerical_recomputation(self):
        execute_source = inspect.getsource(execution.execute_task)
        self.assertEqual(
            execute_source.count("evaluate_terminal_evidence_descriptor("), 1
        )
        self.assertNotIn("validate_terminal_evidence_bundle_source(", execute_source)

        forbidden = (
            "evaluate_terminal_evidence_descriptor(",
            "validate_terminal_evidence_bundle_source(",
            "solve_terminal_reference_a(",
            "solve_terminal_reference_b(",
            "optimize_terminal_allocation(",
        )
        for function in (
            execution.recompute_provisional,
            execution.finalize_post_job,
            execution.independent_readback,
            execution.audit_formal_smoke,
        ):
            source = inspect.getsource(function)
            with self.subTest(function=function.__name__):
                self.assertTrue(all(token not in source for token in forbidden))

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
        stage_resources = resources()
        if stage == "smoke":
            stage_resources["h_rt_seconds"] = 86400
            stage_resources["throttle"] = 16
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
                        resources=stage_resources,
                        compute_ceiling_report_hash="1" * 64,
                    )
        return manifest, suites

    def assert_manifest_rejected(self, manifest, suites, pattern="source-reconstructed|mismatch|invalid|subshard|multiple|task IDs"):
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
        tasks[0], tasks[1] = tasks[1], tasks[0]
        reordered["tasks"] = tuple(tasks)
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
        self.assertEqual(manifest["task_count"], 16)
        self.assertEqual(manifest["task_descriptor_limit"], 1)
        self.assertTrue(manifest["array_required"])
        self.assertTrue(all(len(task["descriptors"]) == 1 for task in manifest["tasks"]))
        self.assertTrue(all(item["subshard_count"] == 4 for item in manifest["tasks"]))

    def test_mixed_slot_partition_and_scheduler_shapes_are_fail_closed(self):
        manifest, _ = self.make_manifest("smoke")
        tasks = [dict(item) for item in manifest["tasks"]]
        first_refs = [dict(item) for item in tasks[0]["descriptors"]]
        first_refs[0]["expected_methods"] = ("production_terminal", "reference_a")
        tasks[0]["descriptors"] = tuple(first_refs)
        tasks[0]["assignment_hash"] = execution._task_hash(tasks[0])
        mixed = dict(manifest, tasks=tuple(tasks))
        mixed["manifest_hash"] = execution.logical_hash(
            execution._without_hash(mixed, "manifest_hash")
        )
        one_slot, shared_two = execution.partition_manifest_task_ids(mixed)
        self.assertEqual(one_slot, (1,))
        self.assertEqual(shared_two, tuple(range(2, 17)))

        one_task = tasks[0]
        self.assertEqual(
            execution._scheduler_task_shape(
                one_task, {"NSLOTS": "1", "JOB_ID": "1", "SGE_TASK_ID": "1"}
            ),
            ("one_slot", 1, None, None, (), ()),
        )
        with tempfile.TemporaryDirectory() as directory, patch.object(
            execution.platform, "node", return_value="n1234"
        ):
            pe = Path(directory) / "pe_hostfile"
            pe.write_text("n1234 2 queue@host UNDEFINED\n", encoding="utf-8")
            environment = {
                "NSLOTS": "2", "PE_HOSTFILE": str(pe),
                **{name: "1" for name in execution.REFERENCE_B_THREAD_ENVIRONMENT},
            }
            shape = execution._scheduler_task_shape(tasks[1], environment)
            self.assertEqual(shape[0:3], ("shared_two", 2, "shared"))
            with self.assertRaisesRegex(RuntimeError, "exactly 2"):
                execution._scheduler_task_shape(tasks[1], dict(environment, NSLOTS="1"))

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            submissions, qacct_paths = self.scheduler_fixture(root, mixed)
            self.assertEqual(len(submissions), 2)
            scheduler = self.create_scheduler(mixed, submissions, root)
            self.assertEqual(
                tuple(tuple(item["manifest_task_ids"]) for item in scheduler["submissions"]),
                (one_slot, shared_two),
            )
            qacct = execution.audit_qacct(
                mixed, scheduler, qacct_paths, evidence_root=root
            )
            self.write_binding_task_artifacts(root, mixed, scheduler, qacct)
            execution.validate_task_scheduler_bindings(
                mixed, task_output_root=root, scheduler=scheduler, qacct=qacct
            )

    def test_smoke_rejects_owner_and_subshard_substitution(self):
        manifest, suites = self.make_manifest("smoke")
        for field, value in (
            ("logical_case_owner", execution.SMOKE_CASE_IDS[1]),
            ("subshard_index", 3),
        ):
            forged = dict(manifest)
            tasks = [dict(item) for item in manifest["tasks"]]
            tasks[0][field] = value
            tasks[0]["assignment_hash"] = execution._task_hash(tasks[0])
            forged["tasks"] = tuple(tasks)
            self.rehash_manifest(forged)
            with self.subTest(field=field):
                self.assert_manifest_rejected(forged, suites, "owner|subshard|source-reconstructed")

    def test_descriptor_rows_and_sidecars_do_not_depend_on_task_grouping(self):
        mdp = one_atom_mdp(FiniteSupportAtom(80.0, 0.5, -1))
        belief = mdp.initial_belief()
        template = descriptor_for(mdp, belief)
        template_bundle = execution.evaluate_terminal_evidence_descriptor(template, mdp, belief)
        template_rows = tuple(
            row for row in template_bundle.rows if row.method != "reference_b"
        )
        template_sidecars = {
            row.sidecar.relative_path: dict(template_bundle.sidecars)[row.sidecar.relative_path]
            for row in template_rows
        }
        descriptors = []
        bundles = {}
        references = []
        for index in range(4):
            descriptor_hash = execution.logical_hash(("shape", index, template.descriptor_hash))
            item = replace(template, descriptor_index=index, descriptor_hash=descriptor_hash)
            rows = []
            sidecars = {}
            for row in template_rows:
                logical_path = f"shape_{index}/{row.sidecar.relative_path}"
                sidecar = replace(row.sidecar, relative_path=logical_path)
                changed = replace(
                    row,
                    descriptor_index=index,
                    descriptor_hash=descriptor_hash,
                    sidecar=sidecar,
                    logical_record_hash="",
                )
                changed = replace(
                    changed,
                    logical_record_hash=execution.terminal_evidence_row_hash(changed),
                )
                rows.append(changed)
                sidecars[logical_path] = template_sidecars[row.sidecar.relative_path]
            descriptors.append(item)
            bundles[descriptor_hash] = SimpleNamespace(rows=tuple(rows), sidecars=sidecars)
            references.append({
                "suite_class": item.suite_class,
                "descriptor_index": item.descriptor_index,
                "descriptor_hash": item.descriptor_hash,
                "expected_methods": tuple(row.method for row in rows),
                "expected_tie_row_count": sum(row.tie_status not in (None, "unique") for row in rows),
                "expected_symmetry_row_count": sum(row.symmetry_required for row in rows),
            })

        def task(refs):
            value = {
                "task_id": 1,
                "logical_case_owner": 1,
                "subshard_index": 0,
                "subshard_count": 1,
                "descriptors": tuple(refs),
                "assignment_hash": "",
            }
            value["assignment_hash"] = execution._task_hash(value)
            return value

        def manifest(refs):
            return {
                "stage": "smoke",
                "array_required": True,
                "manifest_hash": HASH,
                "task_count": 1,
                "tasks": (task(refs),),
                "source_identity": {"identity_hash": "2" * 64},
                "provider_hash": self.provider.provider_hash,
                "scientific_spec_hash": "a" * 64,
                "numerical_method_config_hash": "b" * 64,
            }

        suites = {template.suite_class: SimpleNamespace(descriptors=tuple(descriptors))}
        with tempfile.TemporaryDirectory() as directory, patch.object(
            execution, "validate_execution_manifest"
        ), patch.object(
            execution, "reconstruct_terminal_evidence_source", return_value=(mdp, belief)
        ), patch.object(
            execution,
            "evaluate_terminal_evidence_descriptor",
            side_effect=lambda item, _mdp, _belief, **_kwargs: bundles[item.descriptor_hash],
        ), patch.object(
            execution, "require_terminal_evidence_plan_parity"
        ), patch.object(
            execution, "validate_terminal_evidence_bundle_structure", return_value=()
        ):
            root = Path(directory)
            grouped = execution.execute_task(
                manifest=manifest(references), suites=suites, provider=self.provider,
                acceptance_validator=self.accepted, output_root=root / "grouped", task_id=1,
                scheduler_environment={"NSLOTS": "1", "JOB_ID": "100", "SGE_TASK_ID": "1"},
            )
            isolated = execution.execute_task(
                manifest=manifest(references[:1]), suites=suites, provider=self.provider,
                acceptance_validator=self.accepted, output_root=root / "isolated", task_id=1,
                scheduler_environment={"NSLOTS": "1", "JOB_ID": "101", "SGE_TASK_ID": "1"},
            )
            grouped_rows = json.loads((grouped / "rows.json").read_text(encoding="utf-8"))
            isolated_rows = json.loads((isolated / "rows.json").read_text(encoding="utf-8"))
            self.assertEqual(grouped_rows[:len(isolated_rows)], isolated_rows)
            for relative_path, expected in bundles[descriptors[0].descriptor_hash].sidecars.items():
                self.assertEqual((grouped / "sidecars" / relative_path).read_bytes(), expected)
                self.assertEqual((isolated / "sidecars" / relative_path).read_bytes(), expected)

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
        root = root.resolve()
        scheduler = root / "scheduler"
        jobs = scheduler / "jobs"
        raw_dir = scheduler / "qsub_raw"
        jobs.mkdir(parents=True)
        raw_dir.mkdir(parents=True)
        one_slot, shared_two = execution.partition_manifest_task_ids(manifest)
        one_throttle, shared_throttle = execution._partition_throttles(
            manifest, one_slot, shared_two
        )
        project = Path(execution.__file__).resolve().parents[2]
        h_rt = int(manifest["resources"]["h_rt_seconds"])
        submissions = []
        qacct_paths = {}
        ended = datetime.now(ZoneInfo("America/Los_Angeles"))
        started = ended - timedelta(seconds=2)
        start_text = started.strftime("%m/%d/%Y %H:%M:%S.%f")[:-3]
        end_text = ended.strftime("%m/%d/%Y %H:%M:%S.%f")[:-3]
        partitions = (
            (one_slot, one_throttle, "one_slot"),
            (shared_two, shared_throttle, "shared_two"),
        )
        job_number = 1000
        expanded_partitions = []
        for task_ids, throttle, slot_class in partitions:
            if not task_ids:
                continue
            try:
                execution._scheduler_task_spec(task_ids)
                groups = (task_ids,)
            except RuntimeError:
                groups = []
                start = 0
                for index in range(1, len(task_ids) + 1):
                    if index < len(task_ids) and task_ids[index] == task_ids[index - 1] + 1:
                        continue
                    groups.append(task_ids[start:index])
                    start = index
            for group in groups:
                expanded_partitions.append(
                    (tuple(group), throttle, slot_class, str(job_number))
                )
                job_number += 1
        allocated = [1] * len(expanded_partitions)
        remaining = min(
            int(manifest["resources"]["throttle"]), int(manifest["task_count"])
        ) - len(allocated)
        while remaining:
            for index, (task_ids, _throttle, _slot_class, _job_id) in enumerate(
                expanded_partitions
            ):
                if allocated[index] < len(task_ids):
                    allocated[index] += 1
                    remaining -= 1
                    if remaining == 0:
                        break
        for allocation, (task_ids, _throttle, slot_class, job_id) in zip(
            allocated, expanded_partitions
        ):
            if not task_ids:
                continue
            throttle = allocation
            job_name = f"terminal{slot_class[0]}{job_id}_{RUN_TAG}"
            task_spec = execution._scheduler_task_spec(task_ids)
            raw = raw_dir / f"{job_name}.txt"
            raw.write_text(f"{job_id}\n", encoding="utf-8")
            status = raw_dir / f"{job_name}.status"
            status.write_text("0\n", encoding="utf-8")
            job = jobs / f"{job_name}.job"
            lines = [
                "#!/usr/bin/env bash", "#$ -cwd", f"#$ -N {job_name}",
                "#$ -q campus", "#$ -j y",
                f"#$ -o {root}/logs/{job_name}.$JOB_ID.$TASK_ID.log",
                f"#$ -l h_rt={h_rt // 3600:02d}:{(h_rt % 3600) // 60:02d}:{h_rt % 60:02d}",
                "#$ -l h_data=2000000000", f"#$ -t {task_spec}",
                f"#$ -tc {throttle}",
            ]
            if slot_class == "shared_two":
                lines.append("#$ -pe shared 2")
            lines.extend(("set -euo pipefail", "export LANG=C", "export LC_ALL=C"))
            if slot_class == "shared_two":
                lines.extend(
                    f"export {name}=1" for name in execution.REFERENCE_B_THREAD_ENVIRONMENT
                )
            lines.extend((
                f'cd "{project}"', "task_id=${SGE_TASK_ID}",
                '"python" scripts/terminal_validation_array.py run-task \\',
                '  --manifest "m" \\', f'  --output-root "{root}" \\',
                '  --task-id "${task_id}"',
            ))
            job.write_text("\n".join(lines) + "\n", encoding="utf-8")
            submissions.append({
                "job_id": job_id, "job_name": job_name, "queue": "campus",
                "array_job": True, "manifest_task_ids": task_ids,
                "qsub_raw_path": str(raw), "qsub_status_path": str(status),
                "job_script_path": str(job),
            })
            slots = 2 if slot_class == "shared_two" else 1
            granted_pe = "granted_pe shared\n" if slots == 2 else ""
            records = []
            for task_id in task_ids:
                records.append(
                    f"jobnumber {job_id}\njobname {job_name}\ntaskid {task_id}\nqname campus\n"
                    f"hostname n1234\nslots {slots}\n{granted_pe}failed 0\nexit_status 0\n"
                    f"start_time {start_text}\nend_time {end_text}\n"
                    "cpu 00:00:01\nru_wallclock 2\nmaxvmem 100M\n"
                )
            account = root / f"qacct_{job_id}.txt"
            account.write_text(
                "==============================================================\n".join(records),
                encoding="utf-8",
            )
            qacct_paths[job_id] = account
        return tuple(submissions), qacct_paths

    def create_scheduler(self, manifest, submissions, root):
        return execution.create_scheduler_evidence(
            manifest,
            submissions,
            evidence_root=root,
            execution_project_root=Path(execution.__file__).resolve().parents[2],
            approved_python_bin=Path("python"),
            authorized_manifest_path=Path("m"),
            scheduler_user="zzl",
            run_tag=RUN_TAG,
        )

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
            requires_b = execution.task_requires_reference_b(task)
            artifact = {
                "schema": execution.TASK_ARTIFACT_SCHEMA,
                "manifest_hash": manifest["manifest_hash"],
                "task_id": task_id,
                "assignment_hash": task["assignment_hash"],
                "logical_case_owner": task["logical_case_owner"],
                "subshard_index": task["subshard_index"],
                "subshard_count": task["subshard_count"],
                "job_id": submission_by_task[task_id]["job_id"],
                "sge_task_id": task_id,
                "slots": 2 if requires_b else 1,
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
                "slot_class": "shared_two" if requires_b else "one_slot",
                "parallel_environment": "shared" if requires_b else None,
                "pe_hostfile_sha256": "a" * 64 if requires_b else None,
                "pe_host_slots": ((binding["hostname"], 2),) if requires_b else (),
                "thread_environment": tuple(
                    (name, "1") for name in execution.REFERENCE_B_THREAD_ENVIRONMENT
                ) if requires_b else (),
                "reference_b_runtime_evidence": (
                    self.reference_b_runtime_evidence(
                        task["descriptors"][0]["descriptor_hash"]
                    ),
                ) if requires_b else (),
                "logical_record_hash": "",
            }
            artifact["logical_record_hash"] = execution.logical_hash(
                execution._without_hash(artifact, "logical_record_hash")
            )
            execution.write_new_json(target / "task.json", artifact)

    @staticmethod
    def reference_b_runtime_evidence(descriptor_hash):
        def worker(role, token):
            return {
                "role": role,
                "command": ("python", "-I", "-B", role),
                "command_hash": token * 64,
                "input_hash": ("3" if role == "traced" else "4") * 64,
                "output_hash": "5" * 64,
                "record_bytes_hash": "6" * 64,
                "source_identity_hash": "7" * 64,
                "interpreter_identity_hash": "8" * 64,
                "peak_rss_bytes": 1024,
                "wall_seconds": 0.05,
            }
        value = {
            "schema": execution.REFERENCE_B_RUNTIME_EVIDENCE_SCHEMA,
            "descriptor_hash": descriptor_hash,
            "traced_worker": worker("traced", "1"),
            "source_worker": worker("source_validation", "2"),
            "coordinator_peak_rss_bytes": 1024,
            "thread_environment": tuple(
                (name, "1") for name in execution.REFERENCE_B_THREAD_ENVIRONMENT
            ),
            "evidence_hash": "",
        }
        value["evidence_hash"] = execution.logical_hash(
            execution._without_hash(value, "evidence_hash")
        )
        return value

    def test_scheduler_rejects_wrong_stage_queue_shape_task_and_duplicates(self):
        manifest, _ = self.make_manifest("smoke")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            submissions, _ = self.scheduler_fixture(root, manifest)
            wrong_queue = [dict(item) for item in submissions]
            wrong_queue[0]["queue"] = "unauthorized"
            with self.assertRaisesRegex(RuntimeError, "queue"):
                self.create_scheduler(manifest, wrong_queue, root)
            wrong_shape = [dict(item) for item in submissions]
            wrong_shape[0]["array_job"] = False
            with self.assertRaisesRegex(RuntimeError, "must be an array"):
                self.create_scheduler(manifest, wrong_shape, root)
            wrong_task = [dict(item) for item in submissions]
            wrong_task[0]["manifest_task_ids"] = tuple(range(1, manifest["task_count"])) + (999,)
            with self.assertRaisesRegex(RuntimeError, "unknown"):
                self.create_scheduler(manifest, wrong_task, root)
            reordered = [dict(submissions[0])]
            reordered[0]["manifest_task_ids"] = tuple(reversed(reordered[0]["manifest_task_ids"]))
            with self.assertRaisesRegex(RuntimeError, "mapping|order|partition"):
                self.create_scheduler(manifest, reordered, root)
            job_path = Path(submissions[0]["job_script_path"])
            original = job_path.read_text(encoding="utf-8")
            job_path.write_text(original.replace("#$ -q campus", "#$ -q unauthorized"), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "queue/name"):
                self.create_scheduler(manifest, submissions, root)
            job_path.write_text(original + "#$ -pe shared 2\n", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "parallel-environment"):
                self.create_scheduler(manifest, submissions, root)

    def test_full_scheduler_requires_exact_slot_partitions(self):
        manifest, _ = self.make_manifest("full")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            submissions, _ = self.scheduler_fixture(root, manifest)
            scheduler = self.create_scheduler(manifest, submissions, root)
            expected = tuple(
                part for part in execution.partition_manifest_task_ids(manifest) if part
            )
            self.assertEqual(
                tuple(tuple(item["manifest_task_ids"]) for item in scheduler["submissions"]),
                expected,
            )
            forged = dict(submissions[0])
            forged["array_job"] = False
            with self.assertRaisesRegex(RuntimeError, "must be an array"):
                self.create_scheduler(manifest, (forged, *submissions[1:]), root)

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
            "sge_task_id": 999,
            "slots": 8,
        }
        for field, value in fields.items():
            with self.subTest(field=field), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                submissions, qacct_paths = self.scheduler_fixture(root, manifest)
                scheduler = self.create_scheduler(manifest, submissions, root)
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
                with self.assertRaisesRegex((RuntimeError, ValueError), "bound|identity|evidence|incomplete"):
                    execution.validate_task_scheduler_bindings(
                        manifest, task_output_root=root, scheduler=scheduler, qacct=qacct
                    )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            submissions, qacct_paths = self.scheduler_fixture(root, manifest)
            scheduler = self.create_scheduler(manifest, submissions, root)
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
            scheduler = self.create_scheduler(manifest, submissions, root)
            first = next(iter(qacct_paths.values()))
            first.write_text(first.read_text(encoding="utf-8") + "\n" + first.read_text(encoding="utf-8"), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "coverage|differs"):
                execution.audit_qacct(manifest, scheduler, qacct_paths, evidence_root=root)

    def test_scheduler_and_qacct_require_exact_successful_one_slot_coverage(self):
        manifest, _ = self.make_manifest("smoke")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            submissions, qacct = self.scheduler_fixture(root, manifest)
            scheduler = self.create_scheduler(manifest, submissions, root)
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
            first_text = first.read_text(encoding="utf-8")
            observed_slots = "slots 1" if "slots 1" in first_text else "slots 2"
            first.write_text(first_text.replace(observed_slots, "slots 8"), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "slot/PE"):
                execution.audit_qacct(manifest, scheduler, qacct, evidence_root=root)
            missing = submissions[:-1]
            with self.assertRaisesRegex(RuntimeError, "slot partitions"):
                self.create_scheduler(manifest, missing, root)

    def test_formal_smoke_qacct_enforces_wall_memory_and_qsub_status(self):
        manifest, _ = self.make_manifest("smoke")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            submissions, qacct_paths = self.scheduler_fixture(root, manifest)
            status = Path(submissions[0]["qsub_status_path"])
            status.write_text("1\n", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "qsub status"):
                self.create_scheduler(manifest, submissions, root)
            status.write_text("0\n", encoding="utf-8")
            scheduler = self.create_scheduler(manifest, submissions, root)
            qacct_path = next(iter(qacct_paths.values()))
            original = qacct_path.read_text(encoding="utf-8")
            qacct_path.write_text(original.replace("ru_wallclock 2", "ru_wallclock 21600.1", 1), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "P3 wall or memory"):
                execution.audit_qacct(manifest, scheduler, qacct_paths, evidence_root=root)
            qacct_path.write_text(original.replace("maxvmem 100M", "maxvmem 7G", 1), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "P3 wall or memory"):
                execution.audit_qacct(manifest, scheduler, qacct_paths, evidence_root=root)

    def test_formal_smoke_audit_requires_exact_chain_logs_and_sixteen_tasks(self):
        manifest, suites = self.make_manifest("smoke")
        ceiling = self.make_ceiling()
        manifest = dict(manifest)
        manifest["compute_ceiling_report_hash"] = ceiling["report_hash"]
        self.rehash_manifest(manifest)
        provisional = self.make_provisional(manifest)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            submissions, qacct_paths = self.scheduler_fixture(root, manifest)
            scheduler = self.create_scheduler(manifest, submissions, root)
            qacct = execution.audit_qacct(manifest, scheduler, qacct_paths, evidence_root=root)
            self.write_binding_task_artifacts(root, manifest, scheduler, qacct)

            scheduler_path = root / "scheduler.json"
            qacct_path = root / "qacct.json"
            provisional_path = root / "provisional.json"
            ceiling_path = root / "ceiling.json"
            execution.write_new_json(scheduler_path, scheduler)
            execution.write_new_json(qacct_path, qacct)
            execution.write_new_json(provisional_path, provisional)
            execution.write_new_json(ceiling_path, ceiling)
            post_path = root / "post.json"
            readback_path = root / "readback.json"
            empty_qstat = (
                "<?xml version='1.0'?><job_info><queue_info></queue_info>"
                "<job_info></job_info></job_info>\n"
            )
            finalization_dir = root / "finalization"
            completed_qstat = SimpleNamespace(stdout=empty_qstat, stderr="", returncode=0)
            with self.no_numerical_sentinels(), patch.object(execution, "validate_clean_source_identity"), patch.object(
                execution, "validate_execution_manifest"
            ), patch.object(execution, "validate_task_scheduler_bindings"), patch.object(
                execution, "recompute_provisional", return_value=provisional
            ), patch.object(execution.platform, "node", return_value="n9999"), patch.object(
                execution.subprocess, "run", return_value=completed_qstat
            ) as run_qstat:
                execution.finalize_and_capture_formal_smoke(
                    manifest=manifest, suites=suites, provider=self.provider,
                    acceptance_validator=self.accepted, task_output_root=root,
                    provisional_path=provisional_path,
                    scheduler_evidence_path=scheduler_path,
                    qacct_audit_path=qacct_path,
                    compute_ceiling_path=ceiling_path,
                    scheduler_evidence_root=root,
                    post_job_path=post_path,
                    finalization_capture_dir=finalization_dir,
                    qstat_bin="qstat",
                    project_root=root,
                )
            run_qstat.assert_called_once_with(
                ("qstat", "-xml", "-u", "zzl"),
                text=True,
                capture_output=True,
                check=False,
            )
            with self.no_numerical_sentinels(), patch.object(execution.platform, "system", return_value="Darwin"), patch.object(
                execution, "validate_clean_source_identity"
            ), patch.object(execution, "validate_execution_manifest"), patch.object(
                execution, "validate_task_scheduler_bindings"
            ), patch.object(execution, "recompute_provisional", return_value=provisional), patch.object(
                execution.platform, "node", return_value="local-mac"
            ):
                execution.independent_readback(
                    manifest=manifest, suites=suites, provider=self.provider,
                    acceptance_validator=self.accepted, task_output_root=root,
                    provisional_path=provisional_path,
                    scheduler_evidence_path=scheduler_path,
                    qacct_audit_path=qacct_path,
                    compute_ceiling_path=ceiling_path,
                    scheduler_evidence_root=root,
                    post_job_path=post_path,
                    final_output_path=readback_path,
                    project_root=root,
                )

            logs = root / "logs"
            logs.mkdir(exist_ok=True)
            submission = scheduler["submissions"][0]
            for task_id in range(1, 17):
                (logs / f"{submission['job_name']}.{submission['job_id']}.{task_id}.log").touch()

            def audit_again(name, recomputed=provisional):
                with self.no_numerical_sentinels(), patch.object(execution, "validate_clean_source_identity"), patch.object(
                    execution, "validate_execution_manifest"
                ), patch.object(execution, "recompute_provisional", return_value=recomputed):
                    return execution.audit_formal_smoke(
                        manifest=manifest,
                        suites=suites,
                        provider=self.provider,
                        acceptance_validator=self.accepted,
                        task_output_root=root,
                        provisional_path=provisional_path,
                        scheduler_evidence_path=scheduler_path,
                        qacct_audit_path=qacct_path,
                        compute_ceiling_path=ceiling_path,
                        scheduler_evidence_root=root,
                        post_job_path=post_path,
                        readback_path=readback_path,
                        finalization_capture_dir=finalization_dir,
                        logs_dir=logs,
                        output_path=root / name,
                        project_root=root,
                    )

            audit = audit_again("formal_audit.json")
            self.assertTrue(audit["audit_pass"])
            self.assertEqual(audit["task_count"], 16)
            self.assertEqual(audit["logical_owner_counts"], tuple((owner, 4) for owner in execution.SMOKE_CASE_IDS))
            self.assertLess(audit["queue_excluded_chain_seconds"], 60.0)

            # A self-rehashed artifact from another run cannot be substituted into the chain.
            original_post = post_path.read_bytes()
            forged_post = execution._decode(dict(execution._load_json(post_path)))
            forged_post["bound_file_hashes"] = tuple(
                (name, "f" * 64 if name == "scheduler" else value)
                for name, value in forged_post["bound_file_hashes"]
            )
            forged_post["logical_record_hash"] = execution.logical_hash(
                execution._without_hash(forged_post, "logical_record_hash")
            )
            post_path.unlink()
            execution.write_new_json(post_path, forged_post)
            with self.assertRaisesRegex(RuntimeError, "bound-file hashes"):
                audit_again("cross_run_post_audit.json")
            post_path.write_bytes(original_post)

            original_readback = readback_path.read_bytes()
            forged_readback = execution._decode(dict(execution._load_json(readback_path)))
            forged_readback["post_job_hash"] = "f" * 64
            forged_readback["logical_record_hash"] = execution.logical_hash(
                execution._without_hash(forged_readback, "logical_record_hash")
            )
            readback_path.unlink()
            execution.write_new_json(readback_path, forged_readback)
            with self.assertRaisesRegex(RuntimeError, "finalization/readback"):
                audit_again("cross_run_readback_audit.json")
            readback_path.write_bytes(original_readback)

            malformed = execution._decode(dict(execution._load_json(readback_path)))
            malformed.pop("observed_row_count")
            malformed["logical_record_hash"] = execution.logical_hash(
                execution._without_hash(malformed, "logical_record_hash")
            )
            readback_path.unlink()
            execution.write_new_json(readback_path, malformed)
            with self.assertRaisesRegex(RuntimeError, "incomplete or mismatched"):
                audit_again("malformed_readback_audit.json")
            readback_path.write_bytes(original_readback)

            # Raw scheduler/qacct and recomputed task evidence remain authoritative.
            raw_qacct_path = next(iter(qacct_paths.values()))
            original_raw_qacct = raw_qacct_path.read_bytes()
            raw_qacct_path.write_bytes(original_raw_qacct.replace(b"maxvmem 100M", b"maxvmem 101M", 1))
            with self.assertRaisesRegex(RuntimeError, "raw|qacct"):
                audit_again("raw_qacct_tamper_audit.json")
            raw_qacct_path.write_bytes(original_raw_qacct)

            for name, relative in (
                ("row", Path("tasks/task_00001/rows.json")),
                ("metric", Path("tasks/task_00001/metrics.json")),
            ):
                path = root / relative
                original = path.read_bytes()
                path.write_bytes(original + b" ")
                changed = dict(provisional, logical_record_hash="f" * 64)
                with self.subTest(tamper=name), self.assertRaisesRegex(
                    RuntimeError, "rows or sidecars changed"
                ):
                    audit_again(f"{name}_tamper_audit.json", recomputed=changed)
                path.write_bytes(original)

            sidecar_path = root / "tasks/task_00001/sidecars/tamper.bin"
            sidecar_path.parent.mkdir(parents=True, exist_ok=True)
            sidecar_path.write_bytes(b"original")
            sidecar_path.write_bytes(b"tampered")
            changed = dict(provisional, logical_record_hash="e" * 64)
            with self.assertRaisesRegex(RuntimeError, "rows or sidecars changed"):
                audit_again("sidecar_tamper_audit.json", recomputed=changed)
            with self.assertRaisesRegex(FileExistsError, "already exist"):
                execution.finalize_and_capture_formal_smoke(
                    manifest=manifest, suites=suites, provider=self.provider,
                    acceptance_validator=self.accepted, task_output_root=root,
                    provisional_path=provisional_path,
                    scheduler_evidence_path=scheduler_path,
                    qacct_audit_path=qacct_path,
                    compute_ceiling_path=ceiling_path,
                    scheduler_evidence_root=root,
                    post_job_path=post_path,
                    finalization_capture_dir=finalization_dir,
                    qstat_bin="qstat",
                    project_root=root,
                )
            warning = SimpleNamespace(
                stdout=empty_qstat,
                stderr="permission warning\n",
                returncode=0,
            )
            warning_capture = root / "warning_finalization"
            with patch.object(execution.subprocess, "run", return_value=warning):
                with self.assertRaisesRegex(RuntimeError, "emitted stderr"):
                    execution._capture_formal_smoke_finalization(
                        manifest=manifest,
                        scheduler_evidence_path=scheduler_path,
                        qacct_audit_path=qacct_path,
                        post_job_path=post_path,
                        output_dir=warning_capture,
                        qstat_bin="qstat",
                    )
            self.assertFalse(warning_capture.exists())

    def test_one_slot_task_and_provisional_collection_use_row_sidecar_source_validation(self):
        mdp = one_atom_mdp(FiniteSupportAtom(80.0, 0.5, -1))
        belief = mdp.initial_belief()
        item = descriptor_for(mdp, belief)
        bundle = execution.evaluate_terminal_evidence_descriptor(item, mdp, belief)
        rows = tuple(row for row in bundle.rows if row.method != "reference_b")
        sidecars = dict(bundle.sidecars)
        bundle = SimpleNamespace(
            descriptor_hash=item.descriptor_hash,
            rows=rows,
            sidecars=tuple(
                (row.sidecar.relative_path, sidecars[row.sidecar.relative_path])
                for row in rows
            ),
        )
        methods = tuple(row.method for row in rows)
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
            "array_required": True,
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
        scheduler = {"NSLOTS": "1", "JOB_ID": "1234", "SGE_TASK_ID": "1"}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch.object(execution, "validate_execution_manifest") as validate:
                with patch.object(execution, "reconstruct_terminal_evidence_source", return_value=(mdp, belief)), patch.object(
                    execution, "evaluate_terminal_evidence_descriptor", return_value=bundle
                ):
                    with self.assertRaisesRegex(RuntimeError, "array task ID"):
                        execution.execute_task(
                            manifest=manifest, suites=suites, provider=self.provider,
                            acceptance_validator=self.accepted, output_root=root, task_id=1,
                            scheduler_environment=dict(scheduler, SGE_TASK_ID="2"),
                        )
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
                    with self.no_numerical_sentinels():
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
            "max_walltime_seconds": 86400,
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
            scheduler = self.create_scheduler(manifest, submissions, root)
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
            scheduler = self.create_scheduler(manifest, submissions, root)
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
        approved_python = Path("/approved/python3.11")
        authorized_manifest = Path("/approved/terminal_smoke_manifest.json")
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
            "approved_python_bin": str(approved_python),
            "authorized_manifest_path": str(authorized_manifest),
            "approved_scheduler_user": "zzl",
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "authorization.json"
            execution.write_new_json(path, authorization)
            approved = execution.sha256_file(path)
            with patch.object(execution, "validate_clean_source_identity"):
                execution.validate_execution_authorization(
                    authorization_path=path, approved_file_hash=approved,
                    manifest=manifest, project_root=project_root,
                    approved_python_bin=approved_python,
                    authorized_manifest_path=authorized_manifest,
                    approved_scheduler_user="zzl",
                )
                for field in ("manifest_hash", "compute_ceiling_report_hash", "provider_hash"):
                    with self.subTest(cross_reuse=field):
                        other = dict(manifest)
                        other[field] = "f" * 64
                        with self.assertRaisesRegex(RuntimeError, "exact manifest/source"):
                            execution.validate_execution_authorization(
                                authorization_path=path, approved_file_hash=approved,
                                manifest=other, project_root=project_root,
                                approved_python_bin=approved_python,
                                authorized_manifest_path=authorized_manifest,
                                approved_scheduler_user="zzl",
                            )
                other = dict(manifest)
                other["resources"] = dict(manifest["resources"], throttle=31)
                with self.assertRaisesRegex(RuntimeError, "exact manifest/source"):
                    execution.validate_execution_authorization(
                        authorization_path=path, approved_file_hash=approved,
                        manifest=other, project_root=project_root,
                        approved_python_bin=approved_python,
                        authorized_manifest_path=authorized_manifest,
                        approved_scheduler_user="zzl",
                    )
                other = dict(manifest)
                other_source = dict(source, identity_hash="f" * 64)
                other["source_identity"] = other_source
                with self.assertRaisesRegex(RuntimeError, "exact manifest/source"):
                    execution.validate_execution_authorization(
                        authorization_path=path, approved_file_hash=approved,
                        manifest=other, project_root=project_root,
                        approved_python_bin=approved_python,
                        authorized_manifest_path=authorized_manifest,
                        approved_scheduler_user="zzl",
                    )
                with self.assertRaisesRegex(RuntimeError, "externally approved"):
                    execution.validate_execution_authorization(
                        authorization_path=path, approved_file_hash="f" * 64,
                        manifest=manifest, project_root=project_root,
                        approved_python_bin=approved_python,
                        authorized_manifest_path=authorized_manifest,
                        approved_scheduler_user="zzl",
                    )
                for changed_python, changed_manifest in (
                    (Path("/wrong/python"), authorized_manifest),
                    (approved_python, Path("/wrong/manifest.json")),
                ):
                    with self.assertRaisesRegex(RuntimeError, "exact manifest/source"):
                        execution.validate_execution_authorization(
                            authorization_path=path,
                            approved_file_hash=approved,
                            manifest=manifest,
                            project_root=project_root,
                            approved_python_bin=changed_python,
                            authorized_manifest_path=changed_manifest,
                            approved_scheduler_user="zzl",
                        )
                with self.assertRaisesRegex(RuntimeError, "exact manifest/source"):
                    execution.validate_execution_authorization(
                        authorization_path=path,
                        approved_file_hash=approved,
                        manifest=manifest,
                        project_root=project_root,
                        approved_python_bin=approved_python,
                        authorized_manifest_path=authorized_manifest,
                        approved_scheduler_user="wrong-user",
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
        create_output = script.index("mkdir -p \\")
        qsub = script.index('"${QSUB_BIN}" -terse')
        self.assertLess(authorization, create_output)
        self.assertLess(ceiling, create_output)
        self.assertLess(create_output, qsub)
        self.assertIn('describe-task-partitions', script)
        self.assertIn('append_partition_groups "2" "${shared_two_ids}" "shared_two"', script)
        self.assertIn('aggregate_throttle="${throttle}"', script)
        self.assertIn("#$ -pe shared 2", script)
        self.assertIn('qsub_status_path', script)
        self.assertIn('rollback left a validation-tagged job', script)

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
