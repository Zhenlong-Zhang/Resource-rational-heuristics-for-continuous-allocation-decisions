"""Test purpose: validate frozen terminal cases, coverage, and aggregate acceptance rules."""

from __future__ import annotations

from collections import Counter, defaultdict
import copy
from dataclasses import replace
import math
import unittest

import src.experiments.terminal_validation_suite as suite_module
from src.experiments.terminal_validation_suite import (
    AUTHORITATIVE_PROVIDER_KIND,
    DEFAULT_NUMERICAL_METHOD_CONFIG_PATH,
    FROZEN_CONSTRUCTION_HASHES,
    FROZEN_NUMERICAL_METHOD_CONFIG_HASH,
    FROZEN_SAMPLE_COSTS,
    FROZEN_SCIENTIFIC_SPEC_HASH,
    FROZEN_Z_OFFSETS,
    LEGACY_NUMERICAL_CASE_HASH,
    LEGACY_SPEC_HASH,
    LOCAL_DIAGNOSTIC_PROVIDER_KIND,
    ORIENTATION_VOCABULARY,
    REFERENCE_B_PRESPECIFIED_ONE_STEP_COUNT,
    SUITE_HARD_CAPS,
    build_local_diagnostic_base_provider,
    build_terminal_base_suite,
    build_terminal_one_step_suite,
    build_terminal_reachable_core_suite,
    build_terminal_scientific_projection,
    canonical_base_provider_hash,
    canonical_base_record_hash,
    canonical_hash,
    expected_terminal_numerical_method_config,
    load_frozen_strategy_mapping_cases,
    load_frozen_strategy_mapping_spec,
    load_terminal_numerical_method_config,
    load_terminal_validation_identities,
    make_canonical_base_provider,
    source_derived_terminal_method_hashes,
    suite_integrity_failures,
    terminal_numerical_method_config_hash,
    terminal_scientific_spec_hash,
    terminal_validation_descriptor_hash,
    terminal_validation_manifest_hash,
    validate_terminal_numerical_method_config,
    validate_terminal_validation_suite,
)
from src.solvers.terminal import (
    production_terminal_numerical_method_config_hash,
)
from src.solvers.terminal_reference import (
    terminal_reference_a_numerical_method_config_hash,
)
from src.solvers.terminal_reference_agreement import (
    terminal_reference_agreement_numerical_method_config_hash,
)
from src.solvers.terminal_reference_b import (
    terminal_reference_b_numerical_method_config_hash,
)


def _rehash_descriptor(descriptor, **changes):
    updated = replace(descriptor, **changes, descriptor_hash="")
    return replace(
        updated,
        descriptor_hash=terminal_validation_descriptor_hash(updated),
    )


def _self_rehash_suite(suite, descriptors=None, **manifest_changes):
    descriptors = tuple(descriptors or suite.descriptors)
    invariant = suite_module.FROZEN_SUITE_INVARIANTS[suite.manifest.suite_class]
    names = tuple(name for name, _ in invariant.partitions)
    manifest = replace(
        suite.manifest,
        hard_cap=len(descriptors),
        pre_dedup_count=len(descriptors),
        post_dedup_count=len(descriptors),
        ordered_construction_hash=canonical_hash(
            tuple(item.construction_hash for item in descriptors)
        ),
        ordered_descriptor_hash=canonical_hash(
            tuple(item.descriptor_hash for item in descriptors)
        ),
        partitions=suite_module._partition(descriptors, names),
        manifest_hash="",
        **manifest_changes,
    )
    manifest = replace(
        manifest,
        manifest_hash=terminal_validation_manifest_hash(manifest),
    )
    return replace(suite, manifest=manifest, descriptors=descriptors)


class TerminalValidationIdentityTests(unittest.TestCase):
    def test_legacy_scientific_and_numerical_identities_are_separate(self):
        spec = load_frozen_strategy_mapping_spec()
        cases = load_frozen_strategy_mapping_cases(spec)
        identities = load_terminal_validation_identities()

        self.assertEqual(len(cases), 90)
        self.assertEqual([case["case_id"] for case in cases], list(range(90)))
        self.assertEqual(identities.legacy_spec_hash, LEGACY_SPEC_HASH)
        self.assertEqual(
            identities.legacy_numerical_case_hash,
            LEGACY_NUMERICAL_CASE_HASH,
        )
        self.assertEqual(identities.scientific_spec_hash, FROZEN_SCIENTIFIC_SPEC_HASH)
        self.assertEqual(
            identities.numerical_method_config_hash,
            FROZEN_NUMERICAL_METHOD_CONFIG_HASH,
        )
        self.assertNotEqual(identities.scientific_spec_hash, identities.legacy_spec_hash)
        self.assertNotEqual(
            identities.scientific_spec_hash,
            identities.numerical_method_config_hash,
        )

    def test_scientific_projection_excludes_method_controls(self):
        projection = build_terminal_scientific_projection()
        serialized = str(projection)
        for numerical_control in (
            "rr_terminal_grid_size",
            "rr_terminal_reference_grid_size",
            "oracle_grid_size",
            "gauss_hermite_reference_order",
            "matched_voi_gauss_hermite_order",
            "evaluation_cap",
            "precision_ladder",
        ):
            self.assertNotIn(numerical_control, serialized)
        self.assertEqual(
            terminal_scientific_spec_hash(),
            FROZEN_SCIENTIFIC_SPEC_HASH,
        )

    def test_numerical_config_is_versioned_and_hash_sensitive(self):
        config = load_terminal_numerical_method_config()
        self.assertEqual(config["schema_version"], 2)
        self.assertEqual(config["scientific_fields_permitted"], False)
        self.assertEqual(
            config["suite_construction"]["hard_caps"],
            dict(SUITE_HARD_CAPS),
        )
        self.assertEqual(
            terminal_numerical_method_config_hash(config),
            FROZEN_NUMERICAL_METHOD_CONFIG_HASH,
        )
        self.assertEqual(validate_terminal_numerical_method_config(config), ())
        self.assertEqual(
            terminal_numerical_method_config_hash(config),
            terminal_numerical_method_config_hash(
                expected_terminal_numerical_method_config()
            ),
        )
        mutations = (
            ("source_derived_method_hashes", "production_terminal"),
            ("source_derived_method_hashes", "reference_a"),
            ("source_derived_method_hashes", "reference_b"),
            ("source_derived_method_hashes", "agreement"),
            ("suite_construction", "hard_caps"),
            ("suite_construction", "orientation_vocabulary"),
            ("suite_construction", "partition_order_rule"),
            ("evidence", "terminal_row_schema"),
            ("evidence", "row_order"),
        )
        for section, field in mutations:
            changed = copy.deepcopy(config)
            changed[section][field] = "tampered"
            with self.subTest(section=section, field=field):
                self.assertTrue(validate_terminal_numerical_method_config(changed))
        self.assertEqual(terminal_scientific_spec_hash(), FROZEN_SCIENTIFIC_SPEC_HASH)

    def test_global_config_binds_source_derived_method_hashes(self):
        expected = {
            "production_terminal": production_terminal_numerical_method_config_hash(),
            "reference_a": terminal_reference_a_numerical_method_config_hash(),
            "reference_b": terminal_reference_b_numerical_method_config_hash(),
            "agreement": terminal_reference_agreement_numerical_method_config_hash(),
        }
        self.assertEqual(source_derived_terminal_method_hashes(), expected)
        self.assertEqual(
            load_terminal_numerical_method_config()["source_derived_method_hashes"],
            expected,
        )
        self.assertNotEqual(
            production_terminal_numerical_method_config_hash(
                value_tolerance=1e-5
            ),
            expected["production_terminal"],
        )
        self.assertNotEqual(
            terminal_reference_a_numerical_method_config_hash(199_999),
            expected["reference_a"],
        )
        self.assertNotEqual(
            terminal_reference_b_numerical_method_config_hash(499_999),
            expected["reference_b"],
        )


class TerminalValidationSuiteTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.identities = load_terminal_validation_identities()
        cls.base = build_terminal_base_suite(cls.identities)
        cls.one_step = build_terminal_one_step_suite(cls.identities)
        cls.reachable = build_terminal_reachable_core_suite(cls.identities)

    def test_base_suite_preserves_all_90_original_cases(self):
        manifest = self.base.manifest
        self.assertEqual(len(self.base.descriptors), 90)
        self.assertEqual(manifest.hard_cap, 90)
        self.assertEqual(manifest.pre_dedup_count, 90)
        self.assertEqual(manifest.post_dedup_count, 90)
        self.assertFalse(manifest.deduplication_applied)
        self.assertEqual(
            [descriptor.source_case_id for descriptor in self.base.descriptors],
            list(range(90)),
        )
        self.assertEqual(
            Counter(descriptor.profile for descriptor in self.base.descriptors),
            {
                "uniform_prior": 18,
                "person1_predictive_mean": 18,
                "both_predictive_means": 18,
                "person1_minimum_support": 18,
                "person1_maximum_support": 18,
            },
        )
        frozen = load_frozen_strategy_mapping_cases()
        for descriptor, case in zip(self.base.descriptors, frozen):
            self.assertEqual(descriptor.legacy_belief_hash, case["belief_hash"])
            self.assertEqual(
                descriptor.legacy_reconstruction_matches,
                descriptor.local_legacy_belief_hash == descriptor.legacy_belief_hash,
            )

    def test_one_step_suite_has_exact_full_coverage_and_actual_time(self):
        descriptors = self.one_step.descriptors
        manifest = self.one_step.manifest
        self.assertEqual(len(descriptors), 35_640)
        self.assertEqual(manifest.hard_cap, 35_640)
        self.assertEqual(manifest.pre_dedup_count, 35_640)
        self.assertEqual(manifest.post_dedup_count, 35_640)
        self.assertFalse(manifest.deduplication_applied)
        by_source = Counter(descriptor.source_case_id for descriptor in descriptors)
        self.assertEqual(set(by_source.values()), {396})
        self.assertEqual(
            Counter(descriptor.profile for descriptor in descriptors),
            {"sample_1": 17_820, "sample_2": 17_820},
        )
        self.assertEqual(
            set(descriptor.z_offset for descriptor in descriptors),
            set(FROZEN_Z_OFFSETS),
        )
        self.assertEqual(
            set(descriptor.component_index for descriptor in descriptors),
            set(range(18)),
        )
        self.assertEqual(
            sum(descriptor.reference_b_prespecified for descriptor in descriptors),
            REFERENCE_B_PRESPECIFIED_ONE_STEP_COUNT,
        )
        for descriptor in descriptors:
            self.assertEqual(descriptor.depth, 1)
            self.assertEqual(len(descriptor.history), 1)
            self.assertEqual(descriptor.history[0].action, descriptor.profile)
            self.assertEqual(descriptor.history[0].cost, descriptor.sample_time_cost)
            self.assertEqual(descriptor.deliberation_time, descriptor.sample_time_cost)
            self.assertEqual(
                descriptor.remaining_time_after_termination,
                max(0.0, 120.0 - descriptor.sample_time_cost - 1.0),
            )

    def test_orientation_vocabulary_and_action_component_semantics_are_exact(self):
        self.assertEqual(ORIENTATION_VOCABULARY, ("-1", "+1", "symmetric", "balanced"))
        self.assertTrue(
            set(item.orientation for item in self.base.descriptors)
            <= set(ORIENTATION_VOCABULARY)
        )
        self.assertEqual(
            Counter(item.orientation for item in self.one_step.descriptors),
            {"-1": 17_820, "+1": 17_820},
        )
        runtime = {
            int(case["case_id"]): (mdp, belief)
            for case, _, mdp, belief in suite_module._runtime_base_cases()
        }
        for descriptor in self.one_step.descriptors:
            mdp, belief = runtime[descriptor.source_case_id]
            atom = belief.states[descriptor.component_index]
            expected_action = descriptor.profile
            expected_observation = (
                mdp._need_for_action(atom, expected_action)
                + descriptor.z_offset * mdp.config.sigma_sample
            )
            self.assertEqual(descriptor.orientation, f"{atom.orientation:+d}")
            self.assertEqual(descriptor.action_sequence, (expected_action,))
            self.assertEqual(descriptor.history[0].action, expected_action)
            self.assertEqual(descriptor.history[0].observation, expected_observation)

    def test_reachable_suite_covers_cost_depth_orientation_and_history_patterns(self):
        descriptors = self.reachable.descriptors
        manifest = self.reachable.manifest
        self.assertEqual(len(descriptors), 648)
        self.assertEqual(manifest.hard_cap, 648)
        self.assertEqual(manifest.pre_dedup_count, 648)
        self.assertEqual(manifest.post_dedup_count, 648)
        self.assertFalse(manifest.deduplication_applied)
        self.assertEqual(
            Counter(descriptor.sample_time_cost for descriptor in descriptors),
            {cost: 81 for cost in FROZEN_SAMPLE_COSTS},
        )
        profiles = Counter(descriptor.profile for descriptor in descriptors)
        self.assertEqual(len(profiles), 9)
        self.assertEqual(set(profiles.values()), {72})
        self.assertEqual(
            set(descriptor.orientation for descriptor in descriptors),
            {"symmetric", "balanced", "-1", "+1"},
        )
        grouped = defaultdict(list)
        for descriptor in descriptors:
            grouped[descriptor.profile].append(descriptor)
            self.assertEqual(descriptor.depth, len(descriptor.action_sequence))
            self.assertEqual(descriptor.depth, len(descriptor.offset_sequence))
            self.assertEqual(descriptor.depth, len(descriptor.history))
            self.assertTrue(
                math.isclose(
                    descriptor.deliberation_time,
                    math.fsum(step.cost for step in descriptor.history),
                    rel_tol=0.0,
                    abs_tol=1e-12,
                )
            )
            self.assertTrue(
                all(step.cost == descriptor.sample_time_cost for step in descriptor.history)
            )
        self.assertEqual({item.depth for item in grouped["initial_symmetric"]}, {0})
        self.assertEqual({item.depth for item in grouped["one_step_orientation_-1"]}, {1})
        self.assertEqual({item.depth for item in grouped["one_step_orientation_+1"]}, {1})
        for name in (
            "balanced_depth_6",
            "concentrated_depth_6_-1",
            "concentrated_depth_6_+1",
        ):
            self.assertEqual({item.depth for item in grouped[name]}, {6})
        for item in grouped["balanced_depth_6"]:
            self.assertEqual(
                item.action_sequence,
                ("sample_1", "sample_2", "sample_1", "sample_2", "sample_1", "sample_2"),
            )
        for item in grouped["concentrated_depth_6_-1"]:
            self.assertEqual(set(item.action_sequence), {"sample_1"})
        for item in grouped["concentrated_depth_6_+1"]:
            self.assertEqual(set(item.action_sequence), {"sample_2"})
        for name in (
            "balanced_late_feasible",
            "concentrated_late_feasible_-1",
            "concentrated_late_feasible_+1",
        ):
            for item in grouped[name]:
                expected = min(19, math.floor((120.0 - 1.0) / item.sample_time_cost))
                self.assertEqual(item.depth, expected)

    def test_stress_suites_are_never_environment_selection_inputs(self):
        for suite in (self.one_step, self.reachable):
            self.assertTrue(suite.manifest.component_validation_only)
            self.assertFalse(suite.manifest.environment_selection_eligible)
            self.assertTrue(
                all(item.component_validation_only for item in suite.descriptors)
            )
            self.assertTrue(
                all(not item.environment_selection_eligible for item in suite.descriptors)
            )

    def test_hashes_counts_and_integrity_are_deterministic(self):
        for suite in (self.base, self.one_step, self.reachable):
            self.assertEqual(
                suite.manifest.ordered_construction_hash,
                FROZEN_CONSTRUCTION_HASHES[suite.manifest.suite_class],
            )
            self.assertEqual(suite_integrity_failures(suite, self.identities), ())
        rebuilt = build_terminal_reachable_core_suite(self.identities)
        self.assertEqual(rebuilt.manifest, self.reachable.manifest)
        self.assertEqual(rebuilt.descriptors, self.reachable.descriptors)

    def test_local_replay_is_diagnostic_and_authoritative_mode_requires_acceptance(self):
        provider = build_local_diagnostic_base_provider()
        diagnostic = validate_terminal_validation_suite(
            self.base,
            self.identities,
            base_provider=provider,
        )
        self.assertEqual(diagnostic.failures, ())
        self.assertEqual(diagnostic.validation_status, "diagnostic_source_validated")
        self.assertFalse(diagnostic.authoritative_source_accepted)

        local_as_authoritative = validate_terminal_validation_suite(
            self.base,
            self.identities,
            base_provider=provider,
            require_authoritative=True,
            authoritative_acceptance_validator=lambda _: True,
        )
        self.assertIn(
            "authoritative_base_provider_required",
            local_as_authoritative.failures,
        )
        self.assertFalse(local_as_authoritative.authoritative_source_accepted)

        accepted_fixture = make_canonical_base_provider(
            provider.records,
            provider_kind=AUTHORITATIVE_PROVIDER_KIND,
            source_identity_hash=canonical_hash(
                {"test_fixture": "independently_accepted_provider_boundary"}
            ),
            diagnostic_only=False,
        )
        missing_acceptance = validate_terminal_validation_suite(
            self.base,
            self.identities,
            base_provider=accepted_fixture,
            require_authoritative=True,
        )
        self.assertIn(
            "authoritative_base_acceptance_rejected",
            missing_acceptance.failures,
        )

        accepted = validate_terminal_validation_suite(
            self.base,
            self.identities,
            base_provider=accepted_fixture,
            require_authoritative=True,
            authoritative_acceptance_validator=(
                lambda candidate: candidate.provider_hash
                == accepted_fixture.provider_hash
            ),
        )
        self.assertIn(
            "authoritative_custom_acceptance_disallowed",
            accepted.failures,
        )
        self.assertIn(
            "authoritative_base_acceptance_rejected",
            accepted.failures,
        )
        self.assertEqual(accepted.validation_status, "validation_failed")
        self.assertFalse(accepted.authoritative_source_accepted)

    def test_one_step_descendants_start_from_supplied_exact_base_weights(self):
        provider = build_local_diagnostic_base_provider()
        original = provider.records[0]
        weights = list(original.posterior_weights)
        delta = min(weights[0], weights[1]) * 0.125
        weights[0] += delta
        weights[1] -= delta
        changed = replace(
            original,
            posterior_weights=tuple(weights),
            record_hash="",
        )
        changed = replace(changed, record_hash=canonical_base_record_hash(changed))
        records = (changed,) + provider.records[1:]
        custom_provider = make_canonical_base_provider(
            records,
            provider_kind=LOCAL_DIAGNOSTIC_PROVIDER_KIND,
            source_identity_hash=canonical_hash(
                {"test_fixture": "exact_supplied_base_weights"}
            ),
            diagnostic_only=True,
        )

        custom_one_step = build_terminal_one_step_suite(
            self.identities,
            custom_provider,
        )
        _, _, _, supplied_belief = suite_module._runtime_base_cases(
            custom_provider
        )[0]
        self.assertEqual(supplied_belief.weights, changed.posterior_weights)
        self.assertNotEqual(
            custom_one_step.descriptors[0].posterior_weight_hash,
            self.one_step.descriptors[0].posterior_weight_hash,
        )
        self.assertEqual(
            suite_integrity_failures(
                custom_one_step,
                self.identities,
                base_provider=custom_provider,
            ),
            (),
        )
        self.assertIn(
            "source_mismatch:posterior_weight_hash",
            suite_integrity_failures(
                custom_one_step,
                self.identities,
                base_provider=provider,
            ),
        )

    def test_self_rehashed_base_semantic_forgeries_are_source_rejected(self):
        descriptor = self.base.descriptors[0]
        tampered = _rehash_descriptor(
            descriptor,
            environment_hash="0" * 64,
            support_hash="1" * 64,
            sigma_sample=descriptor.sigma_sample + 0.25,
            sample_time_cost=descriptor.sample_time_cost + 0.5,
            deliberation_time=descriptor.deliberation_time + 0.5,
            remaining_time_after_termination=(
                descriptor.remaining_time_after_termination + 1.0
            ),
            posterior_weight_hash="2" * 64,
            canonical_belief_hash="3" * 64,
            legacy_reconstruction_matches=(
                not descriptor.legacy_reconstruction_matches
            ),
        )
        forged = _self_rehash_suite(
            self.base,
            (tampered,) + self.base.descriptors[1:],
        )
        failures = suite_integrity_failures(forged, self.identities)
        for field in (
            "environment_hash",
            "support_hash",
            "sigma_sample",
            "sample_time_cost",
            "deliberation_time",
            "remaining_time_after_termination",
            "posterior_weight_hash",
            "canonical_belief_hash",
            "legacy_reconstruction_matches",
        ):
            with self.subTest(field=field):
                self.assertIn(f"source_mismatch:{field}", failures)

    def test_self_rehashed_one_step_semantic_forgeries_are_source_rejected(self):
        descriptor = self.one_step.descriptors[0]
        history = (
            replace(
                descriptor.history[0],
                observation=descriptor.history[0].observation + 0.125,
            ),
        )
        tampered = _rehash_descriptor(
            descriptor,
            history=history,
            history_hash=canonical_hash(history),
            posterior_weight_hash="4" * 64,
            canonical_belief_hash="5" * 64,
        )
        forged = _self_rehash_suite(
            self.one_step,
            (tampered,) + self.one_step.descriptors[1:],
        )
        failures = suite_integrity_failures(forged, self.identities)
        for field in (
            "history",
            "history_hash",
            "posterior_weight_hash",
            "canonical_belief_hash",
        ):
            with self.subTest(field=field):
                self.assertIn(f"source_mismatch:{field}", failures)

    def test_self_rehashed_reachable_semantic_forgeries_are_source_rejected(self):
        descriptor = self.reachable.descriptors[1]
        history = (
            replace(
                descriptor.history[0],
                observation=descriptor.history[0].observation - 0.125,
            ),
        )
        tampered = _rehash_descriptor(
            descriptor,
            history=history,
            history_hash=canonical_hash(history),
            posterior_weight_hash="6" * 64,
            canonical_belief_hash="7" * 64,
        )
        forged_descriptors = list(self.reachable.descriptors)
        forged_descriptors[1] = tampered
        forged = _self_rehash_suite(self.reachable, tuple(forged_descriptors))
        failures = suite_integrity_failures(forged, self.identities)
        for field in (
            "history",
            "history_hash",
            "posterior_weight_hash",
            "canonical_belief_hash",
        ):
            with self.subTest(field=field):
                self.assertIn(f"source_mismatch:{field}", failures)

    def test_identity_count_and_descriptor_tampering_fail_closed(self):
        descriptor = self.base.descriptors[0]
        tampered_descriptor = replace(descriptor, profile="tampered")
        tampered_suite = replace(
            self.base,
            descriptors=(tampered_descriptor,) + self.base.descriptors[1:],
        )
        self.assertIn(
            "descriptor_hash",
            suite_integrity_failures(tampered_suite, self.identities),
        )
        self.assertNotEqual(
            terminal_validation_descriptor_hash(tampered_descriptor),
            tampered_descriptor.descriptor_hash,
        )

        wrong_identity = replace(
            self.base,
            manifest=replace(self.base.manifest, scientific_spec_hash="0" * 64),
        )
        failures = suite_integrity_failures(wrong_identity, self.identities)
        self.assertIn("manifest_identity", failures)
        self.assertIn("manifest_hash", failures)

        wrong_count = replace(
            self.base,
            manifest=replace(self.base.manifest, pre_dedup_count=89),
        )
        failures = suite_integrity_failures(wrong_count, self.identities)
        self.assertIn("pre_dedup_count", failures)
        self.assertIn("manifest_hash", failures)

    def test_self_rehashed_deletion_insertion_and_reorder_are_rejected(self):
        deleted = _self_rehash_suite(
            self.reachable,
            self.reachable.descriptors[:-1],
        )
        self.assertIn(
            "descriptor_count",
            suite_integrity_failures(deleted, self.identities),
        )

        inserted_descriptor = _rehash_descriptor(
            self.reachable.descriptors[-1],
            descriptor_index=len(self.reachable.descriptors),
        )
        inserted = _self_rehash_suite(
            self.reachable,
            self.reachable.descriptors + (inserted_descriptor,),
        )
        self.assertIn(
            "descriptor_count",
            suite_integrity_failures(inserted, self.identities),
        )

        descriptors = list(self.reachable.descriptors)
        first = _rehash_descriptor(descriptors[1], descriptor_index=0)
        second = _rehash_descriptor(descriptors[0], descriptor_index=1)
        descriptors[:2] = (first, second)
        reordered = _self_rehash_suite(self.reachable, tuple(descriptors))
        failures = suite_integrity_failures(reordered, self.identities)
        self.assertIn("descriptor_construction_sequence", failures)

    def test_self_rehashed_selection_suite_identity_and_partition_forgeries_are_rejected(self):
        manifest_selection = _self_rehash_suite(
            self.reachable,
            component_validation_only=False,
            environment_selection_eligible=True,
        )
        self.assertIn(
            "manifest_selection_flags",
            suite_integrity_failures(manifest_selection, self.identities),
        )

        first = _rehash_descriptor(
            self.reachable.descriptors[0],
            environment_selection_eligible=True,
        )
        descriptor_selection = _self_rehash_suite(
            self.reachable,
            (first,) + self.reachable.descriptors[1:],
        )
        self.assertIn(
            "descriptor_selection_flags",
            suite_integrity_failures(descriptor_selection, self.identities),
        )

        suite_identity = _self_rehash_suite(
            self.reachable,
            schema="forged_schema",
            suite_version="forged_version",
            construction_rule="forged_rule",
        )
        failures = suite_integrity_failures(suite_identity, self.identities)
        self.assertIn("manifest_schema", failures)
        self.assertIn("manifest_suite_identity", failures)
        self.assertIn("manifest_construction_rule", failures)

        descriptor_schema = _rehash_descriptor(
            self.reachable.descriptors[0],
            schema="forged_schema",
        )
        descriptor_schema_forgery = _self_rehash_suite(
            self.reachable,
            (descriptor_schema,) + self.reachable.descriptors[1:],
        )
        self.assertIn(
            "descriptor_schema",
            suite_integrity_failures(descriptor_schema_forgery, self.identities),
        )

        partitions = list(self.reachable.manifest.partitions)
        partitions[0] = replace(partitions[0], count=partitions[0].count - 1)
        manifest = replace(
            self.reachable.manifest,
            partitions=tuple(partitions),
            manifest_hash="",
        )
        manifest = replace(
            manifest,
            manifest_hash=terminal_validation_manifest_hash(manifest),
        )
        partition_forgery = replace(self.reachable, manifest=manifest)
        self.assertIn(
            "frozen_partitions",
            suite_integrity_failures(partition_forgery, self.identities),
        )


if __name__ == "__main__":
    unittest.main()
    LOCAL_DIAGNOSTIC_PROVIDER_KIND,
    build_local_diagnostic_base_provider,
    canonical_base_provider_hash,
    canonical_base_record_hash,
    make_canonical_base_provider,
