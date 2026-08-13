from __future__ import annotations

from dataclasses import replace
import gzip
import hashlib
import json
import math
import unittest
from unittest.mock import patch

import src.experiments.terminal_evidence_rows as evidence_module
from src.experiments.terminal_evidence_rows import (
    TerminalEvidenceBundle,
    decode_terminal_certificate_sidecar,
    evaluate_terminal_evidence_descriptor,
    recompute_terminal_evidence_summary,
    terminal_descriptor_source_failures,
    terminal_evidence_row_hash,
    validate_terminal_certificate_sidecar,
    validate_terminal_evidence_bundle_source,
    validate_terminal_evidence_bundle_structure,
    validate_terminal_evidence_row,
)
from src.experiments.terminal_validation_suite import (
    AUTHORITATIVE_PROVIDER_KIND,
    BASE_CONSTRUCTION_RULE,
    BASE_SUITE_VERSION,
    DESCRIPTOR_SCHEMA,
    TerminalValidationSuite,
    TerminalValidationDescriptor,
    build_local_diagnostic_base_provider,
    build_terminal_base_suite,
    canonical_base_provider_hash,
    canonical_hash,
    load_terminal_validation_identities,
    terminal_scientific_spec_hash,
    terminal_validation_descriptor_hash,
    terminal_validation_manifest_hash,
)
from src.mdp.finite_support import FiniteSupportAtom
from src.solvers.terminal_reference_agreement import (
    TERMINAL_PRODUCTION_REGRET_TOLERANCE,
)
from tests.test_terminal_optimizer import one_atom_mdp


GLOBAL_NUMERICAL_HASH = "1" * 64


def descriptor_for(mdp, belief, *, suite_class="base", prespecified=False):
    history_hash = canonical_hash(())
    descriptor = TerminalValidationDescriptor(
        schema=DESCRIPTOR_SCHEMA,
        suite_class=suite_class,
        suite_version=BASE_SUITE_VERSION if suite_class == "base" else "test_stress_v1",
        descriptor_index=0,
        component_validation_only=suite_class != "base",
        environment_selection_eligible=False,
        legacy_spec_hash="2" * 64,
        legacy_numerical_case_hash="3" * 64,
        scientific_spec_hash=terminal_scientific_spec_hash(),
        numerical_method_config_hash=GLOBAL_NUMERICAL_HASH,
        source_case_id=0,
        environment="test",
        environment_hash="4" * 64,
        support_hash=mdp.prior.support_hash,
        sigma_sample=float(mdp.config.sigma_sample),
        sample_time_cost=float(mdp.config.sample_time_cost),
        profile="uniform_prior",
        orientation="symmetric",
        depth=0,
        deliberation_time=float(belief.deliberation_time),
        remaining_time_after_termination=float(mdp.remaining_time_after_termination(belief)),
        action_sequence=(),
        offset_sequence=(),
        history=(),
        history_hash=history_hash,
        posterior_weight_hash=canonical_hash(tuple(belief.weights)),
        canonical_belief_hash=canonical_hash({
            "support_hash": mdp.prior.support_hash,
            "posterior_weights": tuple(belief.weights),
            "deliberation_time": belief.deliberation_time,
            "history_hash": history_hash,
        }),
        legacy_belief_hash=None,
        local_legacy_belief_hash=None,
        legacy_reconstruction_matches=None,
        component_index=0 if prespecified else None,
        z_offset=0 if prespecified else None,
        reference_b_prespecified=prespecified,
        construction_rule=BASE_CONSTRUCTION_RULE,
        construction_hash="5" * 64,
        descriptor_hash="",
    )
    return replace(descriptor, descriptor_hash=terminal_validation_descriptor_hash(descriptor))


def _replace_bundle_row(bundle, changed):
    rows = tuple(changed if row.method == changed.method else row for row in bundle.rows)
    return TerminalEvidenceBundle(bundle.descriptor_hash, rows, bundle.sidecars)


def _self_rehash_sidecar(bundle, method, mutate):
    row = next(item for item in bundle.rows if item.method == method)
    sidecars = dict(bundle.sidecars)
    payload = json.loads(gzip.decompress(sidecars[row.sidecar.relative_path]).decode("utf-8"))
    mutate(payload)
    payload["certificate_logical_hash"] = evidence_module._logical_hash(
        payload["certificate_payload"]
    )
    payload["complete_trace_logical_hash"] = evidence_module._logical_hash(
        payload["complete_trace_payload"]
    )
    logical_payload = dict(payload)
    logical_payload.pop("logical_record_hash")
    payload["logical_record_hash"] = evidence_module._logical_hash(logical_payload)
    compressed = gzip.compress(
        evidence_module._canonical_json_bytes(payload), compresslevel=9, mtime=0
    )
    sidecar = replace(
        row.sidecar,
        sha256=hashlib.sha256(compressed).hexdigest(),
        byte_count=len(compressed),
        logical_record_hash=payload["logical_record_hash"],
    )
    changed_row = replace(row, sidecar=sidecar, logical_record_hash="")
    changed_row = replace(
        changed_row, logical_record_hash=terminal_evidence_row_hash(changed_row)
    )
    rows = tuple(changed_row if item.method == method else item for item in bundle.rows)
    sidecars[sidecar.relative_path] = compressed
    return TerminalEvidenceBundle(bundle.descriptor_hash, rows, tuple(sorted(sidecars.items())))


class TerminalEvidenceRowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mdp = one_atom_mdp(FiniteSupportAtom(80.0, 0.5, -1))
        cls.belief = cls.mdp.initial_belief()
        cls.descriptor = descriptor_for(cls.mdp, cls.belief)
        cls.bundle = evaluate_terminal_evidence_descriptor(
            cls.descriptor, cls.mdp, cls.belief
        )
        cls.sidecars = dict(cls.bundle.sidecars)

    def assert_source_rejected(self, bundle):
        failures = validate_terminal_evidence_bundle_source(
            bundle, self.descriptor, self.mdp, self.belief
        )
        self.assertTrue(failures)
        return failures

    def test_fixture_uses_frozen_suite_scientific_identity(self):
        identities = load_terminal_validation_identities()
        self.assertEqual(
            self.descriptor.scientific_spec_hash,
            identities.scientific_spec_hash,
        )
        self.assertNotIn(
            "scientific_spec_hash_mismatch",
            terminal_descriptor_source_failures(
                self.descriptor, self.mdp, self.belief
            ),
        )

    def test_honest_bundle_has_exact_schemas_full_traces_and_source_recomputation(self):
        self.assertEqual(
            terminal_descriptor_source_failures(self.descriptor, self.mdp, self.belief), ()
        )
        self.assertEqual(validate_terminal_evidence_bundle_source(
            self.bundle, self.descriptor, self.mdp, self.belief
        ), ())
        expected_trace_schemas = {
            "production_terminal": "terminal_production_complete_trace_v1",
            "reference_a": "terminal_reference_a_complete_trace_v1",
            "reference_b": "terminal_reference_b_complete_trace_v1",
            "agreement": "terminal_agreement_complete_trace_v1",
        }
        for row in self.bundle.rows:
            decoded = decode_terminal_certificate_sidecar(
                row.sidecar,
                self.sidecars[row.sidecar.relative_path],
                descriptor=self.descriptor,
                method=row.method,
                method_numerical_hash=row.method_numerical_hash,
            )
            self.assertEqual(decoded.complete_trace["schema"], expected_trace_schemas[row.method])
            self.assertTrue(decoded.complete_trace["complete"])
            if row.method in ("production_terminal", "reference_a", "reference_b"):
                if row.method == "production_terminal":
                    self.assertTrue(decoded.complete_trace["created_nodes"])
                    self.assertTrue(decoded.complete_trace["pop_events"])
                else:
                    levels = decoded.complete_trace["precision_levels"]
                    self.assertTrue(levels)
                    self.assertTrue(levels[-1]["created_nodes"])
                    self.assertTrue(levels[-1]["pop_events"])

    def test_structural_validation_accepts_honest_bundle_without_recomputation(self):
        with patch.object(
            evidence_module,
            "evaluate_terminal_evidence_descriptor",
            side_effect=AssertionError("structural validation launched a solver"),
        ):
            self.assertEqual(
                validate_terminal_evidence_bundle_structure(
                    self.bundle, self.descriptor
                ),
                (),
            )

        forged_row = replace(
            self.bundle.rows[0], status="forged", logical_record_hash=""
        )
        forged_row = replace(
            forged_row,
            logical_record_hash=terminal_evidence_row_hash(forged_row),
        )
        forged = _replace_bundle_row(self.bundle, forged_row)
        self.assertTrue(
            validate_terminal_evidence_bundle_structure(forged, self.descriptor)
        )

        sidecars = dict(self.bundle.sidecars)
        path = self.bundle.rows[0].sidecar.relative_path
        sidecars[path] = sidecars[path] + b"tamper"
        tampered = TerminalEvidenceBundle(
            self.bundle.descriptor_hash,
            self.bundle.rows,
            tuple(sorted(sidecars.items())),
        )
        self.assertTrue(
            validate_terminal_evidence_bundle_structure(tampered, self.descriptor)
        )

    def test_evaluator_reuses_each_completed_source_validation_downstream(self):
        import src.solvers.terminal_reference_agreement as agreement_module

        with (
            patch.object(
                evidence_module,
                "source_validate_terminal_reference_record",
                wraps=evidence_module.source_validate_terminal_reference_record,
            ) as validate_a,
            patch.object(
                evidence_module,
                "source_validate_terminal_reference_b_record",
                wraps=evidence_module.source_validate_terminal_reference_b_record,
            ) as validate_b,
            patch.object(
                agreement_module,
                "validate_terminal_reference_record",
                side_effect=AssertionError("agreement recomputed Reference A"),
            ),
            patch.object(
                agreement_module,
                "validate_terminal_reference_b_record",
                side_effect=AssertionError("agreement recomputed Reference B"),
            ),
        ):
            bundle = evaluate_terminal_evidence_descriptor(
                self.descriptor, self.mdp, self.belief
            )
        self.assertTrue(bundle.rows)
        self.assertEqual(validate_a.call_count, 1)
        self.assertEqual(validate_b.call_count, 1)

    def test_false_source_proofs_fail_closed_without_hidden_retry(self):
        import src.solvers.terminal_reference as reference_a_module
        import src.solvers.terminal_reference_agreement as agreement_module
        import src.solvers.terminal_reference_b as reference_b_module

        real_a = evidence_module.source_validate_terminal_reference_record
        real_b = evidence_module.source_validate_terminal_reference_b_record

        def false_a(*args, **kwargs):
            return replace(real_a(*args, **kwargs), valid=False)

        def false_b(*args, **kwargs):
            return replace(real_b(*args, **kwargs), valid=False)

        with (
            patch.object(
                reference_a_module,
                "validate_terminal_reference_record",
                wraps=reference_a_module.validate_terminal_reference_record,
            ) as validate_a,
            patch.object(
                reference_b_module,
                "validate_terminal_reference_b_record",
                wraps=reference_b_module.validate_terminal_reference_b_record,
            ) as validate_b,
            patch.object(evidence_module, "source_validate_terminal_reference_record", false_a),
            patch.object(evidence_module, "source_validate_terminal_reference_b_record", false_b),
            patch.object(
                agreement_module,
                "validate_terminal_reference_record",
                side_effect=AssertionError("agreement retried Reference A"),
            ),
            patch.object(
                agreement_module,
                "validate_terminal_reference_b_record",
                side_effect=AssertionError("agreement retried Reference B"),
            ),
        ):
            bundle = evaluate_terminal_evidence_descriptor(
                self.descriptor, self.mdp, self.belief
            )
        self.assertTrue(any(not row.pass_status for row in bundle.rows))
        self.assertEqual(validate_a.call_count, 1)
        self.assertEqual(validate_b.call_count, 1)

    def test_b1_unknown_certificate_type_fails_after_every_hash_is_recomputed(self):
        forged = _self_rehash_sidecar(
            self.bundle,
            "production_terminal",
            lambda payload: payload.update({
                "certificate_type": "ForgedCertificate",
                "certificate_payload": {"allocation": {"float_hex": 0.5.hex()}, "status": "accepted"},
            }),
        )
        row = next(item for item in forged.rows if item.method == "production_terminal")
        failures = validate_terminal_certificate_sidecar(
            row.sidecar, dict(forged.sidecars)[row.sidecar.relative_path],
            descriptor=self.descriptor, method=row.method,
            method_numerical_hash=row.method_numerical_hash,
        )
        self.assertTrue(failures)
        self.assert_source_rejected(forged)

    def test_b1_finite_certificate_mutation_fails_after_self_rehash(self):
        forged = _self_rehash_sidecar(
            self.bundle,
            "production_terminal",
            lambda payload: payload["certificate_payload"].update(
                allocation={"float_hex": 0.25.hex()}
            ),
        )
        self.assert_source_rejected(forged)

    def test_b2_row_semantics_must_equal_source_validated_certificate(self):
        row = self.bundle.rows[0]
        changed = replace(row, production_allocation=0.25, logical_record_hash="")
        changed = replace(changed, logical_record_hash=terminal_evidence_row_hash(changed))
        failures = self.assert_source_rejected(_replace_bundle_row(self.bundle, changed))
        self.assertTrue(any("row_semantic_mismatch" in item for item in failures))

        coherently_forged = _self_rehash_sidecar(
            self.bundle,
            "production_terminal",
            lambda payload: payload["certificate_payload"].update(
                allocation={"float_hex": 0.25.hex()}
            ),
        )
        forged_row = next(
            item for item in coherently_forged.rows
            if item.method == "production_terminal"
        )
        forged_row = replace(
            forged_row,
            production_allocation=0.25,
            canonical_allocation_interval=(0.25, 0.25),
            logical_record_hash="",
        )
        forged_row = replace(
            forged_row,
            logical_record_hash=terminal_evidence_row_hash(forged_row),
        )
        failures = self.assert_source_rejected(
            _replace_bundle_row(coherently_forged, forged_row)
        )
        self.assertTrue(any("source_recomputation_mismatch" in item for item in failures))

    def test_b3_adversarial_rows_fail_method_rules_and_thresholds(self):
        base = self.bundle.rows[0]
        attacks = {
            "allocation": {"production_allocation": 2.0},
            "allocation_gap": {
                "production_allocation": 0.0,
                "canonical_allocation_interval": (0.5, 0.5),
            },
            "canonical": {"canonical_allocation_interval": (2.0, 3.0)},
            "regret": {"production_regret_interval": (0.0, 1.0)},
            "status": {"status": "banana"},
            "global": {"global_value_interval": (-100.0, 100.0)},
            "value_gap": {"production_value_interval": (100.0, 100.0)},
        }
        for name, changes in attacks.items():
            changed = replace(base, **changes, logical_record_hash="")
            changed = replace(changed, logical_record_hash=terminal_evidence_row_hash(changed))
            with self.subTest(name=name):
                row_failures = validate_terminal_evidence_row(changed, self.descriptor)
                self.assertTrue(row_failures)
                self.assert_source_rejected(_replace_bundle_row(self.bundle, changed))
        self.assertLessEqual(
            self.bundle.rows[-1].production_regret_interval[1],
            TERMINAL_PRODUCTION_REGRET_TOLERANCE,
        )

    def test_b4_escalation_comes_from_validated_source_not_rehashed_summaries(self):
        stress = descriptor_for(self.mdp, self.belief, suite_class="one_step")
        honest = evaluate_terminal_evidence_descriptor(stress, self.mdp, self.belief)
        self.assertEqual([row.method for row in honest.rows], ["production_terminal", "reference_a"])
        forged = _self_rehash_sidecar(
            honest,
            "production_terminal",
            lambda payload: payload["certificate_payload"]["structural_symmetry"].update(
                valid=True,
                proof_hash="9" * 64,
            ),
        )
        failures = validate_terminal_evidence_bundle_source(
            forged, stress, self.mdp, self.belief
        )
        self.assertTrue(failures)

        production = honest.rows[0]
        forged_row = replace(
            production,
            reference_b_required=True,
            reference_b_trigger_reasons=("structural_symmetry",),
            logical_record_hash="",
        )
        forged_row = replace(
            forged_row,
            logical_record_hash=terminal_evidence_row_hash(forged_row),
        )
        self.assert_source_rejected(_replace_bundle_row(honest, forged_row))

    def test_b5_authoritative_migration_unavailable_fails_closed(self):
        identities = load_terminal_validation_identities()
        provider = build_local_diagnostic_base_provider()
        suite = build_terminal_base_suite(identities, provider)
        summary = recompute_terminal_evidence_summary(
            (), {}, suite, base_provider=provider, require_authoritative=True
        )
        self.assertFalse(summary.authoritative_source_accepted)
        self.assertFalse(summary.evidence_valid)
        self.assertFalse(summary.stage_complete)
        self.assertFalse(summary.candidate_pass)
        self.assertTrue(any("authoritative" in reason for reason in summary.failure_reasons))

        relabeled = replace(
            provider,
            provider_kind=AUTHORITATIVE_PROVIDER_KIND,
            diagnostic_only=False,
            provider_hash="",
        )
        relabeled = replace(
            relabeled,
            provider_hash=canonical_base_provider_hash(relabeled),
        )
        relabeled_summary = recompute_terminal_evidence_summary(
            (), {}, suite, base_provider=relabeled, require_authoritative=True
        )
        self.assertFalse(relabeled_summary.authoritative_source_accepted)
        self.assertTrue(any(
            "authoritative_base_acceptance_rejected" in reason
            for reason in relabeled_summary.invalid_row_keys
        ))

    def test_b5_self_rehashed_suite_identity_changes_fail_closed(self):
        identities = load_terminal_validation_identities()
        provider = build_local_diagnostic_base_provider()
        suite = build_terminal_base_suite(identities, provider)
        for field_name in ("scientific_spec_hash", "numerical_method_config_hash"):
            manifest = replace(
                suite.manifest,
                **{field_name: "9" * 64},
                manifest_hash="",
            )
            manifest = replace(
                manifest,
                manifest_hash=terminal_validation_manifest_hash(manifest),
            )
            forged_suite = TerminalValidationSuite(manifest, suite.descriptors)
            with self.subTest(field=field_name):
                summary = recompute_terminal_evidence_summary(
                    (), {}, forged_suite, base_provider=provider,
                    require_authoritative=False,
                )
                self.assertFalse(summary.evidence_valid)
                self.assertFalse(summary.stage_complete)
                self.assertFalse(summary.candidate_pass)
                self.assertIn("suite:manifest_identity", summary.invalid_row_keys)

    def test_b6_self_rehashed_trace_deletion_fails(self):
        forged = _self_rehash_sidecar(
            self.bundle,
            "reference_a",
            lambda payload: payload["complete_trace_payload"].update(
                precision_levels=[]
            ),
        )
        self.assert_source_rejected(forged)

    def test_missing_duplicate_cross_version_nonfinite_and_corrupt_still_fail(self):
        row = self.bundle.rows[0]
        corrupted = self.sidecars[row.sidecar.relative_path][:-1]
        self.assertTrue(validate_terminal_certificate_sidecar(
            row.sidecar, corrupted, descriptor=self.descriptor, method=row.method,
            method_numerical_hash=row.method_numerical_hash,
        ))
        duplicate = TerminalEvidenceBundle(
            self.bundle.descriptor_hash,
            self.bundle.rows + (self.bundle.rows[0],),
            self.bundle.sidecars,
        )
        self.assert_source_rejected(duplicate)
        missing_path = self.bundle.rows[0].sidecar.relative_path
        missing_sidecar = TerminalEvidenceBundle(
            self.bundle.descriptor_hash,
            self.bundle.rows,
            tuple(
                (path, payload) for path, payload in self.bundle.sidecars
                if path != missing_path
            ),
        )
        missing_failures = self.assert_source_rejected(missing_sidecar)
        self.assertTrue(any("bundle_sidecar_set_mismatch" in item for item in missing_failures))
        cross = replace(row, numerical_method_config_hash="9" * 64, logical_record_hash="")
        cross = replace(cross, logical_record_hash=terminal_evidence_row_hash(cross))
        self.assert_source_rejected(_replace_bundle_row(self.bundle, cross))
        nonfinite = replace(row, production_allocation=math.nan, logical_record_hash="")
        with self.assertRaises(ValueError):
            terminal_evidence_row_hash(nonfinite)


if __name__ == "__main__":
    unittest.main()
