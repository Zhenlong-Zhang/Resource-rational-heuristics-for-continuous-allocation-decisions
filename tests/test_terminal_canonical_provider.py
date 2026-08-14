"""Test purpose: validate selection and loading of the accepted canonical terminal provider."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

import src.experiments.terminal_canonical_provider as provider_module
import src.experiments.terminal_evidence_rows as evidence_module
import src.experiments.terminal_execution as execution
import scripts.terminal_validation_array as terminal_cli
from src.experiments.terminal_canonical_provider import (
    ACCEPTED_CANONICAL_BASE_ARTIFACT_SHA256,
    ACCEPTED_CANONICAL_BASE_SEMANTIC_HASH,
    load_accepted_canonical_base_provider,
)
from src.experiments.terminal_evidence_rows import recompute_terminal_evidence_summary
from src.experiments.terminal_validation_suite import (
    AUTHORITATIVE_PROVIDER_KIND,
    build_terminal_base_suite,
    build_terminal_one_step_suite,
    canonical_base_provider_hash,
    canonical_base_record_hash,
    canonical_hash,
    load_terminal_validation_identities,
    reconstruct_canonical_base_record,
    validate_terminal_validation_suite,
)


def self_rehashed_provider(provider):
    first = provider.records[0]
    weights = list(first.posterior_weights)
    delta = min(weights[1] * 0.5, 1e-6)
    weights[0] += delta
    weights[1] -= delta
    record = replace(first, posterior_weights=tuple(weights), record_hash="")
    record = replace(record, record_hash=canonical_base_record_hash(record))
    records = (record,) + provider.records[1:]
    forged = replace(
        provider,
        records=records,
        records_hash=canonical_hash(tuple(item.record_hash for item in records)),
        provider_hash="",
    )
    return replace(forged, provider_hash=canonical_base_provider_hash(forged))


class AcceptedCanonicalProviderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.provider, accepted = load_accepted_canonical_base_provider()
        cls.accepted = staticmethod(accepted)
        cls.identities = load_terminal_validation_identities()

    def test_trust_roots_and_all_90_records_reconstruct(self):
        self.assertIs(
            self.accepted,
            provider_module.accepted_canonical_base_provider,
        )
        self.assertEqual(
            ACCEPTED_CANONICAL_BASE_ARTIFACT_SHA256,
            "59f327defb5e7e931214140ab9e0264fc75b2a6d63a46f4d3c85a18cf0fde997",
        )
        self.assertEqual(
            ACCEPTED_CANONICAL_BASE_SEMANTIC_HASH,
            "0e453ecc8b1247decb369d7a7587ea744a07e9629606ee706fec24b8cc26381c",
        )
        self.assertEqual(self.provider.provider_kind, AUTHORITATIVE_PROVIDER_KIND)
        self.assertFalse(self.provider.diagnostic_only)
        self.assertEqual(len(self.provider.records), 90)
        self.assertEqual(tuple(item.case_id for item in self.provider.records), tuple(range(90)))
        for record in self.provider.records:
            with self.subTest(case_id=record.case_id):
                self.assertEqual(record.record_hash, canonical_base_record_hash(record))
                prior, belief = reconstruct_canonical_base_record(record)
                self.assertEqual(prior.support_hash, record.support_hash)
                self.assertEqual(tuple(prior.weights), record.prior_weights)
                self.assertEqual(tuple(belief.weights), record.posterior_weights)
                self.assertEqual(belief.deliberation_time, record.deliberation_time)
        self.assertTrue(self.accepted(self.provider))

    def test_tamper_wrong_hash_wrong_kind_and_missing_file_fail_closed(self):
        first = self.provider.records[0]
        weights = list(first.posterior_weights)
        delta = min(weights[1] * 0.5, 1e-6)
        weights[0] += delta
        weights[1] -= delta
        tampered_record = replace(first, posterior_weights=tuple(weights), record_hash="")
        tampered_record = replace(
            tampered_record,
            record_hash=canonical_base_record_hash(tampered_record),
        )
        tampered_records = (tampered_record,) + self.provider.records[1:]
        tampered_provider = replace(
            self.provider,
            records=tampered_records,
            records_hash=canonical_hash(tuple(item.record_hash for item in tampered_records)),
            provider_hash="",
        )
        tampered_provider = replace(
            tampered_provider,
            provider_hash=canonical_base_provider_hash(tampered_provider),
        )
        wrong_source = replace(
            self.provider,
            source_identity_hash="0" * 64,
            provider_hash="",
        )
        wrong_source = replace(
            wrong_source,
            provider_hash=canonical_base_provider_hash(wrong_source),
        )
        wrong_kind = replace(
            self.provider,
            provider_kind="local_deterministic_reconstruction_diagnostic_only",
            diagnostic_only=True,
            provider_hash="",
        )
        wrong_kind = replace(
            wrong_kind,
            provider_hash=canonical_base_provider_hash(wrong_kind),
        )
        self.assertFalse(self.accepted(tampered_provider))
        self.assertFalse(self.accepted(wrong_source))
        self.assertFalse(self.accepted(wrong_kind))

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            missing = root / "missing.json"
            with patch.object(
                provider_module, "DEFAULT_ACCEPTED_CANONICAL_BASE_PATH", missing
            ):
                with self.assertRaises(FileNotFoundError):
                    load_accepted_canonical_base_provider()

            altered = root / "altered.json"
            shutil.copyfile(
                provider_module.DEFAULT_ACCEPTED_CANONICAL_BASE_PATH,
                altered,
            )
            altered.write_bytes(altered.read_bytes() + b"\n")
            with patch.object(
                provider_module, "DEFAULT_ACCEPTED_CANONICAL_BASE_PATH", altered
            ):
                with self.assertRaisesRegex(RuntimeError, "byte SHA-256 mismatch"):
                    load_accepted_canonical_base_provider()

        with patch.object(
            provider_module,
            "ACCEPTED_CANONICAL_BASE_ARTIFACT_SHA256",
            "0" * 64,
        ):
            with self.assertRaisesRegex(RuntimeError, "byte SHA-256 mismatch"):
                load_accepted_canonical_base_provider()
        with patch.object(
            provider_module,
            "ACCEPTED_CANONICAL_BASE_SEMANTIC_HASH",
            "0" * 64,
        ):
            with self.assertRaisesRegex(RuntimeError, "output hash is not approved"):
                load_accepted_canonical_base_provider()

    def test_all_90_cases_have_source_valid_base_and_one_step_descendants(self):
        base = build_terminal_base_suite(self.identities, self.provider)
        one_step = build_terminal_one_step_suite(self.identities, self.provider)
        self.assertEqual(len(base.descriptors), 90)
        self.assertEqual(len(one_step.descriptors), 90 * 2 * 18 * 11)
        for case_id in range(90):
            descendants = tuple(
                item for item in one_step.descriptors if item.source_case_id == case_id
            )
            with self.subTest(case_id=case_id):
                self.assertEqual(len(descendants), 2 * 18 * 11)
                self.assertTrue(all(item.depth == 1 for item in descendants))
        for suite in (base, one_step):
            validation = validate_terminal_validation_suite(
                suite,
                self.identities,
                base_provider=self.provider,
                require_authoritative=True,
                authoritative_acceptance_validator=self.accepted,
            )
            self.assertEqual(validation.failures, ())
            self.assertTrue(validation.authoritative_source_accepted)

    def test_suite_evidence_and_execution_share_the_exact_acceptance_gate(self):
        suites = execution.build_terminal_suites(self.provider, self.accepted)
        wrong_hash = self_rehashed_provider(self.provider)
        self.assertFalse(self.accepted(wrong_hash))
        fake_validator = lambda _provider: True
        failed = validate_terminal_validation_suite(
            suites["base"],
            self.identities,
            base_provider=wrong_hash,
            require_authoritative=True,
            authoritative_acceptance_validator=fake_validator,
        )
        self.assertFalse(failed.authoritative_source_accepted)
        self.assertIn("authoritative_custom_acceptance_disallowed", failed.failures)
        self.assertIn("authoritative_base_acceptance_rejected", failed.failures)

        evidence_summary = recompute_terminal_evidence_summary(
            (),
            {},
            suites["base"],
            base_provider=wrong_hash,
            authoritative_acceptance_validator=fake_validator,
            require_authoritative=True,
        )
        self.assertFalse(evidence_summary.authoritative_source_accepted)
        self.assertFalse(evidence_summary.candidate_pass)
        self.assertTrue(any(
            "authoritative_custom_acceptance_disallowed" in reason
            for reason in evidence_summary.failure_reasons
        ))

        source = {
            "schema": execution.SOURCE_IDENTITY_SCHEMA,
            "commit": "1" * 40,
            "tree": "2" * 40,
            "source_hashes": (("source.py", "3" * 64),),
            "source_hashes_hash": execution.logical_hash(
                (("source.py", "3" * 64),)
            ),
            "identity_hash": "",
        }
        source["identity_hash"] = execution.logical_hash(
            execution._without_hash(source, "identity_hash")
        )
        with patch.object(
            execution,
            "_expected_descriptor_plan",
            return_value=(execution.TERMINAL_METHOD_ORDER, 1, 1),
        ):
            manifest = execution.create_execution_manifest(
                stage="smoke",
                suites=suites,
                provider=self.provider,
                acceptance_validator=self.accepted,
                source_identity=source,
                max_descriptors_per_subshard=450,
                resources={
                    "queue": "campus",
                    "h_rt_seconds": 3600,
                    "memory_bytes": 2 * 1024**3,
                    "throttle": 32,
                },
                compute_ceiling_report_hash="4" * 64,
            )
        self.assertEqual(manifest["provider_hash"], self.provider.provider_hash)
        self.assertEqual(
            manifest["provider_source_identity_hash"],
            self.provider.source_identity_hash,
        )
        with self.assertRaisesRegex(RuntimeError, "custom authoritative"):
            execution.create_execution_manifest(
                stage="smoke",
                suites=suites,
                provider=wrong_hash,
                acceptance_validator=fake_validator,
                source_identity=source,
                max_descriptors_per_subshard=450,
                resources={
                    "queue": "campus",
                    "h_rt_seconds": 3600,
                    "memory_bytes": 2 * 1024**3,
                    "throttle": 32,
                },
                compute_ceiling_report_hash="4" * 64,
            )

    def test_accepted_empty_evidence_bundle_fails_closed_without_index_error(self):
        suite = build_terminal_base_suite(self.identities, self.provider)
        def empty_bundle(descriptor, _mdp, _belief):
            return evidence_module.TerminalEvidenceBundle(
                descriptor.descriptor_hash,
                (),
                (),
            )

        with patch.object(
            evidence_module,
            "evaluate_terminal_evidence_descriptor",
            side_effect=empty_bundle,
        ):
            summary = recompute_terminal_evidence_summary(
                (),
                {},
                suite,
                base_provider=self.provider,
                authoritative_acceptance_validator=self.accepted,
                require_authoritative=True,
            )

        self.assertTrue(summary.authoritative_source_accepted)
        self.assertFalse(summary.evidence_valid)
        self.assertFalse(summary.stage_complete)
        self.assertFalse(summary.candidate_pass)
        self.assertIn("invalid_rows", summary.failure_reasons)
        self.assertTrue(
            all(
                "source_reconstruction:empty_evidence_bundle" in failure
                for failure in summary.invalid_row_keys
            )
        )
        self.assertEqual(len(summary.invalid_row_keys), len(suite.descriptors))

    def test_official_cli_uses_only_the_accepted_loader_and_binds_its_sources(self):
        sentinel = (self.provider, self.accepted)
        with patch.object(
            execution,
            "load_accepted_canonical_base_provider",
            return_value=sentinel,
        ) as accepted_loader:
            self.assertIs(terminal_cli.load_provider(), sentinel)
        accepted_loader.assert_called_once_with()
        self.assertFalse(hasattr(execution, "load_approved_canonical_provider"))

        parser_help = terminal_cli.build_parser().format_help()
        for legacy_flag in (
            "--migration",
            "--approved-migration-hash",
            "--provider-acceptance",
        ):
            self.assertNotIn(legacy_flag, parser_help)
        self.assertIn(
            "src/experiments/terminal_canonical_provider.py",
            terminal_cli.SOURCE_PATHS,
        )
        self.assertIn(
            "configs/terminal_base_beliefs_7376c5d_v1.json",
            terminal_cli.SOURCE_PATHS,
        )

        with patch.object(
            execution,
            "load_accepted_canonical_base_provider",
            side_effect=FileNotFoundError("canonical artifact absent"),
        ):
            with self.assertRaises(FileNotFoundError):
                terminal_cli.load_provider()

        submitter = (
            Path(__file__).resolve().parents[1]
            / "scripts"
            / "submit_hoffman2_terminal_validation.sh"
        ).read_text(encoding="utf-8")
        for legacy_name in (
            "APPROVED_MIGRATION_HASH",
            "MIGRATION_EXECUTION_APPROVAL",
            "PROVIDER_ACCEPTANCE",
        ):
            self.assertNotIn(legacy_name, submitter)


if __name__ == "__main__":
    unittest.main()
