"""Test purpose: validate canonical terminal-base migration and immutable provenance."""

from __future__ import annotations

from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

from src.experiments import terminal_base_migration as migration
from src.experiments.terminal_validation_suite import (
    FROZEN_LEGACY_BELIEF_HASH_OVERRIDES,
)
from src.mdp.finite_support import FiniteSupportBeliefState


def _rehash_artifact(
    artifact: migration.BaseBeliefMigration,
) -> migration.BaseBeliefMigration:
    cleared = replace(artifact, output_hash="")
    return replace(cleared, output_hash=migration.migration_output_hash(cleared))


def _rehash_record(
    record: migration.MigrationRecord,
) -> migration.MigrationRecord:
    cleared = replace(record, record_hash="")
    return replace(cleared, record_hash=migration.migration_record_hash(cleared))


def _replace_record(
    artifact: migration.BaseBeliefMigration,
    index: int,
    record: migration.MigrationRecord,
) -> migration.BaseBeliefMigration:
    records = list(artifact.records)
    records[index] = record
    updated = replace(
        artifact,
        records=tuple(records),
        records_hash=migration.canonical_hash(
            tuple(item.record_hash for item in records)
        ),
        output_hash="",
    )
    return _rehash_artifact(updated)


class TerminalBaseMigrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.artifact = migration.build_synthetic_fixture_for_tests()
        (
            cls.execution_approval,
            cls.execution_approval_hash,
        ) = migration.synthetic_execution_approval_for_tests(cls.artifact)

    def _validate(self, artifact=None, *, approved_hash=None) -> None:
        selected = artifact or self.artifact
        migration.validate_migration(
            selected,
            approved_output_hash=approved_hash or selected.output_hash,
            allow_synthetic_fixture=True,
            execution_approval=self.execution_approval,
            approved_execution_approval_file_hash=self.execution_approval_hash,
        )

    def test_fixture_is_permanently_non_authoritative_and_has_exact_case_order(self):
        artifact = self.artifact
        self.assertEqual(artifact.migration_status, migration.SYNTHETIC_STATUS)
        self.assertFalse(artifact.authoritative)
        self.assertEqual([case.case_id for case in artifact.original_case_descriptors], list(range(90)))
        self.assertEqual(
            tuple(record.case for record in artifact.records),
            artifact.original_case_descriptors,
        )
        self.assertEqual(len(artifact.records), 90)
        self.assertEqual(
            set(FROZEN_LEGACY_BELIEF_HASH_OVERRIDES),
            {24, 29, 83, 88},
        )

        mismatch_ids = {
            record.case.case_id
            for record in artifact.records
            if not record.original_belief_hash_matches_payload
        }
        self.assertLessEqual(
            mismatch_ids,
            set(FROZEN_LEGACY_BELIEF_HASH_OVERRIDES),
        )
        for record in artifact.records:
            _, belief = migration.reconstruct_exact_belief(record.belief)
            self.assertEqual(
                record.original_belief_hash_matches_payload,
                migration.legacy_belief_hash(belief) == record.case.belief_hash,
            )

    def test_every_base_belief_reconstructs_exact_binary64_payload(self):
        for record in self.artifact.records:
            prior, belief = migration.reconstruct_exact_belief(record.belief)
            support = record.belief.support
            self.assertEqual(
                tuple(state.total_need.hex() for state in prior.states),
                tuple(atom.total_need_hex for atom in support.states),
            )
            self.assertEqual(
                tuple(state.gap_fraction.hex() for state in prior.states),
                tuple(atom.gap_fraction_hex for atom in support.states),
            )
            self.assertEqual(
                tuple(state.orientation for state in prior.states),
                tuple(atom.orientation for atom in support.states),
            )
            self.assertEqual(
                tuple(weight.hex() for weight in prior.weights),
                support.prior_weights_hex,
            )
            self.assertEqual(
                tuple(weight.hex() for weight in belief.weights),
                record.belief.posterior_weights_hex,
            )
            self.assertEqual(
                belief.deliberation_time.hex(),
                record.belief.deliberation_time_hex,
            )
            self.assertEqual(belief.history, [])
            self.assertEqual(
                migration.make_belief_payload(
                    prior,
                    belief,
                    record.belief.original_belief_hash,
                ),
                record.belief,
            )

    def test_nonempty_history_round_trips_using_float_hex(self):
        prior, original = migration.reconstruct_exact_belief(
            self.artifact.records[0].belief
        )
        belief = FiniteSupportBeliefState(
            original.states,
            original.weights,
            deliberation_time=3.0,
            history=[
                {"action": 1.0, "observation": 12.25, "cost": 1.0},
                {"action": 2.0, "observation": -0.5, "cost": 2.0},
            ],
        )
        belief.weights = original.weights
        payload = migration.make_belief_payload(prior, belief, "test-only")
        self.assertEqual(payload.history[0].action_hex, 1.0.hex())
        self.assertEqual(payload.history[1].action_hex, 2.0.hex())
        _, reconstructed = migration.reconstruct_exact_belief(payload)
        self.assertEqual(reconstructed.deliberation_time.hex(), 3.0.hex())
        self.assertEqual(reconstructed.history, belief.history)

    def test_importer_requires_explicit_synthetic_permission_and_approved_hash(self):
        with self.assertRaisesRegex(RuntimeError, "Reviewer-approved"):
            migration.validate_migration(
                self.artifact,
                approved_output_hash=None,
                allow_synthetic_fixture=True,
                execution_approval=self.execution_approval,
                approved_execution_approval_file_hash=self.execution_approval_hash,
            )
        with self.assertRaisesRegex(RuntimeError, "not approved"):
            migration.validate_migration(
                self.artifact,
                approved_output_hash="0" * 64,
                allow_synthetic_fixture=True,
                execution_approval=self.execution_approval,
                approved_execution_approval_file_hash=self.execution_approval_hash,
            )
        with self.assertRaisesRegex(RuntimeError, "not authoritative"):
            migration.validate_migration(
                self.artifact,
                approved_output_hash=self.artifact.output_hash,
                execution_approval=self.execution_approval,
                approved_execution_approval_file_hash=self.execution_approval_hash,
            )
        self._validate()

    def test_file_import_round_trip_and_absent_file_rejection(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "synthetic.json"
            path.write_text(
                json.dumps(
                    migration.migration_to_dict(self.artifact),
                    sort_keys=True,
                    allow_nan=False,
                ),
                encoding="utf-8",
            )
            loaded = migration.load_migration(
                path,
                approved_output_hash=self.artifact.output_hash,
                allow_synthetic_fixture=True,
                execution_approval=self.execution_approval,
                approved_execution_approval_file_hash=self.execution_approval_hash,
            )
            self.assertEqual(loaded, self.artifact)
            with self.assertRaises(FileNotFoundError):
                migration.load_migration(
                    Path(directory) / "absent.json",
                    approved_output_hash=self.artifact.output_hash,
                    allow_synthetic_fixture=True,
                    execution_approval=self.execution_approval,
                    approved_execution_approval_file_hash=self.execution_approval_hash,
                )

    def test_duplicate_json_keys_and_unknown_fields_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            duplicate = Path(directory) / "duplicate.json"
            duplicate.write_text('{"schema":"x","schema":"y"}', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "duplicate JSON key"):
                migration.load_migration(
                    duplicate,
                    approved_output_hash=self.artifact.output_hash,
                    allow_synthetic_fixture=True,
                    execution_approval=self.execution_approval,
                    approved_execution_approval_file_hash=self.execution_approval_hash,
                )
        raw = migration.migration_to_dict(self.artifact)
        raw["unexpected"] = "field"
        with self.assertRaisesRegex(ValueError, "fields mismatch"):
            migration.parse_migration(raw)

    def test_json_type_changes_are_not_silently_normalized(self):
        case_id_string = migration.migration_to_dict(self.artifact)
        case_id_string["original_case_descriptors"][0]["case_id"] = "0"
        with self.assertRaisesRegex(ValueError, "case_id must be an integer"):
            migration.parse_migration(case_id_string)

        numeric_hex = migration.migration_to_dict(self.artifact)
        numeric_hex["records"][0]["belief"]["posterior_weights_hex"][0] = 1.0
        with self.assertRaisesRegex(ValueError, "posterior weight must be a string"):
            migration.parse_migration(numeric_hex)

    def test_source_provenance_tamper_fails_even_after_internal_rehash(self):
        source_hashes = list(self.artifact.source_hashes)
        source_hashes[0] = (source_hashes[0][0], "0" * 64)
        forged = replace(
            self.artifact,
            source_hashes=tuple(source_hashes),
            source_hashes_hash=migration.canonical_hash(tuple(source_hashes)),
            output_hash="",
        )
        forged = _rehash_artifact(forged)
        with self.assertRaisesRegex(RuntimeError, "differ from the original manifest"):
            self._validate(forged, approved_hash=forged.output_hash)

    def test_every_fixed_provenance_identity_is_enforced(self):
        mutations = (
            ("source_commit", "forged", "synthetic fixture provenance"),
            ("source_tree_hash", "forged", "synthetic fixture provenance"),
            ("original_manifest_path", "forged.json", "original provenance"),
            ("original_manifest_file_hash", "0" * 64, "original provenance"),
            ("original_manifest_hash", "0" * 64, "original provenance"),
            ("original_spec_hash", "0" * 64, "original provenance"),
            ("original_case_hash", "0" * 64, "original provenance"),
        )
        for field_name, value, message in mutations:
            forged = _rehash_artifact(
                replace(self.artifact, **{field_name: value, "output_hash": ""})
            )
            with self.subTest(field=field_name):
                with self.assertRaisesRegex(RuntimeError, message):
                    self._validate(forged, approved_hash=forged.output_hash)

    def test_importer_refuses_absent_or_wrong_original_manifest_provenance(self):
        with tempfile.TemporaryDirectory(dir=migration.PROJECT_ROOT) as directory:
            absent = Path(directory) / "absent-manifest.json"
            with mock.patch.object(
                migration,
                "DEFAULT_ORIGINAL_MANIFEST_PATH",
                absent,
            ):
                with self.assertRaises(FileNotFoundError):
                    self._validate(
                        self.artifact,
                        approved_hash=self.artifact.output_hash,
                    )

            wrong = Path(directory) / "wrong-manifest.json"
            wrong.write_text("{}", encoding="utf-8")
            with mock.patch.object(
                migration,
                "DEFAULT_ORIGINAL_MANIFEST_PATH",
                wrong,
            ):
                with self.assertRaisesRegex(RuntimeError, "file hash mismatch"):
                    self._validate(
                        self.artifact,
                        approved_hash=self.artifact.output_hash,
                    )

    def test_case_deletion_reorder_and_descriptor_tamper_fail_closed(self):
        deleted = replace(
            self.artifact,
            original_case_descriptors=self.artifact.original_case_descriptors[:-1],
            records=self.artifact.records[:-1],
            records_hash=migration.canonical_hash(
                tuple(record.record_hash for record in self.artifact.records[:-1])
            ),
            output_hash="",
        )
        deleted = _rehash_artifact(deleted)
        with self.assertRaisesRegex(RuntimeError, "case descriptors"):
            self._validate(deleted, approved_hash=deleted.output_hash)

        reordered = replace(
            self.artifact,
            original_case_descriptors=tuple(
                reversed(self.artifact.original_case_descriptors)
            ),
            records=tuple(reversed(self.artifact.records)),
            records_hash=migration.canonical_hash(
                tuple(record.record_hash for record in reversed(self.artifact.records))
            ),
            output_hash="",
        )
        reordered = _rehash_artifact(reordered)
        with self.assertRaisesRegex(RuntimeError, "case descriptors"):
            self._validate(reordered, approved_hash=reordered.output_hash)

        first = replace(self.artifact.original_case_descriptors[0], environment="forged")
        descriptors = (first,) + self.artifact.original_case_descriptors[1:]
        forged = _rehash_artifact(
            replace(
                self.artifact,
                original_case_descriptors=descriptors,
                output_hash="",
            )
        )
        with self.assertRaisesRegex(RuntimeError, "case descriptors"):
            self._validate(forged, approved_hash=forged.output_hash)

    def test_runtime_dependency_and_tool_hash_tamper_is_rejected(self):
        cases = (
            replace(self.artifact, runtime_identity_hash="0" * 64),
            replace(self.artifact, dependency_identity_hash="0" * 64),
            replace(self.artifact, migration_tool_hashes_hash="0" * 64),
        )
        expected = ("runtime", "dependency", "tool-source")
        for artifact, message in zip(cases, expected):
            forged = _rehash_artifact(replace(artifact, output_hash=""))
            with self.subTest(message=message):
                with self.assertRaisesRegex(RuntimeError, message):
                    self._validate(forged, approved_hash=forged.output_hash)

    def test_self_rehashed_execution_identity_tamper_fails_against_external_profile(self):
        identity_fields = (
            ("migration_tool_hashes", "migration_tool_hashes_hash", "tool identity"),
            ("runtime_identity", "runtime_identity_hash", "runtime identity"),
            ("dependency_identity", "dependency_identity_hash", "dependency identity"),
        )
        for field_name, aggregate_name, message in identity_fields:
            original = getattr(self.artifact, field_name)
            for index, (key, _) in enumerate(original):
                pairs = list(original)
                replacement = "0" * 64 if "hashes" in field_name else "forged"
                pairs[index] = (key, replacement)
                changed = tuple(pairs)
                forged = _rehash_artifact(
                    replace(
                        self.artifact,
                        **{
                            field_name: changed,
                            aggregate_name: migration.canonical_hash(changed),
                            "output_hash": "",
                        },
                    )
                )
                with self.subTest(field=field_name, key=key):
                    with self.assertRaisesRegex(RuntimeError, message):
                        self._validate(forged, approved_hash=forged.output_hash)

        forged = _rehash_artifact(
            replace(
                self.artifact,
                execution_approval_file_hash="0" * 64,
                output_hash="",
            )
        )
        with self.assertRaisesRegex(RuntimeError, "approval hash"):
            self._validate(forged, approved_hash=forged.output_hash)

    def test_cross_gap_support_swap_fails_after_complete_internal_rehash(self):
        source = self.artifact.records[30].belief.support
        target = self.artifact.records[0]
        self.assertNotEqual(target.belief.support.support_hash, source.support_hash)
        belief = replace(target.belief, support=source, payload_hash="")
        belief = replace(belief, payload_hash=migration.belief_payload_hash(belief))
        changed_record = _rehash_record(
            replace(target, belief=belief, record_hash="")
        )
        forged = _replace_record(self.artifact, 0, changed_record)
        with self.assertRaisesRegex(RuntimeError, "support differs from frozen environment"):
            self._validate(forged, approved_hash=forged.output_hash)

    def test_execution_approval_requires_external_file_hash(self):
        raw = migration._plain(self.execution_approval)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "approval.json"
            path.write_text(
                json.dumps(raw, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            approved_hash = hashlib.sha256(path.read_bytes()).hexdigest()
            loaded = migration.load_execution_approval(
                path, approved_file_hash=approved_hash
            )
            self.assertEqual(loaded, self.execution_approval)
            with self.assertRaisesRegex(RuntimeError, "not approved"):
                migration.load_execution_approval(
                    path, approved_file_hash="0" * 64
                )

    def test_dirty_tracked_dependency_and_package_init_fail_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = (
                root / "src" / "experiments" / "helper_dependency.py",
                root / "src" / "experiments" / "__init__.py",
            )
            for path in paths:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("ORIGINAL = True\n", encoding="utf-8")
            commands = (
                ("git", "init", "-q"),
                ("git", "config", "user.email", "migration-test@example.invalid"),
                ("git", "config", "user.name", "Migration Test"),
                ("git", "add", "."),
                ("git", "commit", "-qm", "fixture"),
            )
            for command in commands:
                subprocess.run(command, cwd=root, check=True)
            migration._require_clean_tracked_worktree(root)
            for path in paths:
                with self.subTest(path=path.name):
                    path.write_text("DIRTY = True\n", encoding="utf-8")
                    with self.assertRaisesRegex(RuntimeError, "tracked worktree"):
                        migration._require_clean_tracked_worktree(root)
                    path.write_text("ORIGINAL = True\n", encoding="utf-8")
            paths[0].write_text("STAGED = True\n", encoding="utf-8")
            subprocess.run(("git", "add", str(paths[0])), cwd=root, check=True)
            with self.assertRaisesRegex(RuntimeError, "tracked"):
                migration._require_clean_tracked_worktree(root)

    def test_frozen_untracked_allowlist_rejects_extra_or_changed_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            subprocess.run(("git", "init", "-q"), cwd=root, check=True)
            allowed_path = root / "reviewed_tool.py"
            allowed_path.write_text("reviewed\n", encoding="utf-8")
            allowed = {"reviewed_tool.py": hashlib.sha256(allowed_path.read_bytes()).hexdigest()}
            migration._require_frozen_untracked_inputs(allowed, root)
            extra = root / "unreviewed.py"
            extra.write_text("extra\n", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "inventory mismatch"):
                migration._require_frozen_untracked_inputs(allowed, root)
            extra.unlink()
            allowed_path.write_text("changed\n", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "input mismatch"):
                migration._require_frozen_untracked_inputs(allowed, root)

    def test_exporter_never_overwrites_existing_output(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "candidate.json"
            sentinel = b"do-not-overwrite\n"
            output.write_bytes(sentinel)
            with self.assertRaises(FileExistsError):
                migration.export_authoritative_base_migration(
                    output,
                    Path(directory) / "controlled-manifest.json",
                    execution_approval_path=Path(directory) / "approval.json",
                    approved_execution_approval_file_hash="0" * 64,
                    scheduled_job_script_path=Path(directory) / "job.sh",
                )
            self.assertEqual(output.read_bytes(), sentinel)

    def test_scheduled_migration_scripts_encode_one_slot_fail_closed_procedure(self):
        job = migration.PROJECT_ROOT / "scripts/hoffman2_terminal_base_migration.job"
        submitter = (
            migration.PROJECT_ROOT
            / "scripts/submit_hoffman2_terminal_base_migration.sh"
        )
        for path in (job, submitter):
            subprocess.run(("bash", "-n", str(path)), check=True)
        job_text = job.read_text(encoding="utf-8")
        submitter_text = submitter.read_text(encoding="utf-8")
        collector_text = (
            migration.PROJECT_ROOT
            / "src/experiments/terminal_migration_evidence.py"
        ).read_text(encoding="utf-8")
        self.assertIn('[[ "${NSLOTS:-}" != "1" ]]', job_text)
        self.assertIn('"${HOST_FQDN%%.*}" == login*', job_text)
        self.assertIn("--execution-approval-file-hash", job_text)
        self.assertIn('mkdir "${RUN_DIR}"', submitter_text)
        self.assertIn("-pe shared 1", submitter_text)
        self.assertIn("-r n", submitter_text)
        self.assertNotIn("-t 1-", submitter_text)
        self.assertNotIn("git -C", submitter_text)
        self.assertIn("git_in_directory", submitter_text)
        for gate in ('"failed": "0"', '"exit_status": "0"', '"slots": "1"'):
            self.assertIn(gate, collector_text)

    def test_historical_submitter_rejects_tools_that_differ_from_approval(self):
        project_root = migration.PROJECT_ROOT
        submitter = project_root / "scripts/submit_hoffman2_terminal_base_migration.sh"
        approval = (
            project_root
            / "configs/reference/terminal_base_migration_execution_approval_v1.json"
        )
        manifest = migration.DEFAULT_ORIGINAL_MANIFEST_PATH
        real_git = shutil.which("git")
        real_shasum = shutil.which("shasum")
        self.assertIsNotNone(real_git)
        self.assertIsNotNone(real_shasum)
        self.assertTrue(approval.is_file())
        self.assertTrue(manifest.is_file())

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fake_bin = root / "bin"
            fake_bin.mkdir()
            git_log = root / "git.log"
            qsub_log = root / "qsub.log"
            git_wrapper = fake_bin / "git"
            git_wrapper.write_text(
                "#!/bin/sh\n"
                "printf '%s\\n' \"$*\" >> \"$GIT_WRAPPER_LOG\"\n"
                "for argument in \"$@\"; do\n"
                "  if [ \"$argument\" = '-C' ] || [ \"$argument\" = 'worktree' ]; then\n"
                "    echo 'unsupported modern Git operation' >&2\n"
                "    exit 97\n"
                "  fi\n"
                "done\n"
                "exec \"$REAL_GIT\" \"$@\"\n",
                encoding="utf-8",
            )
            git_wrapper.chmod(0o755)
            sha256sum = fake_bin / "sha256sum"
            sha256sum.write_text(
                "#!/bin/sh\n"
                "exec \"$REAL_SHASUM\" -a 256 \"$@\"\n",
                encoding="utf-8",
            )
            sha256sum.chmod(0o755)
            qsub = fake_bin / "qsub"
            qsub.write_text(
                "#!/bin/sh\n"
                "printf '%s\\n' \"$@\" > \"$QSUB_LOG\"\n"
                "printf '888888\\n'\n",
                encoding="utf-8",
            )
            qsub.chmod(0o755)

            runs_root = root / "runs"
            run_id = "old_git_compatible_submit"
            run_dir = runs_root / run_id
            stage_root = run_dir / "authoritative_checkout"
            environment = dict(os.environ)
            environment.update(
                {
                    "PATH": f"{fake_bin}:{environment['PATH']}",
                    "REAL_GIT": str(real_git),
                    "REAL_SHASUM": str(real_shasum),
                    "GIT_WRAPPER_LOG": str(git_log),
                    "QSUB_LOG": str(qsub_log),
                    "RUN_ID": run_id,
                    "SOURCE_ROOT": str(project_root),
                    "RUNS_ROOT": str(runs_root),
                    "APPROVAL_PATH": str(approval),
                    "APPROVAL_HASH": hashlib.sha256(approval.read_bytes()).hexdigest(),
                    "PYTHON_BIN": sys.executable,
                    "CONDA_SH": "/mock/conda.sh",
                    "CONDA_ENV_PATH": "/mock/conda/env",
                    "ORIGINAL_MANIFEST": str(manifest),
                    "QSUB_BIN": "qsub",
                }
            )
            completed = subprocess.run(
                ("bash", str(submitter), "submit"),
                cwd=project_root,
                env=environment,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertNotEqual(completed.returncode, 0)
            self.assertIn(
                "staged migration tool hashes differ from Reviewer approval",
                completed.stderr,
            )
            self.assertFalse(qsub_log.exists())
            git_calls = git_log.read_text(encoding="utf-8")
            self.assertNotIn(" -C ", f" {git_calls} ")
            self.assertNotIn("worktree", git_calls)
            self.assertIn("clone --no-checkout", git_calls)
            self.assertIn(
                f"checkout --detach {migration.AUTHORITATIVE_COMMIT}", git_calls
            )

    def test_payload_tamper_is_rejected_by_approved_output_trust_root(self):
        record = self.artifact.records[0]
        weights = list(record.belief.posterior_weights_hex)
        weights[0] = (float.fromhex(weights[0]) * 0.5).hex()
        belief = replace(
            record.belief,
            posterior_weights_hex=tuple(weights),
            payload_hash="",
        )
        belief = replace(belief, payload_hash=migration.belief_payload_hash(belief))
        changed_record = _rehash_record(replace(record, belief=belief, record_hash=""))
        forged = _replace_record(self.artifact, 0, changed_record)
        with self.assertRaisesRegex(RuntimeError, "not approved"):
            self._validate(forged, approved_hash=self.artifact.output_hash)

    def test_malformed_float_hex_and_support_hash_are_rejected(self):
        record = self.artifact.records[0]
        weights = list(record.belief.posterior_weights_hex)
        weights[0] = "1.0"
        belief = replace(
            record.belief,
            posterior_weights_hex=tuple(weights),
            payload_hash="",
        )
        belief = replace(belief, payload_hash=migration.belief_payload_hash(belief))
        changed_record = _rehash_record(replace(record, belief=belief, record_hash=""))
        forged = _replace_record(self.artifact, 0, changed_record)
        with self.assertRaisesRegex(ValueError, "canonical finite float-hex"):
            self._validate(forged, approved_hash=forged.output_hash)

        support = replace(record.belief.support, support_hash="0" * 64, payload_hash="")
        support = replace(support, payload_hash=migration.support_payload_hash(support))
        belief = replace(record.belief, support=support, payload_hash="")
        belief = replace(belief, payload_hash=migration.belief_payload_hash(belief))
        changed_record = _rehash_record(replace(record, belief=belief, record_hash=""))
        forged = _replace_record(self.artifact, 0, changed_record)
        with self.assertRaisesRegex(RuntimeError, "support differs from frozen environment"):
            self._validate(forged, approved_hash=forged.output_hash)

    def test_authoritative_context_rejects_wrong_commit_tree_and_source(self):
        manifest = migration._load_original_manifest(
            migration.DEFAULT_ORIGINAL_MANIFEST_PATH
        )
        with mock.patch.object(migration, "_git_output", return_value="wrong"):
            with self.assertRaisesRegex(RuntimeError, "authoritative commit"):
                migration.validate_authoritative_export_context(
                    manifest, self.execution_approval
                )

        def git_tree(*args):
            return (
                migration.AUTHORITATIVE_COMMIT
                if args == ("rev-parse", "HEAD")
                else "wrong-tree"
            )

        with mock.patch.object(migration, "_git_output", side_effect=git_tree):
            with self.assertRaisesRegex(RuntimeError, "tree mismatch"):
                migration.validate_authoritative_export_context(
                    manifest, self.execution_approval
                )

        def valid_git(*args):
            return (
                migration.AUTHORITATIVE_COMMIT
                if args == ("rev-parse", "HEAD")
                else migration.AUTHORITATIVE_TREE
            )

        with mock.patch.object(
            migration, "_git_output", side_effect=valid_git
        ), mock.patch.object(
            migration, "_require_clean_tracked_worktree"
        ), mock.patch.object(
            migration, "_require_frozen_untracked_inputs"
        ), mock.patch.object(migration, "_file_hash", return_value="0" * 64):
            with self.assertRaisesRegex(RuntimeError, "authoritative source mismatch"):
                migration.validate_authoritative_export_context(
                    manifest, self.execution_approval
                )


if __name__ == "__main__":
    unittest.main()
