"""Test purpose: validate terminal migration evidence construction and source-code provenance."""

from __future__ import annotations

import ast
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

from src.experiments import terminal_base_migration as migration
from src.experiments import terminal_migration_evidence as evidence


def _write_json(path: Path, value, *, canonical: bool = False) -> None:
    if canonical:
        encoded = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    else:
        encoded = json.dumps(value, indent=2, sort_keys=True, allow_nan=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(encoded + ("" if canonical else "\n"), encoding="utf-8")


def _qacct(job_id: str = "123", slots: str = "1", hostname: str = "n123") -> str:
    return (
        "==============================================================\n"
        f"jobnumber      {job_id}\n"
        f"hostname       {hostname}\n"
        "failed         0\n"
        "exit_status    0\n"
        f"slots          {slots}\n"
    )


class TerminalMigrationEvidenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.artifact = migration.build_synthetic_fixture_for_tests()
        cls.approval, cls.approval_hash = (
            migration.synthetic_execution_approval_for_tests(cls.artifact)
        )

    def _approved_probe_hashes(self, root: Path | None = None) -> dict[str, str]:
        source_root = migration.PROJECT_ROOT if root is None else root
        return {
            relative_path: evidence.sha256_file(source_root / relative_path)
            for relative_path in evidence.RUNTIME_PROBE_FILE_PATHS
        }

    def _collect_runtime_probe(
        self,
        *,
        run_dir: Path,
        qacct_path: Path,
        gate_path: Path,
        approved_hashes: dict[str, str] | None = None,
    ):
        return evidence.collect_runtime_probe_scheduler_evidence(
            run_dir=run_dir,
            qacct_path=qacct_path,
            submitted_job_id="777",
            approved_probe_file_hashes=(
                self._approved_probe_hashes()
                if approved_hashes is None
                else approved_hashes
            ),
            gate_path=gate_path,
        )

    def _migration_fixture(self, root: Path) -> tuple[Path, Path, dict]:
        run_dir = root / "run"
        run_dir.mkdir()
        approval_path = run_dir / "submission_evidence" / "execution_approval.json"
        _write_json(approval_path, migration._plain(self.approval), canonical=True)
        self.assertEqual(evidence.sha256_file(approval_path), self.approval_hash)
        submission = run_dir / "submission_evidence"
        (submission / "job_id.txt").write_text("123\n", encoding="utf-8")
        (submission / "qsub.stdout").write_text("123\n", encoding="utf-8")
        (submission / "qsub.exit_status").write_text("0\n", encoding="utf-8")
        (submission / "qsub.command").write_text("qsub -terse job.sh\n", encoding="utf-8")
        (submission / "approved_execution_approval_sha256.txt").write_text(
            self.approval_hash + "\n", encoding="utf-8"
        )

        artifact_path = run_dir / "terminal_base_beliefs.candidate.json"
        _write_json(artifact_path, migration.migration_to_dict(self.artifact))
        artifact_hash = evidence.sha256_file(artifact_path)
        (run_dir / "artifact_sha256.txt").write_text(
            artifact_hash + "\n", encoding="utf-8"
        )
        (run_dir / "semantic_output_hash.txt").write_text(
            self.artifact.output_hash + "\n", encoding="utf-8"
        )
        job_evidence = {
            "schema": evidence.MIGRATION_JOB_EVIDENCE_SCHEMA,
            "artifact_file": artifact_path.name,
            "artifact_sha256": artifact_hash,
            "semantic_output_hash": self.artifact.output_hash,
            "artifact_semantic_output_hash": self.artifact.output_hash,
            "execution_approval_file_hash": self.approval_hash,
            "python_executable": dict(self.artifact.runtime_identity)[
                "python_executable"
            ],
            "runtime_identity": [list(item) for item in self.artifact.runtime_identity],
            "dependency_identity": [
                list(item) for item in self.artifact.dependency_identity
            ],
            "migration_tool_hashes": [
                list(item) for item in self.artifact.migration_tool_hashes
            ],
            "hostname": "N123.hoffman2.idre.ucla.edu.",
            "canonical_hostname": "n123",
            "job_id": "123",
            "slots": 1,
            "start_utc": "2026-08-10T01:00:00Z",
            "end_utc": "2026-08-10T01:00:01Z",
        }
        _write_json(run_dir / "job_evidence.json", job_evidence)
        qacct_path = run_dir / "submission_evidence" / "qacct.raw"
        qacct_path.write_text(
            _qacct(hostname="n123.hoffman2.idre.ucla.edu"),
            encoding="utf-8",
        )
        return run_dir, qacct_path, job_evidence

    def test_valid_migration_evidence_is_strictly_cross_bound(self):
        with tempfile.TemporaryDirectory() as directory:
            run_dir, qacct_path, _ = self._migration_fixture(Path(directory))
            gate_path = run_dir / "scheduler_gate.json"
            gate = evidence.collect_migration_scheduler_evidence(
                run_dir=run_dir,
                qacct_path=qacct_path,
                submitted_job_id="123",
                approved_execution_approval_file_hash=self.approval_hash,
                gate_path=gate_path,
            )
            self.assertEqual(gate["job_id"], "123")
            self.assertEqual(gate["canonical_hostname"], "n123")
            self.assertTrue(gate["candidate_only_not_reviewer_approved"])

    def test_shared_schema_constants_match_migration_contract(self):
        self.assertEqual(
            evidence.AUTHORITATIVE_RUNTIME_KEYS,
            migration.AUTHORITATIVE_RUNTIME_KEYS,
        )
        self.assertEqual(
            evidence.AUTHORITATIVE_DEPENDENCY_KEYS,
            migration.AUTHORITATIVE_DEPENDENCY_KEYS,
        )
        self.assertEqual(evidence.MIGRATION_TOOL_PATHS, migration.MIGRATION_TOOL_PATHS)

    def test_mismatched_job_id_slots_and_host_are_permanently_rejected(self):
        mutations = (
            ("job_id", "999", "job evidence ID"),
            ("slots", 81, "slots must be"),
            ("hostname", "login4.hoffman2.idre.ucla.edu", "compute host"),
        )
        for field, value, message in mutations:
            with self.subTest(field=field), tempfile.TemporaryDirectory() as directory:
                run_dir, qacct_path, job_evidence = self._migration_fixture(
                    Path(directory)
                )
                job_evidence[field] = value
                if field == "hostname":
                    job_evidence["canonical_hostname"] = "login4"
                _write_json(run_dir / "job_evidence.json", job_evidence)
                with self.assertRaisesRegex(ValueError, message):
                    evidence.collect_migration_scheduler_evidence(
                        run_dir=run_dir,
                        qacct_path=qacct_path,
                        submitted_job_id="123",
                        approved_execution_approval_file_hash=self.approval_hash,
                        gate_path=run_dir / "scheduler_gate.json",
                    )

    def test_job_evidence_missing_extra_and_nested_identity_tamper_fail(self):
        mutations = ("missing", "extra", "tools", "runtime", "dependencies")
        for mutation in mutations:
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as directory:
                run_dir, qacct_path, job_evidence = self._migration_fixture(
                    Path(directory)
                )
                if mutation == "missing":
                    job_evidence.pop("end_utc")
                elif mutation == "extra":
                    job_evidence["unreviewed"] = True
                elif mutation == "tools":
                    job_evidence["migration_tool_hashes"][0][1] = "0" * 64
                elif mutation == "runtime":
                    job_evidence["runtime_identity"][0][1] = "forged"
                else:
                    job_evidence["dependency_identity"][0][1] = "forged"
                _write_json(run_dir / "job_evidence.json", job_evidence)
                with self.assertRaises((ValueError, RuntimeError)):
                    evidence.collect_migration_scheduler_evidence(
                        run_dir=run_dir,
                        qacct_path=qacct_path,
                        submitted_job_id="123",
                        approved_execution_approval_file_hash=self.approval_hash,
                        gate_path=run_dir / "scheduler_gate.json",
                    )

    def test_artifact_standalone_and_approval_hash_cross_checks_fail_closed(self):
        targets = (
            "artifact_sha256.txt",
            "semantic_output_hash.txt",
            "execution_approval.json",
            "approved_execution_approval_sha256.txt",
        )
        for target in targets:
            with self.subTest(target=target), tempfile.TemporaryDirectory() as directory:
                run_dir, qacct_path, _ = self._migration_fixture(Path(directory))
                path = (
                    run_dir / "submission_evidence" / target
                    if target in {
                        "execution_approval.json",
                        "approved_execution_approval_sha256.txt",
                    }
                    else run_dir / target
                )
                path.write_text("0" * 64 + "\n", encoding="utf-8")
                with self.assertRaises((ValueError, RuntimeError, json.JSONDecodeError)):
                    evidence.collect_migration_scheduler_evidence(
                        run_dir=run_dir,
                        qacct_path=qacct_path,
                        submitted_job_id="123",
                        approved_execution_approval_file_hash=self.approval_hash,
                        gate_path=run_dir / "scheduler_gate.json",
                    )

    def test_raw_qsub_job_identity_and_exit_status_are_cross_bound(self):
        mutations = (
            ("job_id.txt", "999\n", "submitted job ID"),
            ("qsub.stdout", "999\n", "submitted job ID"),
            ("qsub.exit_status", "1\n", "exit status"),
            ("qsub.command", "", "command evidence"),
        )
        for filename, value, message in mutations:
            with self.subTest(filename=filename), tempfile.TemporaryDirectory() as directory:
                run_dir, qacct_path, _ = self._migration_fixture(Path(directory))
                (run_dir / "submission_evidence" / filename).write_text(
                    value, encoding="utf-8"
                )
                with self.assertRaisesRegex(ValueError, message):
                    evidence.collect_migration_scheduler_evidence(
                        run_dir=run_dir,
                        qacct_path=qacct_path,
                        submitted_job_id="123",
                        approved_execution_approval_file_hash=self.approval_hash,
                        gate_path=run_dir / "scheduler_gate.json",
                    )

    def test_qacct_job_slots_and_hostname_must_match(self):
        qacct_mutations = (
            (_qacct(job_id="999"), "jobnumber"),
            (_qacct(slots="81"), "slots"),
            (_qacct(hostname="login4"), "compute host"),
        )
        for qacct_text, message in qacct_mutations:
            with self.subTest(message=message), tempfile.TemporaryDirectory() as directory:
                run_dir, qacct_path, _ = self._migration_fixture(Path(directory))
                qacct_path.write_text(qacct_text, encoding="utf-8")
                with self.assertRaisesRegex(ValueError, message):
                    evidence.collect_migration_scheduler_evidence(
                        run_dir=run_dir,
                        qacct_path=qacct_path,
                        submitted_job_id="123",
                        approved_execution_approval_file_hash=self.approval_hash,
                        gate_path=run_dir / "scheduler_gate.json",
                    )

    def _runtime_probe_fixture(self, root: Path) -> tuple[Path, Path, dict]:
        run_dir = root / "probe"
        run_dir.mkdir()
        profile = {
            "schema": evidence.RUNTIME_PROFILE_SCHEMA,
            "runtime_identity": [list(item) for item in self.artifact.runtime_identity],
            "dependency_identity": [
                list(item) for item in self.artifact.dependency_identity
            ],
        }
        profile_path = run_dir / "runtime_profile.candidate.json"
        _write_json(profile_path, profile)
        profile_hash = evidence.sha256_file(profile_path)
        (run_dir / "runtime_profile_sha256.txt").write_text(
            profile_hash + "\n", encoding="utf-8"
        )
        approved_hashes = self._approved_probe_hashes()
        submission = run_dir / "submission_evidence"
        for relative_path in evidence.RUNTIME_PROBE_FILE_PATHS:
            destination = submission / relative_path
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(migration.PROJECT_ROOT / relative_path, destination)
        evidence.write_runtime_probe_preflight(
            source_root=migration.PROJECT_ROOT,
            evidence_root=submission,
            approved_probe_file_hashes=approved_hashes,
            python_executable=dict(self.artifact.runtime_identity)[
                "python_executable"
            ],
            conda_env_path="/reviewed/conda/env",
            preflight_path=submission / "preflight.json",
        )
        job = submission / "scripts/hoffman2_terminal_runtime_profile_probe.job"
        job_hash = evidence.sha256_file(job)
        (submission / "job_id.txt").write_text("777\n", encoding="utf-8")
        (submission / "qsub.stdout").write_text("777\n", encoding="utf-8")
        (submission / "qsub.exit_status").write_text("0\n", encoding="utf-8")
        (submission / "qsub.command").write_text("qsub -terse probe.job\n", encoding="utf-8")
        (submission / "approved_probe_job_script_sha256.txt").write_text(
            job_hash + "\n", encoding="utf-8"
        )
        job_evidence = {
            "schema": evidence.RUNTIME_PROBE_JOB_EVIDENCE_SCHEMA,
            "profile_file": profile_path.name,
            "profile_sha256": profile_hash,
            "job_script_sha256": job_hash,
            "python_executable": dict(self.artifact.runtime_identity)[
                "python_executable"
            ],
            "hostname": "n777.hoffman2.idre.ucla.edu",
            "canonical_hostname": "n777",
            "job_id": "777",
            "slots": 1,
            "start_utc": "2026-08-10T02:00:00Z",
            "end_utc": "2026-08-10T02:00:01Z",
        }
        _write_json(run_dir / "runtime_probe_job_evidence.json", job_evidence)
        qacct_path = run_dir / "qacct.raw"
        qacct_path.write_text(_qacct(job_id="777", hostname="n777"), encoding="utf-8")
        return run_dir, qacct_path, job_evidence

    def test_valid_runtime_profile_probe_is_candidate_only_and_exact(self):
        with tempfile.TemporaryDirectory() as directory:
            run_dir, qacct_path, _ = self._runtime_probe_fixture(Path(directory))
            gate = evidence.collect_runtime_probe_scheduler_evidence(
                run_dir=run_dir,
                qacct_path=qacct_path,
                submitted_job_id="777",
                approved_probe_file_hashes=self._approved_probe_hashes(),
                gate_path=run_dir / "runtime_profile_scheduler_gate.json",
            )
            self.assertEqual(gate["canonical_hostname"], "n777")
            self.assertEqual(
                dict(gate["approved_file_hashes"]),
                self._approved_probe_hashes(),
            )
            self.assertEqual(
                gate["approved_file_hashes"], gate["evidence_copy_hashes"]
            )
            self.assertTrue(gate["candidate_only_not_reviewer_approved"])
            validated = evidence.validate_runtime_probe_scheduler_gate(
                run_dir=run_dir,
                qacct_path=qacct_path,
                submitted_job_id="777",
                approved_probe_file_hashes=self._approved_probe_hashes(),
                gate_path=run_dir / "runtime_profile_scheduler_gate.json",
            )
            self.assertEqual(validated, gate)

    def test_runtime_probe_collector_runs_from_exact_four_file_minimal_stage(self):
        project_root = migration.PROJECT_ROOT
        reviewed_paths = tuple(Path(path) for path in evidence.RUNTIME_PROBE_FILE_PATHS)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            stage = root / "minimal-stage"
            for relative_path in reviewed_paths:
                destination = stage / relative_path
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(project_root / relative_path, destination)

            staged_files = tuple(
                sorted(path.relative_to(stage) for path in stage.rglob("*") if path.is_file())
            )
            self.assertEqual(staged_files, tuple(sorted(reviewed_paths)))
            self.assertFalse(
                (stage / "src/experiments/terminal_base_migration.py").exists()
            )

            run_dir = root / "probe-run"
            run_dir.mkdir()
            submission = run_dir / "submission_evidence"
            for relative_path in reviewed_paths:
                destination = submission / relative_path
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(stage / relative_path, destination)
            approved_hashes = self._approved_probe_hashes(stage)
            approved_args = (
                "--approved-evidence-module-hash",
                approved_hashes["src/experiments/terminal_migration_evidence.py"],
                "--approved-collector-hash",
                approved_hashes["scripts/collect_hoffman2_runtime_profile_probe.py"],
                "--approved-probe-job-script-hash",
                approved_hashes["scripts/hoffman2_terminal_runtime_profile_probe.job"],
                "--approved-submitter-hash",
                approved_hashes[
                    "scripts/submit_hoffman2_terminal_runtime_profile_probe.sh"
                ],
            )
            collector = (
                submission / "scripts/collect_hoffman2_runtime_profile_probe.py"
            )
            guard = """
import builtins
import runpy
import sys

original_import = builtins.__import__
def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
    requested = (name, *(str(item) for item in (fromlist or ())))
    if any("terminal_base_migration" in item for item in requested):
        raise RuntimeError("runtime probe attempted to import migration")
    return original_import(name, globals, locals, fromlist, level)

builtins.__import__ = guarded_import
sys.argv = [sys.argv[1], *sys.argv[2:]]
runpy.run_path(sys.argv[0], run_name="__main__")
"""
            environment = dict(os.environ)
            environment.pop("PYTHONPATH", None)
            environment.pop("PYTHONHOME", None)
            environment["PYTHONNOUSERSITE"] = "1"

            def run_collector(*arguments: str) -> subprocess.CompletedProcess[str]:
                return subprocess.run(
                    (sys.executable, "-I", "-c", guard, str(collector), *arguments),
                    cwd=stage,
                    env=environment,
                    capture_output=True,
                    text=True,
                    check=False,
                )

            python_executable = "/reviewed/conda/env/bin/python"
            preflight_path = submission / "preflight.json"
            preflight_result = run_collector(
                "preflight",
                "--source-root",
                str(stage),
                "--evidence-root",
                str(submission),
                "--python-executable",
                python_executable,
                "--conda-env-path",
                "/reviewed/conda/env",
                "--preflight",
                str(preflight_path),
                *approved_args,
            )
            self.assertEqual(
                preflight_result.returncode,
                0,
                msg=(
                    f"stdout={preflight_result.stdout}\n"
                    f"stderr={preflight_result.stderr}"
                ),
            )
            runtime_values = {
                "byteorder": "little",
                "libc": "glibc 2.17",
                "platform_machine": "x86_64",
                "platform_release": "5.14.0",
                "platform_system": "Linux",
                "python_build": "main Aug 10 2026",
                "python_executable": python_executable,
                "python_implementation": "CPython",
                "python_version": "3.11.9",
            }
            profile = {
                "schema": evidence.RUNTIME_PROFILE_SCHEMA,
                "runtime_identity": [
                    [key, runtime_values[key]]
                    for key in evidence.AUTHORITATIVE_RUNTIME_KEYS
                ],
                "dependency_identity": [["numpy", "2.1.0"], ["scipy", "1.14.0"]],
            }
            profile_path = run_dir / "runtime_profile.candidate.json"
            _write_json(profile_path, profile)
            profile_hash = hashlib.sha256(profile_path.read_bytes()).hexdigest()
            (run_dir / "runtime_profile_sha256.txt").write_text(
                profile_hash + "\n", encoding="utf-8"
            )

            frozen_job = (
                submission / "scripts/hoffman2_terminal_runtime_profile_probe.job"
            )
            job_hash = hashlib.sha256(frozen_job.read_bytes()).hexdigest()
            (submission / "job_id.txt").write_text("777\n", encoding="utf-8")
            (submission / "qsub.stdout").write_text("777\n", encoding="utf-8")
            (submission / "qsub.exit_status").write_text("0\n", encoding="utf-8")
            (submission / "qsub.command").write_text(
                "qsub -terse probe.job\n", encoding="utf-8"
            )
            (submission / "approved_probe_job_script_sha256.txt").write_text(
                job_hash + "\n", encoding="utf-8"
            )
            _write_json(
                run_dir / "runtime_probe_job_evidence.json",
                {
                    "schema": evidence.RUNTIME_PROBE_JOB_EVIDENCE_SCHEMA,
                    "profile_file": profile_path.name,
                    "profile_sha256": profile_hash,
                    "job_script_sha256": job_hash,
                    "python_executable": python_executable,
                    "hostname": "n777.hoffman2.idre.ucla.edu",
                    "canonical_hostname": "n777",
                    "job_id": "777",
                    "slots": 1,
                    "start_utc": "2026-08-10T02:00:00Z",
                    "end_utc": "2026-08-10T02:00:01Z",
                },
            )
            qacct_path = run_dir / "qacct.raw"
            qacct_path.write_text(
                _qacct(job_id="777", hostname="n777"), encoding="utf-8"
            )
            gate_path = run_dir / "runtime_profile_scheduler_gate.json"
            completed = run_collector(
                "collect",
                "--run-dir",
                str(run_dir),
                "--qacct",
                str(qacct_path),
                "--submitted-job-id",
                "777",
                "--gate",
                str(gate_path),
                *approved_args,
            )
            self.assertEqual(
                completed.returncode,
                0,
                msg=f"stdout={completed.stdout}\nstderr={completed.stderr}",
            )
            self.assertTrue(gate_path.is_file())
            gate = json.loads(gate_path.read_text(encoding="utf-8"))
            self.assertEqual(gate["job_id"], "777")
            self.assertEqual(gate["slots"], 1)
            self.assertEqual(gate["canonical_hostname"], "n777")
            self.assertEqual(
                dict(gate["evidence_copy_hashes"]), approved_hashes
            )

    def test_migration_dependency_is_not_imported_at_module_scope(self):
        module_path = migration.PROJECT_ROOT / (
            "src/experiments/terminal_migration_evidence.py"
        )
        tree = ast.parse(module_path.read_text(encoding="utf-8"))
        top_level_imports = [
            node
            for node in tree.body
            if isinstance(node, (ast.Import, ast.ImportFrom))
        ]
        self.assertFalse(
            any("terminal_base_migration" in ast.unparse(node) for node in top_level_imports)
        )
        lazy_loader = next(
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "_load_terminal_base_migration"
        )
        self.assertIn("terminal_base_migration", ast.unparse(lazy_loader))

    def test_runtime_profile_rejects_extra_missing_reordered_or_absent_dependency(self):
        profile = {
            "schema": evidence.RUNTIME_PROFILE_SCHEMA,
            "runtime_identity": [list(item) for item in self.artifact.runtime_identity],
            "dependency_identity": [
                list(item) for item in self.artifact.dependency_identity
            ],
        }
        mutations = []
        extra = json.loads(json.dumps(profile))
        extra["unreviewed"] = True
        mutations.append(extra)
        missing = json.loads(json.dumps(profile))
        missing["runtime_identity"] = missing["runtime_identity"][:-1]
        mutations.append(missing)
        reordered = json.loads(json.dumps(profile))
        reordered["runtime_identity"] = list(reversed(reordered["runtime_identity"]))
        mutations.append(reordered)
        absent = json.loads(json.dumps(profile))
        absent["dependency_identity"][0][1] = "not-installed"
        mutations.append(absent)
        for index, mutation in enumerate(mutations):
            with self.subTest(index=index):
                with self.assertRaises(ValueError):
                    evidence.validate_runtime_profile(mutation)

    def test_runtime_probe_rejects_missing_or_altered_evidence_module_copy(self):
        for mutation in ("missing", "altered"):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as directory:
                run_dir, qacct_path, _ = self._runtime_probe_fixture(Path(directory))
                module_copy = (
                    run_dir
                    / "submission_evidence/src/experiments/terminal_migration_evidence.py"
                )
                if mutation == "missing":
                    module_copy.unlink()
                else:
                    module_copy.write_bytes(module_copy.read_bytes() + b"\n# altered\n")
                with self.assertRaises((FileNotFoundError, ValueError)):
                    self._collect_runtime_probe(
                        run_dir=run_dir,
                        qacct_path=qacct_path,
                        gate_path=run_dir / "runtime_profile_scheduler_gate.json",
                    )

    def test_self_rehashed_preflight_and_gate_cannot_replace_external_approval(self):
        relative_module = "src/experiments/terminal_migration_evidence.py"

        def replace_hash(pairs, replacement):
            return [
                [path, replacement if path == relative_module else item_hash]
                for path, item_hash in pairs
            ]

        with tempfile.TemporaryDirectory() as directory:
            run_dir, qacct_path, _ = self._runtime_probe_fixture(Path(directory))
            module_copy = run_dir / "submission_evidence" / relative_module
            module_copy.write_bytes(module_copy.read_bytes() + b"\n# forged\n")
            altered_hash = evidence.sha256_file(module_copy)
            preflight_path = run_dir / "submission_evidence/preflight.json"
            preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
            for key in (
                "approved_file_hashes",
                "source_file_hashes",
                "evidence_copy_hashes",
            ):
                preflight[key] = replace_hash(preflight[key], altered_hash)
            _write_json(preflight_path, preflight)
            with self.assertRaisesRegex(ValueError, "externally approved"):
                self._collect_runtime_probe(
                    run_dir=run_dir,
                    qacct_path=qacct_path,
                    gate_path=run_dir / "runtime_profile_scheduler_gate.json",
                )

        with tempfile.TemporaryDirectory() as directory:
            run_dir, qacct_path, _ = self._runtime_probe_fixture(Path(directory))
            gate_path = run_dir / "runtime_profile_scheduler_gate.json"
            self._collect_runtime_probe(
                run_dir=run_dir,
                qacct_path=qacct_path,
                gate_path=gate_path,
            )
            module_copy = run_dir / "submission_evidence" / relative_module
            module_copy.write_bytes(module_copy.read_bytes() + b"\n# forged\n")
            altered_hash = evidence.sha256_file(module_copy)
            preflight_path = run_dir / "submission_evidence/preflight.json"
            preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
            for key in (
                "approved_file_hashes",
                "source_file_hashes",
                "evidence_copy_hashes",
            ):
                preflight[key] = replace_hash(preflight[key], altered_hash)
            _write_json(preflight_path, preflight)

            gate = json.loads(gate_path.read_text(encoding="utf-8"))
            for key in (
                "approved_file_hashes",
                "source_file_hashes",
                "evidence_copy_hashes",
            ):
                gate[key] = replace_hash(gate[key], altered_hash)
            gate["preflight_sha256"] = evidence.sha256_file(preflight_path)
            _write_json(gate_path, gate)
            with self.assertRaisesRegex(ValueError, "externally approved"):
                evidence.validate_runtime_probe_scheduler_gate(
                    run_dir=run_dir,
                    qacct_path=qacct_path,
                    submitted_job_id="777",
                    approved_probe_file_hashes=self._approved_probe_hashes(),
                    gate_path=gate_path,
                )

    def test_runtime_probe_adversarial_scheduler_and_hash_mismatches_fail(self):
        mutations = (
            ("job_id", "999", "job ID"),
            ("slots", 2, "slots must be"),
            ("hostname", "n999", "hostname differs"),
            ("profile_sha256", "0" * 64, "hashes do not agree"),
        )
        for field, value, message in mutations:
            with self.subTest(field=field), tempfile.TemporaryDirectory() as directory:
                run_dir, qacct_path, job_evidence = self._runtime_probe_fixture(
                    Path(directory)
                )
                job_evidence[field] = value
                if field == "hostname":
                    job_evidence["canonical_hostname"] = "n999"
                _write_json(run_dir / "runtime_probe_job_evidence.json", job_evidence)
                with self.assertRaisesRegex(ValueError, message):
                    self._collect_runtime_probe(
                        run_dir=run_dir,
                        qacct_path=qacct_path,
                        gate_path=run_dir / "runtime_profile_scheduler_gate.json",
                    )

        with tempfile.TemporaryDirectory() as directory:
            run_dir, qacct_path, _ = self._runtime_probe_fixture(Path(directory))
            approved = self._approved_probe_hashes()
            approved["scripts/hoffman2_terminal_runtime_profile_probe.job"] = "0" * 64
            with self.assertRaises(ValueError):
                self._collect_runtime_probe(
                    run_dir=run_dir,
                    qacct_path=qacct_path,
                    approved_hashes=approved,
                    gate_path=run_dir / "runtime_profile_scheduler_gate.json",
                )

    def test_shell_paths_are_one_slot_nonarray_and_probe_never_runs_migration(self):
        scripts = (
            "hoffman2_terminal_base_migration.job",
            "submit_hoffman2_terminal_base_migration.sh",
            "hoffman2_terminal_runtime_profile_probe.job",
            "submit_hoffman2_terminal_runtime_profile_probe.sh",
        )
        for name in scripts:
            path = migration.PROJECT_ROOT / "scripts" / name
            subprocess.run(("bash", "-n", str(path)), check=True)
        migration_submitter = (
            migration.PROJECT_ROOT
            / "scripts/submit_hoffman2_terminal_base_migration.sh"
        ).read_text(encoding="utf-8")
        probe_job = (
            migration.PROJECT_ROOT
            / "scripts/hoffman2_terminal_runtime_profile_probe.job"
        ).read_text(encoding="utf-8")
        probe_submitter = (
            migration.PROJECT_ROOT
            / "scripts/submit_hoffman2_terminal_runtime_profile_probe.sh"
        ).read_text(encoding="utf-8")
        self.assertIn("collect_terminal_base_migration_evidence.py", migration_submitter)
        self.assertIn("-pe shared 1", probe_submitter)
        self.assertIn("-r n", probe_submitter)
        self.assertNotIn("-t 1-", probe_submitter)
        self.assertIn("APPROVED_EVIDENCE_MODULE_HASH", probe_submitter)
        self.assertIn("APPROVED_PROBE_COLLECTOR_HASH", probe_submitter)
        self.assertIn("APPROVED_PROBE_SUBMITTER_HASH", probe_submitter)
        self.assertIn('"${EVIDENCE_DIR}/src/experiments"', probe_submitter)
        self.assertIn('python3 "${FROZEN_COLLECTOR}" collect', probe_submitter)
        self.assertNotIn("export_terminal_base_migration", probe_job)
        self.assertNotIn("terminal_base_beliefs", probe_job)


if __name__ == "__main__":
    unittest.main()
