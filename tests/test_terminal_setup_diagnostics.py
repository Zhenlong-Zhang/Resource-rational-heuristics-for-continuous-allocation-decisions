"""Test purpose: validate pre-execution terminal setup and scheduler diagnostics."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from src.experiments import terminal_execution as execution
from src.experiments import terminal_setup_diagnostics as setup


HASH = "1" * 64
EMPTY_QSTAT = (
    "<?xml version='1.0'?><job_info><queue_info></queue_info>"
    "<job_info></job_info></job_info>\n"
)


class TerminalSetupDiagnosticTests(unittest.TestCase):
    def _ceiling_evidence(self, root: Path):
        files = {
            "myresources.raw": (
                "User zzl is in the following resource group(s): campus\n"
                "jobs are limited to 24 hours run-time\n"
            ),
            "qconf_sconf_global.raw": "max_aj_instances 100\nmax_aj_tasks 200000\n",
            "qconf_sq_campus2.raw": "qname campus2.q\nslots 128\nh_rt 24:00:00\n",
            "qhost.raw": "HOSTNAME NCPU MEMTOT\nn123 32 128G\n",
            "qquota.raw": "resource quota rule limit filter\n",
            "qstat_g_c.raw": (
                "CLUSTER QUEUE CQLOAD USED RES AVAIL TOTAL aoACDS cdsuE\n"
                "campus2.q 0.4 1 0 127 128 0 0\n"
            ),
            "qstat_user.xml": EMPTY_QSTAT,
        }
        for name, text in files.items():
            (root / name).write_text(text, encoding="utf-8")

    def test_compute_ceiling_is_derived_from_and_bound_to_raw_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._ceiling_evidence(root)
            report = setup.create_compute_ceiling_report(
                root,
                queue="campus2.q",
                captured_at_utc="2026-08-12T00:00:00+00:00",
            )
            self.assertEqual(report["max_throttle"], 100)
            self.assertEqual(report["max_walltime_seconds"], 86400)
            self.assertEqual(
                tuple(name for name, _hash in report["raw_evidence_hashes"]),
                setup.COMPUTE_CEILING_EVIDENCE_FILES,
            )
            (root / "qstat_g_c.raw").write_text("malformed\n", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "queue"):
                setup.create_compute_ceiling_report(root, queue="campus2.q")

    def test_output_audit_calls_current_fragment_validator_signature(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for replicate in ("a", "b"):
                (root / f"plan_{replicate}").mkdir()
                (root / f"profiles_{replicate}").mkdir()
                for task_id in range(1, 17):
                    (root / f"plan_{replicate}" / f"fragment_{task_id:03d}.json").touch()
                    (root / f"profiles_{replicate}" / f"fragment_{task_id:03d}.json").touch()
            (root / "profiles_merge").mkdir()
            (root / "profiles_merge/merge.json").touch()
            manifest_path = root / "terminal_smoke_manifest.json"
            assembly_path = root / "manifest_plan_assembly.json"
            ceiling_path = root / "ceiling.json"
            for path in (manifest_path, assembly_path, ceiling_path):
                path.touch()
            (root / "setup_submissions.tsv").write_text(
                "plan_a_001\t101\t/a.job\t/a.qsub\n"
                "plan_b_002\t102\t/b.job\t/b.qsub\n"
                "plan_merge\t103\t/merge.job\t/merge.qsub\n",
                encoding="utf-8",
            )
            source = {"identity_hash": HASH}
            manifest = {
                "schema": execution.EXECUTION_MANIFEST_SCHEMA,
                "manifest_hash": HASH,
                "max_descriptors_per_subshard": 450,
                "task_count": 16,
                "task_descriptor_limit": 1,
                "array_required": True,
                "tasks": tuple(
                    {
                        "logical_case_owner": execution.SMOKE_CASE_IDS[(index - 1) // 4],
                        "descriptors": ({"descriptor_index": index},),
                    }
                    for index in range(1, 17)
                ),
                "case_owners": tuple(
                    {"logical_case_owner": owner}
                    for owner in execution.SMOKE_CASE_IDS
                ),
                "resources": {
                    "queue": "campus2.q",
                    "h_rt_seconds": 86400,
                    "memory_bytes": 8589934592,
                    "throttle": 16,
                },
            }
            assembly = {"assembly_hash": HASH}
            ceiling = {"captured_at_utc": "2026-08-12T00:00:00+00:00", "report_hash": HASH}

            def load(path):
                if path == manifest_path:
                    return manifest
                if path == assembly_path:
                    return assembly
                if path == ceiling_path:
                    return ceiling
                task_id = int(path.stem.rsplit("_", 1)[1])
                return {
                    "stage": "smoke",
                    "shard_index": task_id,
                    "shard_count": 16,
                    "source_identity": source,
                    "fragment_hash": HASH,
                }

            with (
                patch.object(setup, "_load", side_effect=load),
                patch.object(setup, "create_compute_ceiling_report", return_value=ceiling),
                patch.object(setup, "_validate_profile", return_value=HASH),
                patch.object(execution, "_validate_self_hash"),
                patch.object(execution, "load_accepted_canonical_base_provider", return_value=(object(), object())),
                patch.object(execution, "build_terminal_suites", return_value={}),
                patch.object(execution, "capture_clean_source_identity", return_value=source),
                patch.object(execution, "validate_manifest_plan_fragment", autospec=True) as validate_fragment,
                patch.object(execution, "merge_manifest_plan_replicates", return_value=(manifest, assembly)),
                patch.object(execution, "validate_compute_ceiling_binding"),
            ):
                setup._validate_outputs(root, ceiling_path, root)
            self.assertEqual(validate_fragment.call_count, 32)
            _args, kwargs = validate_fragment.call_args
            self.assertEqual(
                kwargs,
                {"reconstruct_expected": False, "validate_suite_contents": False},
            )

    def test_p2_audit_aggregates_exact_scheduler_shape(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "final_qstat").mkdir()
            (root / "logs").mkdir()
            (root / "final_qstat" / "snapshot.xml").write_text(
                EMPTY_QSTAT, encoding="utf-8"
            )
            (root / "final_qstat" / "snapshot.status").write_text(
                "0\n", encoding="utf-8"
            )
            ceiling = root / "ceiling.json"
            ceiling.write_text("{}\n", encoding="utf-8")
            submissions = (
                {"role": "plan_a_001", "job_id": "101"},
                {"role": "plan_b_002", "job_id": "102"},
                {"role": "plan_merge", "job_id": "103"},
            )
            for row in submissions:
                count = 1 if row["role"] == "plan_merge" else 16
                for task_id in range(1, count + 1):
                    suffix = "" if count == 1 else f".{task_id}"
                    (root / "logs" / f"{row['role']}.{row['job_id']}{suffix}.log").write_text(
                        "", encoding="utf-8"
                    )

            def usage(row):
                count = 1 if row["role"] == "plan_merge" else 16
                wall = 20.0 if count == 1 else 100.0
                return tuple({
                    "task_id": None if count == 1 else task_id,
                    "hostname": "n1",
                    "wall_seconds": wall,
                    "cpu_seconds": wall - 1.0,
                    "max_memory_bytes": 100 * 1024**2,
                } for task_id in range(1, count + 1)), HASH

            with (
                patch.object(
                    setup,
                    "_validate_outputs",
                    return_value=({"manifest_hash": HASH, "expected_descriptor_count": 16}, submissions, (HASH,)),
                ),
                patch.object(setup, "_validate_job_script", return_value=("job", HASH, HASH)),
                patch.object(setup, "_validate_qacct", side_effect=lambda _root, row, _name: usage(row)),
                patch.object(execution, "sha256_file", return_value=HASH),
            ):
                result = setup.audit_manifest_setup(
                    root,
                    compute_ceiling_path=ceiling,
                    compute_ceiling_evidence_root=root,
                )
            self.assertTrue(result["audit_pass"])
            self.assertEqual(result["fragment_task_count"], 32)
            self.assertEqual(result["critical_path_seconds"], 120.0)

    def test_p2_audit_rejects_failure_log_and_critical_path(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "final_qstat").mkdir()
            (root / "logs").mkdir()
            (root / "final_qstat" / "snapshot.xml").write_text(EMPTY_QSTAT, encoding="utf-8")
            (root / "final_qstat" / "snapshot.status").write_text("0\n", encoding="utf-8")
            ceiling = root / "ceiling.json"
            ceiling.write_text("{}\n", encoding="utf-8")
            submissions = (
                {"role": "plan_a_001", "job_id": "101"},
                {"role": "plan_b_002", "job_id": "102"},
                {"role": "plan_merge", "job_id": "103"},
            )
            for row in submissions:
                count = 1 if row["role"] == "plan_merge" else 16
                for task_id in range(1, count + 1):
                    suffix = "" if count == 1 else f".{task_id}"
                    marker = "Killed\n" if row["role"] == "plan_a_001" and task_id == 1 else ""
                    (root / "logs" / f"{row['role']}.{row['job_id']}{suffix}.log").write_text(
                        marker, encoding="utf-8"
                    )
            records = tuple({
                "task_id": task_id,
                "hostname": "n1",
                "wall_seconds": 1.0,
                "cpu_seconds": 1.0,
                "max_memory_bytes": 1,
            } for task_id in range(1, 17))
            with (
                patch.object(
                    setup,
                    "_validate_outputs",
                    return_value=({"manifest_hash": HASH, "expected_descriptor_count": 16}, submissions, (HASH,)),
                ),
                patch.object(setup, "_validate_job_script", return_value=("job", HASH, HASH)),
                patch.object(setup, "_validate_qacct", return_value=(records, HASH)),
            ):
                with self.assertRaisesRegex(RuntimeError, "forbidden"):
                    setup.audit_manifest_setup(
                        root,
                        compute_ceiling_path=ceiling,
                        compute_ceiling_evidence_root=root,
                    )

    def test_smoke_submitter_uses_minutes_scale_limits_and_audit_is_immutable(self):
        project = Path(__file__).resolve().parents[1]
        submitter = (project / "scripts/submit_hoffman2_terminal_manifest_setup.sh").read_text(
            encoding="utf-8"
        )
        self.assertIn('fragment_h_rt="00:10:00"', submitter)
        self.assertIn('merge_h_rt="00:05:00"', submitter)
        self.assertEqual(submitter.count('formal_h_rt_seconds=86400'), 2)
        self.assertIn('formal_h_rt_seconds=86400', submitter)
        self.assertIn('export LANG=C LC_ALL=C', submitter)
        self.assertIn('${submitted_jobs[@]+"${submitted_jobs[@]}"}', submitter)
        auditor = (project / "scripts/audit_hoffman2_terminal_manifest_setup.sh").read_text(
            encoding="utf-8"
        )
        for token in ("Refusing to overwrite", "qstat", "qacct", "setup_audit.json"):
            self.assertIn(token, auditor)
        self.assertTrue({
            "scripts/audit_hoffman2_terminal_manifest_setup.sh",
            "scripts/audit_terminal_manifest_setup.py",
            "scripts/capture_hoffman2_compute_ceiling.sh",
            "scripts/create_hoffman2_compute_ceiling_report.py",
            "src/experiments/terminal_setup_diagnostics.py",
        }.issubset(execution.TERMINAL_SOURCE_PATHS))

    def test_partial_submission_rolls_back_and_proves_absence(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bin_dir = root / "bin"
            bin_dir.mkdir()
            state = root / "qsub_count"
            qdel_log = root / "qdel_args"
            ceiling = root / "ceiling.json"
            ceiling.write_text("{}\n", encoding="utf-8")
            scripts = {
                "git": "#!/bin/sh\nexit 0\n",
                "qsub": (
                    "#!/bin/sh\n"
                    'n=0; [ ! -f "$FAKE_STATE" ] || n=$(cat "$FAKE_STATE")\n'
                    'n=$((n+1)); printf "%s\\n" "$n" > "$FAKE_STATE"\n'
                    '[ "$n" -eq 1 ] && { echo "101.1-16:1"; exit 0; }\n'
                    "exit 2\n"
                ),
                "qdel": '#!/bin/sh\nprintf "%s\\n" "$*" > "$QDEL_LOG"\nexit 0\n',
                "qstat": f"#!/bin/sh\ncat <<'EOF'\n{EMPTY_QSTAT}EOF\n",
            }
            for name, content in scripts.items():
                path = bin_dir / name
                path.write_text(content, encoding="utf-8")
                path.chmod(0o700)
            setup_root = root / "setup"
            env = dict(os.environ)
            env.update({
                "STAGE": "smoke",
                "SETUP_ROOT": str(setup_root),
                "COMPUTE_CEILING": str(ceiling),
                "PYTHON_BIN": sys.executable,
                "GIT_BIN": str(bin_dir / "git"),
                "QSUB_BIN": str(bin_dir / "qsub"),
                "QDEL_BIN": str(bin_dir / "qdel"),
                "QSTAT_BIN": str(bin_dir / "qstat"),
                "FAKE_STATE": str(state),
                "QDEL_LOG": str(qdel_log),
                "LANG": "C",
                "LC_ALL": "C",
            })
            completed = subprocess.run(
                ["bash", str(Path(__file__).resolve().parents[1] / "scripts/submit_hoffman2_terminal_manifest_setup.sh")],
                env=env,
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertNotEqual(completed.returncode, 0)
            self.assertTrue(
                qdel_log.is_file(),
                completed.stdout + "\n" + completed.stderr,
            )
            self.assertEqual(qdel_log.read_text(encoding="utf-8").strip(), "101")
            self.assertEqual(
                (setup_root / "rollback/status").read_text(encoding="utf-8").strip(),
                "all_submitted_jobs_absent",
            )
            state.unlink()
            qdel_log.unlink()
            (bin_dir / "qstat").write_text(
                "#!/bin/sh\necho 'permission denied'\nexit 1\n",
                encoding="utf-8",
            )
            uncertain_root = root / "uncertain_setup"
            env["SETUP_ROOT"] = str(uncertain_root)
            uncertain = subprocess.run(
                ["bash", str(Path(__file__).resolve().parents[1] / "scripts/submit_hoffman2_terminal_manifest_setup.sh")],
                env=env,
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(uncertain.returncode, 97)
            self.assertEqual(
                (uncertain_root / "rollback/status").read_text(encoding="utf-8").strip(),
                "cleanup_uncertain",
            )

    def test_merge_dependency_substitution_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "jobs").mkdir()
            (root / "qsub_raw").mkdir()
            ceiling = root / "ceiling.json"
            ceiling.write_text("{}\n", encoding="utf-8")
            role = "plan_merge"
            job_id = "103"
            project = Path(__file__).resolve().parents[1]
            script = "\n".join((
                "#$ -N tv_smerge",
                "#$ -l h_rt=00:05:00",
                "#$ -l h_data=4G",
                "#$ -hold_jid 999,998",
                f'cd "{project}"',
                '"/u/home/z/zzl/.conda/envs/rr-allocation/bin/python" scripts/terminal_validation_array.py merge-plan-fragments',
                '--stage "smoke"',
                f'--replicate-a-dir "{root}/plan_a"',
                f'--replicate-b-dir "{root}/plan_b" --shard-count "16"',
                f'--output "{root}/terminal_smoke_manifest.json"',
                f'--assembly-output "{root}/manifest_plan_assembly.json"',
                f'--compute-ceiling "{ceiling}"',
                f'--profile-output "{root}/profiles_merge/merge.json"',
            ))
            job_path = root / "jobs" / f"{role}.job"
            qsub_path = root / "qsub_raw" / f"{role}.txt"
            qsub_status_path = root / "qsub_raw" / f"{role}.status"
            job_path.write_text(script, encoding="utf-8")
            qsub_path.write_text(f"{job_id}\n", encoding="utf-8")
            qsub_status_path.write_text("0\n", encoding="utf-8")
            submissions = (
                {"role": "plan_a_001", "job_id": "101"},
                {"role": "plan_b_002", "job_id": "102"},
                {"role": role, "job_id": job_id, "job_file": str(job_path), "qsub_file": str(qsub_path)},
            )
            with self.assertRaisesRegex(RuntimeError, "semantics"):
                setup._validate_job_script(
                    root,
                    submissions[-1],
                    submissions,
                    compute_ceiling_path=ceiling,
                )

    def test_generated_smoke_job_scripts_match_exact_audit_templates(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bin_dir = root / "bin"
            bin_dir.mkdir()
            state = root / "qsub_count"
            ceiling = root / "ceiling.json"
            ceiling.write_text("{}\n", encoding="utf-8")
            scripts = {
                "git": "#!/bin/sh\nexit 0\n",
                "qsub": (
                    "#!/bin/sh\n"
                    'n=100; [ ! -f "$FAKE_STATE" ] || n=$(cat "$FAKE_STATE")\n'
                    'n=$((n+1)); printf "%s\\n" "$n" > "$FAKE_STATE"\n'
                    'printf "%s\\n" "$n"\n'
                ),
                "qstat": f"#!/bin/sh\ncat <<'EOF'\n{EMPTY_QSTAT}EOF\n",
            }
            for name, content in scripts.items():
                path = bin_dir / name
                path.write_text(content, encoding="utf-8")
                path.chmod(0o700)
            setup_root = root / "setup"
            env = dict(os.environ)
            env.update({
                "STAGE": "smoke",
                "SETUP_ROOT": str(setup_root),
                "COMPUTE_CEILING": str(ceiling),
                "PYTHON_BIN": sys.executable,
                "GIT_BIN": str(bin_dir / "git"),
                "QSUB_BIN": str(bin_dir / "qsub"),
                "QSTAT_BIN": str(bin_dir / "qstat"),
                "FAKE_STATE": str(state),
                "LANG": "C",
                "LC_ALL": "C",
            })
            completed = subprocess.run(
                ["bash", str(Path(__file__).resolve().parents[1] / "scripts/submit_hoffman2_terminal_manifest_setup.sh")],
                env=env,
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(
                completed.returncode,
                0,
                completed.stdout + "\n" + completed.stderr,
            )
            canonical_setup_root = setup_root.resolve()
            submissions = setup._submissions(canonical_setup_root)
            self.assertEqual(tuple(row["job_id"] for row in submissions), ("101", "102", "103"))
            with patch.object(setup, "APPROVED_PYTHON_BIN", sys.executable):
                for row in submissions:
                    setup._validate_job_script(
                        canonical_setup_root,
                        row,
                        submissions,
                        compute_ceiling_path=ceiling.resolve(),
                    )

    def test_successful_unparseable_qsub_is_recovered_and_cancelled(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bin_dir = root / "bin"
            bin_dir.mkdir()
            active = root / "active_job_name"
            qdel_log = root / "qdel_args"
            ceiling = root / "ceiling.json"
            ceiling.write_text("{}\n", encoding="utf-8")
            scripts = {
                "git": "#!/bin/sh\nexit 0\n",
                "qsub": (
                    "#!/bin/sh\n"
                    'grep "^#\\$ -N " "$2" | awk \'{print $3}\' > "$ACTIVE_JOB"\n'
                    "echo 'submission accepted but malformed output'\n"
                    "exit 0\n"
                ),
                "qdel": (
                    '#!/bin/sh\nprintf "%s\\n" "$*" > "$QDEL_LOG"\n'
                    'rm -f "$ACTIVE_JOB"\nexit 0\n'
                ),
                "qstat": (
                    "#!/bin/sh\n"
                    "echo \"<?xml version='1.0'?><job_info><queue_info>\"\n"
                    'if [ -f "$ACTIVE_JOB" ]; then n=$(cat "$ACTIVE_JOB"); '
                    'echo "<job_list><JB_job_number>777</JB_job_number><JB_name>$n</JB_name></job_list>"; fi\n'
                    'echo "</queue_info><job_info></job_info></job_info>"\n'
                ),
            }
            for name, content in scripts.items():
                path = bin_dir / name
                path.write_text(content, encoding="utf-8")
                path.chmod(0o700)
            setup_root = root / "setup"
            env = dict(os.environ)
            env.update({
                "STAGE": "smoke",
                "SETUP_ROOT": str(setup_root),
                "COMPUTE_CEILING": str(ceiling),
                "PYTHON_BIN": sys.executable,
                "GIT_BIN": str(bin_dir / "git"),
                "QSUB_BIN": str(bin_dir / "qsub"),
                "QDEL_BIN": str(bin_dir / "qdel"),
                "QSTAT_BIN": str(bin_dir / "qstat"),
                "ACTIVE_JOB": str(active),
                "QDEL_LOG": str(qdel_log),
                "LANG": "C",
                "LC_ALL": "C",
            })
            completed = subprocess.run(
                ["bash", str(Path(__file__).resolve().parents[1] / "scripts/submit_hoffman2_terminal_manifest_setup.sh")],
                env=env,
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertNotEqual(completed.returncode, 0)
            self.assertEqual(qdel_log.read_text(encoding="utf-8").strip(), "777")
            self.assertEqual(
                (setup_root / "rollback/plan_a_001.recovered_job_ids").read_text(
                    encoding="utf-8"
                ).strip(),
                "777",
            )
            self.assertFalse(active.exists())

    def test_profile_and_qacct_attacks_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            profile_path = root / "profile.json"
            profile = {
                "schema": "terminal_validation_phase_profile_v1",
                "command": "freeze-plan-fragment",
                "bindings": {"stage": "smoke"},
                "phase_seconds": (("command_total", 1.0),),
                "profile_hash": "",
            }
            profile["profile_hash"] = execution.logical_hash(
                execution._without_hash(profile, "profile_hash")
            )
            execution.write_new_json(profile_path, profile)
            with self.assertRaisesRegex(RuntimeError, "timing"):
                setup._validate_profile(
                    profile_path,
                    command="freeze-plan-fragment",
                    bindings={"stage": "smoke"},
                )

        attacks = (
            ("exit", "0", "1", "10", "100M", "prove success"),
            ("wall", "0", "0", "301", "100M", "wall-time"),
            ("memory", "0", "0", "10", "1.1G", "memory"),
        )
        for name, failed, exit_status, wall, memory, message in attacks:
            with self.subTest(name=name), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                (root / "qacct").mkdir()
                records = []
                for task_id in range(1, 17):
                    records.append("\n".join((
                        "qname campus2.q@n1",
                        "hostname n1",
                        "jobname tv_sa1",
                        "jobnumber 101",
                        f"taskid {task_id}",
                        "slots 1",
                        f"failed {failed if task_id == 1 else '0'}",
                        f"exit_status {exit_status if task_id == 1 else '0'}",
                        "cpu 9",
                        f"ru_wallclock {wall if task_id == 1 else '10'}",
                        f"maxvmem {memory if task_id == 1 else '100M'}",
                    )))
                (root / "qacct/plan_a_001.raw").write_text(
                    "\n==============================================================\n".join(records) + "\n",
                    encoding="utf-8",
                )
                row = {"role": "plan_a_001", "job_id": "101"}
                with self.assertRaisesRegex(RuntimeError, message):
                    setup._validate_qacct(root, row, "tv_sa1")


if __name__ == "__main__":
    unittest.main()
