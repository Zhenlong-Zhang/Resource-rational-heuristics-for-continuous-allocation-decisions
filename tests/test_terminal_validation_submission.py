from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import textwrap
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SUBMITTER = PROJECT_ROOT / "scripts" / "submit_hoffman2_terminal_validation.sh"


class TerminalValidationSubmissionTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.bin_dir = self.root / "bin"
        self.bin_dir.mkdir()
        self.state_dir = self.root / "state"
        self.state_dir.mkdir()
        (self.state_dir / "jobs").write_text("", encoding="utf-8")
        (self.state_dir / "deleted").write_text("", encoding="utf-8")
        self._write_fake_commands()

        self.manifest = self.root / "terminal_smoke_manifest.json"
        self.manifest_hash = "a" * 64
        self.manifest.write_text(
            json.dumps({
                "stage": "smoke",
                "task_count": 16,
                "resources": {
                    "queue": "campus2.q",
                    "h_rt_seconds": 7200,
                    "memory_bytes": 8589934592,
                    "throttle": 4,
                },
                "manifest_hash": self.manifest_hash,
            }) + "\n",
            encoding="utf-8",
        )
        self.verdict = self.root / "verdict.txt"
        self.verdict.write_text(
            "ACCEPT TERMINAL IMPLEMENTATION FOR SCHEDULED SMOKE\n",
            encoding="utf-8",
        )
        self.authorization = self.root / "authorization.json"
        self.authorization.write_text("{}\n", encoding="utf-8")
        self.ceiling = self.root / "ceiling.json"
        self.ceiling.write_text("{}\n", encoding="utf-8")

    def tearDown(self):
        self.temporary.cleanup()

    def _write_executable(self, name: str, body: str) -> None:
        path = self.bin_dir / name
        path.write_text(textwrap.dedent(body).lstrip(), encoding="utf-8")
        path.chmod(0o755)

    def _write_fake_commands(self) -> None:
        self._write_executable("git", "#!/usr/bin/env bash\nexit 0\n")
        self._write_executable(
            "python",
            f"""
            #!/usr/bin/env bash
            set -euo pipefail
            if [[ "$#" -ge 2 && "$1" == *terminal_validation_array.py ]]; then
              case "$2" in
                validate-manifest|validate-authorization|validate-compute-ceiling)
                  exit 0
                  ;;
                record-scheduler)
                  output=""
                  shift 2
                  while [[ "$#" -gt 0 ]]; do
                    if [[ "$1" == "--output" ]]; then output="$2"; break; fi
                    shift
                  done
                  mkdir -p "$(dirname "$output")"
                  printf '{{}}\n' > "$output"
                  exit 0
                  ;;
              esac
            fi
            exec {sys.executable!r} "$@"
            """,
        )
        self._write_executable(
            "qsub",
            """
            #!/usr/bin/env bash
            set -euo pipefail
            if [[ "${QSUB_MODE:-valid}" == "failure" ]]; then
              printf 'qsub transport failure\n'
              exit 2
            fi
            job_file="${@: -1}"
            name="$(awk '$1 == "#$" && $2 == "-N" {print $3}' "$job_file")"
            printf '9001\t%s\n' "$name" >> "${FAKE_STATE}/jobs"
            if [[ "${QSUB_MODE:-valid}" == "malformed" ]]; then
              printf 'submission accepted but identifier unavailable\n'
            else
              printf '9001.1-16:1\n'
            fi
            """,
        )
        self._write_executable(
            "qstat",
            """
            #!/usr/bin/env bash
            set -euo pipefail
            case "${QSTAT_MODE:-valid}" in
              failure) printf 'permission denied\n'; exit 1 ;;
              malformed) printf '<not_qstat/>\n'; exit 0 ;;
            esac
            printf '<job_info><queue_info>'
            while IFS=$'\t' read -r job_id name; do
              [[ -n "$job_id" ]] || continue
              printf '<job_list><JB_job_number>%s</JB_job_number><JB_name>%s</JB_name></job_list>' "$job_id" "$name"
            done < "${FAKE_STATE}/jobs"
            printf '</queue_info><job_info></job_info></job_info>\n'
            """,
        )
        self._write_executable(
            "qdel",
            """
            #!/usr/bin/env bash
            set -euo pipefail
            if [[ "${QDEL_MODE:-valid}" == "failure" ]]; then exit 2; fi
            tmp="${FAKE_STATE}/jobs.tmp"
            cp "${FAKE_STATE}/jobs" "$tmp"
            for target in "$@"; do
              awk -F '\t' -v target="$target" '$1 != target' "$tmp" > "$tmp.next"
              mv "$tmp.next" "$tmp"
              printf '%s\n' "$target" >> "${FAKE_STATE}/deleted"
            done
            mv "$tmp" "${FAKE_STATE}/jobs"
            """,
        )

    def _output_root(self, label: str) -> Path:
        return self.root / label

    def _run(self, label: str, **overrides: str) -> subprocess.CompletedProcess[str]:
        environment = os.environ.copy()
        environment.update({
            "PATH": f"{self.bin_dir}:{environment.get('PATH', '')}",
            "MANIFEST": str(self.manifest),
            "OUTPUT_ROOT": str(self._output_root(label)),
            "REVIEW_VERDICT_FILE": str(self.verdict),
            "APPROVED_REVIEW_VERDICT_HASH": hashlib.sha256(self.verdict.read_bytes()).hexdigest(),
            "EXECUTION_AUTHORIZATION": str(self.authorization),
            "APPROVED_EXECUTION_AUTHORIZATION_HASH": hashlib.sha256(self.authorization.read_bytes()).hexdigest(),
            "COMPUTE_CEILING": str(self.ceiling),
            "PYTHON_BIN": str(self.bin_dir / "python"),
            "QSUB_BIN": str(self.bin_dir / "qsub"),
            "QSTAT_BIN": str(self.bin_dir / "qstat"),
            "QDEL_BIN": str(self.bin_dir / "qdel"),
            "FAKE_STATE": str(self.state_dir),
            "LANG": "C",
            "LC_ALL": "C",
        })
        environment.update(overrides)
        return subprocess.run(
            ["bash", str(SUBMITTER)],
            cwd=PROJECT_ROOT,
            env=environment,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=30,
        )

    def test_exact_array_submission_is_recorded(self):
        result = self._run("valid")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        job = next((self._output_root("valid") / "scheduler/jobs").glob("*.job"))
        text = job.read_text(encoding="utf-8")
        self.assertIn("#$ -t 1-16", text)
        self.assertIn("#$ -tc 4", text)
        self.assertIn("task_id=${SGE_TASK_ID}", text)
        command_start = text.index(
            f'"{self.bin_dir / "python"}" scripts/terminal_validation_array.py run-task'
        )
        self.assertEqual(
            text[command_start:],
            (
                f'"{self.bin_dir / "python"}" scripts/terminal_validation_array.py run-task \\\n'
                f'  --manifest "{self.manifest}" \\\n'
                f'  --output-root "{self._output_root("valid")}" \\\n'
                '  --task-id "${task_id}"\n'
            ),
        )
        status = next((self._output_root("valid") / "scheduler/qsub_raw").glob("*.status"))
        self.assertEqual(status.read_text(encoding="utf-8"), "0\n")

    def test_malformed_successful_qsub_is_recovered_and_deleted(self):
        result = self._run("malformed", QSUB_MODE="malformed")
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual((self.state_dir / "jobs").read_text(encoding="utf-8"), "")
        self.assertEqual((self.state_dir / "deleted").read_text(encoding="utf-8"), "9001\n")
        status = self._output_root("malformed") / "scheduler/rollback/status"
        self.assertEqual(status.read_text(encoding="utf-8").strip(), "all_submitted_jobs_absent")

    def test_qsub_failure_with_authoritative_absence_is_not_cleanup_uncertain(self):
        result = self._run("qsub_failure", QSUB_MODE="failure")
        self.assertNotEqual(result.returncode, 0)
        self.assertNotEqual(result.returncode, 97)
        self.assertEqual((self.state_dir / "jobs").read_text(encoding="utf-8"), "")

    def test_qstat_malformed_or_failed_is_cleanup_uncertain(self):
        for mode in ("malformed", "failure"):
            with self.subTest(mode=mode):
                (self.state_dir / "jobs").write_text("", encoding="utf-8")
                result = self._run(f"qstat_{mode}", QSTAT_MODE=mode)
                self.assertEqual(result.returncode, 97, result.stdout + result.stderr)

    def test_qdel_failure_cannot_certify_cleanup(self):
        result = self._run("qdel_failure", QSUB_MODE="malformed", QDEL_MODE="failure")
        self.assertEqual(result.returncode, 97, result.stdout + result.stderr)
        self.assertIn("9001", (self.state_dir / "jobs").read_text(encoding="utf-8"))

    def test_preexisting_tag_collision_is_never_deleted(self):
        output_root = self._output_root("collision").resolve()
        token = hashlib.sha256(
            os.fsencode(output_root) + b"\0" + self.manifest_hash.encode("ascii")
        ).hexdigest()[:16]
        (self.state_dir / "jobs").write_text(
            f"777\ttvsmoke_{token}\n", encoding="utf-8"
        )
        result = self._run("collision")
        self.assertEqual(result.returncode, 97, result.stdout + result.stderr)
        self.assertEqual(
            (self.state_dir / "jobs").read_text(encoding="utf-8"),
            f"777\ttvsmoke_{token}\n",
        )
        self.assertEqual((self.state_dir / "deleted").read_text(encoding="utf-8"), "")


if __name__ == "__main__":
    unittest.main()
