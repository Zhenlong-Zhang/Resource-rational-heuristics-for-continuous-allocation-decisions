import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import textwrap
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCHEDULER_LIBRARY = PROJECT_ROOT / "scripts" / "r6_scheduler.sh"


class R6SubmissionTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.bin_dir = self.root / "bin"
        self.bin_dir.mkdir()
        self.state_dir = self.root / "state"
        self.state_dir.mkdir()
        self.evidence_dir = self.root / "evidence"
        self.job_file = self.root / "job.sh"
        self.job_file.write_text("#!/usr/bin/env bash\ntrue\n", encoding="utf-8")
        self._write_fake_scheduler()

    def tearDown(self):
        self.temporary.cleanup()

    def _write_executable(self, name: str, body: str) -> None:
        path = self.bin_dir / name
        path.write_text(textwrap.dedent(body).lstrip(), encoding="utf-8")
        path.chmod(0o755)

    def _write_fake_scheduler(self) -> None:
        self._write_executable(
            "qsub",
            """
            #!/usr/bin/env bash
            set -euo pipefail
            name=""
            while [[ "$#" -gt 0 ]]; do
              case "$1" in
                -terse) shift ;;
                -N) name="$2"; shift 2 ;;
                *) shift ;;
              esac
            done
            next_file="${FAKE_SCHEDULER_STATE}/next_id"
            job_id="$(cat "${next_file}")"
            printf '%s' "$((job_id + 1))" >"${next_file}"
            printf '%s\t%s\n' "${job_id}" "${name}" >>"${FAKE_SCHEDULER_STATE}/jobs"
            if [[ "${FAKE_QSUB_MODE:-valid}" == "malformed" ]]; then
              printf 'submission accepted but identifier unavailable\n'
            else
              printf '%s.1-3:1\n' "${job_id}"
            fi
            """,
        )
        self._write_executable(
            "qstat",
            """
            #!/usr/bin/env bash
            set -euo pipefail
            printf '<job_info><queue_info>'
            while IFS=$'\\t' read -r job_id name; do
              [[ -n "${job_id}" ]] || continue
              printf '<job_list><JB_job_number>%s</JB_job_number><JB_name>%s</JB_name></job_list>' "${job_id}" "${name}"
            done <"${FAKE_SCHEDULER_STATE}/jobs"
            printf '</queue_info></job_info>\n'
            """,
        )
        self._write_executable(
            "qdel",
            """
            #!/usr/bin/env bash
            set -euo pipefail
            tmp="${FAKE_SCHEDULER_STATE}/jobs.tmp"
            cp "${FAKE_SCHEDULER_STATE}/jobs" "${tmp}"
            for target in "$@"; do
              awk -F '\\t' -v target="${target}" '$1 != target && $2 != target' "${tmp}" >"${tmp}.next"
              mv "${tmp}.next" "${tmp}"
              printf '%s\n' "${target}" >>"${FAKE_SCHEDULER_STATE}/deleted"
            done
            mv "${tmp}" "${FAKE_SCHEDULER_STATE}/jobs"
            """,
        )
        (self.state_dir / "next_id").write_text("9001", encoding="utf-8")
        (self.state_dir / "jobs").write_text("", encoding="utf-8")
        (self.state_dir / "deleted").write_text("", encoding="utf-8")

    def _run(self, script: str, mode: str = "valid") -> subprocess.CompletedProcess[str]:
        environment = os.environ.copy()
        environment.update(
            {
                "FAKE_SCHEDULER_STATE": str(self.state_dir),
                "FAKE_QSUB_MODE": mode,
                "R6_QSUB_BIN": str(self.bin_dir / "qsub"),
                "R6_QSTAT_BIN": str(self.bin_dir / "qstat"),
                "R6_QDEL_BIN": str(self.bin_dir / "qdel"),
                "R6_SCHEDULER_PYTHON": shutil.which("python3") or "python3",
            }
        )
        return subprocess.run(
            ["bash", "-c", script],
            cwd=PROJECT_ROOT,
            env=environment,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=30,
        )

    def test_valid_terse_submission_is_tracked(self):
        result = self._run(
            f"""
            set -euo pipefail
            source {str(SCHEDULER_LIBRARY)!r}
            r6_init_submission_tracking {str(self.evidence_dir)!r}
            r6_submit_job family unique_valid {str(self.job_file)!r}
            [[ "$R6_LAST_JOB_ID" == "9001" ]]
            [[ "${{R6_SUBMITTED_JOB_IDS[*]}}" == "9001" ]]
            """
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(
            (self.state_dir / "jobs").read_text(encoding="utf-8"),
            "9001\tunique_valid\n",
        )
        self.assertEqual(
            (self.evidence_dir / "family.qsub.stdout").read_text(encoding="utf-8"),
            "9001.1-3:1\n",
        )

    def test_malformed_successful_qsub_output_leaves_no_live_job(self):
        result = self._run(
            f"""
            set -euo pipefail
            source {str(SCHEDULER_LIBRARY)!r}
            r6_init_submission_tracking {str(self.evidence_dir)!r}
            if r6_submit_job family unique_malformed {str(self.job_file)!r}; then
              exit 99
            fi
            """,
            mode="malformed",
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual((self.state_dir / "jobs").read_text(encoding="utf-8"), "")
        deleted = (self.state_dir / "deleted").read_text(encoding="utf-8")
        self.assertIn("unique_malformed", deleted)
        self.assertIn(
            "submission accepted but identifier unavailable",
            (self.evidence_dir / "family.qsub.stdout").read_text(encoding="utf-8"),
        )

    def test_later_failure_rolls_back_every_tracked_job(self):
        result = self._run(
            f"""
            set -euo pipefail
            source {str(SCHEDULER_LIBRARY)!r}
            r6_init_submission_tracking {str(self.evidence_dir)!r}
            r6_submit_job first unique_first {str(self.job_file)!r}
            r6_submit_job second unique_second {str(self.job_file)!r}
            r6_rollback_partial_submission
            """
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual((self.state_dir / "jobs").read_text(encoding="utf-8"), "")
        deleted = (self.state_dir / "deleted").read_text(encoding="utf-8")
        self.assertIn("9001", deleted)
        self.assertIn("9002", deleted)

    def test_failed_qdel_and_qstat_cannot_certify_malformed_submission_cleanup(self):
        result = self._run(
            f"""
            set -euo pipefail
            source {str(SCHEDULER_LIBRARY)!r}
            r6_init_submission_tracking {str(self.evidence_dir)!r}
            R6_QDEL_BIN=/usr/bin/false
            R6_QSTAT_BIN=/usr/bin/false
            set +e
            r6_submit_job family unique_unverifiable {str(self.job_file)!r}
            status=$?
            set -e
            [[ "${{status}}" -eq 70 ]]
            """,
            mode="malformed",
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("manual intervention is required", result.stderr)
        self.assertEqual(
            (self.state_dir / "jobs").read_text(encoding="utf-8"),
            "9001\tunique_unverifiable\n",
        )


if __name__ == "__main__":
    unittest.main()
