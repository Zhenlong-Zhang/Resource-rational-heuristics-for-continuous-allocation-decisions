#!/usr/bin/env python3
"""Freeze, execute, account for, and read back terminal validation evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import src.experiments.terminal_execution as execution  # noqa: E402


SOURCE_PATHS = (
    "configs/terminal_base_beliefs_7376c5d_v1.json",
    "configs/terminal_evidence_numerical_method_v2.json",
    "scripts/terminal_validation_array.py",
    "scripts/submit_hoffman2_terminal_validation.sh",
    "scripts/submit_hoffman2_terminal_manifest_setup.sh",
    "src/experiments/terminal_evidence_rows.py",
    "src/experiments/terminal_execution.py",
    "src/experiments/terminal_canonical_provider.py",
    "src/experiments/terminal_validation_suite.py",
    "src/mdp/finite_support.py",
    "src/solvers/terminal.py",
    "src/solvers/terminal_reference.py",
    "src/solvers/terminal_reference_agreement.py",
    "src/solvers/terminal_reference_b.py",
)


def load(path: Path):
    return execution._decode(dict(execution._load_json(path)))


def load_provider(_args=None):
    return execution.load_accepted_canonical_base_provider()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    freeze = commands.add_parser("freeze-manifest")
    freeze.add_argument("--stage", choices=("smoke", "full"), required=True)
    freeze.add_argument("--output", type=Path, required=True)
    freeze.add_argument("--compute-ceiling", type=Path, required=True)
    freeze.add_argument("--max-descriptors-per-subshard", type=int, default=450)
    freeze.add_argument("--queue", required=True)
    freeze.add_argument("--h-rt-seconds", type=int, required=True)
    freeze.add_argument("--memory-bytes", type=int, required=True)
    freeze.add_argument("--throttle", type=int, required=True)

    fragment = commands.add_parser("freeze-plan-fragment")
    fragment.add_argument("--stage", choices=("smoke", "full"), required=True)
    fragment.add_argument("--shard-index", type=int, required=True)
    fragment.add_argument("--shard-count", type=int, required=True)
    fragment.add_argument("--output", type=Path, required=True)

    merge = commands.add_parser("merge-plan-fragments")
    merge.add_argument("--stage", choices=("smoke", "full"), required=True)
    merge.add_argument("--replicate-a-dir", type=Path, required=True)
    merge.add_argument("--replicate-b-dir", type=Path, required=True)
    merge.add_argument("--shard-count", type=int, required=True)
    merge.add_argument("--output", type=Path, required=True)
    merge.add_argument("--assembly-output", type=Path, required=True)
    merge.add_argument("--compute-ceiling", type=Path, required=True)
    merge.add_argument("--max-descriptors-per-subshard", type=int, default=450)
    merge.add_argument("--queue", required=True)
    merge.add_argument("--h-rt-seconds", type=int, required=True)
    merge.add_argument("--memory-bytes", type=int, required=True)
    merge.add_argument("--throttle", type=int, required=True)

    run = commands.add_parser("run-task")
    run.add_argument("--manifest", type=Path, required=True)
    run.add_argument("--output-root", type=Path, required=True)
    run.add_argument("--task-id", type=int)

    collect = commands.add_parser("collect-provisional")
    collect.add_argument("--manifest", type=Path, required=True)
    collect.add_argument("--output-root", type=Path, required=True)
    collect.add_argument("--output", type=Path, required=True)

    record = commands.add_parser("record-scheduler")
    record.add_argument("--manifest", type=Path, required=True)
    record.add_argument("--submissions", type=Path, required=True)
    record.add_argument("--evidence-root", type=Path, required=True)
    record.add_argument("--output", type=Path, required=True)

    audit = commands.add_parser("audit-qacct")
    audit.add_argument("--manifest", type=Path, required=True)
    audit.add_argument("--scheduler-evidence", type=Path, required=True)
    audit.add_argument("--evidence-root", type=Path, required=True)
    audit.add_argument("--qacct", action="append", required=True, metavar="JOB_ID:PATH")
    audit.add_argument("--output", type=Path, required=True)

    finalize = commands.add_parser("finalize-post-job")
    finalize.add_argument("--manifest", type=Path, required=True)
    finalize.add_argument("--task-output-root", type=Path, required=True)
    finalize.add_argument("--provisional", type=Path, required=True)
    finalize.add_argument("--scheduler-evidence", type=Path, required=True)
    finalize.add_argument("--qacct-audit", type=Path, required=True)
    finalize.add_argument("--compute-ceiling", type=Path, required=True)
    finalize.add_argument("--scheduler-evidence-root", type=Path, required=True)
    finalize.add_argument("--output", type=Path, required=True)

    readback = commands.add_parser("independent-readback")
    readback.add_argument("--manifest", type=Path, required=True)
    readback.add_argument("--task-output-root", type=Path, required=True)
    readback.add_argument("--provisional", type=Path, required=True)
    readback.add_argument("--scheduler-evidence", type=Path, required=True)
    readback.add_argument("--qacct-audit", type=Path, required=True)
    readback.add_argument("--compute-ceiling", type=Path, required=True)
    readback.add_argument("--scheduler-evidence-root", type=Path, required=True)
    readback.add_argument("--post-job", type=Path, required=True)
    readback.add_argument("--output", type=Path, required=True)

    validate = commands.add_parser("validate-manifest")
    validate.add_argument("--manifest", type=Path, required=True)
    validate.add_argument("--structural-only", action="store_true")

    authorize = commands.add_parser("validate-authorization")
    authorize.add_argument("--manifest", type=Path, required=True)
    authorize.add_argument("--authorization", type=Path, required=True)
    authorize.add_argument("--approved-authorization-hash", required=True)

    ceiling = commands.add_parser("validate-compute-ceiling")
    ceiling.add_argument("--manifest", type=Path, required=True)
    ceiling.add_argument("--compute-ceiling", type=Path, required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "freeze-plan-fragment":
        provider, accepted = load_provider(args)
        suites = execution.build_terminal_suites(provider, accepted)
        source = execution.capture_clean_source_identity(PROJECT_ROOT, SOURCE_PATHS)
        fragment = execution.create_manifest_plan_fragment(
            stage=args.stage,
            shard_index=args.shard_index,
            shard_count=args.shard_count,
            suites=suites,
            provider=provider,
            acceptance_validator=accepted,
            source_identity=source,
        )
        execution.write_new_json(args.output, fragment)
    elif args.command == "merge-plan-fragments":
        provider, accepted = load_provider(args)
        suites = execution.build_terminal_suites(provider, accepted)
        source = execution.capture_clean_source_identity(PROJECT_ROOT, SOURCE_PATHS)
        ceiling = load(args.compute_ceiling)
        execution._validate_self_hash(ceiling, "report_hash", "compute ceiling report")
        names = tuple(f"fragment_{index:03d}.json" for index in range(1, args.shard_count + 1))
        replicate_a = tuple(load(args.replicate_a_dir / name) for name in names)
        replicate_b = tuple(load(args.replicate_b_dir / name) for name in names)
        manifest, assembly = execution.merge_manifest_plan_replicates(
            stage=args.stage,
            replicate_a=replicate_a,
            replicate_b=replicate_b,
            suites=suites,
            provider=provider,
            acceptance_validator=accepted,
            source_identity=source,
            max_descriptors_per_subshard=args.max_descriptors_per_subshard,
            resources={
                "queue": args.queue,
                "h_rt_seconds": args.h_rt_seconds,
                "memory_bytes": args.memory_bytes,
                "throttle": args.throttle,
            },
            compute_ceiling_report_hash=ceiling["report_hash"],
        )
        execution.validate_compute_ceiling_binding(manifest, ceiling)
        execution.write_new_json(args.output, manifest)
        execution.write_new_json(args.assembly_output, assembly)
    elif args.command == "freeze-manifest":
        provider, accepted = load_provider(args)
        suites = execution.build_terminal_suites(provider, accepted)
        source = execution.capture_clean_source_identity(PROJECT_ROOT, SOURCE_PATHS)
        ceiling = load(args.compute_ceiling)
        execution._validate_self_hash(ceiling, "report_hash", "compute ceiling report")
        ceiling_hash = ceiling["report_hash"]
        manifest = execution.create_execution_manifest(
            stage=args.stage,
            suites=suites,
            provider=provider,
            acceptance_validator=accepted,
            source_identity=source,
            max_descriptors_per_subshard=args.max_descriptors_per_subshard,
            resources={
                "queue": args.queue,
                "h_rt_seconds": args.h_rt_seconds,
                "memory_bytes": args.memory_bytes,
                "throttle": args.throttle,
            },
            compute_ceiling_report_hash=ceiling_hash,
        )
        execution.validate_compute_ceiling_binding(manifest, ceiling)
        execution.write_new_json(args.output, manifest)
    elif args.command in ("run-task", "collect-provisional", "validate-manifest"):
        manifest = load(args.manifest)
        provider, accepted = load_provider(args)
        suites = execution.build_terminal_suites(provider, accepted)
        execution.validate_clean_source_identity(PROJECT_ROOT, manifest["source_identity"])
        if args.command == "run-task":
            task_id = args.task_id
            if task_id is None:
                raw = __import__("os").environ.get("SGE_TASK_ID")
                if raw is None or not raw.isdigit():
                    raise RuntimeError("task ID is absent")
                task_id = int(raw)
            execution.execute_task(
                manifest=manifest,
                suites=suites,
                provider=provider,
                acceptance_validator=accepted,
                output_root=args.output_root,
                task_id=task_id,
            )
        elif args.command == "collect-provisional":
            execution.collect_provisional(
                manifest=manifest,
                suites=suites,
                provider=provider,
                acceptance_validator=accepted,
                output_root=args.output_root,
                provisional_path=args.output,
            )
        else:
            execution.validate_execution_manifest(
                manifest,
                suites,
                provider,
                accepted,
                reconstruct_expected=not args.structural_only,
            )
    elif args.command == "record-scheduler":
        manifest = load(args.manifest)
        raw = json.loads(args.submissions.read_text(encoding="utf-8"))
        scheduler = execution.create_scheduler_evidence(
            manifest, raw["submissions"], evidence_root=args.evidence_root
        )
        execution.write_new_json(args.output, scheduler)
    elif args.command == "audit-qacct":
        manifest = load(args.manifest)
        scheduler = load(args.scheduler_evidence)
        paths = {}
        for value in args.qacct:
            job_id, separator, path = value.partition(":")
            if not separator or job_id in paths:
                raise RuntimeError("qacct arguments must be unique JOB_ID:PATH pairs")
            paths[job_id] = Path(path)
        audit = execution.audit_qacct(
            manifest, scheduler, paths, evidence_root=args.evidence_root
        )
        execution.write_new_json(args.output, audit)
    elif args.command == "validate-authorization":
        execution.validate_clean_source_identity(
            PROJECT_ROOT, load(args.manifest)["source_identity"]
        )
        execution.validate_execution_authorization(
            authorization_path=args.authorization,
            approved_file_hash=args.approved_authorization_hash,
            manifest=load(args.manifest),
            project_root=PROJECT_ROOT,
        )
    elif args.command == "validate-compute-ceiling":
        execution.validate_compute_ceiling_binding(
            load(args.manifest), load(args.compute_ceiling)
        )
    elif args.command == "finalize-post-job":
        provider, accepted = load_provider(args)
        suites = execution.build_terminal_suites(provider, accepted)
        execution.finalize_post_job(
            manifest=load(args.manifest),
            suites=suites,
            provider=provider,
            acceptance_validator=accepted,
            task_output_root=args.task_output_root,
            provisional_path=args.provisional,
            scheduler_evidence_path=args.scheduler_evidence,
            qacct_audit_path=args.qacct_audit,
            compute_ceiling_path=args.compute_ceiling,
            scheduler_evidence_root=args.scheduler_evidence_root,
            output_path=args.output,
            project_root=PROJECT_ROOT,
        )
    elif args.command == "independent-readback":
        provider, accepted = load_provider(args)
        suites = execution.build_terminal_suites(provider, accepted)
        execution.independent_readback(
            manifest=load(args.manifest),
            suites=suites,
            provider=provider,
            acceptance_validator=accepted,
            task_output_root=args.task_output_root,
            provisional_path=args.provisional,
            scheduler_evidence_path=args.scheduler_evidence,
            qacct_audit_path=args.qacct_audit,
            compute_ceiling_path=args.compute_ceiling,
            scheduler_evidence_root=args.scheduler_evidence_root,
            post_job_path=args.post_job,
            final_output_path=args.output,
            project_root=PROJECT_ROOT,
        )


if __name__ == "__main__":
    main()
