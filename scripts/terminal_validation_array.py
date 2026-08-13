#!/usr/bin/env python3
"""Freeze, execute, account for, and read back terminal validation evidence."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import sys
import time

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import src.experiments.terminal_execution as execution  # noqa: E402


SOURCE_PATHS = execution.TERMINAL_SOURCE_PATHS

PHASE_PROFILE_SCHEMA = "terminal_validation_phase_profile_v1"


def load(path: Path):
    return execution._decode(dict(execution._load_json(path)))


def load_provider(_args=None):
    return execution.load_accepted_canonical_base_provider()


def phase_profile(command: str, phases, **bindings):
    normalized = tuple(sorted((str(name), float(value)) for name, value in phases.items()))
    if any(not math.isfinite(value) or value < 0.0 for _, value in normalized):
        raise ValueError("phase profile times must be finite and nonnegative")
    payload = {
        "schema": PHASE_PROFILE_SCHEMA,
        "command": command,
        "bindings": dict(bindings),
        "phase_seconds": normalized,
        "profile_hash": "",
    }
    payload["profile_hash"] = execution.logical_hash(
        execution._without_hash(payload, "profile_hash")
    )
    return payload


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
    fragment.add_argument("--profile-output", type=Path)

    diagnostic = commands.add_parser("diagnose-plan")
    diagnostic.add_argument("--stage", choices=("smoke", "full"), required=True)
    diagnostic.add_argument("--descriptor-position", type=int)
    diagnostic.add_argument("--mode", choices=("plan-only", "parity"), required=True)
    diagnostic.add_argument("--output", type=Path, required=True)

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
    merge.add_argument("--profile-output", type=Path)

    run = commands.add_parser("run-task")
    run.add_argument("--manifest", type=Path, required=True)
    run.add_argument("--output-root", type=Path, required=True)
    run.add_argument("--task-id", type=int)

    partition = commands.add_parser("describe-task-partitions")
    partition.add_argument("--manifest", type=Path, required=True)

    collect = commands.add_parser("collect-provisional")
    collect.add_argument("--manifest", type=Path, required=True)
    collect.add_argument("--output-root", type=Path, required=True)
    collect.add_argument("--output", type=Path, required=True)

    record = commands.add_parser("record-scheduler")
    record.add_argument("--manifest", type=Path, required=True)
    record.add_argument("--submissions", type=Path, required=True)
    record.add_argument("--evidence-root", type=Path, required=True)
    record.add_argument("--execution-project-root", type=Path, required=True)
    record.add_argument("--approved-python-bin", type=Path, required=True)
    record.add_argument("--scheduler-user", required=True)
    record.add_argument("--run-tag", required=True)
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

    marker = commands.add_parser("finalize-and-capture-formal-smoke")
    marker.add_argument("--manifest", type=Path, required=True)
    marker.add_argument("--task-output-root", type=Path, required=True)
    marker.add_argument("--provisional", type=Path, required=True)
    marker.add_argument("--scheduler-evidence", type=Path, required=True)
    marker.add_argument("--qacct-audit", type=Path, required=True)
    marker.add_argument("--compute-ceiling", type=Path, required=True)
    marker.add_argument("--scheduler-evidence-root", type=Path, required=True)
    marker.add_argument("--post-job-output", type=Path, required=True)
    marker.add_argument("--capture-output-dir", type=Path, required=True)

    formal_audit = commands.add_parser("audit-formal-smoke")
    formal_audit.add_argument("--manifest", type=Path, required=True)
    formal_audit.add_argument("--task-output-root", type=Path, required=True)
    formal_audit.add_argument("--provisional", type=Path, required=True)
    formal_audit.add_argument("--scheduler-evidence", type=Path, required=True)
    formal_audit.add_argument("--qacct-audit", type=Path, required=True)
    formal_audit.add_argument("--compute-ceiling", type=Path, required=True)
    formal_audit.add_argument("--scheduler-evidence-root", type=Path, required=True)
    formal_audit.add_argument("--post-job", type=Path, required=True)
    formal_audit.add_argument("--readback", type=Path, required=True)
    formal_audit.add_argument("--finalization-capture-dir", type=Path, required=True)
    formal_audit.add_argument("--logs-dir", type=Path, required=True)
    formal_audit.add_argument("--output", type=Path, required=True)

    validate = commands.add_parser("validate-manifest")
    validate.add_argument("--manifest", type=Path, required=True)
    validate.add_argument("--structural-only", action="store_true")

    authorize = commands.add_parser("validate-authorization")
    authorize.add_argument("--manifest", type=Path, required=True)
    authorize.add_argument("--authorization", type=Path, required=True)
    authorize.add_argument("--approved-authorization-hash", required=True)
    authorize.add_argument("--approved-python-bin", type=Path, required=True)
    authorize.add_argument("--authorized-manifest-path", type=Path, required=True)
    authorize.add_argument("--approved-scheduler-user", required=True)

    ceiling = commands.add_parser("validate-compute-ceiling")
    ceiling.add_argument("--manifest", type=Path, required=True)
    ceiling.add_argument("--compute-ceiling", type=Path, required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "describe-task-partitions":
        manifest = load(args.manifest)
        one_slot, shared_two = execution.partition_manifest_task_ids(manifest)
        one_throttle, shared_throttle = execution._partition_throttles(
            manifest, one_slot, shared_two
        )
        print(json.dumps({
            "one_slot_task_ids": one_slot,
            "shared_two_task_ids": shared_two,
            "one_slot_throttle": one_throttle,
            "shared_two_throttle": shared_throttle,
        }, sort_keys=True, separators=(",", ":")))
    elif args.command == "finalize-and-capture-formal-smoke":
        provider, accepted = load_provider(args)
        suites = execution.build_terminal_suites(
            provider, accepted, validate_contents=False
        )
        execution.finalize_and_capture_formal_smoke(
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
            post_job_path=args.post_job_output,
            finalization_capture_dir=args.capture_output_dir,
            qstat_bin="qstat",
            project_root=PROJECT_ROOT,
        )
    elif args.command == "audit-formal-smoke":
        provider, accepted = load_provider(args)
        suites = execution.build_terminal_suites(
            provider, accepted, validate_contents=False
        )
        execution.audit_formal_smoke(
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
            readback_path=args.readback,
            finalization_capture_dir=args.finalization_capture_dir,
            logs_dir=args.logs_dir,
            output_path=args.output,
            project_root=PROJECT_ROOT,
        )
    elif args.command == "diagnose-plan":
        phases = {}
        total_started = time.perf_counter()
        started = time.perf_counter()
        provider, accepted = load_provider(args)
        phases["provider_load"] = time.perf_counter() - started
        started = time.perf_counter()
        suites = execution.build_terminal_suites(
            provider, accepted, validate_contents=False
        )
        phases["suite_reconstruction"] = time.perf_counter() - started
        started = time.perf_counter()
        source = execution.capture_clean_source_identity(PROJECT_ROOT, SOURCE_PATHS)
        phases["source_identity_capture"] = time.perf_counter() - started
        selected = execution._selected_descriptors(args.stage, suites)
        position = args.descriptor_position
        if position is None:
            raw_position = os.environ.get("SGE_TASK_ID")
            if raw_position is None or not raw_position.isdigit():
                raise RuntimeError("descriptor position is absent")
            position = int(raw_position)
        if not 1 <= position <= len(selected):
            raise RuntimeError("descriptor position is outside the frozen selection")
        phases["preparation_total"] = time.perf_counter() - total_started
        diagnostic = execution.create_terminal_plan_diagnostic(
            selected[position - 1],
            provider,
            source_identity_hash=source["identity_hash"],
            include_full_evidence=args.mode == "parity",
            preparation_phase_seconds=phases,
        )
        execution.write_new_json(args.output, diagnostic)
    elif args.command == "freeze-plan-fragment":
        phases = {}
        total_started = time.perf_counter()
        started = time.perf_counter()
        provider, accepted = load_provider(args)
        phases["provider_load"] = time.perf_counter() - started
        started = time.perf_counter()
        suites = execution.build_terminal_suites(
            provider, accepted, validate_contents=False
        )
        phases["suite_reconstruction"] = time.perf_counter() - started
        started = time.perf_counter()
        source = execution.capture_clean_source_identity(PROJECT_ROOT, SOURCE_PATHS)
        phases["source_identity_capture"] = time.perf_counter() - started
        fragment = execution.create_manifest_plan_fragment(
            stage=args.stage,
            shard_index=args.shard_index,
            shard_count=args.shard_count,
            suites=suites,
            provider=provider,
            acceptance_validator=accepted,
            source_identity=source,
            phase_seconds=phases,
        )
        started = time.perf_counter()
        execution.write_new_json(args.output, fragment)
        phases["fragment_serialization"] = time.perf_counter() - started
        phases["command_total"] = time.perf_counter() - total_started
        if args.profile_output is not None:
            execution.write_new_json(
                args.profile_output,
                phase_profile(
                    args.command,
                    phases,
                    stage=args.stage,
                    shard_index=args.shard_index,
                    shard_count=args.shard_count,
                    fragment_hash=fragment["fragment_hash"],
                    source_identity_hash=source["identity_hash"],
                ),
            )
    elif args.command == "merge-plan-fragments":
        phases = {}
        total_started = time.perf_counter()
        started = time.perf_counter()
        provider, accepted = load_provider(args)
        phases["provider_load"] = time.perf_counter() - started
        started = time.perf_counter()
        suites = execution.build_terminal_suites(
            provider, accepted, validate_contents=False
        )
        phases["suite_reconstruction"] = time.perf_counter() - started
        started = time.perf_counter()
        source = execution.capture_clean_source_identity(PROJECT_ROOT, SOURCE_PATHS)
        phases["source_identity_capture"] = time.perf_counter() - started
        started = time.perf_counter()
        ceiling = load(args.compute_ceiling)
        execution._validate_self_hash(ceiling, "report_hash", "compute ceiling report")
        names = tuple(f"fragment_{index:03d}.json" for index in range(1, args.shard_count + 1))
        replicate_a = tuple(load(args.replicate_a_dir / name) for name in names)
        replicate_b = tuple(load(args.replicate_b_dir / name) for name in names)
        phases["fragment_loading"] = time.perf_counter() - started
        started = time.perf_counter()
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
        phases["merge_validation_and_assembly"] = time.perf_counter() - started
        started = time.perf_counter()
        execution.write_new_json(args.output, manifest)
        execution.write_new_json(args.assembly_output, assembly)
        phases["merge_serialization"] = time.perf_counter() - started
        phases["command_total"] = time.perf_counter() - total_started
        if args.profile_output is not None:
            execution.write_new_json(
                args.profile_output,
                phase_profile(
                    args.command,
                    phases,
                    stage=args.stage,
                    shard_count=args.shard_count,
                    manifest_hash=manifest["manifest_hash"],
                    assembly_hash=assembly["assembly_hash"],
                    source_identity_hash=source["identity_hash"],
                ),
            )
    elif args.command == "freeze-manifest":
        provider, accepted = load_provider(args)
        suites = execution.build_terminal_suites(
            provider, accepted, validate_contents=False
        )
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
        suites = execution.build_terminal_suites(
            provider, accepted, validate_contents=False
        )
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
            manifest,
            raw["submissions"],
            evidence_root=args.evidence_root,
            execution_project_root=args.execution_project_root,
            approved_python_bin=args.approved_python_bin,
            authorized_manifest_path=args.manifest,
            scheduler_user=args.scheduler_user,
            run_tag=args.run_tag,
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
            approved_python_bin=args.approved_python_bin,
            authorized_manifest_path=args.authorized_manifest_path,
            approved_scheduler_user=args.approved_scheduler_user,
        )
    elif args.command == "validate-compute-ceiling":
        execution.validate_compute_ceiling_binding(
            load(args.manifest), load(args.compute_ceiling)
        )
    elif args.command == "finalize-post-job":
        provider, accepted = load_provider(args)
        suites = execution.build_terminal_suites(
            provider, accepted, validate_contents=False
        )
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
        suites = execution.build_terminal_suites(
            provider, accepted, validate_contents=False
        )
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
