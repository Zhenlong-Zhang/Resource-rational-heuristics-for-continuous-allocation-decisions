from __future__ import annotations

"""Accepted canonical base-belief provider for terminal validation.

The migration was independently reviewed after generation.  This module is the
post-review trust boundary: only the exact accepted artifact bytes and semantic
identity can produce an authoritative provider.
"""

from dataclasses import replace
import hashlib
from pathlib import Path
from typing import Callable, Tuple

from .terminal_base_migration import (
    AUTHORITATIVE_COMMIT,
    AUTHORITATIVE_TREE,
    DEFAULT_MIGRATION_PATH,
    EXECUTION_APPROVAL_SCHEMA,
    EXECUTION_APPROVAL_STATUS,
    MigrationExecutionApproval,
    load_migration,
    reconstruct_exact_belief,
)
from .terminal_validation_suite import (
    AUTHORITATIVE_PROVIDER_KIND,
    CanonicalBaseProvider,
    CanonicalBaseRecord,
    TerminalHistoryStep,
    canonical_base_provider_failures,
    canonical_base_record_hash,
    canonical_hash,
    make_canonical_base_provider,
)


ACCEPTED_CANONICAL_BASE_ARTIFACT_SHA256 = (
    "59f327defb5e7e931214140ab9e0264fc75b2a6d63a46f4d3c85a18cf0fde997"
)
ACCEPTED_CANONICAL_BASE_SEMANTIC_HASH = (
    "0e453ecc8b1247decb369d7a7587ea744a07e9629606ee706fec24b8cc26381c"
)
ACCEPTED_MIGRATION_EXECUTION_APPROVAL_SHA256 = (
    "55a680a350d366bef12618fa52550eb5150bcf7a409201d81e68b74ab30ff8e0"
)
ACCEPTED_PROVIDER_SOURCE_SCHEMA = "terminal_accepted_canonical_base_source_v1"
DEFAULT_ACCEPTED_CANONICAL_BASE_PATH = DEFAULT_MIGRATION_PATH
ACCEPTED_CANONICAL_PROVIDER_HASH = (
    "01f1dd981e1164eacf10ceb4c44a27548944cd5e32e59537b8755240b6c15897"
)
ACCEPTED_CANONICAL_RECORDS_HASH = (
    "cd84e5fe9d2128a327f9c5ff26148a0895eef77171403e268ea3a26736b9acf7"
)
ACCEPTED_CANONICAL_SOURCE_IDENTITY_HASH = (
    "da68e2c815249374230d41523a2766436d0988d6ea3fd48881e9fd9204e82cea"
)
ACCEPTED_CANONICAL_PROVIDER_ACCEPTANCE_HASH = (
    "a6b1f42a8a83a95f33c871fe52431d30f3b93b0dd8fe470386a5d8b2501c5d20"
)


_ACCEPTED_MIGRATION_TOOL_HASHES = (
    (
        "scripts/export_terminal_base_migration.py",
        "f2c69c4c29b70cbc70ae528dcb2a3070eaf711463a7e2eaadd85551eb1afe3e7",
    ),
    (
        "scripts/hoffman2_terminal_base_migration.job",
        "803af10150da187ac007c265e371a36ae3d5ac43ed738b4ba06288ee57b48584",
    ),
    (
        "src/experiments/terminal_base_migration.py",
        "a25ed54801c7d8fc5300e9d2301038e4f6318de1bf7aed3619f5000c37941353",
    ),
)
_ACCEPTED_RUNTIME_IDENTITY = (
    ("byteorder", "little"),
    ("libc", "glibc|2.17"),
    ("platform_machine", "x86_64"),
    ("platform_release", "3.10.0-1160.108.1.el7.x86_64"),
    ("platform_system", "Linux"),
    ("python_build", "main|Jun  5 2025 13:12:00"),
    ("python_executable", "/u/home/z/zzl/.conda/envs/rr-allocation/bin/python3.11"),
    ("python_implementation", "CPython"),
    ("python_version", "3.11.13"),
)
_ACCEPTED_DEPENDENCY_IDENTITY = (("numpy", "1.26.4"), ("scipy", "1.11.4"))
_ACCEPTED_ALLOWED_UNTRACKED_HASHES = (
    (
        "results/r6_prefeedback_quadrature_7376c5d_v1/"
        "r6_quadrature_diagnostic_manifest.json",
        "9215d3e3823c1f01b070d6a575f214e2bff0f1617262a9445b66a451d02753d2",
    ),
    (
        "scripts/export_terminal_base_migration.py",
        "f2c69c4c29b70cbc70ae528dcb2a3070eaf711463a7e2eaadd85551eb1afe3e7",
    ),
    (
        "src/experiments/terminal_base_migration.py",
        "a25ed54801c7d8fc5300e9d2301038e4f6318de1bf7aed3619f5000c37941353",
    ),
)


def _accepted_execution_approval() -> MigrationExecutionApproval:
    """Reconstruct the execution identity already bound by the accepted bytes."""

    return MigrationExecutionApproval(
        schema=EXECUTION_APPROVAL_SCHEMA,
        approval_status=EXECUTION_APPROVAL_STATUS,
        source_commit=AUTHORITATIVE_COMMIT,
        source_tree_hash=AUTHORITATIVE_TREE,
        original_manifest_file_hash=(
            "9215d3e3823c1f01b070d6a575f214e2bff0f1617262a9445b66a451d02753d2"
        ),
        migration_tool_hashes=_ACCEPTED_MIGRATION_TOOL_HASHES,
        runtime_identity=_ACCEPTED_RUNTIME_IDENTITY,
        dependency_identity=_ACCEPTED_DEPENDENCY_IDENTITY,
        allowed_untracked_hashes=_ACCEPTED_ALLOWED_UNTRACKED_HASHES,
        slots=1,
        array_job=False,
        rerunnable=False,
    )


def accepted_canonical_source_identity_hash(*, records_hash: str) -> str:
    return canonical_hash(
        {
            "schema": ACCEPTED_PROVIDER_SOURCE_SCHEMA,
            "canonical_path": "configs/terminal_base_beliefs_7376c5d_v1.json",
            "artifact_sha256": ACCEPTED_CANONICAL_BASE_ARTIFACT_SHA256,
            "semantic_output_hash": ACCEPTED_CANONICAL_BASE_SEMANTIC_HASH,
            "source_commit": AUTHORITATIVE_COMMIT,
            "source_tree": AUTHORITATIVE_TREE,
            "records_hash": records_hash,
        }
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def accepted_canonical_base_provider(provider: CanonicalBaseProvider) -> bool:
    """Validate one provider against the immutable accepted artifact trust roots."""

    try:
        path = DEFAULT_ACCEPTED_CANONICAL_BASE_PATH.resolve()
        return (
            path.is_file()
            and _sha256_file(path) == ACCEPTED_CANONICAL_BASE_ARTIFACT_SHA256
            and not canonical_base_provider_failures(provider)
            and provider.provider_kind == AUTHORITATIVE_PROVIDER_KIND
            and provider.diagnostic_only is False
            and provider.source_identity_hash
            == ACCEPTED_CANONICAL_SOURCE_IDENTITY_HASH
            and provider.records_hash == ACCEPTED_CANONICAL_RECORDS_HASH
            and provider.provider_hash == ACCEPTED_CANONICAL_PROVIDER_HASH
        )
    except Exception:
        return False


def load_accepted_canonical_base_provider(
    path: Path | None = None,
) -> Tuple[
    CanonicalBaseProvider,
    Callable[[CanonicalBaseProvider], bool],
]:
    """Load the exact Reviewer-accepted canonical migration as a provider."""

    canonical_path = DEFAULT_ACCEPTED_CANONICAL_BASE_PATH.resolve()
    selected_path = canonical_path if path is None else Path(path).resolve()
    if selected_path != canonical_path:
        raise RuntimeError("canonical base provider path differs from the accepted path")
    if not selected_path.is_file():
        raise FileNotFoundError(f"accepted canonical base artifact is absent: {selected_path}")
    observed_sha256 = _sha256_file(selected_path)
    if observed_sha256 != ACCEPTED_CANONICAL_BASE_ARTIFACT_SHA256:
        raise RuntimeError("canonical base artifact byte SHA-256 mismatch")

    migration = load_migration(
        selected_path,
        approved_output_hash=ACCEPTED_CANONICAL_BASE_SEMANTIC_HASH,
        execution_approval=_accepted_execution_approval(),
        approved_execution_approval_file_hash=(
            ACCEPTED_MIGRATION_EXECUTION_APPROVAL_SHA256
        ),
    )
    if migration.output_hash != ACCEPTED_CANONICAL_BASE_SEMANTIC_HASH:
        raise RuntimeError("canonical base artifact semantic hash mismatch")

    records = []
    for expected_case_id, migration_record in enumerate(migration.records):
        if migration_record.case.case_id != expected_case_id:
            raise RuntimeError("canonical migration record order changed")
        prior, belief = reconstruct_exact_belief(migration_record.belief)
        history = tuple(
            TerminalHistoryStep(
                action=(
                    "sample_1"
                    if float.fromhex(step.action_hex) == 1.0
                    else "sample_2"
                ),
                observation=float.fromhex(step.observation_hex),
                cost=float.fromhex(step.cost_hex),
            )
            for step in migration_record.belief.history
        )
        record = CanonicalBaseRecord(
            case_id=migration_record.case.case_id,
            environment=migration_record.case.environment,
            environment_hash=migration_record.case.environment_hash,
            belief_kind=migration_record.case.belief_kind,
            legacy_belief_hash=migration_record.case.belief_hash,
            states=tuple(prior.states),
            prior_weights=tuple(prior.weights),
            posterior_weights=tuple(belief.weights),
            deliberation_time=float(belief.deliberation_time),
            history=history,
            support_hash=prior.support_hash,
            record_hash="",
        )
        records.append(replace(record, record_hash=canonical_base_record_hash(record)))

    if len(records) != 90:
        raise RuntimeError("accepted canonical provider must contain exactly 90 records")
    records_hash = canonical_hash(tuple(record.record_hash for record in records))
    if records_hash != ACCEPTED_CANONICAL_RECORDS_HASH:
        raise RuntimeError("canonical base provider records hash mismatch")
    source_identity_hash = accepted_canonical_source_identity_hash(
        records_hash=records_hash
    )
    if source_identity_hash != ACCEPTED_CANONICAL_SOURCE_IDENTITY_HASH:
        raise RuntimeError("canonical base provider source identity mismatch")
    provider = make_canonical_base_provider(
        records,
        provider_kind=AUTHORITATIVE_PROVIDER_KIND,
        source_identity_hash=source_identity_hash,
        diagnostic_only=False,
    )
    failures = canonical_base_provider_failures(provider)
    if failures:
        raise RuntimeError(
            "accepted canonical provider failed reconstruction: " + ",".join(failures)
        )
    if provider.provider_hash != ACCEPTED_CANONICAL_PROVIDER_HASH:
        raise RuntimeError("canonical base provider hash mismatch")
    if not accepted_canonical_base_provider(provider):
        raise RuntimeError("accepted canonical provider failed its external trust gate")
    return provider, accepted_canonical_base_provider


__all__ = [
    "ACCEPTED_CANONICAL_BASE_ARTIFACT_SHA256",
    "ACCEPTED_CANONICAL_BASE_SEMANTIC_HASH",
    "ACCEPTED_CANONICAL_PROVIDER_ACCEPTANCE_HASH",
    "ACCEPTED_CANONICAL_PROVIDER_HASH",
    "ACCEPTED_CANONICAL_RECORDS_HASH",
    "ACCEPTED_CANONICAL_SOURCE_IDENTITY_HASH",
    "ACCEPTED_MIGRATION_EXECUTION_APPROVAL_SHA256",
    "ACCEPTED_PROVIDER_SOURCE_SCHEMA",
    "DEFAULT_ACCEPTED_CANONICAL_BASE_PATH",
    "accepted_canonical_base_provider",
    "accepted_canonical_source_identity_hash",
    "load_accepted_canonical_base_provider",
]
