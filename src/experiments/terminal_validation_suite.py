from __future__ import annotations

"""Deterministic identities and belief descriptors for terminal numerical evidence."""

from dataclasses import dataclass, fields, is_dataclass, replace
from functools import lru_cache
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

from ..mdp.finite_support import (
    FiniteSupportAtom,
    FiniteSupportBeliefState,
    FiniteSupportMetaMDP,
    FiniteSupportPrior,
)
from ..solvers.terminal import production_terminal_numerical_method_config_hash
from ..solvers.terminal_reference import (
    terminal_reference_a_numerical_method_config_hash,
)
from ..solvers.terminal_reference_agreement import (
    terminal_reference_agreement_numerical_method_config_hash,
)
from ..solvers.terminal_reference_b import (
    terminal_reference_b_numerical_method_config_hash,
)
from .r6_prefeedback_positive_need import (
    DEFAULT_SPEC_PATH,
    PositiveNeedEnvironment,
    _belief_hash as legacy_belief_hash,
    _numerical_belief,
    build_development_environments,
    build_numerical_validation_cases,
    load_positive_need_spec,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_NUMERICAL_METHOD_CONFIG_PATH = (
    PROJECT_ROOT / "configs" / "terminal_evidence_numerical_method_v2.json"
)
LEGACY_SPEC_HASH = "9f3152419a6c3ef46cd33baefe6a13f00dc80004025f1fb2f1d2b08f9e307e10"
LEGACY_NUMERICAL_CASE_HASH = (
    "90354e48a36283225b221360c91b07c21ad369fb7f6a8f8191d7f5ed85ef132b"
)
SCIENTIFIC_PROJECTION_SCHEMA = "terminal_scientific_projection_v1"
NUMERICAL_METHOD_CONFIG_NAME = "terminal_evidence_numerical_method_v2"
DESCRIPTOR_SCHEMA = "terminal_validation_descriptor_v1"
MANIFEST_SCHEMA = "terminal_validation_suite_manifest_v1"
BASE_SUITE_VERSION = "terminal_base_90_v1"
ONE_STEP_SUITE_VERSION = "terminal_one_step_z_v1"
REACHABLE_CORE_SUITE_VERSION = "terminal_reachable_core_v1"
FROZEN_SAMPLE_COSTS = (0.02, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0)
FROZEN_Z_OFFSETS = (-8, -6, -4, -2, -1, 0, 1, 2, 4, 6, 8)
FROZEN_HISTORY_OFFSET_CYCLE = (-2, -1, 0, 1, 2, 0)
SUITE_HARD_CAPS = (("base", 90), ("one_step", 35_640), ("reachable_core", 648))
REFERENCE_B_PRESPECIFIED_ONE_STEP_COUNT = 180
ORIENTATION_VOCABULARY = ("-1", "+1", "symmetric", "balanced")
SUITE_CLASS_ORDER = ("base", "one_step", "reachable_core")
LOCAL_DIAGNOSTIC_PROVIDER_KIND = "local_deterministic_reconstruction_diagnostic_only"
AUTHORITATIVE_PROVIDER_KIND = "independently_accepted_migration"
FROZEN_SCIENTIFIC_SPEC_HASH = (
    "bd1f3a549c4f864efa65a0b949037117d5224d7b57f293e54e2eeafc88169dc2"
)
FROZEN_NUMERICAL_METHOD_CONFIG_HASH = (
    "182bdca5df90b4b241d3ce4a2e4d689f7822bd7b281627880532c3be50685a43"
)
FROZEN_CONSTRUCTION_HASHES = {
    "base": "b7745e7b323a078c4e03bce540b6676890cef9aa1a5ec26b943c185973760063",
    "one_step": "be88282012d6f1b9ffdfcb12583ea0148f706bb8cd9d79a01f776e2625031f06",
    "reachable_core": "05b6570f39078dfc8e63f16a00cc4872e7ed51176ea2e6f09f1a55de67d71866",
}
FROZEN_LEGACY_BELIEF_HASH_OVERRIDES = {
    24: "f8bd999fb547310e7383a4db816a5a43e813ddb5d56d831f53d46d2f0a83906f",
    29: "f8bd999fb547310e7383a4db816a5a43e813ddb5d56d831f53d46d2f0a83906f",
    83: "8ca8c15263c58e15be8208376ec7ef11b2642aa1fb4ca13745a2923d75a8f795",
    88: "8ca8c15263c58e15be8208376ec7ef11b2642aa1fb4ca13745a2923d75a8f795",
}
BASE_CONSTRUCTION_RULE = "unchanged_original_r6_numerical_case"
ONE_STEP_CONSTRUCTION_RULE = "component_need_plus_z_sigma_actual_time"
REACHABLE_CONSTRUCTION_RULE = "median_total_largest_gap_frozen_history"
BASE_MANIFEST_RULE = "unchanged_original_90_base_beliefs"
ONE_STEP_MANIFEST_RULE = "90x2x18x11_component_z_offset_posteriors"
REACHABLE_MANIFEST_RULE = "72_environments_x_9_frozen_reachable_histories"
BASE_PARTITIONS = (
    ("uniform_prior", 18),
    ("person1_predictive_mean", 18),
    ("both_predictive_means", 18),
    ("person1_minimum_support", 18),
    ("person1_maximum_support", 18),
)
ONE_STEP_PARTITIONS = (("sample_1", 17_820), ("sample_2", 17_820))
REACHABLE_PARTITIONS = (
    ("initial_symmetric", 72),
    ("one_step_orientation_-1", 72),
    ("one_step_orientation_+1", 72),
    ("balanced_depth_6", 72),
    ("concentrated_depth_6_-1", 72),
    ("concentrated_depth_6_+1", 72),
    ("balanced_late_feasible", 72),
    ("concentrated_late_feasible_-1", 72),
    ("concentrated_late_feasible_+1", 72),
)


@dataclass(frozen=True)
class FrozenSuiteInvariant:
    suite_class: str
    suite_version: str
    manifest_rule: str
    descriptor_rule: str
    hard_cap: int
    component_validation_only: bool
    environment_selection_eligible: bool
    ordered_construction_hash: str
    partitions: Tuple[Tuple[str, int], ...]
    allowed_orientations: Tuple[str, ...]


FROZEN_SUITE_INVARIANTS = {
    "base": FrozenSuiteInvariant(
        "base",
        BASE_SUITE_VERSION,
        BASE_MANIFEST_RULE,
        BASE_CONSTRUCTION_RULE,
        90,
        False,
        False,
        FROZEN_CONSTRUCTION_HASHES["base"],
        BASE_PARTITIONS,
        ("-1", "+1", "symmetric"),
    ),
    "one_step": FrozenSuiteInvariant(
        "one_step",
        ONE_STEP_SUITE_VERSION,
        ONE_STEP_MANIFEST_RULE,
        ONE_STEP_CONSTRUCTION_RULE,
        35_640,
        True,
        False,
        FROZEN_CONSTRUCTION_HASHES["one_step"],
        ONE_STEP_PARTITIONS,
        ("-1", "+1"),
    ),
    "reachable_core": FrozenSuiteInvariant(
        "reachable_core",
        REACHABLE_CORE_SUITE_VERSION,
        REACHABLE_MANIFEST_RULE,
        REACHABLE_CONSTRUCTION_RULE,
        648,
        True,
        False,
        FROZEN_CONSTRUCTION_HASHES["reachable_core"],
        REACHABLE_PARTITIONS,
        ORIENTATION_VOCABULARY,
    ),
}


def _canonical_value(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("canonical evidence cannot contain a non-finite float")
        return {"float_hex": value.hex()}
    if is_dataclass(value):
        return {field.name: _canonical_value(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise TypeError("canonical evidence mapping keys must be strings")
        return {key: _canonical_value(value[key]) for key in sorted(value)}
    if isinstance(value, (tuple, list)):
        return tuple(_canonical_value(item) for item in value)
    raise TypeError(f"unsupported canonical evidence type: {type(value).__name__}")


def canonical_hash(value: Any) -> str:
    payload = json.dumps(
        _canonical_value(value), sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _legacy_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_frozen_r6_spec() -> Dict[str, object]:
    if _file_hash(DEFAULT_SPEC_PATH) != LEGACY_SPEC_HASH:
        raise RuntimeError("the immutable original R6 spec hash changed")
    return load_positive_need_spec(DEFAULT_SPEC_PATH)


def load_frozen_r6_cases(spec: Optional[Mapping[str, object]] = None) -> Tuple[Dict[str, object], ...]:
    generated = [dict(case) for case in build_numerical_validation_cases(spec)]
    by_id = {int(case["case_id"]): case for case in generated}
    if sorted(by_id) != list(range(90)) or len(generated) != 90:
        raise RuntimeError("the original R6 case IDs must be exactly 0..89")
    for case_id, belief_hash in FROZEN_LEGACY_BELIEF_HASH_OVERRIDES.items():
        by_id[case_id]["belief_hash"] = belief_hash
    frozen = tuple(by_id[index] for index in range(90))
    if _legacy_hash(frozen) != LEGACY_NUMERICAL_CASE_HASH:
        raise RuntimeError("the immutable original R6 case hash changed")
    return frozen


def build_terminal_scientific_projection(spec: Optional[Mapping[str, object]] = None) -> Dict[str, object]:
    frozen_spec = dict(spec or load_frozen_r6_spec())
    cases = load_frozen_r6_cases(frozen_spec)
    environments = build_development_environments(frozen_spec)
    numerical = dict(frozen_spec["numerical_settings"])
    grid = dict(frozen_spec["environment_grid"])
    support_hashes = {}
    for environment in environments:
        support_hashes.setdefault(environment.gap_class, environment.prior.support_hash)
    return {
        "schema": SCIENTIFIC_PROJECTION_SCHEMA,
        "ordered_sections": (
            "legacy_source", "generator_and_prior", "environment", "actions_and_time",
            "case_construction", "decision_semantics", "scientific_thresholds",
        ),
        "legacy_source": {
            "legacy_full_file_spec_hash": LEGACY_SPEC_HASH,
            "legacy_numerical_case_hash": LEGACY_NUMERICAL_CASE_HASH,
        },
        "generator_and_prior": {
            "generator": frozen_spec["generator"],
            "support_hash_by_gap_class": support_hashes,
        },
        "environment": {
            "scientific_grid": {key: grid[key] for key in (
                "sigma_sample", "sample_time_cost", "total_time", "terminate_cost",
                "max_meta_samples", "learning_per_unit_of_tutoring",
                "delta_learning_per_unit_tutoring", "utility_exponent", "lambda_shortfall",
                "prior_sample_count_1", "prior_sample_count_2",
            )},
            "ordered_environment_identities": tuple({
                "name": env.name, "gap_class": env.gap_class,
                "sigma_sample": env.sigma_sample, "sample_time_cost": env.sample_time_cost,
                "support_hash": env.prior.support_hash,
                "legacy_environment_hash": env.environment_hash,
            } for env in environments),
        },
        "actions_and_time": {
            "actions": ("terminate", "sample_1", "sample_2"),
            "sampling_cost_semantics": "time_only",
            "sample_time_is_environment_specific": True,
            "termination_time": grid["terminate_cost"],
        },
        "case_construction": {
            "ordered_original_cases": cases,
            "belief_kinds": (
                "uniform_prior", "person1_predictive_mean", "both_predictive_means",
                "person1_minimum_support", "person1_maximum_support",
            ),
            "base_sample_time_costs": (0.02, 8.0),
        },
        "decision_semantics": {
            "terminal_value_tie_scale": "1e-12*max(1,abs(best_value))",
            "terminal_canonical_rule": "closest_to_half_then_lower",
            "action_canonical_rule": "terminate_then_sample_1_then_sample_2",
        },
        "scientific_thresholds": {
            "allocation_tolerance": numerical["allocation_tolerance"],
            "action_tie_tolerance": numerical["action_tie_tolerance"],
            "action_value_convergence_tolerance": numerical["action_value_convergence_tolerance"],
            "allocation_convergence_tolerance": numerical["allocation_convergence_tolerance"],
            "selection": frozen_spec["selection"],
            "strict_target_thresholds": frozen_spec["strict_target_thresholds"],
            "strict_control_thresholds": frozen_spec["strict_control_thresholds"],
        },
    }


def terminal_scientific_spec_hash(spec: Optional[Mapping[str, object]] = None) -> str:
    return canonical_hash(build_terminal_scientific_projection(spec))


def source_derived_terminal_method_hashes() -> Dict[str, str]:
    """Recompute nested method identities from the implementation constants."""

    return {
        "production_terminal": production_terminal_numerical_method_config_hash(),
        "reference_a": terminal_reference_a_numerical_method_config_hash(),
        "reference_b": terminal_reference_b_numerical_method_config_hash(),
        "agreement": terminal_reference_agreement_numerical_method_config_hash(),
    }


def expected_terminal_numerical_method_config() -> Dict[str, object]:
    """Build the complete v2 global numerical identity from frozen source controls."""

    return {
        "schema_version": 2,
        "config_name": NUMERICAL_METHOD_CONFIG_NAME,
        "scope": "terminal_validation_only",
        "scientific_fields_permitted": False,
        "source_derived_method_hash_order": (
            "production_terminal",
            "reference_a",
            "reference_b",
            "agreement",
        ),
        "source_derived_method_hashes": source_derived_terminal_method_hashes(),
        "suite_construction": {
            "descriptor_schema": DESCRIPTOR_SCHEMA,
            "manifest_schema": MANIFEST_SCHEMA,
            "partition_schema": "terminal_suite_partition_v1",
            "suite_class_order": SUITE_CLASS_ORDER,
            "suite_versions": {
                "base": BASE_SUITE_VERSION,
                "one_step": ONE_STEP_SUITE_VERSION,
                "reachable_core": REACHABLE_CORE_SUITE_VERSION,
                "boundary": "terminal_boundary_status_v1",
            },
            "construction_rules": {
                key: {
                    "manifest": value.manifest_rule,
                    "descriptor": value.descriptor_rule,
                }
                for key, value in FROZEN_SUITE_INVARIANTS.items()
            },
            "hard_caps": dict(SUITE_HARD_CAPS),
            "frozen_ordered_construction_hashes": dict(FROZEN_CONSTRUCTION_HASHES),
            "partition_order_and_counts": {
                key: value.partitions
                for key, value in FROZEN_SUITE_INVARIANTS.items()
            },
            "manifest_selection_flags": {
                key: {
                    "component_validation_only": value.component_validation_only,
                    "environment_selection_eligible": value.environment_selection_eligible,
                }
                for key, value in FROZEN_SUITE_INVARIANTS.items()
            },
            "descriptor_selection_flags": "must_equal_manifest_flags_on_every_row",
            "descriptor_index_rule": "exact_contiguous_zero_based_manifest_order",
            "partition_order_rule": "frozen_order_then_ordered_descriptor_hash",
            "orientation_vocabulary": ORIENTATION_VOCABULARY,
            "base_orientation_rule": (
                "exact_sign_of_posterior_mean_1_minus_mean_2_with_zero_symmetric"
            ),
            "orientation_semantics": {
                "-1": "latent_or_posterior_need_orientation_toward_recipient_2",
                "+1": "latent_or_posterior_need_orientation_toward_recipient_1",
                "symmetric": "recipient_symmetric_or_equal_posterior_mean",
                "balanced": "alternating_recipient_sampling_history",
            },
            "z_offsets": FROZEN_Z_OFFSETS,
            "history_offset_cycle": FROZEN_HISTORY_OFFSET_CYCLE,
            "deduplication": "disabled_for_descriptor_foundation",
            "late_depth_cap": 19,
            "boundary_grid_points": 129,
            "boundary_bisection_cap": 40,
            "boundary_proximity_tolerance": 0.0005,
        },
        "reference_b_tiers": {
            "all_base_beliefs": True,
            "first_component_z0_per_base_action": True,
            "structural_symmetry": True,
            "reference_a_near_tie_separation": 1e-6,
            "reference_a_unresolved": True,
            "production_disagreement_or_failure": True,
            "tier_order": (
                "all_base_beliefs",
                "first_component_z0_per_base_action",
                "structural_symmetry",
                "reference_a_near_tie_separation",
                "reference_a_unresolved",
                "production_disagreement_or_failure",
            ),
        },
        "evidence": {
            "terminal_row_schema": "terminal_evidence_row_v1",
            "certificate_sidecar_schema": "terminal_certificate_sidecar_v1",
            "suite_descriptor_schema": DESCRIPTOR_SCHEMA,
            "suite_manifest_schema": MANIFEST_SCHEMA,
            "compression": "gzip_mtime_0",
            "hash": "sha256_canonical_json_float_hex_v1",
            "row_order": "suite_class_then_descriptor_index_then_method_priority",
            "method_priority": (
                "production_terminal",
                "reference_a",
                "reference_b",
                "agreement",
            ),
            "sidecar_index_order": "relative_path_lexicographic",
            "sidecar_required_metadata": (
                "sha256",
                "byte_count",
                "schema_version",
                "logical_record_hash",
            ),
        },
        "integration": {
            "status": "not_implemented_in_terminal_evidence_foundation",
            "numerical_identity_requires_new_reviewed_config_version": True,
        },
    }


def validate_terminal_numerical_method_config(
    config: Mapping[str, object],
) -> Tuple[str, ...]:
    expected = expected_terminal_numerical_method_config()
    failures = []
    for key in expected:
        if key not in config:
            failures.append(f"missing:{key}")
        elif _canonical_value(config[key]) != _canonical_value(expected[key]):
            failures.append(f"mismatch:{key}")
    for key in config:
        if key not in expected:
            failures.append(f"unexpected:{key}")
    return tuple(failures)


def load_terminal_numerical_method_config(
    path: Path = DEFAULT_NUMERICAL_METHOD_CONFIG_PATH,
) -> Dict[str, object]:
    config = json.loads(path.read_text(encoding="utf-8"))
    failures = validate_terminal_numerical_method_config(config)
    if failures:
        raise ValueError(
            "terminal numerical-method config differs from source controls: "
            + ",".join(failures)
        )
    return config


def terminal_numerical_method_config_hash(config: Optional[Mapping[str, object]] = None) -> str:
    return canonical_hash(config or load_terminal_numerical_method_config())


@dataclass(frozen=True)
class TerminalValidationIdentities:
    legacy_spec_hash: str
    legacy_numerical_case_hash: str
    scientific_spec_hash: str
    numerical_method_config_name: str
    numerical_method_config_hash: str


def load_terminal_validation_identities() -> TerminalValidationIdentities:
    spec = load_frozen_r6_spec()
    load_frozen_r6_cases(spec)
    identities = TerminalValidationIdentities(
        LEGACY_SPEC_HASH, LEGACY_NUMERICAL_CASE_HASH,
        terminal_scientific_spec_hash(spec), NUMERICAL_METHOD_CONFIG_NAME,
        terminal_numerical_method_config_hash(),
    )
    if identities.scientific_spec_hash != FROZEN_SCIENTIFIC_SPEC_HASH:
        raise RuntimeError("scientific projection changed")
    if identities.numerical_method_config_hash != FROZEN_NUMERICAL_METHOD_CONFIG_HASH:
        raise RuntimeError("numerical config changed")
    return identities


@dataclass(frozen=True)
class TerminalHistoryStep:
    action: str
    observation: float
    cost: float


@dataclass(frozen=True)
class CanonicalBaseRecord:
    case_id: int
    environment: str
    environment_hash: str
    belief_kind: str
    legacy_belief_hash: str
    states: Tuple[FiniteSupportAtom, ...]
    prior_weights: Tuple[float, ...]
    posterior_weights: Tuple[float, ...]
    deliberation_time: float
    history: Tuple[TerminalHistoryStep, ...]
    support_hash: str
    record_hash: str


@dataclass(frozen=True)
class CanonicalBaseProvider:
    provider_kind: str
    source_identity_hash: str
    diagnostic_only: bool
    records: Tuple[CanonicalBaseRecord, ...]
    records_hash: str
    provider_hash: str


@dataclass(frozen=True)
class TerminalDescriptorReconstruction:
    expected_descriptor: Optional["TerminalValidationDescriptor"]
    failures: Tuple[str, ...]


@dataclass(frozen=True)
class TerminalSuiteValidationResult:
    failures: Tuple[str, ...]
    validation_status: str
    provider_hash: str
    authoritative_source_accepted: bool


@dataclass(frozen=True)
class TerminalValidationDescriptor:
    schema: str
    suite_class: str
    suite_version: str
    descriptor_index: int
    component_validation_only: bool
    environment_selection_eligible: bool
    legacy_spec_hash: str
    legacy_numerical_case_hash: str
    scientific_spec_hash: str
    numerical_method_config_hash: str
    source_case_id: Optional[int]
    environment: str
    environment_hash: str
    support_hash: str
    sigma_sample: float
    sample_time_cost: float
    profile: str
    orientation: str
    depth: int
    deliberation_time: float
    remaining_time_after_termination: float
    action_sequence: Tuple[str, ...]
    offset_sequence: Tuple[int, ...]
    history: Tuple[TerminalHistoryStep, ...]
    history_hash: str
    posterior_weight_hash: str
    canonical_belief_hash: str
    legacy_belief_hash: Optional[str]
    local_legacy_belief_hash: Optional[str]
    legacy_reconstruction_matches: Optional[bool]
    component_index: Optional[int]
    z_offset: Optional[int]
    reference_b_prespecified: bool
    construction_rule: str
    construction_hash: str
    descriptor_hash: str


@dataclass(frozen=True)
class TerminalSuitePartition:
    name: str
    count: int
    ordered_descriptor_hash: str


@dataclass(frozen=True)
class TerminalValidationSuiteManifest:
    schema: str
    suite_class: str
    suite_version: str
    component_validation_only: bool
    environment_selection_eligible: bool
    legacy_spec_hash: str
    legacy_numerical_case_hash: str
    scientific_spec_hash: str
    numerical_method_config_hash: str
    construction_rule: str
    hard_cap: int
    pre_dedup_count: int
    post_dedup_count: int
    deduplication_applied: bool
    ordered_construction_hash: str
    ordered_descriptor_hash: str
    partitions: Tuple[TerminalSuitePartition, ...]
    manifest_hash: str


@dataclass(frozen=True)
class TerminalValidationSuite:
    manifest: TerminalValidationSuiteManifest
    descriptors: Tuple[TerminalValidationDescriptor, ...]


def _history(belief: FiniteSupportBeliefState) -> Tuple[TerminalHistoryStep, ...]:
    result = []
    for step in belief.history:
        action = "sample_1" if float(step["action"]) == 1.0 else "sample_2"
        result.append(TerminalHistoryStep(action, float(step["observation"]), float(step["cost"])))
    return tuple(result)


def canonical_base_record_hash(record: CanonicalBaseRecord) -> str:
    return canonical_hash(replace(record, record_hash=""))


def canonical_base_provider_hash(provider: CanonicalBaseProvider) -> str:
    return canonical_hash(replace(provider, provider_hash=""))


def _make_canonical_base_record(
    case: Mapping[str, object],
    environment: PositiveNeedEnvironment,
    belief: FiniteSupportBeliefState,
) -> CanonicalBaseRecord:
    record = CanonicalBaseRecord(
        case_id=int(case["case_id"]),
        environment=str(case["environment"]),
        environment_hash=str(case["environment_hash"]),
        belief_kind=str(case["belief_kind"]),
        legacy_belief_hash=str(case["belief_hash"]),
        states=tuple(belief.states),
        prior_weights=tuple(float(value) for value in environment.prior.weights),
        posterior_weights=tuple(float(value) for value in belief.weights),
        deliberation_time=float(belief.deliberation_time),
        history=_history(belief),
        support_hash=str(environment.prior.support_hash),
        record_hash="",
    )
    return replace(record, record_hash=canonical_base_record_hash(record))


def make_canonical_base_provider(
    records: Sequence[CanonicalBaseRecord],
    *,
    provider_kind: str,
    source_identity_hash: str,
    diagnostic_only: bool,
) -> CanonicalBaseProvider:
    ordered = tuple(records)
    records_hash = canonical_hash(tuple(record.record_hash for record in ordered))
    provider = CanonicalBaseProvider(
        provider_kind=str(provider_kind),
        source_identity_hash=str(source_identity_hash),
        diagnostic_only=bool(diagnostic_only),
        records=ordered,
        records_hash=records_hash,
        provider_hash="",
    )
    return replace(provider, provider_hash=canonical_base_provider_hash(provider))


def reconstruct_canonical_base_record(
    record: CanonicalBaseRecord,
) -> Tuple[FiniteSupportPrior, FiniteSupportBeliefState]:
    if canonical_base_record_hash(record) != record.record_hash:
        raise ValueError("canonical base record hash mismatch")
    prior = FiniteSupportPrior(record.states, record.prior_weights)
    object.__setattr__(prior, "weights", tuple(record.prior_weights))
    if prior.support_hash != record.support_hash:
        raise ValueError("canonical base support hash mismatch")
    history = [
        {
            "action": 1.0 if step.action == "sample_1" else 2.0,
            "observation": float(step.observation),
            "cost": float(step.cost),
        }
        for step in record.history
    ]
    if any(step.action not in ("sample_1", "sample_2") for step in record.history):
        raise ValueError("canonical base history action mismatch")
    belief = FiniteSupportBeliefState(
        record.states,
        record.posterior_weights,
        deliberation_time=float(record.deliberation_time),
        history=history,
    )
    belief.weights = tuple(record.posterior_weights)
    return prior, belief


def canonical_base_provider_failures(
    provider: CanonicalBaseProvider,
) -> Tuple[str, ...]:
    failures: List[str] = []
    if provider.provider_kind not in (
        LOCAL_DIAGNOSTIC_PROVIDER_KIND,
        AUTHORITATIVE_PROVIDER_KIND,
    ):
        failures.append("provider_kind")
    if (
        provider.provider_kind == LOCAL_DIAGNOSTIC_PROVIDER_KIND
        and not provider.diagnostic_only
    ):
        failures.append("local_provider_must_be_diagnostic")
    if (
        provider.provider_kind == AUTHORITATIVE_PROVIDER_KIND
        and provider.diagnostic_only
    ):
        failures.append("authoritative_provider_cannot_be_diagnostic")
    if not isinstance(provider.source_identity_hash, str) or (
        len(provider.source_identity_hash) != 64
        or any(character not in "0123456789abcdef" for character in provider.source_identity_hash)
    ):
        failures.append("provider_source_identity_hash")
    if canonical_base_provider_hash(provider) != provider.provider_hash:
        failures.append("provider_hash")
    if provider.records_hash != canonical_hash(
        tuple(record.record_hash for record in provider.records)
    ):
        failures.append("provider_records_hash")
    if len(provider.records) != 90:
        failures.append("provider_record_count")
    if tuple(record.case_id for record in provider.records) != tuple(range(90)):
        failures.append("provider_case_order")
    cases = load_frozen_r6_cases()
    environments = {
        environment.name: environment
        for environment in build_development_environments(load_frozen_r6_spec())
    }
    for index, record in enumerate(provider.records):
        if index >= len(cases):
            break
        if not isinstance(record.record_hash, str) or (
            len(record.record_hash) != 64
            or any(character not in "0123456789abcdef" for character in record.record_hash)
        ):
            _append_failure(failures, "provider_record_hash_format")
        case = cases[index]
        expected_case = (
            int(case["case_id"]),
            str(case["environment"]),
            str(case["environment_hash"]),
            str(case["belief_kind"]),
            str(case["belief_hash"]),
        )
        actual_case = (
            record.case_id,
            record.environment,
            record.environment_hash,
            record.belief_kind,
            record.legacy_belief_hash,
        )
        if actual_case != expected_case:
            _append_failure(failures, "provider_case_descriptor")
        try:
            prior, belief = reconstruct_canonical_base_record(record)
        except (TypeError, ValueError):
            _append_failure(failures, "provider_record_reconstruction")
            continue
        environment = environments.get(record.environment)
        if environment is None or environment.environment_hash != record.environment_hash:
            _append_failure(failures, "provider_environment")
            continue
        if prior.support_hash != record.support_hash:
            _append_failure(failures, "provider_support")
        if tuple(belief.states) != tuple(prior.states):
            _append_failure(failures, "provider_belief_support")
    return tuple(failures)


@lru_cache(maxsize=1)
def build_local_diagnostic_base_provider() -> CanonicalBaseProvider:
    """Rebuild local cases for diagnostics without claiming authoritative identity."""

    spec = load_frozen_r6_spec()
    environments = {
        environment.name: environment
        for environment in build_development_environments(spec)
    }
    records = []
    for case in load_frozen_r6_cases(spec):
        environment = environments[str(case["environment"])]
        belief = _numerical_belief(environment, str(case["belief_kind"]))
        records.append(_make_canonical_base_record(case, environment, belief))
    provider = make_canonical_base_provider(
        records,
        provider_kind=LOCAL_DIAGNOSTIC_PROVIDER_KIND,
        source_identity_hash=canonical_hash(
            {
                "provider_kind": LOCAL_DIAGNOSTIC_PROVIDER_KIND,
                "legacy_spec_hash": LEGACY_SPEC_HASH,
                "legacy_case_hash": LEGACY_NUMERICAL_CASE_HASH,
                "generator": "current_platform_deterministic_replay",
            }
        ),
        diagnostic_only=True,
    )
    if canonical_base_provider_failures(provider):
        raise RuntimeError("local diagnostic base provider failed self-validation")
    return provider


def terminal_validation_descriptor_hash(descriptor: TerminalValidationDescriptor) -> str:
    return canonical_hash(replace(descriptor, descriptor_hash=""))


def _make_descriptor(*, suite_class: str, suite_version: str, descriptor_index: int,
                     component_validation_only: bool, identities: TerminalValidationIdentities,
                     environment: PositiveNeedEnvironment, mdp: FiniteSupportMetaMDP,
                     belief: FiniteSupportBeliefState, source_case_id: Optional[int],
                     profile: str, orientation: str, action_sequence: Sequence[str],
                     offset_sequence: Sequence[int], construction_rule: str,
                     construction_payload: Mapping[str, object], legacy_hash: Optional[str] = None,
                     component_index: Optional[int] = None, z_offset: Optional[int] = None,
                     reference_b_prespecified: bool = False) -> TerminalValidationDescriptor:
    history = _history(belief)
    history_hash = canonical_hash(history)
    local_hash = legacy_belief_hash(belief) if legacy_hash is not None else None
    descriptor = TerminalValidationDescriptor(
        DESCRIPTOR_SCHEMA, suite_class, suite_version, descriptor_index,
        component_validation_only, False, identities.legacy_spec_hash,
        identities.legacy_numerical_case_hash, identities.scientific_spec_hash,
        identities.numerical_method_config_hash, source_case_id, environment.name,
        environment.environment_hash, mdp.prior.support_hash, float(environment.sigma_sample),
        float(environment.sample_time_cost), profile, orientation, len(history),
        float(belief.deliberation_time), float(mdp.remaining_time_after_termination(belief)),
        tuple(action_sequence), tuple(offset_sequence), history, history_hash,
        canonical_hash(tuple(belief.weights)),
        canonical_hash({"support_hash": mdp.prior.support_hash,
                        "posterior_weights": tuple(belief.weights),
                        "deliberation_time": belief.deliberation_time,
                        "history_hash": history_hash}),
        legacy_hash, local_hash, None if legacy_hash is None else local_hash == legacy_hash,
        component_index, z_offset, reference_b_prespecified, construction_rule,
        canonical_hash({"suite_version": suite_version,
                        "scientific_spec_hash": identities.scientific_spec_hash,
                        "construction": construction_payload}), "",
    )
    return replace(descriptor, descriptor_hash=terminal_validation_descriptor_hash(descriptor))


def _partition(descriptors, names):
    return tuple(TerminalSuitePartition(
        name, sum(item.profile == name for item in descriptors),
        canonical_hash(tuple(item.descriptor_hash for item in descriptors if item.profile == name))
    ) for name in names)


def terminal_validation_manifest_hash(manifest: TerminalValidationSuiteManifest) -> str:
    return canonical_hash(replace(manifest, manifest_hash=""))


def _make_suite(suite_class, version, component_only, rule, descriptors, partitions, identities):
    hard_cap = dict(SUITE_HARD_CAPS)[suite_class]
    if len(descriptors) != hard_cap:
        raise RuntimeError("descriptor count differs from hard cap")
    construction_hash = canonical_hash(tuple(item.construction_hash for item in descriptors))
    if construction_hash != FROZEN_CONSTRUCTION_HASHES[suite_class]:
        raise RuntimeError(
            "suite construction changed: "
            f"{suite_class}={construction_hash}"
        )
    manifest = TerminalValidationSuiteManifest(
        MANIFEST_SCHEMA, suite_class, version, component_only, False,
        identities.legacy_spec_hash, identities.legacy_numerical_case_hash,
        identities.scientific_spec_hash, identities.numerical_method_config_hash,
        rule, hard_cap, hard_cap, hard_cap, False, construction_hash,
        canonical_hash(tuple(item.descriptor_hash for item in descriptors)),
        _partition(descriptors, partitions), "",
    )
    manifest = replace(manifest, manifest_hash=terminal_validation_manifest_hash(manifest))
    return TerminalValidationSuite(manifest, tuple(descriptors))


def _runtime_base_cases(
    base_provider: Optional[CanonicalBaseProvider] = None,
):
    provider = base_provider or build_local_diagnostic_base_provider()
    provider_failures = canonical_base_provider_failures(provider)
    if provider_failures:
        raise RuntimeError(
            "canonical base provider failed: " + ",".join(provider_failures)
        )
    environments = {
        env.name: env
        for env in build_development_environments(load_frozen_r6_spec())
    }
    result = []
    for record in provider.records:
        env = environments[record.environment]
        prior, belief = reconstruct_canonical_base_record(record)
        mdp = FiniteSupportMetaMDP(env.config, prior)
        case = {
            "case_id": record.case_id,
            "environment": record.environment,
            "environment_hash": record.environment_hash,
            "belief_kind": record.belief_kind,
            "belief_hash": record.legacy_belief_hash,
        }
        result.append((case, env, mdp, belief))
    return result


def _posterior_mean_orientation(belief: FiniteSupportBeliefState) -> str:
    if belief.mean_1 == belief.mean_2:
        return "symmetric"
    return "+1" if belief.mean_1 > belief.mean_2 else "-1"


def build_terminal_base_suite(
    identities: Optional[TerminalValidationIdentities] = None,
    base_provider: Optional[CanonicalBaseProvider] = None,
):
    identities = identities or load_terminal_validation_identities()
    descriptors = []
    for case, env, mdp, belief in _runtime_base_cases(base_provider):
        descriptors.append(_make_descriptor(
            suite_class="base", suite_version=BASE_SUITE_VERSION,
            descriptor_index=len(descriptors), component_validation_only=False,
            identities=identities, environment=env, mdp=mdp, belief=belief,
            source_case_id=int(case["case_id"]), profile=str(case["belief_kind"]),
            orientation=_posterior_mean_orientation(belief),
            action_sequence=(), offset_sequence=(),
            construction_rule=BASE_CONSTRUCTION_RULE,
            construction_payload={"legacy_case": case}, legacy_hash=str(case["belief_hash"])))
    return _make_suite("base", BASE_SUITE_VERSION, False,
                       BASE_MANIFEST_RULE, descriptors,
                       ("uniform_prior", "person1_predictive_mean", "both_predictive_means",
                        "person1_minimum_support", "person1_maximum_support"), identities)


def build_terminal_one_step_suite(
    identities: Optional[TerminalValidationIdentities] = None,
    base_provider: Optional[CanonicalBaseProvider] = None,
):
    identities = identities or load_terminal_validation_identities()
    descriptors = []
    for case, env, mdp, belief in _runtime_base_cases(base_provider):
        for action in ("sample_1", "sample_2"):
            for component_index, atom in enumerate(belief.states):
                need = mdp._need_for_action(atom, action)
                for z in FROZEN_Z_OFFSETS:
                    posterior = mdp.posterior_transition(
                        belief, action, need + z * mdp.config.sigma_sample,
                        advance_time=True, record=True)
                    descriptors.append(_make_descriptor(
                        suite_class="one_step", suite_version=ONE_STEP_SUITE_VERSION,
                        descriptor_index=len(descriptors), component_validation_only=True,
                        identities=identities, environment=env, mdp=mdp, belief=posterior,
                        source_case_id=int(case["case_id"]), profile=action,
                        orientation=f"{atom.orientation:+d}", action_sequence=(action,),
                        offset_sequence=(z,), construction_rule=ONE_STEP_CONSTRUCTION_RULE,
                        construction_payload={"source_case_id": int(case["case_id"]),
                                              "action": action, "component_index": component_index,
                                              "z_offset": z}, component_index=component_index,
                        z_offset=z, reference_b_prespecified=component_index == 0 and z == 0))
    return _make_suite("one_step", ONE_STEP_SUITE_VERSION, True,
                       ONE_STEP_MANIFEST_RULE, descriptors,
                       ("sample_1", "sample_2"), identities)


def _anchor_atom(env, orientation):
    totals = sorted({float(atom.total_need) for atom in env.prior.states})
    total = totals[len(totals) // 2]
    gap = max(float(atom.absolute_gap) for atom in env.prior.states)
    matches = [atom for atom in env.prior.states if float(atom.total_need) == total
               and float(atom.absolute_gap) == gap and atom.orientation == orientation]
    if len(matches) != 1:
        raise RuntimeError("reachable anchor is not unique")
    return matches[0]


def _offsets(depth):
    return tuple(FROZEN_HISTORY_OFFSET_CYCLE[i % len(FROZEN_HISTORY_OFFSET_CYCLE)] for i in range(depth))


def _profiles(mdp):
    late = min(19, math.floor((mdp.config.total_time - mdp.config.terminate_cost) /
                              mdp.config.sample_time_cost))
    balanced6 = tuple("sample_1" if i % 2 == 0 else "sample_2" for i in range(6))
    balanced_late = tuple("sample_1" if i % 2 == 0 else "sample_2" for i in range(late))
    return (
        ("initial_symmetric", "symmetric", -1, (), ()),
        ("one_step_orientation_-1", "-1", -1, ("sample_1",), (0,)),
        ("one_step_orientation_+1", "+1", 1, ("sample_2",), (0,)),
        ("balanced_depth_6", "balanced", -1, balanced6, _offsets(6)),
        ("concentrated_depth_6_-1", "-1", -1, ("sample_1",) * 6, _offsets(6)),
        ("concentrated_depth_6_+1", "+1", 1, ("sample_2",) * 6, _offsets(6)),
        ("balanced_late_feasible", "balanced", -1, balanced_late, _offsets(late)),
        ("concentrated_late_feasible_-1", "-1", -1, ("sample_1",) * late, _offsets(late)),
        ("concentrated_late_feasible_+1", "+1", 1, ("sample_2",) * late, _offsets(late)),
    )


def _apply_history(mdp, anchor, actions, offsets):
    belief = mdp.initial_belief()
    for action, offset in zip(actions, offsets):
        observation = mdp._need_for_action(anchor, action) + offset * mdp.config.sigma_sample
        belief = mdp.posterior_transition(belief, action, observation, advance_time=True, record=True)
    return belief


def build_terminal_reachable_core_suite(identities: Optional[TerminalValidationIdentities] = None):
    identities = identities or load_terminal_validation_identities()
    descriptors = []
    for env in build_development_environments(load_frozen_r6_spec()):
        mdp = FiniteSupportMetaMDP(env.config, env.prior)
        for profile, orientation, anchor_orientation, actions, offsets in _profiles(mdp):
            belief = _apply_history(mdp, _anchor_atom(env, anchor_orientation), actions, offsets)
            descriptors.append(_make_descriptor(
                suite_class="reachable_core", suite_version=REACHABLE_CORE_SUITE_VERSION,
                descriptor_index=len(descriptors), component_validation_only=True,
                identities=identities, environment=env, mdp=mdp, belief=belief,
                source_case_id=None, profile=profile, orientation=orientation,
                action_sequence=actions, offset_sequence=offsets,
                construction_rule=REACHABLE_CONSTRUCTION_RULE,
                construction_payload={"environment": env.name, "profile": profile,
                                      "anchor_rule": "median_total_largest_gap",
                                      "anchor_orientation": anchor_orientation,
                                      "action_sequence": actions, "offset_sequence": offsets}))
    names = tuple(item[0] for item in _profiles(FiniteSupportMetaMDP(
        build_development_environments(load_frozen_r6_spec())[0].config,
        build_development_environments(load_frozen_r6_spec())[0].prior)))
    return _make_suite("reachable_core", REACHABLE_CORE_SUITE_VERSION, True,
                       REACHABLE_MANIFEST_RULE, descriptors,
                       names, identities)


def _append_failure(failures: List[str], reason: str) -> None:
    if reason not in failures:
        failures.append(reason)


def _descriptor_semantic_failures(
    descriptor: TerminalValidationDescriptor,
    invariant: FrozenSuiteInvariant,
    identities: TerminalValidationIdentities,
) -> Tuple[str, ...]:
    failures: List[str] = []
    expected_identity = (
        identities.legacy_spec_hash,
        identities.legacy_numerical_case_hash,
        identities.scientific_spec_hash,
        identities.numerical_method_config_hash,
    )
    if descriptor.schema != DESCRIPTOR_SCHEMA:
        failures.append("descriptor_schema")
    if (
        descriptor.suite_class != invariant.suite_class
        or descriptor.suite_version != invariant.suite_version
    ):
        failures.append("descriptor_suite_identity")
    if (
        descriptor.component_validation_only
        != invariant.component_validation_only
        or descriptor.environment_selection_eligible
        != invariant.environment_selection_eligible
    ):
        failures.append("descriptor_selection_flags")
    if (
        descriptor.legacy_spec_hash,
        descriptor.legacy_numerical_case_hash,
        descriptor.scientific_spec_hash,
        descriptor.numerical_method_config_hash,
    ) != expected_identity:
        failures.append("descriptor_identity")
    if descriptor.construction_rule != invariant.descriptor_rule:
        failures.append("descriptor_construction_rule")
    if descriptor.orientation not in invariant.allowed_orientations:
        failures.append("descriptor_orientation")
    if descriptor.depth != len(descriptor.history):
        failures.append("descriptor_depth")
    if descriptor.depth != len(descriptor.action_sequence):
        failures.append("descriptor_action_depth")
    if descriptor.depth != len(descriptor.offset_sequence):
        failures.append("descriptor_offset_depth")
    if tuple(step.action for step in descriptor.history) != descriptor.action_sequence:
        failures.append("descriptor_history_actions")
    if descriptor.history_hash != canonical_hash(descriptor.history):
        failures.append("descriptor_history_hash")
    if any(step.cost != descriptor.sample_time_cost for step in descriptor.history):
        failures.append("descriptor_history_cost")
    if not math.isclose(
        descriptor.deliberation_time,
        math.fsum(step.cost for step in descriptor.history),
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        failures.append("descriptor_deliberation_time")

    if invariant.suite_class == "base":
        if descriptor.source_case_id != descriptor.descriptor_index:
            failures.append("base_case_order")
        if descriptor.action_sequence or descriptor.offset_sequence or descriptor.history:
            failures.append("base_history")
        if descriptor.component_index is not None or descriptor.z_offset is not None:
            failures.append("base_component_fields")
        if descriptor.reference_b_prespecified:
            failures.append("base_reference_b_marker")
    elif invariant.suite_class == "one_step":
        block_size = 2 * 18 * len(FROZEN_Z_OFFSETS)
        case_id, remainder = divmod(descriptor.descriptor_index, block_size)
        action_index, remainder = divmod(
            remainder, 18 * len(FROZEN_Z_OFFSETS)
        )
        component_index, z_index = divmod(remainder, len(FROZEN_Z_OFFSETS))
        expected_action = ("sample_1", "sample_2")[action_index]
        expected_z = FROZEN_Z_OFFSETS[z_index]
        if descriptor.source_case_id != case_id:
            failures.append("one_step_case_order")
        if descriptor.profile != expected_action:
            failures.append("one_step_action_order")
        if descriptor.component_index != component_index:
            failures.append("one_step_component_order")
        if descriptor.z_offset != expected_z:
            failures.append("one_step_z_order")
        if descriptor.action_sequence != (expected_action,):
            failures.append("one_step_action_sequence")
        if descriptor.offset_sequence != (expected_z,):
            failures.append("one_step_offset_sequence")
        if descriptor.reference_b_prespecified != (
            component_index == 0 and expected_z == 0
        ):
            failures.append("one_step_reference_b_marker")
    elif invariant.suite_class == "reachable_core":
        expected_profile, expected_orientation, _, _, _ = _profiles_for_integrity()[
            descriptor.descriptor_index % 9
        ]
        if descriptor.source_case_id is not None:
            failures.append("reachable_source_case")
        if descriptor.profile != expected_profile:
            failures.append("reachable_profile_order")
        if descriptor.orientation != expected_orientation:
            failures.append("reachable_orientation")
        if descriptor.component_index is not None or descriptor.z_offset is not None:
            failures.append("reachable_component_fields")
        if descriptor.reference_b_prespecified:
            failures.append("reachable_reference_b_marker")
    return tuple(failures)


def _profiles_for_integrity() -> Tuple[Tuple[str, str, int, Tuple[str, ...], Tuple[int, ...]], ...]:
    return (
        ("initial_symmetric", "symmetric", -1, (), ()),
        ("one_step_orientation_-1", "-1", -1, (), ()),
        ("one_step_orientation_+1", "+1", 1, (), ()),
        ("balanced_depth_6", "balanced", -1, (), ()),
        ("concentrated_depth_6_-1", "-1", -1, (), ()),
        ("concentrated_depth_6_+1", "+1", 1, (), ()),
        ("balanced_late_feasible", "balanced", -1, (), ()),
        ("concentrated_late_feasible_-1", "-1", -1, (), ()),
        ("concentrated_late_feasible_+1", "+1", 1, (), ()),
    )


def _structural_suite_integrity_failures(
    suite: TerminalValidationSuite,
    identities: TerminalValidationIdentities,
) -> Tuple[str, ...]:
    failures: List[str] = []
    manifest = suite.manifest
    invariant = FROZEN_SUITE_INVARIANTS.get(manifest.suite_class)
    if invariant is None:
        return ("unknown_suite_class",)
    expected_identity = (
        identities.legacy_spec_hash,
        identities.legacy_numerical_case_hash,
        identities.scientific_spec_hash,
        identities.numerical_method_config_hash,
    )
    if manifest.schema != MANIFEST_SCHEMA:
        failures.append("manifest_schema")
    if (
        manifest.suite_class != invariant.suite_class
        or manifest.suite_version != invariant.suite_version
    ):
        failures.append("manifest_suite_identity")
    if manifest.construction_rule != invariant.manifest_rule:
        failures.append("manifest_construction_rule")
    if (
        manifest.component_validation_only != invariant.component_validation_only
        or manifest.environment_selection_eligible
        != invariant.environment_selection_eligible
    ):
        failures.append("manifest_selection_flags")
    if (
        manifest.legacy_spec_hash,
        manifest.legacy_numerical_case_hash,
        manifest.scientific_spec_hash,
        manifest.numerical_method_config_hash,
    ) != expected_identity:
        failures.append("manifest_identity")
    if manifest.hard_cap != invariant.hard_cap:
        failures.append("hard_cap")
    if len(suite.descriptors) != invariant.hard_cap:
        failures.append("descriptor_count")
    if manifest.pre_dedup_count != invariant.hard_cap:
        failures.append("pre_dedup_count")
    if manifest.post_dedup_count != invariant.hard_cap:
        failures.append("post_dedup_count")
    if manifest.deduplication_applied:
        failures.append("deduplication")
    if tuple(item.descriptor_index for item in suite.descriptors) != tuple(
        range(invariant.hard_cap)
    ):
        failures.append("descriptor_indices")

    for descriptor in suite.descriptors:
        for reason in _descriptor_semantic_failures(
            descriptor, invariant, identities
        ):
            _append_failure(failures, reason)
        if terminal_validation_descriptor_hash(descriptor) != descriptor.descriptor_hash:
            _append_failure(failures, "descriptor_hash")

    construction_hash = canonical_hash(
        tuple(item.construction_hash for item in suite.descriptors)
    )
    if manifest.ordered_construction_hash != invariant.ordered_construction_hash:
        failures.append("frozen_ordered_construction_hash")
    if construction_hash != invariant.ordered_construction_hash:
        failures.append("descriptor_construction_sequence")
    if manifest.ordered_construction_hash != construction_hash:
        failures.append("ordered_construction_hash")
    descriptor_hash = canonical_hash(
        tuple(item.descriptor_hash for item in suite.descriptors)
    )
    if manifest.ordered_descriptor_hash != descriptor_hash:
        failures.append("ordered_descriptor_hash")

    partition_names = tuple(name for name, _ in invariant.partitions)
    partition_counts = tuple(
        (partition.name, partition.count) for partition in manifest.partitions
    )
    if partition_counts != invariant.partitions:
        failures.append("frozen_partitions")
    if manifest.partitions != _partition(suite.descriptors, partition_names):
        failures.append("partitions")
    if terminal_validation_manifest_hash(manifest) != manifest.manifest_hash:
        failures.append("manifest_hash")
    return tuple(failures)


@lru_cache(maxsize=12)
def _expected_source_suite(
    suite_class: str,
    identities: TerminalValidationIdentities,
    base_provider: CanonicalBaseProvider,
) -> TerminalValidationSuite:
    if suite_class == "base":
        return build_terminal_base_suite(identities, base_provider)
    if suite_class == "one_step":
        return build_terminal_one_step_suite(identities, base_provider)
    if suite_class == "reachable_core":
        return build_terminal_reachable_core_suite(identities)
    raise ValueError(f"unknown terminal suite class: {suite_class}")


def reconstruct_terminal_validation_descriptor(
    descriptor: TerminalValidationDescriptor,
    *,
    base_provider: CanonicalBaseProvider,
    identities: Optional[TerminalValidationIdentities] = None,
) -> TerminalDescriptorReconstruction:
    identities = identities or load_terminal_validation_identities()
    if descriptor.suite_class not in FROZEN_SUITE_INVARIANTS:
        return TerminalDescriptorReconstruction(None, ("unknown_suite_class",))
    hard_cap = FROZEN_SUITE_INVARIANTS[descriptor.suite_class].hard_cap
    if not 0 <= descriptor.descriptor_index < hard_cap:
        return TerminalDescriptorReconstruction(None, ("descriptor_index_range",))
    try:
        expected = _expected_source_suite(
            descriptor.suite_class,
            identities,
            base_provider,
        ).descriptors[descriptor.descriptor_index]
    except (KeyError, RuntimeError, TypeError, ValueError) as error:
        return TerminalDescriptorReconstruction(
            None,
            (f"source_reconstruction:{type(error).__name__}",),
        )
    failures = tuple(
        f"source_mismatch:{field.name}"
        for field in fields(TerminalValidationDescriptor)
        if getattr(descriptor, field.name) != getattr(expected, field.name)
    )
    return TerminalDescriptorReconstruction(expected, failures)


def validate_terminal_validation_suite(
    suite: TerminalValidationSuite,
    identities: Optional[TerminalValidationIdentities] = None,
    *,
    base_provider: Optional[CanonicalBaseProvider] = None,
    require_authoritative: bool = False,
    authoritative_acceptance_validator: Optional[
        Callable[[CanonicalBaseProvider], bool]
    ] = None,
) -> TerminalSuiteValidationResult:
    identities = identities or load_terminal_validation_identities()
    provider = base_provider or build_local_diagnostic_base_provider()
    failures = list(_structural_suite_integrity_failures(suite, identities))
    for reason in canonical_base_provider_failures(provider):
        _append_failure(failures, f"base_provider:{reason}")

    accepted = False
    if require_authoritative:
        from .terminal_canonical_provider import (  # pylint: disable=import-outside-toplevel
            accepted_canonical_base_provider,
        )

        if provider.diagnostic_only or provider.provider_kind != AUTHORITATIVE_PROVIDER_KIND:
            _append_failure(failures, "authoritative_base_provider_required")
        else:
            if (
                authoritative_acceptance_validator is not None
                and authoritative_acceptance_validator
                is not accepted_canonical_base_provider
            ):
                _append_failure(
                    failures, "authoritative_custom_acceptance_disallowed"
                )
            accepted = accepted_canonical_base_provider(provider)
            if not accepted:
                _append_failure(failures, "authoritative_base_acceptance_rejected")

    if not any(reason.startswith("base_provider:") for reason in failures):
        for descriptor in suite.descriptors:
            reconstruction = reconstruct_terminal_validation_descriptor(
                descriptor,
                base_provider=provider,
                identities=identities,
            )
            for reason in reconstruction.failures:
                _append_failure(failures, reason)

    status = (
        "authoritative_source_validated"
        if require_authoritative and accepted and not failures
        else "diagnostic_source_validated"
        if not require_authoritative and not failures
        else "validation_failed"
    )
    return TerminalSuiteValidationResult(
        failures=tuple(failures),
        validation_status=status,
        provider_hash=provider.provider_hash,
        authoritative_source_accepted=(
            require_authoritative and accepted and not failures
        ),
    )


def suite_integrity_failures(
    suite: TerminalValidationSuite,
    identities: Optional[TerminalValidationIdentities] = None,
    *,
    base_provider: Optional[CanonicalBaseProvider] = None,
    require_authoritative: bool = False,
    authoritative_acceptance_validator: Optional[
        Callable[[CanonicalBaseProvider], bool]
    ] = None,
) -> Tuple[str, ...]:
    return validate_terminal_validation_suite(
        suite,
        identities,
        base_provider=base_provider,
        require_authoritative=require_authoritative,
        authoritative_acceptance_validator=authoritative_acceptance_validator,
    ).failures


__all__ = [name for name in globals() if name.startswith(("build_terminal_", "load_terminal_", "terminal_", "FROZEN_", "LEGACY_", "SUITE_", "REFERENCE_", "DEFAULT_", "Terminal"))] + [
    "AUTHORITATIVE_PROVIDER_KIND",
    "CanonicalBaseProvider",
    "CanonicalBaseRecord",
    "LOCAL_DIAGNOSTIC_PROVIDER_KIND",
    "build_local_diagnostic_base_provider",
    "canonical_base_provider_failures",
    "canonical_base_provider_hash",
    "canonical_base_record_hash",
    "canonical_hash",
    "load_frozen_r6_cases",
    "load_frozen_r6_spec",
    "make_canonical_base_provider",
    "reconstruct_canonical_base_record",
    "reconstruct_terminal_validation_descriptor",
    "suite_integrity_failures",
    "validate_terminal_validation_suite",
]
