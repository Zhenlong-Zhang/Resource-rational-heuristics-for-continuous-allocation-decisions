from __future__ import annotations

"""Pure report-model builders for the four-row heuristic map."""

import html
import json
import math
from typing import Dict, List, Mapping, Sequence, Tuple


MAP_COLUMNS = (
    "heuristic_name",
    "information_acquisition_and_final_allocation_definition",
    "qualitative_environmental_conditions",
    "most_prototypical_evaluated_environment",
)
MAP_STRATEGIES = (
    "immediate_equal_split",
    "immediate_equal_outcome",
    "active_search_equal_outcome",
    "scarcity_lower_need",
)
ALLOWED_CLAIM_TYPES = {"suggestion", "decision", "result", "interpretation"}
ALLOWED_BOUNDARY_LABELS = {
    "supported_qualitative",
    "grid_edge_open",
    "confounded",
    "unresolved",
}
R5_AGGREGATE_REFERENCE = "historical aggregate package provided separately"


class HeuristicMapReportError(RuntimeError):
    """Raised when report inputs cannot support the frozen map contract."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise HeuristicMapReportError(message)


def _prototype_map(evidence_audit: Mapping[str, object]) -> Dict[str, Mapping[str, object]]:
    prototypes = evidence_audit.get("heuristic_prototypes")
    _require(isinstance(prototypes, list), "Evidence audit omits prototypes")
    mapped = {str(row["strategy"]): row for row in prototypes}
    _require(
        set(mapped)
        == {
            "immediate_equal_split",
            "immediate_equal_outcome",
            "active_search_equal_outcome",
        },
        "Established prototype identities changed",
    )
    return mapped


def _classification_by_class(
    classifications: Sequence[Mapping[str, object]],
) -> Dict[str, Mapping[str, object]]:
    mapped = {str(row["acquisition_class"]): row for row in classifications}
    _require(
        not classifications
        or set(mapped) == {"no_search", "active_search"},
        "Scarcity classifications do not cover both acquisition classes",
    )
    return mapped


def scarcity_row_name(
    object_classification: str,
    classifications: Sequence[Mapping[str, object]],
) -> str:
    meta = _classification_by_class(classifications)
    exact = [row for row in meta.values() if row["classification"] == "supported_exact"]
    if exact:
        rules = sorted({str(row["target_rule"]) for row in exact})
        labels = {
            "all_to_lower": "all to lower-need / closer-to-goal recipient",
            "meet_lower_first": "meet lower need first, then allocate the remainder",
        }
        if len(rules) == 1 and rules[0] in labels:
            return f"{labels[rules[0]]} (exact held-out support)"
        return "lower-need / closer-to-goal exact patterns (acquisition-class specific)"
    if any(row["classification"] == "partial_direction_only" for row in meta.values()):
        return "allocate more to the lower-need / closer-to-goal recipient (direction-only partial)"
    if classifications and all(
        row["classification"] == "not_supported_in_frozen_scope"
        for row in classifications
    ):
        return "lower-need-first / closer-to-goal (hypothesis; unsupported in frozen scope)"
    if not classifications and object_classification == "not_supported_in_frozen_scope":
        return "lower-need-first / closer-to-goal (hypothesis; unsupported in frozen scope)"
    return "lower-need-first / closer-to-goal (hypothesis; inconclusive in frozen scope)"


def _scarcity_prototype_cell(
    selection: Mapping[str, object] | None,
    selected_anchors: Sequence[Mapping[str, object]],
) -> str:
    if selection is not None:
        targets = selection.get("targets")
        _require(isinstance(targets, list) and len(targets) == 2, "Scarcity selection is malformed")
        return "; ".join(
            f"{row['acquisition_class']}: {row['environment_id']} [{row['selection_status']}]"
            for row in targets
        )
    if selected_anchors:
        return "; ".join(
            f"object anchor {row['environment_id']} [{row['anchor_support_kind']}]"
            for row in selected_anchors
        )
    return "No eligible scarcity prototype; all frozen object-level configurations failed the anchor gate."


def build_heuristic_map(
    evidence_audit: Mapping[str, object],
    *,
    object_gate: Mapping[str, object],
    selected_anchors: Sequence[Mapping[str, object]],
    selection: Mapping[str, object] | None,
    classifications: Sequence[Mapping[str, object]],
) -> List[Dict[str, str]]:
    _require(evidence_audit.get("audit_pass") is True, "Historical evidence audit did not pass")
    _require(
        evidence_audit.get("no_episode_simulation_or_new_R5_inference") is True,
        "R5 aggregate scope guard is absent",
    )
    prototypes = _prototype_map(evidence_audit)
    r5 = evidence_audit["round_audits"]["R5"]  # type: ignore[index]
    r6 = evidence_audit["round_audits"]["R6"]  # type: ignore[index]
    _require(int(r5["point_joint_count"]) == 11, "R5 point count changed")
    _require(int(r5["strict_joint_count"]) == 0, "R5 strict count changed")
    _require(
        r6["recovery_classification"] == "higher_utility_behaviorally_different_strategy",
        "R6 classification changed",
    )
    class_map = _classification_by_class(classifications)
    scarcity_conditions = (
        f"Object level: {object_gate['object_level_classification']}. "
        + (
            "; ".join(
                f"{name} held-out: {row['classification']}"
                for name, row in sorted(class_map.items())
            )
            if class_map
            else "Metalevel stages stopped by the frozen object gate."
        )
        + " [supported_qualitative only where the corresponding frozen bounds pass; otherwise unresolved]."
    )
    rows = [
        {
            "heuristic_name": "Immediate 50/50 split [supported_qualitative]",
            "information_acquisition_and_final_allocation_definition": (
                "Terminate without new samples and allocate one half of remaining time to each recipient."
            ),
            "qualitative_environmental_conditions": (
                "Symmetric or nearly symmetric beliefs where information has little value; equal split may overlap "
                "belief-based equal outcome. Some confirmed cells have negative mean utility, so support is behavioral/resource-rational within the tested grid, not a welfare guarantee. [supported_qualitative; grid_edge_open]"
            ),
            "most_prototypical_evaluated_environment": str(
                prototypes["immediate_equal_split"]["environment"]
            ),
        },
        {
            "heuristic_name": "Equal outcome without new search [supported_qualitative]",
            "information_acquisition_and_final_allocation_definition": (
                "Use informative asymmetric prior beliefs, acquire no new samples, and allocate to equalize predicted final outcomes."
            ),
            "qualitative_environmental_conditions": (
                "Positive-utility environments with informative asymmetric prior knowledge; this is final-choice equivalence, not active information search. [supported_qualitative; grid_edge_open]"
            ),
            "most_prototypical_evaluated_environment": str(
                prototypes["immediate_equal_outcome"]["environment"]
            ),
        },
        {
            "heuristic_name": "Equal outcome with active search [point-rule replication; strict unresolved]",
            "information_acquisition_and_final_allocation_definition": (
                "Acquire information from both recipients, then choose a final allocation close to equal outcome and closer to equal outcome than to 50/50."
            ),
            "qualitative_environmental_conditions": (
                "11 of 12 frozen R5 confirmation points replicated the two point-estimate rule; 0 of 12 passed the strict two-lower-bound rule. R6's three related noise conditions gave higher RR utility but behavioral difference, not three independent replications. Raw sigma_need is confounded by negative needs and feasibility; fixed-total separation is a constructed mechanism diagnostic. [supported_qualitative for existence; unresolved strict boundary; confounded; grid_edge_open]"
                f" The {R5_AGGREGATE_REFERENCE} is aggregate-only, with no raw episode archive and no new R5 inference."
            ),
            "most_prototypical_evaluated_environment": str(
                prototypes["active_search_equal_outcome"]["environment"]
            ),
        },
        {
            "heuristic_name": scarcity_row_name(
                str(object_gate["object_level_classification"]),
                classifications,
            ),
            "information_acquisition_and_final_allocation_definition": (
                "No-search: terminate immediately; active-search benchmark: three observations per recipient. Final choice separately tests all-to-lower, meet-lower-first, and the weaker more-to-lower direction using terminal posterior means."
            ),
            "qualitative_environmental_conditions": scarcity_conditions,
            "most_prototypical_evaluated_environment": _scarcity_prototype_cell(
                selection,
                selected_anchors,
            ),
        },
    ]
    _require(len(rows) == 4, "Heuristic map must contain exactly four rows")
    _require(all(tuple(row) == MAP_COLUMNS for row in rows), "Heuristic map columns changed")
    return rows


def build_claim_ledger(
    evidence_audit: Mapping[str, object],
    *,
    object_gate: Mapping[str, object],
    classifications: Sequence[Mapping[str, object]],
) -> List[Dict[str, str]]:
    claims: List[Dict[str, str]] = [
        {
            "claim_id": "R6-D01",
            "claim_type": "decision",
            "statement": "Utility is primary; information acquisition and final choice are reported separately.",
            "source_artifact": "scarcity_frozen_settings.json",
            "source_keys": "settings.thresholds;settings.numerics",
            "evidence_status": "frozen_before_confirmation",
            "limitations": "This is a design decision, not an empirical result.",
        },
        {
            "claim_id": "R6-R01",
            "claim_type": "result",
            "statement": "R5 reproduced the joint point rule at 11 of 12 frozen points and the strict two-lower-bound rule at 0 of 12 points.",
            "source_artifact": "historical_r5_aggregate_summary.json",
            "source_keys": "round_audits.R5.point_joint_count;round_audits.R5.strict_joint_count",
            "evidence_status": "hash_bound_aggregate_only",
            "limitations": "The accepted Drive package is not a raw episode archive and supports no new R5 inference.",
        },
        {
            "claim_id": "R6-R02",
            "claim_type": "result",
            "statement": "The frozen R6 comparison classified RR as higher utility but behaviorally different from the manual active-search equal-outcome strategy.",
            "source_artifact": "r6_behavior_summary.json",
            "source_keys": "round_audits.R6.recovery_classification",
            "evidence_status": "strict_serious_readback",
            "limitations": "The three noise conditions are related conditions, not independent replications.",
        },
        {
            "claim_id": "R6-S01",
            "claim_type": "suggestion",
            "statement": "Falk proposed prioritizing the lower-need or closer-to-goal recipient under scarcity.",
            "source_artifact": "scarcity_frozen_settings.json",
            "source_keys": "scientific_scope;thresholds",
            "evidence_status": "tested_in_frozen_scarcity_scope",
            "limitations": "The suggestion is not treated as a result before the frozen gates pass.",
        },
        {
            "claim_id": "R6-R03",
            "claim_type": "result",
            "statement": f"The frozen object-level scarcity grid classification is {object_gate['object_level_classification']}.",
            "source_artifact": "scarcity_object_gate.json",
            "source_keys": "object_level_classification;selected_anchor_count",
            "evidence_status": "validated_scarcity_object_gate",
            "limitations": "The near-certain-positive generator is still an unbounded Gaussian; positive-only claims use the both-positive stratum.",
        },
    ]
    for index, row in enumerate(classifications, start=1):
        claims.append(
            {
                "claim_id": f"R6-R1{index}",
                "claim_type": "result",
                "statement": (
                    f"The held-out {row['acquisition_class']} scarcity target is classified "
                    f"{row['classification']}."
                ),
                "source_artifact": "scarcity_confirmation_classifications.json",
                "source_keys": f"classifications.{row['acquisition_class']}",
                "evidence_status": "frozen_heldout_target_only",
                "limitations": "Diagnostic contrasts cannot establish or rescue support.",
            }
        )
    claims.append(
        {
            "claim_id": "R6-I01",
            "claim_type": "interpretation",
            "statement": "The map describes qualitative regions within tested grids and does not establish universal thresholds.",
            "source_artifact": "heuristic_map.csv",
            "source_keys": "all_rows.qualitative_environmental_conditions",
            "evidence_status": "bounded_interpretation",
            "limitations": "Open sides remain open; confounded diagnostics are labeled.",
        }
    )
    _require(all(row["claim_type"] in ALLOWED_CLAIM_TYPES for row in claims), "Claim type outside contract")
    _require(len({row["claim_id"] for row in claims}) == len(claims), "Claim IDs are not unique")
    return claims


def prototype_cost_audit(
    evidence_audit: Mapping[str, object],
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for prototype in evidence_audit["heuristic_prototypes"]:  # type: ignore[index]
        environment = str(prototype["environment"])
        parameters: Dict[str, float] = {}
        for field in ("sample_time_cost", "sample_time_cost_percent", "total_time", "mu_need"):
            marker = f"{field}="
            if marker in environment:
                try:
                    parameters[field] = float(environment.split(marker, 1)[1].split("_", 1)[0])
                except ValueError:
                    pass
        if prototype.get("sample_time_cost", "") != "":
            parameters["sample_time_cost"] = float(prototype["sample_time_cost"])
        if prototype.get("total_time", "") != "":
            parameters["total_time"] = float(prototype["total_time"])
        if prototype.get("sample_time_cost_percent", "") != "":
            parameters["sample_time_cost_percent"] = float(
                prototype["sample_time_cost_percent"]
            )
        percent = parameters.get("sample_time_cost_percent")
        if percent is None and "sample_time_cost" in parameters and "total_time" in parameters:
            percent = 100.0 * parameters["sample_time_cost"] / parameters["total_time"]
        sample_time_cost = parameters.get("sample_time_cost")
        if sample_time_cost is None and percent is not None and "total_time" in parameters:
            sample_time_cost = parameters["total_time"] * percent / 100.0
        rows.append(
            {
                "strategy": prototype["strategy"],
                "environment": environment,
                "sample_time_cost": sample_time_cost if sample_time_cost is not None else "",
                "total_time": parameters.get("total_time", ""),
                "sample_time_cost_percent": percent if percent is not None else "",
                "exact_zero_sampling_cost": percent == 0.0 if percent is not None else "",
                "scope_note": "Parsed from the validated observed prototype identity; blank means the identity does not encode the field.",
            }
        )
    return rows


def markdown_table(rows: Sequence[Mapping[str, str]]) -> str:
    header = "| Heuristic name | Information acquisition and final allocation | Qualitative conditions | Most prototypical evaluated environment |"
    separator = "|---|---|---|---|"
    body = []
    for row in rows:
        values = [str(row[field]).replace("|", "\\|").replace("\n", " ") for field in MAP_COLUMNS]
        body.append("| " + " | ".join(values) + " |")
    return "\n".join((header, separator, *body))


def html_table(rows: Sequence[Mapping[str, str]]) -> str:
    headers = (
        "Heuristic name",
        "Information acquisition and final allocation",
        "Qualitative environmental conditions",
        "Most prototypical evaluated environment",
    )
    head = "".join(f"<th>{html.escape(value)}</th>" for value in headers)
    body = "".join(
        "<tr>"
        + "".join(f"<td>{html.escape(str(row[field]))}</td>" for field in MAP_COLUMNS)
        + "</tr>"
        for row in rows
    )
    return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"


__all__ = (
    "ALLOWED_BOUNDARY_LABELS",
    "ALLOWED_CLAIM_TYPES",
    "HeuristicMapReportError",
    "MAP_COLUMNS",
    "MAP_STRATEGIES",
    "R5_AGGREGATE_REFERENCE",
    "build_claim_ledger",
    "build_heuristic_map",
    "html_table",
    "markdown_table",
    "prototype_cost_audit",
    "scarcity_row_name",
)
