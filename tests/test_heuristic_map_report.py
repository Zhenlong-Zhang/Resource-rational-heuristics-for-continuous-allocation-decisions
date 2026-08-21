"""Focused pure tests for the frozen four-row Falk report model."""

from __future__ import annotations

import unittest

from src.experiments.heuristic_map_report import (
    ALLOWED_CLAIM_TYPES,
    MAP_COLUMNS,
    build_claim_ledger,
    build_heuristic_map,
    prototype_cost_audit,
    scarcity_row_name,
)


def evidence_fixture() -> dict[str, object]:
    return {
        "audit_pass": True,
        "no_episode_simulation_or_new_historical_inference": True,
        "heuristic_prototypes": [
            {
                "strategy": "immediate_equal_split",
                "environment": "near_50_50_sample_time_cost=16_total_time=10",
                "sample_time_cost": 16.0,
                "total_time": 10.0,
                "sample_time_cost_percent": 160.0,
            },
            {
                "strategy": "immediate_equal_outcome",
                "environment": "equal_outcome_total_time=120",
                "sample_time_cost": 1.0,
                "total_time": 120.0,
                "sample_time_cost_percent": 100.0 / 120.0,
            },
            {
                "strategy": "active_search_equal_outcome",
                "environment": (
                    "confirmation_sigma_need=30_sigma_sample=10_total_time=180_"
                    "utility_exponent=0.5_sample_time_cost_percent=0.005"
                ),
                "sample_time_cost": 0.009,
                "total_time": 180.0,
                "sample_time_cost_percent": 0.005,
            },
        ],
        "analysis_audits": {
            "active_search_confirmation": {
                "point_joint_count": 11,
                "strict_joint_count": 0,
            },
            "scarcity_behavior": {
                "recovery_classification": (
                    "higher_utility_behaviorally_different_strategy"
                )
            },
        },
    }


class HeuristicMapReportTests(unittest.TestCase):
    def test_map_is_exactly_four_by_four_and_preserves_analysis_limits(self) -> None:
        classifications = [
            {
                "acquisition_class": "no_search",
                "classification": "supported_exact",
                "target_rule": "meet_lower_first",
            },
            {
                "acquisition_class": "active_search",
                "classification": "partial_direction_only",
                "target_rule": "more_to_lower",
            },
        ]
        rows = build_heuristic_map(
            evidence_fixture(),
            object_gate={"object_level_classification": "supported_exact"},
            selected_anchors=[
                {
                    "environment_id": "oracle_anchor",
                    "anchor_support_kind": "exact",
                }
            ],
            selection={
                "targets": [
                    {
                        "acquisition_class": "no_search",
                        "environment_id": "target_no_search",
                        "selection_status": "exact_candidate",
                    },
                    {
                        "acquisition_class": "active_search",
                        "environment_id": "target_active_search",
                        "selection_status": "direction_candidate",
                    },
                ]
            },
            classifications=classifications,
        )
        self.assertEqual(len(rows), 4)
        self.assertTrue(all(tuple(row) == MAP_COLUMNS for row in rows))
        active = rows[2]["qualitative_environmental_conditions"]
        self.assertIn("11 of 12", active)
        self.assertIn("0 of 12", active)
        self.assertIn("Three related scarcity noise conditions", active)
        self.assertIn("not three independent replications", active)
        self.assertIn("aggregate", active.lower())

    def test_scarcity_name_follows_exact_partial_rejection_and_precision(self) -> None:
        exact = [
            {
                "acquisition_class": "no_search",
                "classification": "supported_exact",
                "target_rule": "all_to_lower",
            },
            {
                "acquisition_class": "active_search",
                "classification": "not_supported_in_frozen_scope",
                "target_rule": "more_to_lower",
            },
        ]
        self.assertIn("all to lower-need", scarcity_row_name("supported_exact", exact))
        partial = [
            {
                "acquisition_class": name,
                "classification": "partial_direction_only",
                "target_rule": "more_to_lower",
            }
            for name in ("no_search", "active_search")
        ]
        self.assertIn("direction-only partial", scarcity_row_name("partial_direction_only", partial))
        self.assertIn(
            "unsupported in frozen scope",
            scarcity_row_name("not_supported_in_frozen_scope", []),
        )
        self.assertIn(
            "inconclusive in frozen scope",
            scarcity_row_name("inconclusive_precision", []),
        )

    def test_claim_ledger_has_only_allowed_types_and_keeps_suggestion_distinct(self) -> None:
        claims = build_claim_ledger(
            evidence_fixture(),
            object_gate={"object_level_classification": "inconclusive_precision"},
            classifications=[],
        )
        self.assertEqual({row["claim_type"] for row in claims}.difference(ALLOWED_CLAIM_TYPES), set())
        lower_need = next(row for row in claims if row["claim_id"] == "MAP-S01")
        self.assertEqual(lower_need["claim_type"], "suggestion")
        self.assertIn("not treated as a result", lower_need["limitations"])

    def test_prototype_costs_report_time_units_percent_and_zero_distinction(self) -> None:
        rows = prototype_cost_audit(evidence_fixture())
        by_strategy = {str(row["strategy"]): row for row in rows}
        self.assertAlmostEqual(
            float(by_strategy["immediate_equal_outcome"]["sample_time_cost"]),
            1.0,
        )
        self.assertAlmostEqual(
            float(
                by_strategy["immediate_equal_outcome"][
                    "sample_time_cost_percent"
                ]
            ),
            100.0 / 120.0,
        )
        self.assertFalse(
            bool(by_strategy["active_search_equal_outcome"]["exact_zero_sampling_cost"])
        )


if __name__ == "__main__":
    unittest.main()
