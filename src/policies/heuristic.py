from __future__ import annotations

import math
from typing import List

try:
    from ..mdp.meta_mdp import Action, BeliefState, ContinuousAllocationMetaMDP, MetaPolicy
except ImportError:  # Allows notebooks to import modules after adding src/ to sys.path.
    from mdp.meta_mdp import Action, BeliefState, ContinuousAllocationMetaMDP, MetaPolicy


def _normal_cdf(value: float) -> float:
    return 0.5 * (1.0 + math.erf(value / math.sqrt(2.0)))


def effort_to_goal(need_estimate: float, learning_rate: float) -> float:
    """Return nonnegative tutoring time needed to reach the estimated goal."""

    if learning_rate <= 0.0:
        raise ValueError("learning_rate must be positive")
    return max(0.0, float(need_estimate) / float(learning_rate))


def lower_effort_identity(
    effort_1: float,
    effort_2: float,
    *,
    relative_tolerance: float = 1e-9,
) -> int:
    """Return 1/2 for the lower effort-to-goal recipient, or 0 for a tie."""

    if effort_1 < 0.0 or effort_2 < 0.0:
        raise ValueError("effort-to-goal values must be nonnegative")
    tolerance = relative_tolerance * max(1.0, effort_1, effort_2)
    if abs(effort_1 - effort_2) <= tolerance:
        return 0
    return 1 if effort_1 < effort_2 else 2


def all_to_lower_allocation(
    need_1: float,
    need_2: float,
    learning_rate_1: float,
    learning_rate_2: float,
) -> float:
    effort_1 = effort_to_goal(need_1, learning_rate_1)
    effort_2 = effort_to_goal(need_2, learning_rate_2)
    identity = lower_effort_identity(effort_1, effort_2)
    if identity == 1:
        return 1.0
    if identity == 2:
        return 0.0
    return 0.5


def meet_lower_first_allocation(
    need_1: float,
    need_2: float,
    learning_rate_1: float,
    learning_rate_2: float,
    remaining_time: float,
) -> float:
    if remaining_time <= 0.0:
        return 0.5
    effort_1 = effort_to_goal(need_1, learning_rate_1)
    effort_2 = effort_to_goal(need_2, learning_rate_2)
    identity = lower_effort_identity(effort_1, effort_2)
    if identity == 1:
        return min(remaining_time, effort_1) / remaining_time
    if identity == 2:
        return (remaining_time - min(remaining_time, effort_2)) / remaining_time
    return 0.5


def greatest_effort_need_allocation(
    need_1: float,
    need_2: float,
    learning_rate_1: float,
    learning_rate_2: float,
) -> float:
    effort_1 = effort_to_goal(need_1, learning_rate_1)
    effort_2 = effort_to_goal(need_2, learning_rate_2)
    lower = lower_effort_identity(effort_1, effort_2)
    if lower == 1:
        return 0.0
    if lower == 2:
        return 1.0
    return 0.5


class _ImmediateLowerEffortPolicy:
    allocation_rule = ""

    def termination_time_cost(self, mdp: ContinuousAllocationMetaMDP, belief: BeliefState) -> float:
        return mdp.config.terminate_cost

    def choose_action(self, mdp: ContinuousAllocationMetaMDP, belief: BeliefState) -> Action:
        return mdp.TERMINATE

    def choose_final_allocation(self, mdp: ContinuousAllocationMetaMDP, belief: BeliefState) -> float:
        rate_1, rate_2 = mdp.learning_rates()
        if self.allocation_rule == "all_to_lower":
            return all_to_lower_allocation(
                belief.mean_1,
                belief.mean_2,
                rate_1,
                rate_2,
            )
        if self.allocation_rule == "meet_lower_first":
            return meet_lower_first_allocation(
                belief.mean_1,
                belief.mean_2,
                rate_1,
                rate_2,
                mdp.remaining_time_after_termination(belief, self),
            )
        raise RuntimeError("Unknown lower-effort allocation rule")


class ImmediateAllToLowerPolicy(_ImmediateLowerEffortPolicy):
    name = "immediate_all_to_lower"
    allocation_rule = "all_to_lower"


class ImmediateMeetLowerFirstPolicy(_ImmediateLowerEffortPolicy):
    name = "immediate_meet_lower_first"
    allocation_rule = "meet_lower_first"


class ScarcityGreatestNeedPolicy:
    """Rate-aware greatest-need comparator used only by the scarcity workflow."""

    name = "scarcity_greatest_need"

    def termination_time_cost(self, mdp: ContinuousAllocationMetaMDP, belief: BeliefState) -> float:
        return mdp.config.terminate_cost

    def choose_action(self, mdp: ContinuousAllocationMetaMDP, belief: BeliefState) -> Action:
        return mdp.TERMINATE

    def choose_final_allocation(self, mdp: ContinuousAllocationMetaMDP, belief: BeliefState) -> float:
        rate_1, rate_2 = mdp.learning_rates()
        return greatest_effort_need_allocation(
            belief.mean_1,
            belief.mean_2,
            rate_1,
            rate_2,
        )


class _ManualActiveSearchLowerEffortPolicy(_ImmediateLowerEffortPolicy):
    """Balanced six-sample acquisition with a separately named final-choice rule."""

    def __init__(self, samples_per_person: int = 3):
        if samples_per_person != 3:
            raise ValueError("Scarcity active lower-effort policies require exactly three samples per person")
        self.samples_per_person = samples_per_person

    def choose_action(self, mdp: ContinuousAllocationMetaMDP, belief: BeliefState) -> Action:
        count_1 = sum(1 for item in belief.history if item["action"] == 1.0)
        count_2 = sum(1 for item in belief.history if item["action"] == 2.0)
        if count_1 >= self.samples_per_person and count_2 >= self.samples_per_person:
            return mdp.TERMINATE
        if count_1 < count_2:
            return mdp.SAMPLE_PERSON_1
        if count_2 < count_1:
            return mdp.SAMPLE_PERSON_2
        rate_1, rate_2 = mdp.learning_rates()
        effort_variance_1 = belief.var_1 / (rate_1 * rate_1)
        effort_variance_2 = belief.var_2 / (rate_2 * rate_2)
        if effort_variance_1 >= effort_variance_2:
            return mdp.SAMPLE_PERSON_1
        return mdp.SAMPLE_PERSON_2


class ManualActiveSearchAllToLowerPolicy(_ManualActiveSearchLowerEffortPolicy):
    name = "manual_active_search_all_to_lower"
    allocation_rule = "all_to_lower"


class ManualActiveSearchMeetLowerFirstPolicy(_ManualActiveSearchLowerEffortPolicy):
    name = "manual_active_search_meet_lower_first"
    allocation_rule = "meet_lower_first"


class EqualDivisionPolicy:
    name = "equal_division"

    def termination_time_cost(self, mdp: ContinuousAllocationMetaMDP, belief: BeliefState) -> float:
        return mdp.config.terminate_cost

    def choose_action(self, mdp: ContinuousAllocationMetaMDP, belief: BeliefState) -> Action:
        return mdp.TERMINATE

    def choose_final_allocation(self, mdp: ContinuousAllocationMetaMDP, belief: BeliefState) -> float:
        return mdp.final_allocation_equal_division(belief)


class EqualSplitBaselinePolicy(EqualDivisionPolicy):
    """Active-search diagnostic alias for the equal-split baseline.

    This policy is behaviorally identical to equal division: it terminates
    without acquiring new information and allocates the remaining resource 50/50.
    The separate name keeps the manual-baseline comparison readable.
    """

    name = "manual_equal_split"


class ManualActiveSearchEqualOutcomePolicy:
    """Transparent hand-coded active-search benchmark for active-search diagnostics.

    The policy deliberately avoids hidden true-state information. It gathers a
    fixed, balanced number of ordinary observations from both recipients, then
    uses the terminal belief to choose the equal-outcome/maximin allocation. The
    resulting allocation is evaluated with true-state metrics elsewhere.
    """

    name = "manual_active_search_equal_outcome"

    def __init__(self, samples_per_person: int = 3):
        if samples_per_person < 0:
            raise ValueError("samples_per_person must be non-negative")
        self.samples_per_person = samples_per_person

    def choose_action(self, mdp: ContinuousAllocationMetaMDP, belief: BeliefState) -> Action:
        count_1 = sum(1 for item in belief.history if item["action"] == 1.0)
        count_2 = sum(1 for item in belief.history if item["action"] == 2.0)
        if count_1 >= self.samples_per_person and count_2 >= self.samples_per_person:
            return mdp.TERMINATE
        if count_1 < count_2:
            return mdp.SAMPLE_PERSON_1
        if count_2 < count_1:
            return mdp.SAMPLE_PERSON_2
        return mdp.SAMPLE_PERSON_1 if belief.var_1 >= belief.var_2 else mdp.SAMPLE_PERSON_2

    def choose_final_allocation(self, mdp: ContinuousAllocationMetaMDP, belief: BeliefState) -> float:
        return mdp.final_allocation_equal_outcome(belief)


class Person1FirstPolicy:
    name = "sample_person1_until_threshold"

    def __init__(self, uncertainty_threshold: float = 40.0):
        self.uncertainty_threshold = uncertainty_threshold

    def choose_action(self, mdp: ContinuousAllocationMetaMDP, belief: BeliefState) -> Action:
        if belief.var_1 + belief.var_2 <= self.uncertainty_threshold:
            return mdp.TERMINATE
        return mdp.SAMPLE_PERSON_1


class BalancedSamplingPolicy:
    name = "balanced_sampling"

    def __init__(self, samples_per_person: int = 2):
        self.samples_per_person = samples_per_person

    def choose_action(self, mdp: ContinuousAllocationMetaMDP, belief: BeliefState) -> Action:
        count_1 = sum(1 for item in belief.history if item["action"] == 1.0)
        count_2 = sum(1 for item in belief.history if item["action"] == 2.0)
        if count_1 < self.samples_per_person:
            return mdp.SAMPLE_PERSON_1
        if count_2 < self.samples_per_person:
            return mdp.SAMPLE_PERSON_2
        return mdp.TERMINATE


class NeediestFirstPolicy:
    name = "neediest_first"

    def __init__(self, uncertainty_stop: float = 25.0):
        self.uncertainty_stop = uncertainty_stop

    def choose_action(self, mdp: ContinuousAllocationMetaMDP, belief: BeliefState) -> Action:
        if belief.var_1 + belief.var_2 <= self.uncertainty_stop:
            return mdp.TERMINATE
        if belief.mean_1 >= belief.mean_2:
            return mdp.SAMPLE_PERSON_1
        return mdp.SAMPLE_PERSON_2


class GiveAllToGreatestNeedPolicy:
    name = "give_all_to_greatest_need"

    def termination_time_cost(self, mdp: ContinuousAllocationMetaMDP, belief: BeliefState) -> float:
        return mdp.config.terminate_cost

    def choose_action(self, mdp: ContinuousAllocationMetaMDP, belief: BeliefState) -> Action:
        return mdp.TERMINATE

    def choose_final_allocation(self, mdp: ContinuousAllocationMetaMDP, belief: BeliefState) -> float:
        return mdp.final_allocation_all_to_greatest_need(belief)


class HelpPoorestImmediatePolicy:
    name = "help_poorest_immediate"

    def termination_time_cost(self, mdp: ContinuousAllocationMetaMDP, belief: BeliefState) -> float:
        return mdp.config.terminate_cost

    def choose_action(self, mdp: ContinuousAllocationMetaMDP, belief: BeliefState) -> Action:
        return mdp.TERMINATE

    def choose_final_allocation(self, mdp: ContinuousAllocationMetaMDP, belief: BeliefState) -> float:
        return mdp.final_allocation_all_to_greatest_need(belief)


class HelpPoorestAfterCertaintyPolicy:
    name = "help_poorest_after_certainty"

    def __init__(self, certainty_threshold: float = 0.85, max_total_samples: int = 6):
        self.certainty_threshold = certainty_threshold
        self.max_total_samples = max_total_samples

    def probability_person_1_is_needier(self, belief: BeliefState) -> float:
        difference_std = math.sqrt(belief.var_1 + belief.var_2)
        if difference_std <= 0:
            return 1.0 if belief.mean_1 >= belief.mean_2 else 0.0
        return _normal_cdf((belief.mean_1 - belief.mean_2) / difference_std)

    def choose_action(self, mdp: ContinuousAllocationMetaMDP, belief: BeliefState) -> Action:
        sample_count = sum(1 for item in belief.history if item["action"] in (1.0, 2.0))
        p_person_1_needier = self.probability_person_1_is_needier(belief)
        certainty = max(p_person_1_needier, 1.0 - p_person_1_needier)
        if sample_count >= self.max_total_samples or certainty >= self.certainty_threshold:
            return mdp.TERMINATE
        if belief.var_1 > belief.var_2:
            return mdp.SAMPLE_PERSON_1
        if belief.var_2 > belief.var_1:
            return mdp.SAMPLE_PERSON_2
        return mdp.SAMPLE_PERSON_1 if belief.mean_1 >= belief.mean_2 else mdp.SAMPLE_PERSON_2

    def choose_final_allocation(self, mdp: ContinuousAllocationMetaMDP, belief: BeliefState) -> float:
        return mdp.final_allocation_all_to_greatest_need(belief)


class FocusAttentionOnNeediestAfterGapPolicy:
    name = "focus_attention_to_neediest_after_gap"

    def __init__(self, gap_threshold: float = 10.0, max_total_samples: int = 4):
        self.gap_threshold = gap_threshold
        self.max_total_samples = max_total_samples

    def choose_action(self, mdp: ContinuousAllocationMetaMDP, belief: BeliefState) -> Action:
        sample_count = sum(1 for item in belief.history if item["action"] in (1.0, 2.0))
        if sample_count >= self.max_total_samples:
            return mdp.TERMINATE
        if abs(belief.mean_1 - belief.mean_2) >= self.gap_threshold:
            return mdp.TERMINATE
        count_1 = sum(1 for item in belief.history if item["action"] == 1.0)
        count_2 = sum(1 for item in belief.history if item["action"] == 2.0)
        return mdp.SAMPLE_PERSON_1 if count_1 <= count_2 else mdp.SAMPLE_PERSON_2

    def choose_final_allocation(self, mdp: ContinuousAllocationMetaMDP, belief: BeliefState) -> float:
        return mdp.final_allocation_all_to_greatest_need(belief)


class ThresholdDifferencePolicy:
    name = "sample_until_need_gap_detected"

    def __init__(self, gap_threshold: float = 10.0, max_total_samples: int = 4):
        self.gap_threshold = gap_threshold
        self.max_total_samples = max_total_samples

    def choose_action(self, mdp: ContinuousAllocationMetaMDP, belief: BeliefState) -> Action:
        sample_count = sum(1 for item in belief.history if item["action"] in (1.0, 2.0))
        if sample_count >= self.max_total_samples:
            return mdp.TERMINATE
        if abs(belief.mean_1 - belief.mean_2) >= self.gap_threshold:
            return mdp.TERMINATE
        count_1 = sum(1 for item in belief.history if item["action"] == 1.0)
        count_2 = sum(1 for item in belief.history if item["action"] == 2.0)
        return mdp.SAMPLE_PERSON_1 if count_1 <= count_2 else mdp.SAMPLE_PERSON_2


class OneAndDonePolicy:
    name = "one_and_done"

    def __init__(self):
        self._sampled_once = False

    def choose_action(self, mdp: ContinuousAllocationMetaMDP, belief: BeliefState) -> Action:
        sample_count = sum(1 for item in belief.history if item["action"] in (1.0, 2.0))
        if sample_count >= 1:
            return mdp.TERMINATE
        return mdp.SAMPLE_PERSON_1 if abs(belief.mean_1 - belief.mean_2) <= mdp.config.equal_perception_tolerance else (
            mdp.SAMPLE_PERSON_1 if belief.mean_1 >= belief.mean_2 else mdp.SAMPLE_PERSON_2
        )


class ProportionalNeedPolicy:
    name = "proportional_to_estimated_need"

    def termination_time_cost(self, mdp: ContinuousAllocationMetaMDP, belief: BeliefState) -> float:
        return mdp.config.terminate_cost + mdp.config.sample_time_cost

    def choose_action(self, mdp: ContinuousAllocationMetaMDP, belief: BeliefState) -> Action:
        return mdp.TERMINATE

    def choose_final_allocation(self, mdp: ContinuousAllocationMetaMDP, belief: BeliefState) -> float:
        return mdp.final_allocation_proportional_to_estimated_needs(belief)


class RectifyThenSplitPolicy:
    name = "rectify_then_split_equally"

    def termination_time_cost(self, mdp: ContinuousAllocationMetaMDP, belief: BeliefState) -> float:
        return mdp.config.terminate_cost + mdp.config.sample_time_cost

    def choose_action(self, mdp: ContinuousAllocationMetaMDP, belief: BeliefState) -> Action:
        return mdp.TERMINATE

    def choose_final_allocation(self, mdp: ContinuousAllocationMetaMDP, belief: BeliefState) -> float:
        return mdp.final_allocation_rectify_then_split_equally(belief)


class EqualOutcomePolicy:
    name = "equal_outcome"

    def termination_time_cost(self, mdp: ContinuousAllocationMetaMDP, belief: BeliefState) -> float:
        return mdp.config.terminate_cost + mdp.config.sample_time_cost

    def choose_action(self, mdp: ContinuousAllocationMetaMDP, belief: BeliefState) -> Action:
        return mdp.TERMINATE

    def choose_final_allocation(self, mdp: ContinuousAllocationMetaMDP, belief: BeliefState) -> float:
        return mdp.final_allocation_equal_outcome(belief)


class MaximinPolicy:
    name = "maximin_equal_outcome"

    def termination_time_cost(self, mdp: ContinuousAllocationMetaMDP, belief: BeliefState) -> float:
        return mdp.config.terminate_cost + mdp.config.sample_time_cost

    def choose_action(self, mdp: ContinuousAllocationMetaMDP, belief: BeliefState) -> Action:
        return mdp.TERMINATE

    def choose_final_allocation(self, mdp: ContinuousAllocationMetaMDP, belief: BeliefState) -> float:
        return mdp.final_allocation_maximin(belief)


def build_policy_library() -> List[MetaPolicy]:
    return [
        EqualDivisionPolicy(),
        GiveAllToGreatestNeedPolicy(),
        HelpPoorestImmediatePolicy(),
        HelpPoorestAfterCertaintyPolicy(certainty_threshold=0.85, max_total_samples=6),
        FocusAttentionOnNeediestAfterGapPolicy(gap_threshold=10.0, max_total_samples=4),
        ProportionalNeedPolicy(),
        RectifyThenSplitPolicy(),
        EqualOutcomePolicy(),
        MaximinPolicy(),
        OneAndDonePolicy(),
        Person1FirstPolicy(uncertainty_threshold=40.0),
        BalancedSamplingPolicy(samples_per_person=2),
        NeediestFirstPolicy(uncertainty_stop=30.0),
        ThresholdDifferencePolicy(gap_threshold=10.0, max_total_samples=4),
    ]


def build_final_choice_heuristics() -> List[MetaPolicy]:
    return [
        EqualDivisionPolicy(),
        GiveAllToGreatestNeedPolicy(),
        HelpPoorestImmediatePolicy(),
        ProportionalNeedPolicy(),
        RectifyThenSplitPolicy(),
        EqualOutcomePolicy(),
        MaximinPolicy(),
    ]
