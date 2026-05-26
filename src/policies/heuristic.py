from __future__ import annotations

import math
from typing import List

try:
    from ..mdp.meta_mdp import Action, BeliefState, ContinuousAllocationMetaMDP, MetaPolicy
except ImportError:  # Allows notebooks to import modules after adding src/ to sys.path.
    from mdp.meta_mdp import Action, BeliefState, ContinuousAllocationMetaMDP, MetaPolicy


def _normal_cdf(value: float) -> float:
    return 0.5 * (1.0 + math.erf(value / math.sqrt(2.0)))


class EqualDivisionPolicy:
    name = "equal_division"

    def termination_time_cost(self, mdp: ContinuousAllocationMetaMDP, belief: BeliefState) -> float:
        return mdp.config.terminate_cost

    def choose_action(self, mdp: ContinuousAllocationMetaMDP, belief: BeliefState) -> Action:
        return mdp.TERMINATE

    def choose_final_allocation(self, mdp: ContinuousAllocationMetaMDP, belief: BeliefState) -> float:
        return mdp.final_allocation_equal_division(belief)


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
