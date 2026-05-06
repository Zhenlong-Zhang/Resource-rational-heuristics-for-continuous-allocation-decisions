from __future__ import annotations

from typing import List

from mdp.meta_mdp import Action, BeliefState, ContinuousAllocationMetaMDP, MetaPolicy


class EqualDivisionPolicy:
    name = "equal_division"

    def choose_action(self, mdp: ContinuousAllocationMetaMDP, belief: BeliefState) -> Action:
        return mdp.TERMINATE


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


def build_policy_library() -> List[MetaPolicy]:
    return [
        EqualDivisionPolicy(),
        Person1FirstPolicy(uncertainty_threshold=40.0),
        BalancedSamplingPolicy(samples_per_person=2),
        NeediestFirstPolicy(uncertainty_stop=30.0),
        ThresholdDifferencePolicy(gap_threshold=10.0, max_total_samples=4),
    ]
