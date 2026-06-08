# Policies

`policies/` contains decision strategies for the metalevel MDP.

`heuristic.py` includes final-choice and information-acquisition heuristics:

- `EqualDivisionPolicy`: terminate immediately and split 50/50
- `GiveAllToGreatestNeedPolicy`: terminate and allocate all remaining help to the greatest estimated need
- `HelpPoorestImmediatePolicy`: named help-poorest version of greatest-need final choice
- `HelpPoorestAfterCertaintyPolicy`: sample until sufficiently certain who is neediest, then help that person
- `FocusAttentionOnNeediestAfterGapPolicy`: sample until a need gap appears, then allocate to the neediest
- `ProportionalNeedPolicy`: allocate proportional to estimated need
- `RectifyThenSplitPolicy`: rectify estimated need difference, then split remaining help equally
- `EqualOutcomePolicy`: choose the allocation that equalizes estimated outcomes in the current two-person model
- `MaximinPolicy`: current operationalization matches equal outcome in this two-person single-resource setting
- `OneAndDonePolicy`: draw one sample and then act
- `BalancedSamplingPolicy`, `Person1FirstPolicy`, `NeediestFirstPolicy`, `ThresholdDifferencePolicy`: information-acquisition baselines

`voi.py` includes:

- `MyopicValueOfInformationPolicy`: one-step value-of-information approximation
- `BlinkeredPolicy`: approximation that repeatedly considers sampling one computation type

Policy classes implement `choose_action(mdp, belief)`. Some final-choice heuristics also implement `choose_final_allocation(mdp, belief)`.
