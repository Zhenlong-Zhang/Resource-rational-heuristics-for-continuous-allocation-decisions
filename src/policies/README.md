# Policies

`heuristic.py` contains hand-coded allocation and sampling strategies.

Final-choice and simple metalevel policies include:

- equal division and immediate equal split
- immediate or certainty-based help for the poorest recipient
- give all help to the estimated greatest need
- equal outcome / maximin
- proportional-to-need allocation
- rectify the estimated need difference, then split
- one-and-done, balanced sampling, threshold stopping, and related sampling rules
- manual balanced active search followed by equal-outcome allocation

`voi.py` contains `MyopicValueOfInformationPolicy` and `BlinkeredPolicy` for Gaussian beliefs.

`finite_support_voi.py` contains `FiniteSupportMyopicVOIPolicy` for controlled finite-support environments.

Policies implement `choose_action(mdp, belief)`. Final-choice equivalence can also be evaluated by applying allocation heuristics to the same terminal belief reached by an RR approximation, without treating every heuristic as a full metalevel policy.
