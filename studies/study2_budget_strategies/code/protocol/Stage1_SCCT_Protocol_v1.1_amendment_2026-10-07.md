# Stage-1 protocol, amendment v1.1 (7 October 2026; pre-outcome, no tuning or confirmatory run completed)

Change (Decision 6, tuning data): before any construction, 15% of each domain's common pool (fixed SHA-256 order,
key 20261007|tuning-reserve) is held out. The frozen RevTgt, reversal-only RevNon and AddDiff items of those sources
are added to the dev-split items to form the robustness-dev set. No Stage-1 condition trains on them. B1 is computed
on the remaining pool and stays 1002 (Restaurant) / 414 (Laptop).

Reason: the 150-item dev split yields too few RevNon items and RT/RN source pairs for W_dev and J_dev to be defined.

Unchanged: pass rule, endpoints, thresholds, grids, seeds, and every other decision of v1.0.
