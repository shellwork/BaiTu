# QC and Stopping Criteria Plan

## Campaign context
This campaign tunes a weighted Battleship acquisition policy in a simulated closed loop.
Each cycle proposes a batch of candidate policies, runs them against the Battleship
simulator plus synthetic image-reading pipeline, and fits an ensemble surrogate over
policy -> mean queries to finish the board.

## Chosen stopping criteria

1. Held-out prediction error
- Metric: MAE on a fixed validation policy set (units: mean queries to finish board).
- Threshold: stop when MAE ≤ 4.0 queries.
- Rationale: an MAE ≤ 4 queries means the surrogate can distinguish good from bad policies
  with sub-step precision — sufficient to confidently identify the optimum.
- Computation: fit the surrogate on all completed cycles, predict the held-out policy
  set, and compare predictions against the simulated ground-truth performance.
- Robustness: held-out policies are fixed at campaign start and never proposed; this
  prevents the surrogate from appearing accurate by overfitting to seen regions.

2. Information gain per cycle
- Metric: percent reduction in mean ensemble variance on a fixed unlabeled policy pool.
- Threshold: stop when 0 < information gain ≤ 2.0%.
- Rationale: below 2% the surrogate is extracting negligible new structure from each
  additional cycle — a reliable signal of diminishing returns.
- Computation: compare mean predictive variance at cycle t vs cycle t-1 on the fixed pool.
- Robustness: zero-gain cycles (variance briefly rising due to noise) are excluded; only
  sustained low-gain cycles trigger this criterion.

3. Prediction stability
- Metric: fraction of pool predictions whose relative shift is < 1% between consecutive cycles.
- Threshold: stop when stability ≥ 90%.
- Rationale: when 90% of candidate policies are predicted identically cycle-to-cycle,
  the LCB acquisition ranking has effectively converged and further cycles cannot
  change the policy recommendation.
- Computation: on the fixed unlabeled policy pool, compare surrogate mean predictions
  between cycle t and t-1; count the proportion with |Δ|/max(|pred|,1) < 0.01.
- Robustness: requires at least 2 completed cycles to avoid triggering on the first cycle.

4. No-improvement streak
- Metric: number of consecutive cycles in which best_so_far_queries did not decrease.
- Threshold: stop when streak ≥ 5 consecutive cycles.
- Rationale: with 8 queries/cycle and 3 replicates, 5 consecutive failures to beat the
  incumbent (~120 board evaluations) is strong evidence that the true optimum has been
  found within measurement noise and additional exploration is unproductive.
- Computation: walk history in reverse; count cycles until best_so_far improved by > 1e-6.
- Robustness: uses best_so_far (cumulative minimum) rather than per-cycle best, so a
  single lucky cycle cannot reset a streak prematurely.

5. Cumulative experiment count
- Metric: al_cycle_count.
- Threshold: stop when al_cycle_count reaches 20.
- Rationale: hard budget cap — guarantees termination regardless of surrogate behaviour.
- Computation: increment once after each lab handoff / cycle completes.

## QC metrics

1. Process consistency checks
- Metric: replicate CV (%) of mean queries for the cycle's policy replicates.
- Acceptable range: < 30.0%.
- Action on failure: retry the offending batch once; if instability persists, flag for
  review before using the observations for model updates.

2. Hardware errors
- Metric: count of simulated hardware errors or read failures per cycle.
- Acceptable range: 0 during normal operation.
- Action on failure: allow operator review; either recover and continue or terminate the
  campaign and decide which completed data points to retain.

3. Ensemble uncertainty
- Metric: mean ensemble predictive variance on the unlabeled policy pool.
- Acceptable range: > 1.0e-08 and < 400.0.
- Action on failure: review surrogate fit, increase exploration, or reduce ensemble size
  if the variance collapses or explodes.

4. Lab throughput / hardware limits
- Metric: query_size per cycle.
- Acceptable range: 8 to 96 candidates per cycle.
- Action on failure: reject the cycle configuration before execution.

## Monitoring strategy

- Per cycle: compute stopping metrics, surrogate diagnostics, replicate CV, and CV readout error.
- Continuously during the cycle: log hardware-error events and detector unknown reads.
- On QC failure: retry once when possible, otherwise flag and pause operator review.
- On stopping trigger: save all figures, JSON history, and final campaign summary.

## Robustness safeguards

- Held-out MAE uses a fixed validation set never seen during proposal — prevents
  the surrogate appearing accurate by overfitting to already-sampled regions.
- Information gain is measured on a fixed unlabeled pool to avoid cherry-picking.
- Prediction stability uses relative change across the full pool, not just the current best.
- No-improvement streak uses best_so_far (cumulative minimum), so a single lucky cycle
  cannot prematurely reset the counter.
- The max-cycle budget criterion guarantees termination even if all other criteria fail.
- All five criteria are evaluated independently; the campaign stops as soon as any one
  is triggered, and the triggering reason is recorded for audit.

## Stopping criteria summary table

| # | Criterion | Metric | Threshold | Direction |
|---|-----------|--------|-----------|-----------|
| 1 | Held-out prediction error | MAE (queries) | ≤ 4.0 | lower is better |
| 2 | Information gain | variance reduction % | ≤ 2.0% | stop at diminishing returns |
| 3 | Prediction stability | % stable predictions | ≥ 90% | stop when converged |
| 4 | No-improvement streak | consecutive cycles | ≥ 5 | stop at plateau |
| 5 | Budget | al_cycle_count | ≥ 20 | hard cap |

## Final-cycle snapshot

- Completed cycles: 8
- Best observed mean queries: 30.67
- Held-out MAE: 6.82
- Information gain: 39.00%
- Prediction stability: 69.53%
- No-improvement streak: 5 cycles