# QC and Stopping-Criteria Plan

## Overview
This document describes the quality control (QC) metrics, acceptable operating
ranges, stopping criteria, and monitoring strategy for the BaiTu automated
enzyme kinetics active learning campaign.

---

## Chosen QC Metrics

### 1. Process Consistency — Replicate Coefficient of Variation (CV)
- **Definition**: Standard deviation / mean of replicate v₀ measurements × 100 %
- **Threshold**: < 10%
- **Monitoring**: Continuously, per queried batch, before lab handoff
- **Action on failure**: Flag batch for human review; optionally re-run assay

### 2. Biological Plausibility Bounds
- **k_cat range**: [1e-04, 1e+08] s⁻¹
- **K_m range**: [1e-07, 1e-01] M
- **Monitoring**: Per cycle, on model predictions over the full pool
- **Action on failure**: Flag out-of-bounds predictions; investigate data quality

### 3. Ensemble Health — Variance > 0
- **Definition**: Mean ensemble variance over the pool must be strictly positive
- **Threshold**: variance > 0 (mode collapse guard)
- **Monitoring**: Per cycle, after ensemble training
- **Action on failure**: Re-initialise ensemble with different random seeds

### 4. Lab Throughput — Query Size
- **Range**: [8, 96] samples per cycle
- **Monitoring**: Before each lab handoff
- **Action on failure**: Adjust query_size to match hardware capacity

### 5. Hardware Errors
- **Definition**: Count of instrument/robot errors per cycle
- **Threshold**: 0 (any error triggers review)
- **Monitoring**: Continuously during experiment execution
- **Action on failure**: Allow user to recover protocol or terminate experiment;
  user selects which data points (if any) to add to the training set

---

## Stopping Criteria

### Criterion 1: Budget Exhausted
- **Condition**: Cycle count ≥ 5
- **Metric**: `al_cycle_count` — incremented after each lab handoff
- **Rationale**: Fixed resource budget prevents runaway experiments

### Criterion 2: Held-Out Prediction Error Below Threshold
- **Condition**: log-scale MAE(k_cat) < 0.3 AND
  log-scale MAE(K_m) < 0.3 (natural log units)
- **Metric**: Computed on fixed held-out validation set at end of each cycle
- **Rationale**: Model has reached sufficient predictive accuracy for the scientific goal

### Criterion 3: Prediction Stability
- **Condition**: ≥ 95% of pool predictions
  shift by < 1 % between consecutive cycles, for 2
  consecutive cycles
- **Metric**: Fraction of pool samples with |Δpred / pred_prev| < 0.01
- **Rationale**: Model has converged — additional experiments yield diminishing returns

---

## Acceptable Operating Ranges (Summary)

| Metric | Acceptable Range | Action if Violated |
|--------|----------------|--------------------|
| Replicate CV | < 10% | Re-run assay / flag for review |
| k_cat prediction | [1e-04, 1e+08] s⁻¹ | Flag out-of-bounds samples |
| K_m prediction | [1e-07, 1e-01] M | Flag out-of-bounds samples |
| Ensemble variance | > 0 | Re-initialise ensemble |
| Query size | [8, 96] | Adjust to hardware capacity |
| Hardware errors | 0 | Halt; user intervention |

---

## Monitoring Strategy

- **Per-cycle**: QC report generated after each cycle; sent to campaign dashboard.
- **Continuous**: Hardware errors monitored in real time during experiment execution.
- **Rolling**: Prediction stability evaluated over a sliding window of 2 cycles.

## Robustness Safeguards

- The prediction-stability criterion requires **2 consecutive** cycles above the
  threshold to prevent premature stopping from a single lucky cycle.
- The MAE criterion requires **both** k_cat AND K_m to be below threshold
  simultaneously, preventing a one-sided fit from triggering early stop.
- Replicate CV is checked **before** lab handoff so that noisy batches are
  caught before they contaminate the training set.

---

## Campaign History Summary

| Cycle | n_labeled | MAE log k_cat | MAE log K_m | Stability % | QC OK |
|-------|-----------|---------------|-------------|-------------|-------|
| 1 | 82 | 0.635 | 1.024 | 0.0% | ✓ |
| 2 | 114 | 0.496 | 0.638 | 0.3% | ✗ |
| 3 | 146 | 0.519 | 0.480 | 0.0% | ✓ |
| 4 | 178 | 0.421 | 0.613 | 0.2% | ✓ |
| 5 | 210 | 0.375 | 0.484 | 0.4% | ✓ |
