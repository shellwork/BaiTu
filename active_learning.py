"""
Active Learning Module — Trypsin Kinetics Optimization
=======================================================
Implements Step 3 of the closed-loop automated science workflow:
Model Query Strategy (MQS) based on epistemic uncertainty from a Deep Ensemble.

Closed-loop iteration:
  1. Train TrypsinEnsemble on current labeled dataset  D_n
  2. Build discrete temperature candidate pool C
  3. Score each T in C by ensemble variance  Var_{ensemble}(v | T)
  4. Query T* = argmax_{T in C} score(T)  → send to wet lab
  5. Measure v at T*  → add (T*, v*) to D_n  → go to step 1
"""

import os
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")          # headless backend — safe in all environments
import matplotlib.pyplot as plt

from config import TrypsinConfig
from model import TrypsinEnsemble


# ---------------------------------------------------------------------------
# Step 3a — Build Candidate Pool
# ---------------------------------------------------------------------------

def build_candidate_pool(
        temp_min:  float = TrypsinConfig.TEMP_MIN,
        temp_max:  float = TrypsinConfig.TEMP_MAX,
        temp_step: float = TrypsinConfig.TEMP_STEP,
        temp_mean: float = TrypsinConfig.TEMP_MEAN,
        temp_std:  float = TrypsinConfig.TEMP_STD,
) -> tuple:
    """
    Discretize the instrument-operable temperature range into a candidate pool.

    Each candidate is a scalar temperature value.  We return both the raw
    (°C) and normalised forms so callers can use either.

    Returns:
        temps_raw  : np.ndarray  shape (N,)   raw temperatures [°C]
        temps_norm : np.ndarray  shape (N,)   z-score normalised temperatures
    """
    temps_raw  = np.arange(temp_min, temp_max + temp_step, temp_step)
    temps_norm = (temps_raw - temp_mean) / temp_std
    return temps_raw, temps_norm


# ---------------------------------------------------------------------------
# Step 3b — Score Candidates by Epistemic Uncertainty
# ---------------------------------------------------------------------------

def score_candidates(
        ensemble:        TrypsinEnsemble,
        temps_norm:      np.ndarray,
        substrate_concs: list = None,
        device:          str  = TrypsinConfig.DEVICE,
) -> np.ndarray:
    """
    Compute an uncertainty score for every candidate temperature.

    For a given T, uncertainty is defined as the *mean variance* of velocity
    predictions across a set of reference substrate concentrations:

        score(T) = (1/|S_ref|) * Σ_{[S] in S_ref}  Var_{members}(v | T, [S])

    Averaging over S_ref gives a temperature-level score and avoids the
    score being dominated by a single extreme substrate concentration.

    Args:
        ensemble       : TrypsinEnsemble in eval mode
        temps_norm     : normalised candidate temperatures  shape (N,)
        substrate_concs: reference [S] values [µM] for scoring
        device         : computation device

    Returns:
        scores : np.ndarray  shape (N,)   higher → more uncertain → more informative
    """
    if substrate_concs is None:
        substrate_concs = TrypsinConfig.QUERY_SUBSTRATE_CONCS

    ensemble.eval()
    scores = np.zeros(len(temps_norm))

    with torch.no_grad():
        for i, t_norm in enumerate(temps_norm):
            T = torch.tensor([[t_norm]], dtype=torch.float32).to(device)

            per_S_variances = []
            for s_val in substrate_concs:
                S      = torch.tensor([[s_val]], dtype=torch.float32).to(device)
                var_v  = ensemble.predict_uncertainty(T, S)   # (1, 1)
                per_S_variances.append(var_v.item())

            scores[i] = float(np.mean(per_S_variances))

    return scores


# ---------------------------------------------------------------------------
# Step 3c — Greedy MQS Query
# ---------------------------------------------------------------------------

def query_next_experiment(
        ensemble: TrypsinEnsemble,
        device:   str  = TrypsinConfig.DEVICE,
        verbose:  bool = True,
) -> tuple:
    """
    MQS greedy query: select the temperature with maximum epistemic uncertainty.

        T* = argmax_{T in Pool}  score(T)

    This is the Maximum Uncertainty Query Strategy — it greedily picks the
    single most informative point, maximising expected information gain
    under the Bayesian active learning framework.

    Args:
        ensemble : trained TrypsinEnsemble
        device   : computation device
        verbose  : print a human-readable query summary

    Returns:
        query_temp : float          recommended temperature [°C]
        scores     : np.ndarray     uncertainty scores for all candidates
        temps_raw  : np.ndarray     raw candidate temperatures [°C]
    """
    temps_raw, temps_norm = build_candidate_pool()
    scores = score_candidates(ensemble, temps_norm, device=device)

    best_idx   = int(np.argmax(scores))
    query_temp = float(temps_raw[best_idx])

    if verbose:
        sep = "=" * 62
        print(f"\n{sep}")
        print("  MQS Active Learning — Next Experiment Query")
        print(sep)
        print(f"  Candidate pool : {temps_raw[0]:.0f}°C → {temps_raw[-1]:.0f}°C "
              f"  (step = {TrypsinConfig.TEMP_STEP:.0f}°C, "
              f"{len(temps_raw)} points)")
        print(f"  Uncertainty range : [{scores.min():.3e}, {scores.max():.3e}]")
        print(f"\n  >> Recommended experiment  :  T = {query_temp:.1f} °C")
        print(f"  >> Epistemic uncertainty   :  {scores[best_idx]:.3e}")
        print(sep)

        top3 = np.argsort(scores)[::-1][:3]
        print("\n  Top-3 most informative temperatures:")
        for rank, idx in enumerate(top3, 1):
            marker = "  <<" if idx == best_idx else ""
            print(f"    {rank}. T = {temps_raw[idx]:5.1f}°C   "
                  f"uncertainty = {scores[idx]:.3e}{marker}")

    return query_temp, scores, temps_raw


# ---------------------------------------------------------------------------
# Visualisation helper
# ---------------------------------------------------------------------------

def plot_uncertainty_landscape(
        ensemble:     TrypsinEnsemble,
        device:       str = TrypsinConfig.DEVICE,
        save_path:    str = "evaluation/uncertainty_landscape.png",
        show:         bool = False,
):
    """
    Plot predicted mean velocity ± std across the temperature candidate pool
    for each reference substrate concentration.

    Saves the figure to `save_path` and optionally displays it.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    temps_raw, temps_norm = build_candidate_pool()
    substrate_concs = TrypsinConfig.QUERY_SUBSTRATE_CONCS

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # --- Left panel: mean v ± std for each [S] ---
    ax = axes[0]
    ensemble.eval()
    with torch.no_grad():
        for s_val in substrate_concs:
            means, stds = [], []
            for t_norm in temps_norm:
                T = torch.tensor([[t_norm]], dtype=torch.float32).to(device)
                S = torch.tensor([[s_val]],  dtype=torch.float32).to(device)
                mean_v, var_v, _, _ = ensemble(T, S)
                means.append(mean_v.item())
                stds.append(var_v.item() ** 0.5)   # std = sqrt(var)

            means = np.array(means)
            stds  = np.array(stds)
            ax.plot(temps_raw, means, label=f"[S]={s_val:.0f} µM")
            ax.fill_between(temps_raw, means - stds, means + stds, alpha=0.15)

    ax.set_xlabel("Temperature (°C)")
    ax.set_ylabel("Predicted velocity v  (µM/s)")
    ax.set_title("Ensemble mean ± std  across temperature")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # --- Right panel: temperature-level uncertainty score (MQS landscape) ---
    scores = score_candidates(ensemble, temps_norm, device=device)
    best_idx = int(np.argmax(scores))

    ax2 = axes[1]
    ax2.bar(temps_raw, scores, width=1.4, color="steelblue", alpha=0.7)
    ax2.axvline(temps_raw[best_idx], color="crimson", linestyle="--", linewidth=2,
                label=f"Query: {temps_raw[best_idx]:.1f}°C")
    ax2.set_xlabel("Temperature (°C)")
    ax2.set_ylabel("Epistemic uncertainty  (mean variance)")
    ax2.set_title("MQS uncertainty landscape")
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"[plot_uncertainty_landscape] Saved to '{save_path}'")
    if show:
        plt.show()
    plt.close()


# ---------------------------------------------------------------------------
# Convenience: load a saved ensemble from checkpoints
# ---------------------------------------------------------------------------

def load_ensemble(
        save_dir:   str = TrypsinConfig.TRYPSIN_CHECKPOINT_DIR,
        n_members:  int = TrypsinConfig.ENSEMBLE_SIZE,
        hidden_dim: int = TrypsinConfig.HIDDEN_DIM,
        device:     str = TrypsinConfig.DEVICE,
) -> TrypsinEnsemble:
    """
    Reconstruct and load a previously trained TrypsinEnsemble from disk.
    """
    ensemble = TrypsinEnsemble(n_members=n_members, hidden_dim=hidden_dim).to(device)
    for i, member in enumerate(ensemble.members):
        ckpt = os.path.join(save_dir, f"member_{i}.pth")
        if not os.path.exists(ckpt):
            raise FileNotFoundError(
                f"Checkpoint not found: '{ckpt}'. Train the ensemble first."
            )
        member.load_state_dict(torch.load(ckpt, map_location=device))
        member.eval()
    print(f"[load_ensemble] Loaded {n_members} members from '{save_dir}'")
    return ensemble


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Loading trained ensemble...")
    ens = load_ensemble()

    query_temp, scores, temps_raw = query_next_experiment(ens)

    print(f"\nGenerating uncertainty landscape plot...")
    plot_uncertainty_landscape(ens)

    print(f"\nDone.  Send the next trypsin assay to: T = {query_temp:.1f} °C")
