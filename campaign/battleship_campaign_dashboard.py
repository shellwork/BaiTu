"""
Streamlit dashboard for the Battleship closed-loop campaign.

Run (from repository root):
  streamlit run campaign/battleship_campaign_dashboard.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import streamlit as st


DEFAULT_RESULTS_DIR = "battleship_campaign_results"

# QC thresholds mirroring battleship_campaign.py THRESHOLDS
_QC_CV_THRESH = 30.0
_QC_VAR_LO    = 1e-8
_QC_VAR_HI    = 400.0


def load_json(path: Path) -> Optional[object]:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def load_history(results_dir: Path) -> Optional[List[Dict]]:
    data = load_json(results_dir / "history.json")
    return data if isinstance(data, list) else None


def load_observations(results_dir: Path) -> Optional[List[Dict]]:
    data = load_json(results_dir / "observations.json")
    return data if isinstance(data, list) else None


# ── helpers ──────────────────────────────────────────────────────────────────

def _delta(history: List[Dict], key: str) -> Optional[float]:
    """Return change of *key* from first to last cycle (last − first)."""
    if len(history) < 2:
        return None
    try:
        return float(history[-1][key]) - float(history[0][key])
    except (KeyError, TypeError):
        return None


def _status_badge(ok: bool) -> str:
    return "🟢 OK" if ok else "🔴 FLAG"


# ── main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    st.set_page_config(
        page_title="Battleship Campaign Dashboard",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # ── sidebar ───────────────────────────────────────────────────────────────
    st.sidebar.title("Campaign Dashboard")
    results_dir = Path(st.sidebar.text_input("Results directory", DEFAULT_RESULTS_DIR))

    history      = load_history(results_dir)
    observations = load_observations(results_dir)
    config       = load_json(results_dir / "config.json")

    if not history:
        st.warning("No `history.json` found. Please run `python battleship_campaign.py` first.")
        return

    last  = history[-1]
    first = history[0]

    # ── top KPI strip ─────────────────────────────────────────────────────────
    st.title("🚢 Battleship Active-Learning Campaign")
    st.caption(f"Results loaded from `{results_dir}`")

    k1, k2, k3, k4, k5, k6 = st.columns(6)
    k1.metric(
        "Completed Cycles",
        int(last["al_cycle_count"]),
        delta=None,
    )
    k2.metric(
        "Best-so-far Queries",
        f"{last['best_so_far_queries']:.1f}",
        delta=f"{_delta(history, 'best_so_far_queries'):+.1f}" if _delta(history, 'best_so_far_queries') is not None else None,
        delta_color="inverse",
    )
    k3.metric(
        "Held-out MAE",
        f"{last['heldout_mae_queries']:.2f}",
        delta=f"{_delta(history, 'heldout_mae_queries'):+.2f}" if _delta(history, 'heldout_mae_queries') is not None else None,
        delta_color="inverse",
    )
    k4.metric(
        "Prediction Stability",
        f"{last['prediction_stability_pct']:.1f}%",
    )
    k5.metric(
        "Mean Ensemble Var",
        f"{last['mean_ensemble_var']:.2f}",
        delta=f"{_delta(history, 'mean_ensemble_var'):+.2f}" if _delta(history, 'mean_ensemble_var') is not None else None,
        delta_color="inverse",
    )

    stop_reason = last.get("stop_reason") or "not_triggered"
    k6.metric("Stop Reason", stop_reason)

    # QC status banner
    qc_ok = last.get("qc_all_ok", False)
    if qc_ok:
        st.success("✅ QC status: All checks passed for the final cycle.")
    else:
        flags = ", ".join(last.get("qc_flags", [])) or "unknown"
        st.warning(f"⚠️  QC flags on final cycle: **{flags}**")

    st.divider()

    # ── config sidebar panel ──────────────────────────────────────────────────
    with st.sidebar.expander("Campaign configuration", expanded=False):
        st.json(config or {})

    best_policy = last.get("best_policy", {})
    if best_policy:
        st.sidebar.subheader("Best policy found")
        for k, v in best_policy.items():
            st.sidebar.progress(float(v), text=f"{k}: {v:.3f}")

    # ── tabs ──────────────────────────────────────────────────────────────────
    tab_overview, tab_learning, tab_stopping, tab_qc, tab_baseline, tab_obs = st.tabs([
        "Overview", "Learning", "Stopping Criteria", "QC Monitoring", "Baseline Comparison", "Observations"
    ])

    # ── Overview tab ──────────────────────────────────────────────────────────
    with tab_overview:
        c_left, c_right = st.columns([2, 1])

        with c_left:
            img_path = results_dir / "learning_curves.png"
            if img_path.exists():
                st.image(str(img_path), caption="Learning curves", width="stretch")

        with c_right:
            st.subheader("Campaign summary")
            st.write(f"**Stop reason:** `{stop_reason}`")
            st.write(f"**Cycles completed:** {int(last['al_cycle_count'])}")
            st.write(f"**Best queries:** {last['best_so_far_queries']:.2f}")
            st.write(f"**Final held-out MAE:** {last['heldout_mae_queries']:.2f}")
            st.write(f"**Final R²:** {last.get('heldout_r2', float('nan')):.3f}")

            st.subheader("Best policy weights")
            if best_policy:
                bp_df = pd.DataFrame(
                    [{"Component": k, "Weight": round(v, 4)} for k, v in best_policy.items()]
                )
                st.dataframe(bp_df, use_container_width=False, hide_index=True)

        st.subheader("Per-cycle summary table")
        summary_df = pd.DataFrame([
            {
                "Cycle":              h["al_cycle_count"],
                "Cycle mean queries": round(h["cycle_mean_queries"], 2),
                "Cycle best queries": round(h["cycle_best_queries"], 2),
                "Best-so-far":        round(h["best_so_far_queries"], 2),
                "MAE":                round(h["heldout_mae_queries"], 2),
                "R²":                 round(h.get("heldout_r2", float("nan")), 3),
                "QC":                 _status_badge(h["qc_all_ok"]),
                "Stop":               "✅" if h["stop_triggered"] else "",
            }
            for h in history
        ])
        st.dataframe(summary_df, use_container_width=False, hide_index=True)

    # ── Learning tab ─────────────────────────────────────────────────────────
    with tab_learning:
        st.subheader("Algorithm learning over simulated cycles")

        img_path = results_dir / "learning_curves.png"
        if img_path.exists():
            st.image(str(img_path), caption="Learning curves (best-so-far, surrogate MAE / R²)", width="stretch")

        with st.expander("Why does the campaign learn?", expanded=True):
            st.markdown("""
**How learning happens**

Each cycle the campaign proposes a batch of candidate acquisition policies
(4-dimensional weight vectors over *prob / entropy / target / checker*),
runs them on the Battleship simulator, and observes the number of queries
needed to finish each board.  A bootstrap **kernel ensemble surrogate** is
refitted on all observations collected so far, mapping the policy space
→ mean query count.

The surrogate drives the next batch via a **Lower-Confidence Bound (LCB)**
acquisition function:
> `score = predicted_mean − 1.5 × √(predicted_variance)`

Low predicted mean (fast solver) and high variance (unexplored region) both
make a policy attractive, balancing exploitation and exploration.

**What limits convergence**

| Factor | Effect |
|--------|--------|
| Small replicate count (n=3) | High board-layout variance inflates CoV and noises the surrogate |
| Kernel bandwidth | Too narrow → over-fits; too wide → under-resolves policy differences |
| 4-D policy simplex | Vast landscape; 8 queries/cycle ≪ required coverage |
| Stochastic boards | Same policy can score very differently on easy vs hard board layouts |

**What accelerates convergence**

* Hunt-Target heuristic (`target` weight) dominates: it focuses queries on
  ship neighbourhoods, which is the most informative region.
* LCB naturally revisits unexplored regions, preventing premature convergence.
* Diversity constraint (L1 ≥ 0.30) prevents the batch from collapsing onto
  a single policy.
            """)

    # ── Stopping Criteria tab ─────────────────────────────────────────────────
    with tab_stopping:
        st.subheader("Stopping-criterion metrics over cycles")

        img_path = results_dir / "stopping_criteria.png"
        if img_path.exists():
            st.image(str(img_path), caption="Stopping criteria trajectories", width="stretch")

        st.subheader("Stopping metrics table")
        stop_df = pd.DataFrame([
            {
                "Cycle":               h["al_cycle_count"],
                "Held-out MAE":        round(h["heldout_mae_queries"], 3),
                "Info gain %":         round(h["information_gain_pct"], 2),
                "Pred. stability %":   round(h["prediction_stability_pct"], 2),
                "Cycle count":         h["al_cycle_count"],
                "Stop triggered":      "✅" if h["stop_triggered"] else "",
                "Reason":              h.get("stop_reason", ""),
            }
            for h in history
        ])
        st.dataframe(stop_df, use_container_width=False, hide_index=True)

        plan_path = results_dir / "qc_stopping_criteria_plan.md"
        if plan_path.exists():
            with st.expander("📄 Written QC & Stopping-Criteria Plan", expanded=True):
                st.markdown(plan_path.read_text(encoding="utf-8"))

    # ── QC Monitoring tab ─────────────────────────────────────────────────────
    with tab_qc:
        st.subheader("QC metric monitoring")

        img_path = results_dir / "qc_timeline.png"
        if img_path.exists():
            st.image(str(img_path), caption="QC timeline (heatmap + trajectories)", width="stretch")

        st.subheader("QC metrics table")
        qc_df = pd.DataFrame([
            {
                "Cycle":             h["al_cycle_count"],
                "Replicate CoV %":   round(h["replicate_cv_pct"], 2),
                "CoV threshold":     _QC_CV_THRESH,
                "CoV pass":          _status_badge(h["replicate_cv_pct"] < _QC_CV_THRESH),
                "Ensemble var":      round(h["mean_ensemble_var"], 4),
                "Var pass":          _status_badge(_QC_VAR_LO < h["mean_ensemble_var"] < _QC_VAR_HI),
                "Readout mismatch %": round(100.0 * h["mean_cv_error_rate"], 3),
                "Unknown rate %":    round(100.0 * h.get("mean_unknown_rate", 0.0), 3),
                "Hardware errors":   h["hardware_errors"],
                "QC overall":        _status_badge(h["qc_all_ok"]),
                "Flags":             ", ".join(h["qc_flags"]) if h["qc_flags"] else "OK",
            }
            for h in history
        ])
        st.dataframe(qc_df, use_container_width=False, hide_index=True)

        with st.expander("QC metric definitions & acceptable ranges"):
            st.markdown(f"""
| Metric | Formula | Acceptable range | Action on failure |
|--------|---------|-----------------|-------------------|
| **Replicate CoV** | std(queries) / mean(queries) × 100 | < {_QC_CV_THRESH:.0f}% | Retry batch once; flag if persists |
| **Ensemble variance** | mean predictive var on unlabelled pool | ({_QC_VAR_LO:.0e}, {_QC_VAR_HI:.0f}) | Review surrogate; increase exploration |
| **Readout mismatch** | fraction of wells where image label ≠ board truth | 0% (ideal) | Check image pipeline; rerun plate read |
| **Unknown rate** | fraction of wells returning "unknown" from RGB classifier | 0% (ideal) | Check RGB tolerance; recalibrate |
| **Hardware errors** | count of simulated hardware failures per cycle | 0 | Operator review; may terminate campaign |
| **Query size** | number of policy candidates per cycle | 8 – 96 | Reject cycle config before execution |
            """)

    # ── Baseline Comparison tab ───────────────────────────────────────────────
    with tab_baseline:
        st.subheader("Bayesian-optimised policy vs baselines")

        baseline_data = load_json(results_dir / "baselines.json")

        # ── KPI strip: one metric card per strategy ───────────────────────────
        if baseline_data and isinstance(baseline_data, dict):
            bd = baseline_data  # type: ignore[assignment]
            opt = bd.get("campaign_optimum", {})
            bl  = bd.get("baselines", {})

            all_strategies = list(bl.items()) + [("campaign_opt", opt)]
            n_cols = len(all_strategies)
            kpi_cols = st.columns(n_cols)

            for col, (key, meta) in zip(kpi_cols, all_strategies):
                mean_q = meta.get("mean_queries", float("nan"))
                std_q  = meta.get("std_queries",  float("nan"))
                impr   = meta.get("improvement_vs_random_pct", float("nan"))
                label  = meta.get("label", key).split("\n")[0]
                col.metric(
                    label=label,
                    value=f"{mean_q:.1f} q",
                    delta=f"{impr:+.1f}% vs random",
                    delta_color="inverse" if key == "random" else "normal",
                    help=f"Mean ± std:  {mean_q:.1f} ± {std_q:.1f} queries",
                )

            st.divider()

        # ── Comparison figure ─────────────────────────────────────────────────
        img_path = results_dir / "baseline_comparison.png"
        if img_path.exists():
            st.image(str(img_path), caption="Baseline comparison: bar chart · CDF · campaign learning curve", width="stretch")
        else:
            st.info("No `baseline_comparison.png` found. Re-run `python battleship_campaign.py` to generate it.")

        # ── Numeric comparison table ──────────────────────────────────────────
        if baseline_data and isinstance(baseline_data, dict):
            bd = baseline_data  # type: ignore[assignment]
            opt = bd.get("campaign_optimum", {})
            bl  = bd.get("baselines", {})
            random_mean = bl.get("random", {}).get("mean_queries", float("nan"))

            rows = []
            for key, meta in bl.items():
                rows.append({
                    "Strategy":              meta.get("label", key),
                    "Type":                  "Baseline",
                    "Mean queries":          round(meta.get("mean_queries", float("nan")), 2),
                    "Std queries":           round(meta.get("std_queries",  float("nan")), 2),
                    "vs Random (%)":         f"{meta.get('improvement_vs_random_pct', float('nan')):+.1f}%",
                    "Policy (prob/ent/tgt/chk)": "  ".join(f"{v:.2f}" for v in (meta.get("policy") or [])),
                })
            rows.append({
                "Strategy":              opt.get("label", "Campaign optimum").split("  ")[0],
                "Type":                  "🏆 Campaign optimum",
                "Mean queries":          round(opt.get("mean_queries", float("nan")), 2),
                "Std queries":           round(opt.get("std_queries",  float("nan")), 2),
                "vs Random (%)":         f"{opt.get('improvement_vs_random_pct', float('nan')):+.1f}%",
                "Policy (prob/ent/tgt/chk)": "  ".join(
                    f"{v:.2f}" for v in (opt.get("policy", {}).values() if isinstance(opt.get("policy"), dict) else [])
                ),
            })

            cmp_df = pd.DataFrame(rows)
            st.subheader("Numeric comparison")
            st.dataframe(cmp_df, use_container_width=False, hide_index=True)

            # ── Per-seed query distribution ───────────────────────────────────
            with st.expander("Per-seed query distribution (all strategies)", expanded=False):
                dist_rows = []
                for key, meta in bl.items():
                    for seed_idx, q in enumerate(meta.get("queries", [])):
                        dist_rows.append({
                            "Strategy": meta.get("label", key),
                            "Type": "Baseline",
                            "Seed index": seed_idx,
                            "Queries": q,
                        })
                opt_qs = opt.get("queries", [])
                for seed_idx, q in enumerate(opt_qs):
                    dist_rows.append({
                        "Strategy": "Campaign optimum",
                        "Type": "Campaign",
                        "Seed index": seed_idx,
                        "Queries": q,
                    })
                if dist_rows:
                    dist_df = pd.DataFrame(dist_rows)
                    st.dataframe(dist_df, use_container_width=False, hide_index=True)

                    # aggregate stats grouped by strategy
                    agg = (
                        dist_df.groupby("Strategy")["Queries"]
                        .agg(["mean", "std", "min", "max", "count"])
                        .round(2)
                        .reset_index()
                    )
                    agg.columns = ["Strategy", "Mean", "Std", "Min", "Max", "N"]
                    st.caption("Aggregated statistics")
                    st.dataframe(agg, use_container_width=False, hide_index=True)

        elif not (results_dir / "baselines.json").exists():
            st.info("No `baselines.json` found. Re-run `python battleship_campaign.py` to generate full comparison data.")

        # ── Rationale ─────────────────────────────────────────────────────────
        with st.expander("Why compare against baselines?"):
            st.markdown("""
The four fixed-policy baselines are the **null hypotheses** against which the
campaign must demonstrate value.  Each represents a different extreme of the
decision-making spectrum:

| Baseline | Policy weights | What it tests |
|----------|---------------|---------------|
| **Random** | Uniform random | Worst-case reference; any informed strategy must beat this |
| **Max Probability** | prob=1 | Does systematic exploration add value over pure exploitation? |
| **Max Entropy** | entropy=1 | Do domain-aware heuristics outperform generic uncertainty sampling? |
| **Hunt-Target** | target=1 | Is the Bayesian optimum better than the strongest human-inspired heuristic? |

The campaign is justified only if its optimised policy **beats all four** baselines.
A failure on any column suggests the surrogate did not learn a useful landscape
model within the given budget.

**Reading the CDF panel**: a curve shifted to the *left* finishes boards faster.
The campaign optimum should dominate all baselines across the full distribution,
not just on the mean — an advantage only on easy boards (right tail) would
not be meaningful.
            """)

    # ── Observations tab ─────────────────────────────────────────────────────
    with tab_obs:
        st.subheader("Raw policy observations")
        if observations:
            obs_df = pd.DataFrame(observations)
            st.dataframe(obs_df, use_container_width=False, hide_index=True)
        else:
            st.info("No observations found.")


if __name__ == "__main__":
    main()
