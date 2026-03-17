"""
qc_dashboard.py — Streamlit Campaign Management Interface for BaiTu.

Displays all QC metrics and campaign progress in a multi-tab dashboard.
Reads simulation history from plots/simulation_history.json (produced by
run_simulation.py), or runs a quick demo simulation on demand.

Usage
-----
  streamlit run qc_dashboard.py
"""

from __future__ import annotations

import json
import math
import os
import subprocess
import sys
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import streamlit as st

from qc_framework import THRESHOLDS, QCMonitor

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="BaiTu — Campaign Management",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

PALETTE = {
    "blue":   "#2563EB",
    "orange": "#EA580C",
    "green":  "#16A34A",
    "red":    "#DC2626",
    "purple": "#7C3AED",
    "grey":   "#6B7280",
}

HISTORY_PATH = "plots/simulation_history.json"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

@st.cache_data
def load_history(path: str) -> Optional[List[Dict]]:
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def _safe(val, default=float("nan")):
    if val is None or (isinstance(val, float) and math.isnan(val)):
        return default
    return val


# ---------------------------------------------------------------------------
# Matplotlib figure helpers (returned as st.pyplot figures)
# ---------------------------------------------------------------------------

def _fig_learning_curves(history: List[Dict]) -> plt.Figure:
    cycles = [h["cycle"] + 1 for h in history]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # R² panel
    ax = axes[0]
    for key, label, color in [
        ("v0_r2",   "v₀ rate",  PALETTE["blue"]),
        ("kcat_r2", "k_cat",    PALETTE["orange"]),
        ("km_r2",   "K_m",      PALETTE["green"]),
    ]:
        vals = [_safe(h.get(key)) for h in history]
        ax.plot(cycles, vals, marker="o", color=color, label=label, linewidth=2)
    ax.axhline(0, color=PALETTE["grey"], linestyle="--", linewidth=0.8)
    ax.set_xlabel("Cycle"); ax.set_ylabel("R²")
    ax.set_title("Held-out R² Score"); ax.legend(); ax.grid(alpha=0.3)
    ax.set_ylim(-0.1, 1.05)

    # Log MAE panel
    ax = axes[1]
    for key, label, color in [
        ("mae_log_kcat", "log k_cat MAE", PALETTE["orange"]),
        ("mae_log_km",   "log K_m MAE",   PALETTE["green"]),
    ]:
        vals = [_safe(h.get(key)) for h in history]
        ax.plot(cycles, vals, marker="s", color=color, label=label, linewidth=2)
    ax.axhline(THRESHOLDS["mae_log_kcat_threshold"], color=PALETTE["red"],
               linestyle="--", label="Stop threshold", linewidth=2)
    ax.set_xlabel("Cycle"); ax.set_ylabel("MAE (log units)")
    ax.set_title("Log-scale MAE"); ax.legend(); ax.grid(alpha=0.3)

    plt.tight_layout()
    return fig


def _fig_uncertainty(history: List[Dict]) -> plt.Figure:
    cycles = np.array([h["cycle"] + 1 for h in history], dtype=float)
    vars_  = np.array([_safe(h.get("mean_ensemble_var"), 0) for h in history])

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(cycles, vars_, marker="o", color=PALETTE["blue"], linewidth=2, label="Mean pool variance")

    valid = vars_ > 0
    if valid.sum() >= 3:
        try:
            log_v = np.log(vars_[valid])
            coeffs = np.polyfit(cycles[valid], log_v, 1)
            xf = np.linspace(cycles[valid].min(), cycles[valid].max(), 100)
            ax.plot(xf, np.exp(np.polyval(coeffs, xf)), "--", color=PALETTE["orange"],
                    label=f"Exp. fit τ={-1/coeffs[0]:.1f} cycles")
        except Exception:
            pass

    ax.set_xlabel("Cycle"); ax.set_ylabel("Mean Ensemble Variance")
    ax.set_title("Epistemic Uncertainty Reduction")
    ax.set_yscale("log"); ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
    return fig


def _fig_info_gain(history: List[Dict]) -> plt.Figure:
    cycles = [h["cycle"] + 1 for h in history]
    gains  = [_safe(h.get("info_gain_nats"), 0) for h in history]
    colors = [PALETTE["green"] if g > 0.01 else PALETTE["grey"] for g in gains]

    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(cycles, gains, color=colors, edgecolor="white", width=0.6)
    for bar, g in zip(bars, gains):
        if g > 0:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.001,
                    f"{g:.3f}", ha="center", va="bottom", fontsize=9)
    ax.set_xlabel("Cycle"); ax.set_ylabel("ΔH (nats)")
    ax.set_title("Information Gain per Cycle")
    ax.set_xticks(cycles); ax.grid(alpha=0.3, axis="y")
    p1 = mpatches.Patch(color=PALETTE["green"], label="Positive gain")
    p2 = mpatches.Patch(color=PALETTE["grey"],  label="Negligible gain")
    ax.legend(handles=[p1, p2])
    plt.tight_layout()
    return fig


def _fig_qc_heatmap(history: List[Dict]) -> plt.Figure:
    qc_keys = [
        ("qc_replicate_cv_ok",    "Replicate CV < 10%"),
        ("qc_kcat_bounds_ok",     "k_cat in bounds"),
        ("qc_km_bounds_ok",       "K_m in bounds"),
        ("qc_ensemble_var_ok",    "Ensemble var > 0"),
        ("qc_query_size_ok",      "Query size valid"),
        ("qc_no_hardware_errors", "No hardware errors"),
    ]
    n_metrics = len(qc_keys)
    n_cycles  = len(history)
    matrix = np.array([[1.0 if h.get(k, True) else 0.0 for h in history]
                        for k, _ in qc_keys])

    fig, ax = plt.subplots(figsize=(max(5, n_cycles * 1.3), max(3, n_metrics * 0.8)))
    cmap = matplotlib.colors.ListedColormap([PALETTE["red"], PALETTE["green"]])
    ax.imshow(matrix, cmap=cmap, vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(n_cycles))
    ax.set_xticklabels([f"Cycle {h['cycle']+1}" for h in history])
    ax.set_yticks(range(n_metrics))
    ax.set_yticklabels([label for _, label in qc_keys])
    ax.set_title("QC Metrics Timeline")
    for i in range(n_metrics):
        for j in range(n_cycles):
            ax.text(j, i, "✓" if matrix[i, j] else "✗",
                    ha="center", va="center", fontsize=14, color="white", fontweight="bold")
    plt.tight_layout()
    return fig


def _fig_stability(history: List[Dict]) -> plt.Figure:
    cycles = [h["cycle"] + 1 for h in history]
    stab   = [_safe(h.get("prediction_stability_pct"), 0) * 100 for h in history]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(cycles, stab, marker="s", color=PALETTE["purple"], linewidth=2, label="Stability %")
    ax.axhline(THRESHOLDS["prediction_stability"] * 100, color=PALETTE["red"],
               linestyle="--", linewidth=2, label=f"Threshold ({THRESHOLDS['prediction_stability']:.0%})")
    ax.set_xlabel("Cycle"); ax.set_ylabel("Prediction Stability (%)")
    ax.set_title("Prediction Stability (fraction of pool predictions changing < 1%)")
    ax.set_ylim(0, 105); ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Main dashboard
# ---------------------------------------------------------------------------

def main():
    st.title("🧬 BaiTu — Automated Enzyme Kinetics Campaign")
    st.caption("Active Learning Campaign Management Interface")

    # ── Sidebar ──────────────────────────────────────────────────────────────
    with st.sidebar:
        st.header("Campaign Status")

        history = load_history(HISTORY_PATH)

        if history is None:
            st.warning("No simulation history found.\nRun `python run_simulation.py` first.")
            n_cycles_done = 0
            n_experiments = 0
        else:
            n_cycles_done  = len(history)
            last           = history[-1]
            n_experiments  = last.get("n_labeled", 0)

            st.metric("Cycles completed", f"{n_cycles_done} / {int(THRESHOLDS['max_cycles'])}")
            st.metric("Experiments run",  n_experiments)
            budget_pct = n_cycles_done / int(THRESHOLDS["max_cycles"])
            st.progress(budget_pct, text=f"Budget: {budget_pct:.0%}")

            # Stopping status
            stop_triggered = any(h.get("stop_triggered", False) for h in history)
            if stop_triggered:
                st.error("🛑 Stopping criterion triggered")
            else:
                st.success("▶ Campaign running")

            # Last cycle QC badge
            qc_ok = last.get("qc_all_ok", True)
            if qc_ok:
                st.success("✅ Last cycle QC: PASS")
            else:
                flags = last.get("qc_flags", [])
                st.error(f"❌ QC flags: {len(flags)} issue(s)")

        st.divider()
        st.subheader("Run Simulation")
        if st.button("▶ Run simulation now", type="primary"):
            with st.spinner("Running closed-loop simulation …"):
                result = subprocess.run(
                    [sys.executable, "run_simulation.py", "--cycles", "5", "--query_size", "32",
                     "--ensemble", "5", "--epochs", "30"],
                    capture_output=True, text=True, cwd=os.path.dirname(__file__) or ".",
                )
            if result.returncode == 0:
                st.success("Simulation complete! Reload to see results.")
                st.cache_data.clear()
            else:
                st.error(f"Simulation failed:\n{result.stderr[-1000:]}")

        st.divider()
        st.subheader("Thresholds")
        st.json({k: v for k, v in THRESHOLDS.items()}, expanded=False)

    # ── No history guard ──────────────────────────────────────────────────
    if history is None:
        st.info(
            "👋 Welcome! No simulation data yet.\n\n"
            "Run `python run_simulation.py` from the terminal, "
            "or use the **Run simulation now** button in the sidebar."
        )
        return

    # ── Tabs ─────────────────────────────────────────────────────────────
    tabs = st.tabs([
        "📈 Learning Progress",
        "🛑 Stopping Criteria",
        "🔍 QC Status",
        "🧠 Model Diagnostics",
        "📋 Experiment Log",
    ])

    # ── Tab 1: Learning Progress ──────────────────────────────────────────
    with tabs[0]:
        st.header("Learning Progress")

        last = history[-1]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("v₀ R²",   f"{_safe(last.get('v0_r2'),0):.4f}")
        c2.metric("k_cat R²", f"{_safe(last.get('kcat_r2'),0):.4f}")
        c3.metric("K_m R²",  f"{_safe(last.get('km_r2'),0):.4f}")
        c4.metric("Labeled", last.get("n_labeled", "—"))

        st.pyplot(_fig_learning_curves(history))

        st.subheader("Per-cycle metrics table")
        df_metrics = pd.DataFrame([{
            "Cycle": h["cycle"] + 1,
            "n_labeled": h.get("n_labeled"),
            "v₀ R²": round(_safe(h.get("v0_r2")), 4),
            "k_cat R²": round(_safe(h.get("kcat_r2")), 4),
            "K_m R²": round(_safe(h.get("km_r2")), 4),
            "MAE log(k_cat)": round(_safe(h.get("mae_log_kcat")), 4),
            "MAE log(K_m)":   round(_safe(h.get("mae_log_km")),   4),
        } for h in history])
        st.dataframe(df_metrics, use_container_width=True)

    # ── Tab 2: Stopping Criteria ──────────────────────────────────────────
    with tabs[1]:
        st.header("Stopping Criteria")

        last = history[-1]
        c1, c2, c3 = st.columns(3)

        mae_kcat = _safe(last.get("mae_log_kcat"), float("inf"))
        c1.metric(
            "Criterion 2: MAE log(k_cat)",
            f"{mae_kcat:.4f} nats",
            delta=f"Threshold: {THRESHOLDS['mae_log_kcat_threshold']}",
            delta_color="off",
        )
        if mae_kcat < THRESHOLDS["mae_log_kcat_threshold"]:
            c1.success("✅ Below threshold — STOP eligible")
        else:
            c1.info(f"▶ Above threshold ({mae_kcat:.4f} > {THRESHOLDS['mae_log_kcat_threshold']})")

        stab = _safe(last.get("prediction_stability_pct"), 0)
        c2.metric(
            "Criterion 3: Prediction Stability",
            f"{stab:.1%}",
            delta=f"Threshold: {THRESHOLDS['prediction_stability']:.0%}",
            delta_color="off",
        )
        if stab >= THRESHOLDS["prediction_stability"]:
            c2.success("✅ Above threshold — STOP eligible (needs 2 consecutive)")
        else:
            c2.info(f"▶ Below threshold ({stab:.1%} < {THRESHOLDS['prediction_stability']:.0%})")

        budget_used = len(history)
        c3.metric(
            "Criterion 1: Cycle Budget",
            f"{budget_used} / {int(THRESHOLDS['max_cycles'])}",
        )
        st.progress(budget_used / int(THRESHOLDS["max_cycles"]))
        if budget_used >= int(THRESHOLDS["max_cycles"]):
            c3.error("🛑 Budget exhausted — STOP triggered")
        else:
            c3.info(f"▶ {int(THRESHOLDS['max_cycles']) - budget_used} cycle(s) remaining")

        st.subheader("Stopping Criteria Trajectories")
        st.pyplot(_fig_stability(history))

        # Evaluate and show decision
        monitor = QCMonitor()
        decision = monitor.should_stop(
            [{
                "cycle": h["cycle"],
                "mae_log_kcat": h.get("mae_log_kcat", float("inf")),
                "mae_log_km":   h.get("mae_log_km",   float("inf")),
                "prediction_stability_pct": h.get("prediction_stability_pct", 0),
                "n_labeled": h.get("n_labeled", 0),
            } for h in history]
        )
        if decision.should_stop:
            st.error(f"🛑 **STOP triggered** — reason: `{decision.reason}`")
            st.json(decision.metrics_at_stop)
        else:
            st.success("▶ Campaign continues — no stopping criterion met")

        with st.expander("Safeguards against premature stopping"):
            st.markdown(
                """
                - **Prediction-stability** requires **2 consecutive** cycles above 95 % — a single
                  lucky cycle does not trigger early stop.
                - **MAE criterion** requires **both** k_cat AND K_m MAE below threshold
                  simultaneously, preventing a one-sided fit from triggering early stop.
                - **Replicate CV** is checked *before* lab handoff so noisy batches are caught
                  before they contaminate the training set.
                """
            )

    # ── Tab 3: QC Status ─────────────────────────────────────────────────
    with tabs[2]:
        st.header("Quality Control Status")

        last = history[-1]

        st.subheader("Current Cycle QC Snapshot")
        c1, c2, c3 = st.columns(3)
        with c1:
            cv = last.get("qc_replicate_cv_ok", True)
            st.metric("Replicate CV < 10%", "PASS ✅" if cv else "FAIL ❌")
            kcat_b = last.get("qc_kcat_bounds_ok", True)
            st.metric("k_cat in bounds", "PASS ✅" if kcat_b else "FAIL ❌")
        with c2:
            km_b = last.get("qc_km_bounds_ok", True)
            st.metric("K_m in bounds", "PASS ✅" if km_b else "FAIL ❌")
            var_ok = last.get("qc_ensemble_var_ok", True)
            st.metric("Ensemble var > 0", "PASS ✅" if var_ok else "FAIL ❌")
        with c3:
            qs_ok = last.get("qc_query_size_ok", True)
            st.metric("Query size valid", "PASS ✅" if qs_ok else "FAIL ❌")
            hw_ok = last.get("qc_no_hardware_errors", True)
            st.metric("No hardware errors", "PASS ✅" if hw_ok else "FAIL ❌")

        flags = last.get("qc_flags", [])
        if flags:
            st.warning("QC flags raised:")
            for f in flags:
                st.write(f"  • {f}")

        st.subheader("QC Timeline (all cycles)")
        st.pyplot(_fig_qc_heatmap(history))

        st.subheader("Acceptable Operating Ranges")
        ranges_df = pd.DataFrame([
            {"Metric": "Replicate CV",     "Min": "—",     "Max": "10%",    "Action": "Re-run assay"},
            {"Metric": "k_cat prediction", "Min": "1e-4 s⁻¹", "Max": "1e8 s⁻¹", "Action": "Flag out-of-bounds"},
            {"Metric": "K_m prediction",   "Min": "1e-7 M",   "Max": "0.1 M",    "Action": "Flag out-of-bounds"},
            {"Metric": "Ensemble variance", "Min": "> 0",  "Max": "—",      "Action": "Re-initialise ensemble"},
            {"Metric": "Query size",        "Min": "8",    "Max": "96",     "Action": "Adjust to hardware"},
            {"Metric": "Hardware errors",   "Min": "0",    "Max": "0",      "Action": "Halt; user intervention"},
        ])
        st.table(ranges_df)

    # ── Tab 4: Model Diagnostics ──────────────────────────────────────────
    with tabs[3]:
        st.header("Model Diagnostics")

        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Epistemic Uncertainty (Ensemble Variance)")
            st.pyplot(_fig_uncertainty(history))
            st.caption(
                "Mean ensemble variance over the unlabeled pool. A log-scale "
                "exponential decay indicates the model is learning efficiently."
            )

        with c2:
            st.subheader("Information Gain per Cycle")
            st.pyplot(_fig_info_gain(history))
            st.caption(
                "ΔH = 0.5 × ln(var_prev / var_curr). Diminishing returns indicate "
                "the campaign is approaching convergence."
            )

        last = history[-1]
        st.subheader("Ensemble Health Indicators")
        d1, d2, d3 = st.columns(3)
        d1.metric("Mean pool variance (last cycle)",
                  f"{_safe(last.get('mean_ensemble_var'), 0):.4e}")
        d2.metric("Variance > 0 (no collapse)",
                  "✅ Yes" if last.get("qc_ensemble_var_ok", True) else "❌ No")
        d3.metric("Info gain (last cycle)",
                  f"{_safe(last.get('info_gain_nats'), 0):.4f} nats")

    # ── Tab 5: Experiment Log ─────────────────────────────────────────────
    with tabs[4]:
        st.header("Experiment Log")

        rows = []
        for h in history:
            rows.append({
                "Cycle":        h["cycle"] + 1,
                "n_labeled":    h.get("n_labeled", "—"),
                "v₀ R²":        round(_safe(h.get("v0_r2")), 4),
                "k_cat R²":     round(_safe(h.get("kcat_r2")), 4),
                "K_m R²":       round(_safe(h.get("km_r2")), 4),
                "MAE log k_cat": round(_safe(h.get("mae_log_kcat")), 4),
                "MAE log K_m":   round(_safe(h.get("mae_log_km")),   4),
                "Stability %":  f"{_safe(h.get('prediction_stability_pct'),0):.1%}",
                "Info gain":    round(_safe(h.get("info_gain_nats"), 0), 4),
                "QC OK":        "✅" if h.get("qc_all_ok", True) else "❌",
                "Flags":        "; ".join(h.get("qc_flags", [])) or "—",
                "Stop?":        "🛑 YES" if h.get("stop_triggered") else "—",
            })

        df_log = pd.DataFrame(rows)
        st.dataframe(df_log, use_container_width=True, height=300)

        # Download button
        csv = df_log.to_csv(index=False)
        st.download_button(
            label="⬇ Download experiment log (CSV)",
            data=csv,
            file_name="baitu_campaign_log.csv",
            mime="text/csv",
        )

        # Show existing plot files
        st.subheader("Generated Plots")
        plot_files = [
            ("learning_curves.png",      "Learning Curves"),
            ("stopping_criteria.png",    "Stopping Criteria"),
            ("uncertainty_reduction.png","Uncertainty Reduction"),
            ("information_gain.png",     "Information Gain"),
            ("qc_timeline.png",          "QC Timeline"),
        ]
        for fname, label in plot_files:
            path = os.path.join("plots", fname)
            if os.path.exists(path):
                with st.expander(f"📊 {label}"):
                    st.image(path, use_container_width=True)

        # QC plan markdown
        plan_path = os.path.join("plots", "qc_stopping_criteria_plan.md")
        if os.path.exists(plan_path):
            with st.expander("📄 QC & Stopping Criteria Plan (Markdown)"):
                with open(plan_path) as f:
                    st.markdown(f.read())


if __name__ == "__main__":
    main()
