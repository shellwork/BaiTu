"""
agent/evaluator.py
──────────────────
The feedback engine of the closed-loop experiment system.

It is the component that:
  1. Checks whether the Michaelis-Menten fit meets acceptance criteria
  2. DIAGNOSES what went wrong when it doesn't
  3. Computes specific PARAMETER ADJUSTMENTS to fix the problem
  4. Builds a REPLANNING CONTEXT (natural language) that gets injected
     into the Planner's next LLM prompt

The evaluator sits between the Scientist (who produces the fit) and the Planner (who designs the next protocol):

    Scientist.analyze()
        │
        │  ExperimentResult (with r_squared, km, vmax)
        ▼
    EvaluatorAgent.evaluate_iteration()
        │
        │  IterationFeedback (with adjustments + replanning_context)
        ▼
    PlannerAgent.create_plan(feedback=...)
        │
        │  New ExperimentPlan (with adjusted concentrations)
        ▼
    Next iteration...

The key scientific principle: for a reliable Michaelis-Menten fit, the substrate concentrations must BRACKET Km — meaning you need 
data points both well below Km (to see the linear, first-order region) and well above Km (to see the saturated, zero-order plateau).
Ideally, concentrations span from ~0.2×Km to ~5×Km.

If all substrate concentrations are above Km, you only see the plateau and the fitter can't determine Km accurately.  If they're
all below Km, you only see the linear region and can't determine Vmax accurately.  Most of the diagnostic logic here is about
detecting and correcting this problem.
"""

from __future__ import annotations

import re
import logging
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional

from schemas.plan import ExperimentPlan
from schemas.result import ExperimentResult
from schemas.feedback import (
    IterationFeedback,
    FailureCategory,
    ParameterAdjustment,
)

logger = logging.getLogger(__name__)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Constants — thresholds and limits for evaluation
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

R_SQUARED_THRESHOLD = 0.8
"""
Minimum acceptable R² for the Michaelis-Menten fit. Below this, the model doesn't adequately describe the data and the
fitted Km/Vmax values are unreliable.

0.8 is a practical threshold — not as strict as 0.95 (which would require very clean data) but strict enough to ensure the curve shape
is clearly hyperbolic rather than linear or noisy.
"""

KM_MIN_MM = 0.001
"""
Lower bound for biologically plausible Km (in mM). Km < 0.001 mM means the enzyme has sub-micromolar affinity, which
is rare and almost certainly a fitting artifact in our system. If the fitter reports Km this low, it's because all substrate
concentrations are far above the true Km, so the curve looks flat and the fitter extrapolates Km ≈ 0.
"""

KM_MAX_MM = 100.0
"""
Upper bound for biologically plausible Km (in mM). Km > 100 mM means the enzyme has extremely low affinity, which is
biologically uncommon.  More likely, all substrate concentrations are below Km, so the curve looks linear and the fitter extrapolates
Km → ∞.

For trypsin with casein substrates, Km is typically 1–10 mM.
"""

VMAX_MIN = 0.0
"""
Vmax must be strictly positive.  A non-positive Vmax is physically impossible — it would mean the enzyme catalyzes the reverse reaction
or does nothing at infinite substrate concentration.
"""

MIN_SUBSTRATE_CONCENTRATIONS = 5
"""
Minimum number of distinct substrate concentrations needed for a reliable Michaelis-Menten fit.

The equation V = Vmax·[S]/(Km+[S]) has 2 free parameters, so mathematically 2 points suffice, but practically:
- 5-6 concentrations give the fitter enough points to distinguish the curve shape from noise
- Points should be spaced across the [S] range that brackets Km
- Each concentration should have replicates (across the 5 plates in our protocol) to estimate variance
"""

KM_BRACKET_LOW_FACTOR = 0.2
"""
The lowest substrate concentration should ideally be ≤ 0.2 × Km. At [S] = 0.2×Km, the reaction rate is only ~17% of Vmax, which
gives you a clear data point in the "rising" part of the curve.

If your lowest [S] is >> 0.2×Km, you're missing the linear region entirely and the fitter struggles to pin down Km.
"""

KM_BRACKET_HIGH_FACTOR = 5.0
"""
The highest substrate concentration should ideally be ≥ 5 × Km. At [S] = 5×Km, the reaction rate is ~83% of Vmax, which is
close enough to the plateau that the fitter can estimate Vmax.

If your highest [S] is << 5×Km, you're missing the saturation region and the fitter can't determine where the curve levels off.
"""

ENZYME_TO_KM_MAX_RATIO = 0.1
"""
Maximum acceptable ratio of [enzyme] to Km.

In Michaelis-Menten kinetics, the derivation assumes [E] << [S], i.e., the enzyme doesn't significantly deplete the substrate.
If [E] > 0.1 × Km, the enzyme may consume substrate so fast that the system never reaches steady state, and the measured rates
don't reflect true V0 (initial velocity).

When this is violated, we recommend lowering [E] to ~0.01 × Km.
"""


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# EvaluatorAgent
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@dataclass
class EvaluatorAgent:
    """
    Evaluates a completed experiment iteration and produces structured feedback that drives the next planning cycle.

    This agent diagnoses and prescribes.

    Usage:
        evaluator = EvaluatorAgent()
        feedback = evaluator.evaluate_iteration(
            iteration=1,
            plan=verified_plan,
            result=experiment_result,
            prompt_used=planner.last_prompt,
            rag_references=["SOP-001", "SOP-002"],
        )
        if not feedback.passed:
            # feedback.replanning_context contains instructions for the
            # next LLM prompt; feed it back to the planner
            new_plan = planner.create_plan(intent, feedback=feedback)
    """

    llm_provider: Optional[Any] = None
    """
    Optional LLM provider for generating richer diagnostic text. Currently unused but reserved for future use where the evaluator
    could ask the LLM to explain results or suggest creative fixes.
    """

    r_squared_threshold: float = R_SQUARED_THRESHOLD
    """
    The R² threshold for this evaluator instance.  Defaults to 0.8 but can be overridden (e.g., set to 0.9 for stricter acceptance).
    """

    # ══════════════════════════════════════════════════════════════════
    # Main entry point
    # ══════════════════════════════════════════════════════════════════

    def evaluate_iteration(
        self,
        iteration: int,
        plan: ExperimentPlan,
        result: ExperimentResult,
        prompt_used: str,
        rag_references: List[str],
    ) -> IterationFeedback:
        """
        Run ALL evaluation checks on a completed iteration and return
        structured feedback.

        This method runs 6 sequential checks:
          1. R² goodness of fit
          2. Km plausibility
          3. Vmax plausibility
          4. Sufficient data points
          5. Prompt quality (only flagged when experiments also fail)
          6. RAG reference quality

        If ALL checks pass → feedback.passed = True, loop stops.
        If ANY check fails → feedback contains diagnosis + adjustments
        + replanning context, loop continues.

        Parameters
        ----------
        iteration : int
            The 1-indexed iteration number.
        plan : ExperimentPlan
            The plan that was executed (used for extracting concentrations).
        result : ExperimentResult
            The analysis result from the ScientistAgent (must contain
            metrics with keys like r_squared, km, vmax, etc.).
        prompt_used : str
            The full LLM prompt the PlannerAgent generated (cached in
            planner.last_prompt).  Evaluated for completeness.
        rag_references : List[str]
            The references returned by the RAG system.  Checked for
            placeholder patterns.

        Returns
        -------
        IterationFeedback
            The structured feedback object.  The orchestrator reads
            feedback.passed to decide whether to continue.  The planner
            reads feedback.replanning_context on the next iteration.
        """
        # Accumulators for all failures and adjustments found across checks
        failures: List[FailureCategory] = []
        adjustments: List[ParameterAdjustment] = []
        diagnosis_parts: List[str] = []

        # Extract the key metrics from the result
        # The ScientistAgent should have populated these during analyze()
        metrics = result.metrics
        r2 = metrics.get("r_squared")
        km = metrics.get("km")
        vmax = metrics.get("vmax")

        # ── Check 1: Goodness of fit (R²) ───────────────────────────
        # This is the PRIMARY acceptance criterion.  If R² < 0.8, the
        # Michaelis-Menten model doesn't fit the data well enough for
        # the Km and Vmax estimates to be trustworthy.
        if r2 is not None and r2 < self.r_squared_threshold:
            failures.append(FailureCategory.POOR_FIT)
            diagnosis_parts.append(
                f"R² = {r2:.4f} is below the {self.r_squared_threshold} threshold. "
                f"The Michaelis-Menten model is not fitting the data well."
            )
            # Diagnose WHY the fit is poor — this is where the real
            # enzyme kinetics reasoning happens
            adjustments.extend(self._diagnose_poor_fit(metrics, plan))

        # ── Check 2: Km plausibility ─────────────────────────────────
        # Even if R² looks okay, a Km outside [0.001, 100] mM suggests
        # the fit converged to a local minimum that isn't biologically
        # meaningful.
        if km is not None:
            if km < KM_MIN_MM or km > KM_MAX_MM:
                failures.append(FailureCategory.KM_OUT_OF_RANGE)
                diagnosis_parts.append(
                    f"Km = {km} mM is outside the biologically plausible range "
                    f"[{KM_MIN_MM}, {KM_MAX_MM}] mM."
                )
                adjustments.extend(self._diagnose_km_issue(km, metrics, plan))

        # ── Check 3: Vmax plausibility ───────────────────────────────
        # Vmax must be strictly positive.  If it's ≤ 0, something is
        # fundamentally wrong with the enzyme or the measurement.
        if vmax is not None and vmax <= VMAX_MIN:
            failures.append(FailureCategory.VMAX_OUT_OF_RANGE)
            diagnosis_parts.append(
                f"Vmax = {vmax} is non-positive, which is physically impossible. "
                f"The enzyme may be inactive or the signal is below detection."
            )
            adjustments.extend(self._diagnose_vmax_issue(vmax, metrics, plan))

        # ── Check 4: Sufficient substrate concentrations ─────────────
        # You need at least 5 distinct [S] values spread across the
        # curve to get a reliable 2-parameter fit.
        n_concentrations = metrics.get("n_substrate_concentrations", 0)
        if n_concentrations < MIN_SUBSTRATE_CONCENTRATIONS:
            failures.append(FailureCategory.INSUFFICIENT_DATA)
            diagnosis_parts.append(
                f"Only {n_concentrations} substrate concentrations tested. "
                f"Michaelis-Menten fitting needs at least "
                f"{MIN_SUBSTRATE_CONCENTRATIONS} for reliability."
            )
            adjustments.append(ParameterAdjustment(
                parameter="n_substrate_concentrations",
                current_value=n_concentrations,
                # Recommend at least 2 more than current, or the minimum,
                # whichever is larger
                recommended_value=max(n_concentrations + 2, MIN_SUBSTRATE_CONCENTRATIONS + 1),
                rationale=(
                    "Increase the number of distinct substrate concentrations. "
                    "Spread them across the range 0.2×Km to 5×Km for best "
                    "curve coverage."
                ),
            ))

        # ── Check 5: Prompt quality ──────────────────────────────────
        # We only flag prompt issues when there are ALSO experimental
        # failures — a mediocre prompt that still produces a passing
        # experiment is fine.  But if the experiment failed AND the
        # prompt was missing critical context, the prompt may have
        # contributed to the failure.
        prompt_issues = self._check_prompt_quality(prompt_used, failures)
        if prompt_issues:
            failures.append(FailureCategory.PROMPT_DEFICIENCY)
            diagnosis_parts.extend(prompt_issues)

        # ── Check 6: RAG reference quality ───────────────────────────
        # If the RAG system is still returning placeholder stubs, the
        # planner has no real SOP knowledge to draw from.
        rag_issues = self._check_rag_quality(rag_references)
        if rag_issues:
            failures.append(FailureCategory.RAG_MISS)
            diagnosis_parts.extend(rag_issues)

        # ── Assemble the feedback object ─────────────────────────────
        passed = len(failures) == 0
        diagnosis_summary = (
            " | ".join(diagnosis_parts) if diagnosis_parts
            else "All evaluation checks passed."
        )

        # Build the replanning context ONLY if the iteration failed.
        # This is the natural-language text block that will be injected
        # into the Planner's next LLM prompt.
        replanning_context = ""
        if not passed:
            replanning_context = self._build_replanning_context(
                iteration=iteration,
                failures=failures,
                adjustments=adjustments,
                diagnosis_summary=diagnosis_summary,
                previous_metrics=metrics,
            )

        feedback = IterationFeedback(
            iteration=iteration,
            passed=passed,
            failure_categories=failures,
            r_squared=r2,
            km=km,
            vmax=vmax,
            parameter_adjustments=adjustments,
            diagnosis_summary=diagnosis_summary,
            replanning_context=replanning_context,
        )

        logger.info(
            "Iteration %d evaluation: %s | R²=%s | Km=%s | Vmax=%s | failures: %s",
            iteration,
            "PASSED" if passed else "FAILED",
            r2, km, vmax,
            [f.value for f in failures],
        )

        return feedback

    # ══════════════════════════════════════════════════════════════════
    # Diagnostic methods — each one figures out WHY a check failed
    # and recommends specific parameter adjustments
    # ══════════════════════════════════════════════════════════════════

    def _diagnose_poor_fit(
        self, metrics: Dict[str, Any], plan: ExperimentPlan
    ) -> List[ParameterAdjustment]:
        """
        Diagnose why R² is below threshold and recommend adjustments.

        The most common causes of a poor Michaelis-Menten fit:

        1. SUBSTRATE RANGE DOESN'T BRACKET Km
           - If all [S] >> Km: you only see the plateau (saturation). The curve looks flat and the fitter can't find Km.
             Fix: add lower [S] values (~0.2 × Km).
           - If all [S] << Km: you only see the linear region. The curve looks straight and the fitter can't find Vmax.
             Fix: add higher [S] values (~5 × Km).

        2. ENZYME CONCENTRATION TOO HIGH
           If [E] is > 10% of Km, the enzyme depletes the substrate before a steady-state rate is established.  The measured
           "initial rates" aren't really initial rates — they're from a period where [S] is already dropping.
           Fix: reduce [E] to ~1% of Km.

        3. NOT ENOUGH DATA POINTS
           (Handled separately in check 4, but compounds the problem.)
        """
        adjustments: List[ParameterAdjustment] = []

        # Pull out the key numbers from the metrics
        km = metrics.get("km")
        substrate_concs = metrics.get("substrate_concentrations", [])
        enzyme_conc = metrics.get("enzyme_concentration")

        # ── Sub-diagnosis A: Does the substrate range bracket Km? ────
        if km and substrate_concs:
            s_min = min(substrate_concs)
            s_max = max(substrate_concs)

            # Calculate the ideal [S] range based on the current Km estimate.
            # Even though this Km came from a bad fit, it's our best guess
            # and using it to bracket better will improve the next fit.
            ideal_low = KM_BRACKET_LOW_FACTOR * km    # 0.2 × Km
            ideal_high = KM_BRACKET_HIGH_FACTOR * km  # 5 × Km

            # Check if the lowest [S] is too high (missing the rising phase)
            # We use 2× ideal_low as the threshold because if s_min is
            # already ≤ 2× ideal, we're reasonably close
            if s_min > ideal_low * 2:
                adjustments.append(ParameterAdjustment(
                    parameter="substrate_concentration_min",
                    current_value=s_min,
                    recommended_value=round(ideal_low, 4),
                    unit="mM",
                    rationale=(
                        f"Lowest substrate concentration ({s_min} mM) is too far above "
                        f"the estimated Km ({km:.4f} mM).  At these concentrations, the "
                        f"enzyme is nearly saturated and the rate is close to Vmax — you "
                        f"only see the plateau.  Add lower concentrations "
                        f"(~{ideal_low:.4f} mM, i.e., 0.2×Km) to capture the rising, "
                        f"first-order region of the Michaelis-Menten curve."
                    ),
                ))

            # Check if the highest [S] is too low (missing the plateau)
            # We use 0.5× ideal_high as the threshold
            if s_max < ideal_high * 0.5:
                adjustments.append(ParameterAdjustment(
                    parameter="substrate_concentration_max",
                    current_value=s_max,
                    recommended_value=round(ideal_high, 4),
                    unit="mM",
                    rationale=(
                        f"Highest substrate concentration ({s_max} mM) may not be "
                        f"saturating.  At [S] = 5×Km = {ideal_high:.4f} mM, the rate "
                        f"reaches ~83%% of Vmax.  Without data near saturation, the "
                        f"fitter can't determine Vmax accurately.  Add higher "
                        f"concentrations to see the plateau."
                    ),
                ))

        # ── Sub-diagnosis B: Is [E] too high relative to Km? ────────
        # The Michaelis-Menten derivation assumes [E] << [S].
        # If [E] > 10% of Km, the enzyme consumes substrate too fast
        # for steady-state kinetics to hold.
        if enzyme_conc and km:
            if enzyme_conc > ENZYME_TO_KM_MAX_RATIO * km:
                # Recommend reducing to 1% of Km
                recommended_e = round(km * 0.01, 6)
                adjustments.append(ParameterAdjustment(
                    parameter="enzyme_concentration",
                    current_value=enzyme_conc,
                    recommended_value=recommended_e,
                    unit="mM",
                    rationale=(
                        f"Enzyme concentration ({enzyme_conc} mM) is >"
                        f" {ENZYME_TO_KM_MAX_RATIO * 100:.0f}% of Km ({km:.4f} mM).  "
                        f"This can cause substrate depletion before a steady-state "
                        f"rate is established, making initial velocity measurements "
                        f"unreliable.  Reduce [E] to ~{recommended_e} mM (1%% of Km)."
                    ),
                ))

        # ── Fallback: generic advice if we couldn't be specific ──────
        # This happens when we don't have enough metadata to do
        # targeted diagnosis (e.g., substrate_concentrations not
        # reported in the metrics)
        if not adjustments:
            adjustments.append(ParameterAdjustment(
                parameter="substrate_concentration_range",
                current_value="unknown",
                recommended_value="broaden range with more points around Km",
                rationale=(
                    "R² is below threshold but there is insufficient metadata "
                    "to diagnose the specific cause.  General recommendation: "
                    "expand the substrate concentration range, ensure points "
                    "are spread from ~0.2×Km to ~5×Km, and include at least "
                    f"{MIN_SUBSTRATE_CONCENTRATIONS} distinct concentrations."
                ),
            ))

        return adjustments

    def _diagnose_km_issue(
        self,
        km: float,
        metrics: Dict[str, Any],
        plan: ExperimentPlan,
    ) -> List[ParameterAdjustment]:
        """
        Diagnose why the fitted Km is outside the plausible range.

        The logic is different for too-low vs. too-high Km:

        Km too LOW (< 0.001 mM):
            → All [S] values are in the saturated region, so the curve looks flat and the fitter pushes Km toward zero.
            → Fix: add LOWER substrate concentrations so you can see where the curve transitions from linear to saturated.

        Km too HIGH (> 100 mM):
            → All [S] values are in the linear region, so the curve looks like a straight line and the fitter pushes Km
              toward infinity.
            → Fix: add HIGHER substrate concentrations to see the saturation plateau.
        """
        adjustments: List[ParameterAdjustment] = []
        substrate_concs = metrics.get("substrate_concentrations", [])

        if km < KM_MIN_MM:
            # Km is unrealistically low — data is all in the plateau
            # Recommend adding very low [S] to resolve the rising phase
            recommended_low = round(km * 0.1, 6) if km > 0 else 0.0001
            adjustments.append(ParameterAdjustment(
                parameter="substrate_concentration_min",
                current_value=min(substrate_concs) if substrate_concs else "unknown",
                recommended_value=recommended_low,
                unit="mM",
                rationale=(
                    f"Km = {km} mM is extremely low, suggesting all tested "
                    f"substrate concentrations are far above the true Km.  "
                    f"The curve appears flat (fully saturated), so the fitter "
                    f"extrapolates Km ≈ 0.  Add much lower concentrations "
                    f"(try {recommended_low} mM) to resolve the transition "
                    f"from first-order to zero-order kinetics."
                ),
            ))

        elif km > KM_MAX_MM:
            # Km is unrealistically high — data is all in the linear region
            # Recommend adding very high [S] to see the saturation plateau
            recommended_high = round(km * 5, 2)
            adjustments.append(ParameterAdjustment(
                parameter="substrate_concentration_max",
                current_value=max(substrate_concs) if substrate_concs else "unknown",
                recommended_value=recommended_high,
                unit="mM",
                rationale=(
                    f"Km = {km} mM is very high, suggesting all tested "
                    f"substrate concentrations are well below Km.  The curve "
                    f"appears linear (first-order region only), so the fitter "
                    f"extrapolates Km → ∞.  Add much higher concentrations "
                    f"(try {recommended_high} mM) to see where the rate "
                    f"levels off at Vmax."
                ),
            ))

        return adjustments

    def _diagnose_vmax_issue(
        self,
        vmax: float,
        metrics: Dict[str, Any],
        plan: ExperimentPlan,
    ) -> List[ParameterAdjustment]:
        """
        Diagnose why Vmax is non-positive.

        Vmax ≤ 0 means no measurable catalytic activity.  Possible causes:
        - Enzyme stock is inactive or denatured (e.g., boiled pineapple juice)
        - Enzyme concentration is too low for the spectrophotometer to detect the change in absorbance
        - Substrate is not accessible to the enzyme (aggregation, wrong buffer)

        The main recommendation is to increase enzyme concentration and
        verify that the enzyme stock is actually active.
        """
        return [ParameterAdjustment(
            parameter="enzyme_concentration",
            current_value=metrics.get("enzyme_concentration", "unknown"),
            recommended_value="increase (verify stock activity first)",
            rationale=(
                f"Vmax = {vmax} is non-positive, indicating no detectable "
                f"enzymatic activity.  Possible causes: (1) enzyme stock is "
                f"inactive or denatured — verify activity with a positive "
                f"control; (2) enzyme concentration is too low for detectable "
                f"signal — increase [E] and confirm absorbance change is above "
                f"the spectrophotometer's noise floor; (3) check that the "
                f"substrate (azocasein) is properly dissolved and accessible."
            ),
        )]

    def _check_prompt_quality(
        self,
        prompt: str,
        existing_failures: List[FailureCategory],
    ) -> List[str]:
        """
        Check whether deficiencies in the Planner's LLM prompt may have contributed to experimental failures.

        We ONLY flag prompt issues when there are already experimental failures (POOR_FIT, KM_OUT_OF_RANGE, etc.).  A prompt that
        produces a passing experiment is fine regardless of quality.

        We check for:
        - Missing concentration specifications (if there was a fit failure)
        - Missing kinetics terminology (km, vmax, michaelis-menten)
        - Prompt too short to be informative

        Returns a list of diagnostic strings (empty if no issues).
        """
        issues: List[str] = []
        prompt_lower = prompt.lower()

        # Only evaluate prompt quality when there are experimental failures
        # that the prompt might have caused.  No point flagging a prompt
        # that led to a passing experiment.
        experimental_failures = {
            FailureCategory.POOR_FIT,
            FailureCategory.KM_OUT_OF_RANGE,
            FailureCategory.VMAX_OUT_OF_RANGE,
            FailureCategory.INSUFFICIENT_DATA,
        }
        has_experimental_failures = bool(
            experimental_failures.intersection(existing_failures)
        )

        if not has_experimental_failures:
            # No experimental failures → prompt is fine, don't nitpick
            return issues

        # Check: did the prompt specify substrate/enzyme concentrations?
        # If not, the LLM had to guess concentrations, which may have
        # been inappropriate for the enzyme system.
        if "concentration" not in prompt_lower:
            issues.append(
                "Prompt did not specify substrate or enzyme concentrations. "
                "The LLM may have chosen arbitrary values that don't bracket "
                "Km appropriately."
            )

        # Check: did the prompt reference kinetics concepts?
        # Without mentioning Michaelis-Menten, Km, or Vmax, the LLM
        # doesn't know the goal is parameter estimation and may design
        # a generic protocol instead of a kinetics-optimized one.
        kinetics_terms = ["michaelis", "km", "vmax", "kinetics", "kinetic"]
        if not any(term in prompt_lower for term in kinetics_terms):
            issues.append(
                "Prompt did not reference Michaelis-Menten kinetics, Km, or "
                "Vmax.  The LLM lacks the context to design a protocol "
                "optimized for kinetic parameter estimation."
            )

        # Check: is the prompt long enough to be informative?
        # A very short prompt (< 100 chars) almost certainly lacks the
        # detail needed for the LLM to generate a good protocol.
        if len(prompt) < 100:
            issues.append(
                f"Prompt is only {len(prompt)} characters — likely too short "
                f"to contain sufficient experimental context."
            )

        return issues

    def _check_rag_quality(self, references: List[str]) -> List[str]:
        """
        Check whether the RAG system returned real references or stubs.

        The SOPRAGClient currently returns hardcoded placeholders like ["SOP-001", "SOP-002"].  If ALL references match this pattern,
        the vector database hasn't been populated with real standard operating procedures, and the Planner is generating protocols
        without domain knowledge.

        This doesn't block execution, but it's a signal that the RAG system needs attention.

        Returns a list of diagnostic strings (empty if references look real).
        """
        issues: List[str] = []

        if not references:
            # No references at all — RAG returned nothing
            issues.append(
                "RAG system returned zero references.  The Planner has no "
                "SOP context for protocol generation."
            )
            return issues

        # Check if ALL references match the placeholder pattern SOP-NNN
        placeholder_pattern = re.compile(r"^SOP-\d{3}$")
        placeholder_count = sum(
            1 for ref in references
            if placeholder_pattern.match(ref)
        )

        if placeholder_count == len(references):
            issues.append(
                f"All {len(references)} RAG references are placeholders "
                f"(SOP-001, SOP-002, etc.).  The vector database does not "
                f"appear to contain real standard operating procedures.  "
                f"The Planner is generating protocols without SOP context."
            )

        return issues

    # ══════════════════════════════════════════════════════════════════
    # Replanning context builder — the critical output
    # ══════════════════════════════════════════════════════════════════

    def _build_replanning_context(
        self,
        iteration: int,
        failures: List[FailureCategory],
        adjustments: List[ParameterAdjustment],
        diagnosis_summary: str,
        previous_metrics: Dict[str, Any],
    ) -> str:
        """
        Build the natural-language text block that gets injected into the Planner's next LLM prompt.

        The text it produces is the mechanism by which evaluation results flow backward through the pipeline into the next planning cycle.
        Without this, the planner would just regenerate the same protocol and the loop would never converge.

        The structure of the output:
        1. Header: "FEEDBACK FROM ITERATION N"
        2. Status: FAILED with list of failure categories
        3. Previous metrics: R², Km, Vmax (so the LLM knows the numbers)
        4. Numbered adjustments: each with current → recommended value
           and scientific rationale
        5. Explicit instruction: "You MUST incorporate these adjustments"

        The tone is directive — we're telling the LLM what it MUST change, not suggesting.  LLMs respond better to explicit constraints.
        """
        lines = [
            f"=== FEEDBACK FROM ITERATION {iteration} ===",
            f"Status: FAILED — requires parameter adjustment before next run.",
            f"Failure categories: {', '.join(f.value for f in failures)}",
            f"",
            f"Diagnosis: {diagnosis_summary}",
            f"",
            f"Previous iteration metrics:",
            f"  R²   = {previous_metrics.get('r_squared', 'N/A')}",
            f"  Km   = {previous_metrics.get('km', 'N/A')} mM",
            f"  Vmax = {previous_metrics.get('vmax', 'N/A')}",
            f"  Substrate concentrations tested: "
            f"{previous_metrics.get('substrate_concentrations', 'N/A')}",
            f"  Enzyme concentration: "
            f"{previous_metrics.get('enzyme_concentration', 'N/A')} mM",
            f"",
            f"Required adjustments for next iteration:",
        ]

        # Add each adjustment as a numbered item with rationale
        for i, adj in enumerate(adjustments, 1):
            # Show the current → recommended value with units
            unit_str = f" {adj.unit}" if adj.unit else ""
            lines.append(
                f"  {i}. {adj.parameter}: "
                f"{adj.current_value} → {adj.recommended_value}{unit_str}"
            )
            lines.append(f"     Reason: {adj.rationale}")
            lines.append("")  # blank line between adjustments for readability

        # Strong directive — LLMs follow explicit instructions better
        lines.append(
            "IMPORTANT: The next experiment protocol MUST incorporate ALL "
            "of the adjustments listed above.  Do NOT reuse the same "
            "concentrations or conditions that produced the failed results.  "
            "The goal is to achieve R² ≥ 0.8 for the Michaelis-Menten fit."
        )

        return "\n".join(lines)
