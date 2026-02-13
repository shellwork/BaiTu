"""
schemas/feedback.py

This module defines the data structures that carry diagnostic information
between the Evaluator agent and the Planner agent across iterations.

The core idea:
    After each experiment iteration, the Evaluator checks whether the results
    are acceptable (R² ≥ 0.8, Km and Vmax plausible, etc.). If they're NOT,
    the Evaluator produces an IterationFeedback object that contains:
        1. What went wrong (failure categories)
        2. The specific numbers that failed (R², Km, Vmax)
        3. What parameters to change and by how much (ParameterAdjustment list)
        4. A human/LLM-readable diagnosis summary
        5. A replanning_context string that gets injected directly into the
           Planner's next LLM prompt so the model knows what to fix

    If results ARE acceptable, the feedback object simply has passed=True
    and the orchestrator stops the loop.

Data flow:
    Evaluator.evaluate_iteration()
        → produces IterationFeedback
            → consumed by Planner.create_plan(feedback=...)
                → feedback.replanning_context is injected into the LLM prompt
                    → LLM generates a new protocol with adjusted concentrations
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from enum import Enum


class FailureCategory(str, Enum):
    """
    Categorizes WHY an iteration failed. Each category triggers different diagnostic logic in the Evaluator and different adjustment strategies.

    The evaluator can flag multiple categories per iteration. For example, a single run might have both POOR_FIT and INSUFFICIENT_DATA if the R²
    is low AND there weren't enough substrate concentrations tested.

    Categories:
        POOR_FIT:
            The Michaelis-Menten curve fit has R² below the threshold (default 0.8). This is the most common failure mode. Usually means the substrate
            concentration range doesn't bracket Km well enough — either all concentrations are in the saturated plateau (above Km) or all are
            in the linear region (below Km).

        KM_OUT_OF_RANGE:
            The fitted Km value is biologically implausible. Typical enzyme Km values range from ~0.001 mM to ~100 mM. Values outside this range
            usually indicate the curve fit is garbage, not that you've discovered a novel enzyme. This often co-occurs with POOR_FIT.

        VMAX_OUT_OF_RANGE:
            The fitted Vmax is non-positive. Since Vmax represents the maximum reaction velocity, it must be > 0. A non-positive value means the
            enzyme is inactive, the concentration is too low to produce signal, or the fitting algorithm diverged.

        INSUFFICIENT_DATA:
            Not enough substrate concentrations or replicates to produce a reliable curve fit. Michaelis-Menten fitting generally needs at
            least 5-6 different substrate concentrations spanning ~0.2*Km to ~5*Km for a good fit.

        PROMPT_DEFICIENCY:
            The LLM prompt used by the Planner was missing critical context (e.g., didn't mention concentrations, didn't reference Michaelis-Menten,
            was too short). This is an agent quality issue, not an experiment issue.

        RAG_MISS:
            The literature/SOP search returned irrelevant or placeholder results. Indicates the vector database isn't populated or the query didn't
            match useful documents.

        EXECUTION_ERROR:
            Something went wrong in the hardware execution stage — code generation failed, simulation caught an error, or the instrument returned bad data.
    """
  
    POOR_FIT = "poor_fit"
    KM_OUT_OF_RANGE = "km_out_of_range"
    VMAX_OUT_OF_RANGE = "vmax_out_of_range"
    INSUFFICIENT_DATA = "insufficient_data"
    PROMPT_DEFICIENCY = "prompt_deficiency"
    RAG_MISS = "rag_miss"
    EXECUTION_ERROR = "execution_error"


@dataclass
class ParameterAdjustment:
    """
    Represents a single recommended change to an experimental parameter for the next iteration.

    The Evaluator produces a list of these when an iteration fails. Each adjustment says: "Change THIS parameter FROM this value TO that value,
    and here's WHY."

    These adjustments serve two purposes:
        1. They're included in the replanning_context string so the LLM can understand what to change in the next protocol.
        2. They provide a structured, machine-readable record of what was adjusted between iterations (for audit trail / logging).

    Examples for enzyme kinetics:
        ParameterAdjustment(
            parameter="substrate_concentration_min",
            current_value=1.0,
            recommended_value=0.06,
            unit="mM",
            rationale="Lowest [S] is far above estimated Km. Add lower concentrations."
        )

        ParameterAdjustment(
            parameter="enzyme_concentration",
            current_value=0.05,
            recommended_value=0.005,
            unit="mM",
            rationale="Enzyme too concentrated; substrate depleted before steady-state."
        )

    Attributes:
        parameter:          Name of the experimental parameter to adjust. Should be a recognizable name like "substrate_concentration_min",
                            "enzyme_concentration", "n_substrate_concentrations", etc.
                            
        current_value:      The value used in the iteration that just failed. Can be a number, string, or "unknown" if not available.
        
        recommended_value:  What the evaluator thinks it should be changed to. Computed from the diagnostic logic (e.g., 0.2 * Km).
        
        unit:               Unit of measurement (e.g., "mM", "µl"). Empty string if the parameter is dimensionless (like a count).
        
        rationale:          Human-readable explanation of WHY this change is needed. This gets included in the LLM prompt so the model
                            understands the scientific reasoning.
    """
    parameter: str
    current_value: Any
    recommended_value: Any
    unit: str = ""
    rationale: str = ""


@dataclass
class IterationFeedback:
    """
    The primary data structure that bridges evaluation and re-planning.

    After each experiment iteration, the Evaluator produces one of these. The Orchestrator checks feedback.passed:
        - If True:  the loop stops, experiment is complete.
        - If False: the Orchestrator passes this feedback object to the Planner, which injects feedback.replanning_context into its next LLM
                    prompt, causing the model to generate an adjusted protocol.

    This is the "closed loop" mechanism:
        Evaluator  →  IterationFeedback  →  Planner  →  new protocol  →  ...

    Attributes:
        iteration:
            Which iteration number (1-indexed) this feedback is for. Used for logging and for the replanning context header.

        passed:
            The single most important field. True = all checks passed, stop. False = at least one check failed, need another iteration.

        failure_categories:
            List of FailureCategory enums indicating what went wrong. Empty if passed=True. Can contain multiple categories if
            several things failed simultaneously.

        r_squared:
            The R² value from the Michaelis-Menten fit, if available. Stored here so that downstream consumers (orchestrator, history log)
            don't have to dig into the ExperimentResult to find it. None if the scientist agent didn't produce this metric.

        km:
            The fitted Km value in mM, if available. Same reasoning as r_squared.

        vmax:
            The fitted Vmax value, if available. Same reasoning as r_squared.

        parameter_adjustments:
            List of ParameterAdjustment objects describing exactly what to change for the next iteration. Computed by the Evaluator's diagnostic methods
            (e.g., _diagnose_poor_fit, _diagnose_km_issue).

        diagnosis_summary:
            A pipe-separated string of all diagnostic messages. Meant for logging and display. Example:
                "R² = 0.52 is below 0.8 threshold | Km = 0.3 mM is plausible"

        replanning_context:
            The critical field for the feedback loop. This is a multi-line natural-language string that gets injected verbatim into the Planner's
            next LLM prompt. It contains:
                - What iteration this is feedback for
                - The previous metrics (R², Km, Vmax)
                - Each parameter adjustment with its rationale
                - An explicit instruction to incorporate the changes

            Example:
                === FEEDBACK FROM ITERATION 1 ===
                Status: FAILED — requires parameter adjustment.

                Diagnosis: R² = 0.52 is below 0.8. Substrate range too narrow.

                Previous metrics:
                  R² = 0.52
                  Km = 0.3 mM
                  Vmax = 0.045

                Required adjustments:
                  1. substrate_concentration_min: 1.0 → 0.06 mM
                     Reason: Lowest [S] far above Km. Add lower concentrations.

                IMPORTANT: The next protocol MUST incorporate these adjustments.
    """
    iteration: int
    passed: bool
    failure_categories: List[FailureCategory] = field(default_factory=list)

    # Kinetics metrics pulled from the result for easy access
    r_squared: Optional[float] = None
    km: Optional[float] = None
    vmax: Optional[float] = None

    # Actionable adjustments for the next iteration
    parameter_adjustments: List[ParameterAdjustment] = field(default_factory=list)

    # Human-readable diagnosis (for logging / display)
    diagnosis_summary: str = ""

    # LLM-consumable context block (injected into the planner's next prompt)
    replanning_context: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize to a plain dictionary for JSON output, logging, and inclusion in the orchestrator's history records.

        All enum values are converted to their string representations, and nested ParameterAdjustment objects are flattened to dicts.
        """
        return {
            "iteration": self.iteration,
            "passed": self.passed,
            "failure_categories": [f.value for f in self.failure_categories],
            "r_squared": self.r_squared,
            "km": self.km,
            "vmax": self.vmax,
            "parameter_adjustments": [
                {
                    "parameter": a.parameter,
                    "current_value": a.current_value,
                    "recommended_value": a.recommended_value,
                    "unit": a.unit,
                    "rationale": a.rationale,
                }
                for a in self.parameter_adjustments
            ],
            "diagnosis_summary": self.diagnosis_summary,
            "replanning_context": self.replanning_context,
        }
