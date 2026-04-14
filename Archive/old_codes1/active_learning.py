"""
Generic active learning utilities for BaiTu.

Implements two key signals for Phase-2 (real-data) selection:
1) Ensemble uncertainty (epistemic)
2) Contribution score (uncertainty + novelty + coverage)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F


@dataclass
class ContributionWeights:
    uncertainty: float = 0.5
    novelty: float = 0.3
    representativeness: float = 0.2


@dataclass
class ContributionResult:
    score: np.ndarray
    uncertainty: np.ndarray
    novelty: np.ndarray
    representativeness: np.ndarray


def _to_numpy(x: torch.Tensor) -> np.ndarray:
    return x.detach().cpu().numpy().reshape(-1)


def _normalize_01(values: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    v_min = float(values.min())
    v_max = float(values.max())
    if abs(v_max - v_min) < eps:
        return np.zeros_like(values)
    return (values - v_min) / (v_max - v_min + eps)


def _batch_encode(model, dataloader, device: str) -> torch.Tensor:
    feats = []
    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            hidden = model.encode(
                batch["enzyme_embed"].to(device),
                batch["substrate_fp"].to(device),
            )
            feats.append(hidden)
    if len(feats) == 0:
        raise ValueError("Empty dataloader: no samples found for feature encoding.")
    return torch.cat(feats, dim=0)


def ensemble_predict(
    ensemble,
    enzyme_embed: torch.Tensor,
    substrate_fp: torch.Tensor,
    substrate_conc: Optional[torch.Tensor] = None,
    enzyme_conc: Optional[torch.Tensor] = None,
) -> Dict[str, torch.Tensor]:
    """Forward each member and return mean/variance for prediction targets."""
    member_outputs = []
    with torch.no_grad():
        for member in ensemble:
            member.eval()
            out = member(
                enzyme_embed,
                substrate_fp,
                substrate_conc=substrate_conc,
                enzyme_conc=enzyme_conc,
            )
            member_outputs.append(out)

    keys = ["log_kcat", "log_km", "kcat", "km"]
    if substrate_conc is not None and enzyme_conc is not None:
        keys.append("v0_pred")

    agg = {}
    for key in keys:
        stacked = torch.stack([m[key] for m in member_outputs], dim=0)
        agg[f"{key}_mean"] = stacked.mean(dim=0)
        agg[f"{key}_var"] = stacked.var(dim=0, unbiased=False)
    return agg


def score_pool_by_uncertainty(
    ensemble,
    pool_loader,
    device: str,
    target: str = "v0_pred",
) -> np.ndarray:
    """Compute uncertainty score for each pool sample using ensemble variance."""
    scores = []
    for batch in pool_loader:
        pred = ensemble_predict(
            ensemble=ensemble,
            enzyme_embed=batch["enzyme_embed"].to(device),
            substrate_fp=batch["substrate_fp"].to(device),
            substrate_conc=batch["substrate_conc"].to(device),
            enzyme_conc=batch["enzyme_conc"].to(device),
        )
        var = pred[f"{target}_var"]
        scores.append(_to_numpy(var))
    if len(scores) == 0:
        return np.array([])
    return np.concatenate(scores, axis=0)


def score_pool_contribution(
    model,
    ensemble,
    pool_loader,
    labeled_loader,
    device: str,
    weights: ContributionWeights = ContributionWeights(),
) -> ContributionResult:
    """
    Compute contribution score for each pool sample.

    score = w_u * uncertainty + w_n * novelty + w_r * representativeness

    - uncertainty: ensemble variance on v0 prediction.
    - novelty: distance to nearest labeled sample in latent space.
    - representativeness: similarity to pool centroid (encourages useful coverage).
    """
    model.eval()

    labeled_feats = _batch_encode(model, labeled_loader, device=device)
    pool_feats = _batch_encode(model, pool_loader, device=device)

    # Uncertainty (epistemic)
    uncertainty = score_pool_by_uncertainty(
        ensemble=ensemble,
        pool_loader=pool_loader,
        device=device,
        target="v0_pred",
    )

    # Novelty: min distance to labeled set
    with torch.no_grad():
        dists = torch.cdist(pool_feats, labeled_feats, p=2)
        min_dist = dists.min(dim=1).values
        novelty = _to_numpy(min_dist)

    # Representativeness: cosine similarity to pool centroid
    with torch.no_grad():
        centroid = pool_feats.mean(dim=0, keepdim=True)
        rep = F.cosine_similarity(pool_feats, centroid, dim=1)
        representativeness = _to_numpy(rep)

    uncertainty_n = _normalize_01(uncertainty)
    novelty_n = _normalize_01(novelty)
    representativeness_n = _normalize_01(representativeness)

    score = (
        weights.uncertainty * uncertainty_n
        + weights.novelty * novelty_n
        + weights.representativeness * representativeness_n
    )

    return ContributionResult(
        score=score,
        uncertainty=uncertainty_n,
        novelty=novelty_n,
        representativeness=representativeness_n,
    )


def select_top_k(result: ContributionResult, k: int) -> np.ndarray:
    if k <= 0:
        raise ValueError("k must be positive")
    if len(result.score) == 0:
        return np.array([], dtype=np.int64)
    idx = np.argsort(result.score)[::-1]
    return idx[:k]
