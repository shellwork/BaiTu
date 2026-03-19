"""
EnzymeKineticsSimulator — ground-truth oracle for the BaiTu closed-loop campaign.

The simulator uses the pre-computed kinetics dataset
(data/kinetics_simulated_with_embeddings.pt) as its ground truth.  It partitions
the dataset into three non-overlapping splits:

  seed       – small initial labeled set given to the algorithm at the start
  held_out   – fixed validation set (never used for training; used only for eval)
  pool       – large unlabeled pool that the algorithm queries iteratively

When the algorithm queries a batch of pool samples the simulator:
  1. Reveals the true k_cat / K_m / v0 labels.
  2. Adds multiplicative Gaussian noise to v0 (coefficient of variation = noise_cv).
  3. Returns a dataset ready for training.

It also simulates replicate measurements for QC variance checks.
"""

from __future__ import annotations

import math
from typing import List, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import ConcatDataset, Dataset

from config import Config
from dataset import KineticsDataset
from utils import smiles_to_morgan_fingerprint


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

class _SimulatorSubset(Dataset):
    """
    Lightweight wrapper around a KineticsDataset that:
      - exposes only the rows at `indices`
      - optionally masks all labels (for the unlabeled pool)
      - optionally applies multiplicative noise to v0 (for oracle labeling)
    """

    def __init__(
        self,
        base: KineticsDataset,
        indices: List[int],
        mask_labels: bool = False,
        noise_cv: float = 0.0,
        rng: Optional[np.random.Generator] = None,
    ):
        self.base = base
        self.indices = indices
        self.mask_labels = mask_labels
        self.noise_cv = noise_cv
        self.rng = rng if rng is not None else np.random.default_rng(0)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, pos: int) -> dict:
        sample = dict(self.base[self.indices[pos]])

        if self.mask_labels:
            sample["has_param_label"] = torch.tensor([0.0])
            sample["has_rate_label"]  = torch.tensor([0.0])
            sample.pop("log_kcat", None)
            sample.pop("log_km",   None)
            sample.pop("v0",       None)
        elif self.noise_cv > 0.0 and "v0" in sample:
            noise_factor = 1.0 + self.noise_cv * float(self.rng.standard_normal())
            sample["v0"] = sample["v0"] * abs(noise_factor)

        return sample

    # Convenience: expose the underlying dataframe rows for QC / reporting
    def get_dataframe_rows(self) -> pd.DataFrame:
        return self.base.data_frame.iloc[self.indices].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Main simulator class
# ---------------------------------------------------------------------------

class EnzymeKineticsSimulator:
    """
    Ground-truth oracle for the closed-loop enzyme kinetics campaign.

    Parameters
    ----------
    pt_path   : path to preprocessed .pt file (dataframe + enzyme embeddings)
    seed_size : number of samples in the initial labeled seed set
    val_size  : number of samples reserved for held-out evaluation
    noise_cv  : coefficient of variation for simulated measurement noise
    seed      : random seed for reproducible splits
    """

    def __init__(
        self,
        pt_path: str = Config.PREPROCESSED_DATA_PATH,
        seed_size: int = 50,
        val_size: int = 100,
        noise_cv: float = Config.RATE_NOISE_STD,
        seed: int = 42,
    ):
        self.pt_path   = pt_path
        self.noise_cv  = noise_cv
        self._rng      = np.random.default_rng(seed)

        print(f"[Simulator] Loading dataset from {pt_path} ...")
        self._base = KineticsDataset(pt_path=pt_path)
        N = len(self._base)

        if seed_size + val_size >= N:
            raise ValueError(
                f"seed_size ({seed_size}) + val_size ({val_size}) must be < N ({N})"
            )

        # Stratified split by first-level EC class to ensure diverse splits
        df = self._base.data_frame
        ec_classes = (
            df["EC"].astype(str).str.split(".").str[0]
            if "EC" in df.columns
            else pd.Series(["0"] * N)
        )

        # Shuffle within each EC class then allocate proportionally
        all_idx = np.arange(N)
        self._rng.shuffle(all_idx)

        # Simple stratified split: sort by EC prefix, then round-robin
        ec_vals = ec_classes.values[all_idx]
        sorted_order = np.argsort(ec_vals, kind="stable")
        all_idx = all_idx[sorted_order]

        self.seed_idx    = all_idx[:seed_size].tolist()
        self.val_idx     = all_idx[seed_size : seed_size + val_size].tolist()
        self.pool_idx    = all_idx[seed_size + val_size :].tolist()

        # Track which *pool positions* have been queried (0-based index into pool_idx)
        self._queried_positions: set = set()

        print(
            f"[Simulator] Split: seed={len(self.seed_idx)}, "
            f"held_out={len(self.val_idx)}, pool={len(self.pool_idx)}"
        )

    # ------------------------------------------------------------------
    # Dataset accessors
    # ------------------------------------------------------------------

    def get_seed_dataset(self) -> _SimulatorSubset:
        """Labeled initial dataset provided to the algorithm at the start."""
        return _SimulatorSubset(self._base, self.seed_idx, mask_labels=False)

    def get_held_out_dataset(self) -> _SimulatorSubset:
        """Fixed validation set — never used for training."""
        return _SimulatorSubset(self._base, self.val_idx, mask_labels=False)

    def get_pool_dataset(self) -> _SimulatorSubset:
        """
        Remaining unlabeled pool (excluding already-queried positions).
        Labels are masked — the algorithm sees only input features.
        """
        remaining_positions = [
            p for p in range(len(self.pool_idx))
            if p not in self._queried_positions
        ]
        remaining_idx = [self.pool_idx[p] for p in remaining_positions]
        return _SimulatorSubset(self._base, remaining_idx, mask_labels=True)

    def get_pool_positions_remaining(self) -> List[int]:
        """Pool positions (into pool_idx) that have not yet been queried."""
        return [p for p in range(len(self.pool_idx)) if p not in self._queried_positions]

    # ------------------------------------------------------------------
    # Oracle query
    # ------------------------------------------------------------------

    def query(self, pool_positions: List[int]) -> _SimulatorSubset:
        """
        Reveal labels for the selected pool positions.

        The oracle adds multiplicative Gaussian noise to v0 measurements,
        mimicking the variability of a real assay.

        Parameters
        ----------
        pool_positions : 0-based positions within pool_idx (NOT global dataset indices).
                         Typically the output of select_top_k applied to the current pool.

        Returns
        -------
        A labeled _SimulatorSubset for the queried samples (ready for ConcatDataset).
        """
        if not pool_positions:
            raise ValueError("pool_positions must be non-empty")
        invalid = [p for p in pool_positions if p not in range(len(self.pool_idx))]
        if invalid:
            raise ValueError(f"Invalid pool positions: {invalid}")

        global_idx = [self.pool_idx[p] for p in pool_positions]
        self._queried_positions.update(pool_positions)

        return _SimulatorSubset(
            self._base,
            global_idx,
            mask_labels=False,
            noise_cv=self.noise_cv,
            rng=np.random.default_rng(int(self._rng.integers(0, 2**31))),
        )

    # ------------------------------------------------------------------
    # QC helpers
    # ------------------------------------------------------------------

    def get_replicate_cv(
        self,
        pool_positions: List[int],
        n_replicates: int = 3,
    ) -> np.ndarray:
        """
        Simulate n_replicates measurements for each queried sample and return
        the coefficient of variation (std / mean) per sample.

        Used by QCMonitor to check that replicate variance < 10 %.
        """
        cvs = []
        for p in pool_positions:
            g_idx = self.pool_idx[p]
            row = self._base.data_frame.iloc[g_idx]
            true_v0 = float(row["v0"]) if "v0" in row and pd.notna(row["v0"]) else 0.0
            if true_v0 <= 0:
                cvs.append(0.0)
                continue
            reps = true_v0 * (1.0 + self.noise_cv * self._rng.standard_normal(n_replicates))
            reps = np.abs(reps)
            cvs.append(float(np.std(reps) / np.mean(reps)) if np.mean(reps) > 0 else 0.0)
        return np.array(cvs)

    def n_pool_remaining(self) -> int:
        return len(self.pool_idx) - len(self._queried_positions)

    def n_queried(self) -> int:
        return len(self._queried_positions)
