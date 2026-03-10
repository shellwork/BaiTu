import argparse
import os
from typing import List

import numpy as np
import pandas as pd


JOIN_KEYS: List[str] = [
    "EC",
    "EnzymeType",
    "Organism",
    "Sequence",
    "Substrate",
    "Smiles",
    "UniProtID",
]


def _read_and_prepare(path: str, value_col: str, new_name: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    cols = [c for c in JOIN_KEYS if c in df.columns] + [value_col]
    missing = [c for c in JOIN_KEYS if c not in df.columns]
    if missing:
        raise ValueError(f"{path} is missing required key columns: {missing}")
    if value_col not in df.columns:
        raise ValueError(f"{path} is missing parameter column: {value_col}")
    out = df[cols].copy()
    out = out.rename(columns={value_col: new_name})
    return out


def build_aligned_table(km_path: str, kcat_path: str) -> pd.DataFrame:
    km = _read_and_prepare(km_path, "Km(M)", "km_m")
    kcat = _read_and_prepare(kcat_path, "kcat(s^-1)", "kcat_s-1")

    merged = km.merge(kcat, on=JOIN_KEYS, how="inner")
    merged = merged.dropna(subset=["km_m", "kcat_s-1"])
    merged = merged[(merged["km_m"] > 0) & (merged["kcat_s-1"] > 0)]
    merged = merged.drop_duplicates(subset=JOIN_KEYS, keep="first")
    return merged.reset_index(drop=True)


def expand_to_kinetics_rows(
    aligned: pd.DataFrame,
    enzyme_conc_m: float,
    n_points: int,
    min_ratio: float,
    max_ratio: float,
    noise_std: float,
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []

    for _, r in aligned.iterrows():
        km = float(r["km_m"])
        kcat = float(r["kcat_s-1"])
        vmax = kcat * enzyme_conc_m

        # Generate log-spaced substrate concentrations centered around Km
        substrate_grid = np.geomspace(
            km * min_ratio,
            km * max_ratio,
            n_points,
            endpoint=True,
        )

        for s in substrate_grid:
            v0 = (vmax * s) / (km + s)
            if noise_std > 0:
                v0 = max(0.0, v0 + rng.normal(0.0, noise_std * v0))

            row = {
                **{k: r[k] for k in JOIN_KEYS},
                "km_m": km,
                "kcat_s-1": kcat,
                "enzyme_conc_m": enzyme_conc_m,
                "substrate_conc_m": float(s),
                "v0": float(v0),
                "data_source": "simulated_from_parameters",
            }
            rows.append(row)

    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description="Align Km/kcat tables and generate simulated kinetics training rows")
    parser.add_argument("--km_path", type=str, default="data/Km-data_0.4simi-10fold.csv")
    parser.add_argument("--kcat_path", type=str, default="data/kcat-data_0.4simi-10fold.csv")
    parser.add_argument("--output_path", type=str, default="data/kinetics_simulated_from_km_kcat.csv")
    parser.add_argument("--aligned_output_path", type=str, default="data/aligned_km_kcat_pairs.csv")
    parser.add_argument("--enzyme_conc_m", type=float, default=1e-6)
    parser.add_argument("--n_points", type=int, default=16)
    parser.add_argument("--min_ratio", type=float, default=0.05)
    parser.add_argument("--max_ratio", type=float, default=20.0)
    parser.add_argument("--noise_std", type=float, default=0.03)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_pairs", type=int, default=0, help=">0 means only using the first N aligned enzyme-substrate pairs")
    parser.add_argument("--sample_fraction", type=float, default=1.0, help="Random sampling ratio over aligned pairs, range (0,1]")
    args = parser.parse_args()

    aligned = build_aligned_table(args.km_path, args.kcat_path)
    if aligned.empty:
        raise RuntimeError("No aligned Km/kcat pairs found. Please verify key columns match across both tables.")
    rng = np.random.default_rng(args.seed)
    if 0 < args.sample_fraction < 1.0:
        n_keep = max(1, int(len(aligned) * args.sample_fraction))
        keep_idx = rng.choice(len(aligned), size=n_keep, replace=False)
        aligned = aligned.iloc[np.sort(keep_idx)].reset_index(drop=True)
    if args.max_pairs > 0:
        aligned = aligned.head(args.max_pairs).copy()

    sim_df = expand_to_kinetics_rows(
        aligned=aligned,
        enzyme_conc_m=args.enzyme_conc_m,
        n_points=args.n_points,
        min_ratio=args.min_ratio,
        max_ratio=args.max_ratio,
        noise_std=args.noise_std,
        seed=args.seed,
    )

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    aligned.to_csv(args.aligned_output_path, index=False)
    sim_df.to_csv(args.output_path, index=False)

    print(f"[generate_kinetics_from_km_kcat] aligned pairs: {len(aligned)}")
    print(f"[generate_kinetics_from_km_kcat] aligned table: {args.aligned_output_path}")
    print(f"[generate_kinetics_from_km_kcat] generated rows: {len(sim_df)}")
    print(f"[generate_kinetics_from_km_kcat] output: {args.output_path}")


if __name__ == "__main__":
    main()
