import os
import torch
import numpy as np
import pandas as pd
try:
    from rdkit import Chem
    from rdkit.Chem import AllChem, rdFingerprintGenerator
    HAS_RDKIT = True
except ImportError:
    HAS_RDKIT = False
    print("Warning: RDKit not found. Using random fingerprints.")

def smiles_to_morgan_fingerprint(smiles, radius=2, nBits=2048):
    """
    Convert SMILES string to Morgan fingerprint (ECFP4) using the new MorganGenerator API.
    """
    if not HAS_RDKIT:
        # If RDKit is not available, return random fingerprints for testing
        return np.random.randint(0, 2, size=(nBits,)).astype(np.float32)

    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return np.zeros((nBits,), dtype=np.float32)
        
        # Use the new MorganGenerator API
        gen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=nBits)
        fp = gen.GetFingerprint(mol)
        
        # Convert to a NumPy array
        arr = np.zeros((nBits,), dtype=np.float32)
        from rdkit import DataStructs
        DataStructs.ConvertToNumpyArray(fp, arr)
        return arr
    except Exception as e:
        print(f"Error processing SMILES {smiles}: {e}")
        return np.zeros((nBits,), dtype=np.float32)

def generate_dummy_data(num_samples=100, output_path="data/kinetics_data.csv"):
    """
    Generate dummy data for testing
    """
    data = {
        "enzyme_seq": ["MKAILV"] * num_samples,  # Simple placeholder sequence
        "substrate_smiles": ["CCO"] * num_samples, # Ethanol SMILES
        "temperature": np.random.uniform(25, 40, num_samples),
        "ph": np.random.uniform(6, 8, num_samples),
        "salt_conc": np.random.uniform(0, 100, num_samples), # mM
        "enzyme_conc": np.random.uniform(0.01, 0.1, num_samples), # uM
        "substrate_conc": np.random.uniform(1, 500, num_samples), # uM
        "v0": np.random.uniform(0.1, 10, num_samples) # uM/s
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    df = pd.DataFrame(data)
    df.to_csv(output_path, index=False)
    print(f"Generated dummy data at {output_path}")


def generate_trypsin_data(num_temps: int = 20,
                          num_substrate_concs: int = 12,
                          noise_level: float = 0.05,
                          seed: int = 42,
                          output_path: str = "data/trypsin_data.csv") -> pd.DataFrame:
    """
    Generate synthetic trypsin kinetics data governed by temperature-dependent
    Hill equation parameters.

    Ground-truth parameter model (thermodynamically motivated):

        V_max(T)  = 15.0 * exp( -0.5 * ((T - 37) / 8)^2 ) + 0.5
            Bell curve centred on trypsin's physiological optimum (37 C).
            Peak ~15.5 uM/s, baseline ~0.5 uM/s at extremes.

        K_half(T) = 100.0 * 2^((T - 37) / 10)
            Q10 relationship: substrate affinity weakens with rising temperature
            (entropy-driven dissociation).  Doubles every 10 C above 37 C.

        n(T)      = 1.8 + 0.015 * (T - 37)
            Mild linear drift: slight increase in cooperativity at higher
            temperatures, consistent with conformational changes in trypsin.

    Args:
        num_temps          : number of distinct temperature points to sample
        num_substrate_concs: number of [S] values per temperature
        noise_level        : multiplicative Gaussian noise std (fraction of v_true)
        seed               : random seed for reproducibility
        output_path        : where to write the CSV

    Returns:
        pd.DataFrame with columns [temperature, substrate_conc, v0]
    """
    rng = np.random.default_rng(seed)

    temps         = rng.uniform(20.0, 60.0, num_temps)
    substrate_concs = rng.uniform(5.0, 500.0, num_substrate_concs)

    records = []
    for T in temps:
        # Ground truth Hill parameters at temperature T
        V_max  = 15.0 * np.exp(-0.5 * ((T - 37.0) / 8.0) ** 2) + 0.5
        K_half = 100.0 * (2.0 ** ((T - 37.0) / 10.0))
        n      = 1.8 + 0.015 * (T - 37.0)

        for S in substrate_concs:
            S_n    = S ** n
            K_n    = K_half ** n
            v_true = (V_max * S_n) / (K_n + S_n)

            # Multiplicative Gaussian noise (heteroscedastic, realistic for assays)
            noise  = rng.normal(0.0, noise_level * v_true)
            v_obs  = max(0.0, v_true + noise)

            records.append({'temperature': T, 'substrate_conc': S, 'v0': v_obs})

    df = pd.DataFrame(records)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"[generate_trypsin_data] {len(df)} records written to '{output_path}'")
    return df
