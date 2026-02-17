import torch
import numpy as np
import pandas as pd
try:
    from rdkit import Chem
    from rdkit.Chem import AllChem
    HAS_RDKIT = True
except ImportError:
    HAS_RDKIT = False
    print("Warning: RDKit not found. Using random fingerprints.")

def smiles_to_morgan_fingerprint(smiles, radius=2, nBits=2048):
    """
    Convert SMILES string to Morgan fingerprint (ECFP4)
    """
    if not HAS_RDKIT:
        # If RDKit is not available, return random fingerprints for testing
        return np.random.randint(0, 2, size=(nBits,)).astype(np.float32)

    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return np.zeros((nBits,), dtype=np.float32)
        
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nBits)
        arr = np.zeros((0,), dtype=np.int8)
        Chem.DataStructs.ConvertToNumpyArray(fp, arr)
        return arr.astype(np.float32)
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
    
    import os
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    df = pd.DataFrame(data)
    df.to_csv(output_path, index=False)
    print(f"Generated dummy data at {output_path}")
