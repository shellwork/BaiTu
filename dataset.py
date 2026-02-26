import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from config import Config, TrypsinConfig
from utils import smiles_to_morgan_fingerprint
import os

class KineticsDataset(Dataset):
    def __init__(self, pt_path=Config.PREPROCESSED_DATA_PATH, transform=None):
        """
        Args:
            pt_path (string): Path to the preprocessed .pt file (containing embeddings and dataframe).
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        if not os.path.exists(pt_path):
            raise FileNotFoundError(f"Preprocessed data not found at {pt_path}. Please run preprocess_esm.py first.")
            
        print(f"Loading dataset from {pt_path}...")
        data_dict = torch.load(pt_path)
        
        self.data_frame = data_dict['dataframe']
        self.enzyme_embeddings = data_dict['enzyme_embeddings'] # Tensor (N, Dim)
        self.transform = transform
        
        # Verify data alignment
        assert len(self.data_frame) == len(self.enzyme_embeddings), "Dataframe and embeddings length mismatch!"

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        row = self.data_frame.iloc[idx]
        
        # 1. Enzyme features (Directly from precomputed Tensor)
        enzyme_embed = self.enzyme_embeddings[idx] # (Dim,)

        # 2. Substrate features (Morgan Fingerprint) - Still computed on-the-fly as it's fast
        substrate_smiles = row['substrate_smiles']
        substrate_fp = smiles_to_morgan_fingerprint(substrate_smiles, nBits=Config.SUBSTRATE_DIM)
        substrate_fp = torch.tensor(substrate_fp, dtype=torch.float32)

        # 3. Environmental conditions (Temperature, pH, Salt)
        conditions = torch.tensor([
            row['temperature'], 
            row['ph'], 
            row['salt_conc']
        ], dtype=torch.float32)
        
        # Simple normalization (Z-score normalization)
        # Assuming mean/std values for demonstration; should be based on dataset statistics in production
        conditions = (conditions - torch.tensor([30.0, 7.0, 50.0])) / torch.tensor([5.0, 1.0, 20.0])

        # 4. Experimental setup (Variables for physics equation)
        enzyme_conc = torch.tensor([row['enzyme_conc']], dtype=torch.float32)
        substrate_conc = torch.tensor([row['substrate_conc']], dtype=torch.float32)

        # 5. Ground truth label v0
        v0 = torch.tensor([row['v0']], dtype=torch.float32)

        sample = {
            'enzyme_embed': enzyme_embed,
            'substrate_fp': substrate_fp,
            'conditions': conditions,
            'enzyme_conc': enzyme_conc,
            'substrate_conc': substrate_conc,
            'v0': v0
        }

        if self.transform:
            sample = self.transform(sample)

        return sample

def get_dataloader(batch_size=Config.BATCH_SIZE, shuffle=True):
    dataset = KineticsDataset()
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0)


# =============================================================================
# Trypsin Active Learning Dataset
# =============================================================================

class TrypsinDataset(Dataset):
    """
    Minimal dataset for the trypsin closed-loop system.

    Features:
        temperature   : scalar, z-score normalised with TEMP_MEAN / TEMP_STD
        substrate_conc: scalar [µM]
    Target:
        v0            : initial reaction velocity [µM/s]

    The only input to the model is temperature — all enzyme-identity and
    substrate-chemistry features from the general KineticsDataset are
    dropped because we are locked to a single enzyme (trypsin) and the
    Hill equation already encodes the substrate saturation curve via [S].
    """
    def __init__(self,
                 csv_path:  str   = TrypsinConfig.TRYPSIN_DATA_PATH,
                 temp_mean: float = TrypsinConfig.TEMP_MEAN,
                 temp_std:  float = TrypsinConfig.TEMP_STD):
        if not os.path.exists(csv_path):
            raise FileNotFoundError(
                f"Trypsin data not found at '{csv_path}'. "
                "Run 'python data/generate_trypsin_data.py' first."
            )
        self.df        = pd.read_csv(csv_path)
        self.temp_mean = temp_mean
        self.temp_std  = temp_std
        print(f"[TrypsinDataset] Loaded {len(self.df)} records from '{csv_path}'")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Z-score normalise temperature so the MLP input lives near [-2, 2]
        temp_norm = (row['temperature'] - self.temp_mean) / self.temp_std
        temperature    = torch.tensor([temp_norm],           dtype=torch.float32)
        substrate_conc = torch.tensor([row['substrate_conc']], dtype=torch.float32)
        v0             = torch.tensor([row['v0']],             dtype=torch.float32)

        return {'temperature': temperature, 'substrate_conc': substrate_conc, 'v0': v0}


def get_trypsin_dataloader(csv_path:  str   = TrypsinConfig.TRYPSIN_DATA_PATH,
                           batch_size: int  = TrypsinConfig.BATCH_SIZE,
                           shuffle:   bool  = True,
                           temp_mean: float = TrypsinConfig.TEMP_MEAN,
                           temp_std:  float = TrypsinConfig.TEMP_STD):
    dataset = TrypsinDataset(csv_path=csv_path, temp_mean=temp_mean, temp_std=temp_std)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0)
