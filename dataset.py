import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from config import Config
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
        
        # 验证数据对齐
        assert len(self.data_frame) == len(self.enzyme_embeddings), "Dataframe and embeddings length mismatch!"

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        row = self.data_frame.iloc[idx]
        
        # 1. 酶特征 (直接从预计算的 Tensor 中获取)
        enzyme_embed = self.enzyme_embeddings[idx] # (Dim,)

        # 2. 底物特征 (Morgan Fingerprint) - 仍然实时计算，因为很快
        substrate_smiles = row['substrate_smiles']
        substrate_fp = smiles_to_morgan_fingerprint(substrate_smiles, nBits=Config.SUBSTRATE_DIM)
        substrate_fp = torch.tensor(substrate_fp, dtype=torch.float32)

        # 3. 环境条件 (Temperature, pH, Salt)
        conditions = torch.tensor([
            row['temperature'], 
            row['ph'], 
            row['salt_conc']
        ], dtype=torch.float32)
        
        # 简单的标准化 (Z-score normalization)
        conditions = (conditions - torch.tensor([30.0, 7.0, 50.0])) / torch.tensor([5.0, 1.0, 20.0])

        # 4. 实验具体设置
        enzyme_conc = torch.tensor([row['enzyme_conc']], dtype=torch.float32)
        substrate_conc = torch.tensor([row['substrate_conc']], dtype=torch.float32)

        # 5. 真实标签 v0
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
