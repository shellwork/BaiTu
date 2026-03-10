import argparse
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

from config import Config
from model import KineticsPredictor
from utils import smiles_to_morgan_fingerprint


DEFAULT_ESM_MODEL = "facebook/esm2_t6_8M_UR50D"
MODEL_CACHE_DIR = "./esm_model_cache"


@dataclass
class InferenceInput:
    sequence: str
    smiles: str
    substrate_conc_m: float
    enzyme_conc_m: float = Config.ENZYME_CONC_M


class InferenceEngine:
    """Inference wrapper aligned with training-time input/output schema."""

    def __init__(
        self,
        checkpoint_path: str,
        esm_model_name: str = DEFAULT_ESM_MODEL,
        device: str = Config.DEVICE,
    ) -> None:
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(esm_model_name, cache_dir=MODEL_CACHE_DIR)
        self.esm_model = AutoModel.from_pretrained(esm_model_name, cache_dir=MODEL_CACHE_DIR).to(self.device)
        self.esm_model.eval()

        self.model = KineticsPredictor().to(self.device)
        state_dict = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model.eval()

    @torch.no_grad()
    def embed_sequence(self, sequence: str) -> torch.Tensor:
        inputs = self.tokenizer(
            [sequence],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=1024,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        outputs = self.esm_model(**inputs)
        hidden = outputs.last_hidden_state
        mask = inputs["attention_mask"].unsqueeze(-1).expand(hidden.size()).float()
        pooled = (hidden * mask).sum(dim=1) / torch.clamp(mask.sum(dim=1), min=1e-9)
        return pooled

    @torch.no_grad()
    def predict(self, sample: InferenceInput) -> Dict[str, float]:
        enzyme_embed = self.embed_sequence(sample.sequence)
        substrate_fp = torch.tensor(
            smiles_to_morgan_fingerprint(sample.smiles, nBits=Config.SUBSTRATE_DIM),
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(0)
        substrate_conc = torch.tensor([[sample.substrate_conc_m]], dtype=torch.float32, device=self.device)
        enzyme_conc = torch.tensor([[sample.enzyme_conc_m]], dtype=torch.float32, device=self.device)

        outputs = self.model(
            enzyme_embed=enzyme_embed,
            substrate_fp=substrate_fp,
            substrate_conc=substrate_conc,
            enzyme_conc=enzyme_conc,
        )
        return {
            "log_kcat": float(outputs["log_kcat"].cpu().item()),
            "log_km": float(outputs["log_km"].cpu().item()),
            "kcat_s-1": float(outputs["kcat"].cpu().item()),
            "km_m": float(outputs["km"].cpu().item()),
            "v0": float(outputs["v0_pred"].cpu().item()),
        }

    @torch.no_grad()
    def predict_rate_curve(
        self,
        sample: InferenceInput,
        conc_min_m: float,
        conc_max_m: float,
        n_points: int = 100,
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
        base_outputs = self.predict(sample)
        substrate_grid = np.logspace(np.log10(conc_min_m), np.log10(conc_max_m), n_points)

        kcat = torch.tensor([[base_outputs["kcat_s-1"]]], dtype=torch.float32, device=self.device)
        km = torch.tensor([[base_outputs["km_m"]]], dtype=torch.float32, device=self.device)
        enzyme_conc = torch.tensor([[sample.enzyme_conc_m]], dtype=torch.float32, device=self.device)
        substrate_tensor = torch.tensor(substrate_grid, dtype=torch.float32, device=self.device).unsqueeze(1)

        kcat_expand = kcat.expand_as(substrate_tensor)
        km_expand = km.expand_as(substrate_tensor)
        enzyme_expand = enzyme_conc.expand_as(substrate_tensor)

        rates = self.model.generate_rate_curve(
            kcat=kcat_expand,
            km=km_expand,
            enzyme_conc=enzyme_expand,
            substrate_grid=substrate_tensor,
        )
        return substrate_grid, rates.squeeze(1).cpu().numpy(), base_outputs


def main() -> None:
    parser = argparse.ArgumentParser(description="Run single-sample kinetics inference.")
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint (.pth)")
    parser.add_argument("--sequence", required=True, help="Enzyme amino-acid sequence")
    parser.add_argument("--smiles", required=True, help="Substrate SMILES")
    parser.add_argument("--substrate_conc_m", required=True, type=float, help="Substrate concentration [M]")
    parser.add_argument("--enzyme_conc_m", type=float, default=Config.ENZYME_CONC_M, help="Enzyme concentration [M]")
    args = parser.parse_args()

    engine = InferenceEngine(checkpoint_path=args.checkpoint)
    result = engine.predict(
        InferenceInput(
            sequence=args.sequence,
            smiles=args.smiles,
            substrate_conc_m=args.substrate_conc_m,
            enzyme_conc_m=args.enzyme_conc_m,
        )
    )

    print("Inference Result (aligned with training targets):")
    for k, v in result.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
