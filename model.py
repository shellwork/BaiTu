import torch
import torch.nn as nn
import torch.nn.functional as F
from config import Config

class MichaelisMentenLayer(nn.Module):
    """
    Module 3: Physics-Informed Layer
    Pure mathematical logic with no trainable parameters.
    """
    def __init__(self):
        super().__init__()

    def forward(self, k_cat, K_m, enzyme_conc, substrate_conc):
        """
        Inputs:
            k_cat: (Batch, 1) Catalytic constant [s^-1]
            K_m:   (Batch, 1) Michaelis constant [uM]
            enzyme_conc: (Batch, 1) Total enzyme concentration [E_total] [uM]
            substrate_conc: (Batch, 1) Substrate concentration [S] [uM]
        Outputs:
            v0: (Batch, 1) Initial reaction velocity [uM/s]
        Formula:
            V_max = k_cat * [E]
            v0 = (V_max * [S]) / (K_m + [S])
        """
        V_max = k_cat * enzyme_conc
        # Add epsilon to denominator for numerical stability to prevent division by zero
        v0 = (V_max * substrate_conc) / (K_m + substrate_conc + Config.EPSILON)
        return v0

class KineticsPredictor(nn.Module):
    """
    Main Model: Includes encoder fusion, parameter prediction MLP, and physics layer.
    """
    def __init__(self, 
                 enzyme_dim=Config.ENZYME_DIM,    
                 substrate_dim=Config.SUBSTRATE_DIM, 
                 condition_dim=Config.CONDITION_DIM,    
                 hidden_dim=Config.HIDDEN_DIM,
                 dropout=Config.DROPOUT):
        super().__init__()
        
        # --- 1. Pre-processing before feature fusion (Addressing modality imbalance) ---
        # Map different modalities to the same dimension for better alignment
        self.enzyme_projector = nn.Sequential(
            nn.Linear(enzyme_dim, hidden_dim),
            nn.LayerNorm(hidden_dim), # LayerNorm helps balance variance across modalities
            nn.GELU()
        )
        self.substrate_projector = nn.Sequential(
            nn.Linear(substrate_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        )
        self.condition_projector = nn.Sequential(
            nn.Linear(condition_dim, hidden_dim // 4), # Condition dimension is usually smaller
            nn.LayerNorm(hidden_dim // 4),
            nn.GELU()
        )

        # Calculate total dimension after fusion
        fusion_dim = hidden_dim + hidden_dim + (hidden_dim // 4)

        # --- 2. Kinetic Parameter Prediction Network (MLP) ---
        self.mlp = nn.Sequential(
            nn.Linear(fusion_dim, 512),
            nn.BatchNorm1d(512), # BatchNorm to prevent overfitting
            nn.GELU(),
            nn.Dropout(dropout),
            
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            
            nn.Linear(256, 2) # Outputs two values: raw_k_cat, raw_K_m
        )

        # --- 3. Physics Layer ---
        self.physics_layer = MichaelisMentenLayer()
        self.predict_log_params = Config.PREDICT_LOG_PARAMS

    def forward(self, enzyme_embed, substrate_fp, conditions, enzyme_conc, substrate_conc):
        """
        Forward pass workflow
        """
        # Step 1: Encoding and Projection
        e_feat = self.enzyme_projector(enzyme_embed)
        s_feat = self.substrate_projector(substrate_fp)
        c_feat = self.condition_projector(conditions)

        # Step 2: Feature Fusion (Concatenation)
        # Shape: (Batch, hidden*2 + hidden//4)
        x_fused = torch.cat([e_feat, s_feat, c_feat], dim=1)

        # Step 3: Predict Kinetic Parameters
        raw_params = self.mlp(x_fused)
        
        # Key Constraint: Addressing scale differences and numerical stability
        if self.predict_log_params:
            # If predicting log values, use exp to restore to positive space
            # This naturally ensures positive outputs and smoother gradients in log space
            k_cat = torch.exp(raw_params[:, 0:1]) 
            K_m = torch.exp(raw_params[:, 1:2])
        else:
            # Otherwise use Softplus to ensure parameters are positive
            k_cat = F.softplus(raw_params[:, 0:1]) 
            K_m = F.softplus(raw_params[:, 1:2])

        # Step 4: Physics Layer Computation (No trainable parameters)
        # Note: Enzyme and substrate concentrations are experimental conditions, not model parameters
        v0_pred = self.physics_layer(k_cat, K_m, enzyme_conc, substrate_conc)

        return v0_pred, k_cat, K_m
