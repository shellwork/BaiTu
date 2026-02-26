import torch
import torch.nn as nn
import torch.nn.functional as F
from config import Config, TrypsinConfig

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


# =============================================================================
# Trypsin-Specific Closed-Loop Active Learning Architecture
# Step 1: Temperature-Conditioned PINN  (Hill Equation)
# Step 2: Deep Ensemble for Uncertainty Quantification
# =============================================================================

class HillLayer(nn.Module):
    """
    Non-trainable physics layer implementing the Hill equation.

    Generalises Michaelis-Menten (n=1) to support cooperative kinetics:

        v = V_max * [S]^n / (K_half^n + [S]^n)

    All parameters are generated by the upstream MLP; this layer only
    applies the thermodynamic constraint — no learnable weights.
    """
    def __init__(self, epsilon: float = TrypsinConfig.EPSILON):
        super().__init__()
        self.epsilon = epsilon

    def forward(self, V_max, K_half, n, substrate_conc):
        """
        Args:
            V_max         : (Batch, 1)  Maximum velocity          [µM/s]
            K_half        : (Batch, 1)  Half-saturation constant  [µM]
            n             : (Batch, 1)  Hill coefficient           [dimensionless, ≥ 0.5]
            substrate_conc: (Batch, 1)  [S]                       [µM]
        Returns:
            v             : (Batch, 1)  Reaction velocity          [µM/s]
        """
        S = torch.clamp(substrate_conc, min=self.epsilon)  # ensure [S] > 0
        S_n   = S.pow(n)
        K_n   = K_half.pow(n)
        v     = (V_max * S_n) / (K_n + S_n + self.epsilon)
        return v


class TrypsinKineticsNet(nn.Module):
    """
    Single ensemble member: Temperature-Conditioned PINN for Trypsin.

    Architecture (Step 1):
        T_norm  ──►  MLP  ──►  Softplus  ──►  (V_max, K_half, n)
                                                       │
                                               HillLayer([S])
                                                       │
                                                    v_pred

    Input : scalar normalized temperature T_norm  (shape: Batch × 1)
    Output: v_pred  (Batch × 1),  plus the three Hill parameters for inspection.

    The final MLP layer feeds into Softplus to guarantee positive parameters.
    n is further offset by +0.5 to keep it physically meaningful (n ≥ 0.5).
    """
    def __init__(self,
                 hidden_dim: int = TrypsinConfig.HIDDEN_DIM,
                 epsilon:    float = TrypsinConfig.EPSILON):
        super().__init__()

        # Parameter-generation MLP: T → raw (V_max, K_half, n)
        self.mlp = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 3),   # raw outputs for three Hill parameters
        )

        self.hill_layer = HillLayer(epsilon=epsilon)

    def predict_params(self, temperature_norm):
        """
        T_norm → (V_max, K_half, n)   — all strictly positive.

        Softplus ensures positivity; n offset by 0.5 avoids degenerate values
        (n→0 would make the Hill curve a step function, numerically unstable).
        """
        raw    = self.mlp(temperature_norm)            # (Batch, 3)
        V_max  = F.softplus(raw[:, 0:1])
        K_half = F.softplus(raw[:, 1:2])
        n      = F.softplus(raw[:, 2:3]) + 0.5        # n ≥ 0.5
        return V_max, K_half, n

    def forward(self, temperature_norm, substrate_conc):
        """Full forward pass: T_norm + [S]  →  v_pred."""
        V_max, K_half, n = self.predict_params(temperature_norm)
        v_pred = self.hill_layer(V_max, K_half, n, substrate_conc)
        return v_pred, V_max, K_half, n


class TrypsinEnsemble(nn.Module):
    """
    Deep Ensemble of N independent TrypsinKineticsNet members.

    Step 2 — Uncertainty Quantification:
        Each member is trained from a different random initialisation,
        producing diverse predictions.  The variance across members
        quantifies *epistemic* uncertainty — i.e., regions where the
        training data is sparse and the model's knowledge is limited.

        Var(v | T) = (1/N) Σ (v_i - v̄)²

    This variance signal is consumed directly by the MQS strategy in
    active_learning.py to select the most informative next experiment.
    """
    def __init__(self,
                 n_members:  int   = TrypsinConfig.ENSEMBLE_SIZE,
                 hidden_dim: int   = TrypsinConfig.HIDDEN_DIM,
                 epsilon:    float = TrypsinConfig.EPSILON):
        super().__init__()
        self.n_members = n_members
        self.members   = nn.ModuleList([
            TrypsinKineticsNet(hidden_dim=hidden_dim, epsilon=epsilon)
            for _ in range(n_members)
        ])

    def forward(self, temperature_norm, substrate_conc):
        """
        Aggregate predictions from all ensemble members.

        Returns:
            mean_v      : (Batch, 1)  Consensus velocity prediction
            var_v       : (Batch, 1)  Velocity variance  ← epistemic uncertainty
            mean_params : (Batch, 3)  Mean [V_max, K_half, n]
            var_params  : (Batch, 3)  Variance of [V_max, K_half, n]
        """
        all_v, all_params = [], []

        for member in self.members:
            v, V_max, K_half, n = member(temperature_norm, substrate_conc)
            all_v.append(v)
            all_params.append(torch.cat([V_max, K_half, n], dim=-1))

        all_v      = torch.stack(all_v,      dim=0)   # (N, Batch, 1)
        all_params = torch.stack(all_params, dim=0)   # (N, Batch, 3)

        mean_v     = all_v.mean(dim=0)
        var_v      = all_v.var(dim=0)
        mean_params = all_params.mean(dim=0)
        var_params  = all_params.var(dim=0)

        return mean_v, var_v, mean_params, var_params

    def predict_uncertainty(self, temperature_norm, substrate_conc):
        """
        Convenience method: return only the velocity variance.
        Called by the MQS active learning query loop.
        """
        _, var_v, _, _ = self.forward(temperature_norm, substrate_conc)
        return var_v   # (Batch, 1)
