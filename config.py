import torch

class Config:
    # Data paths
    DATA_PATH = "data/kinetics_data.csv"
    PREPROCESSED_DATA_PATH = "data/kinetics_data_with_embeddings.pt"
    CHECKPOINT_DIR = "checkpoints"

    # Model hyperparameters
    # If using a larger model, update ENZYME_DIM accordingly:
    # t6_8M_UR50D:   320 (current default)
    # t12_35M_UR50D:  480
    # t30_150M_UR50D: 640
    # t33_650M_UR50D: 1280
    ENZYME_DIM = 320        # ESM-2 8M dimension
    SUBSTRATE_DIM = 2048    # Morgan Fingerprint dimension (radius=3, nBits=2048)
    CONDITION_DIM = 3       # Temperature(T), pH, Salt concentration
    HIDDEN_DIM = 256        # Projection layer dimension
    DROPOUT = 0.2

    # Training hyperparameters
    BATCH_SIZE = 32
    LEARNING_RATE = 1e-4
    NUM_EPOCHS = 100
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # Physics constraints and stability
    PREDICT_LOG_PARAMS = True  # Predict log(k_cat) and log(Km) for numerical stability
    EPSILON = 1e-8             # Small value to prevent division by zero


class TrypsinConfig:
    """
    Configuration for the closed-loop trypsin active learning system.
    Covers Step 1 (PINN), Step 2 (Deep Ensemble UQ), and Step 3 (MQS).
    """
    # --- Data paths ---
    TRYPSIN_DATA_PATH = "data/trypsin_data.csv"
    TRYPSIN_CHECKPOINT_DIR = "checkpoints/trypsin_ensemble"

    # --- Step 1: Temperature-Conditioned PINN ---
    # Single scalar input (T_norm), three Hill-equation parameter outputs
    HIDDEN_DIM = 64           # Width of each MLP hidden layer

    # Temperature normalization (z-score standardization)
    # Centered on trypsin's physiological optimum (~37°C)
    TEMP_MEAN = 37.0          # °C
    TEMP_STD  = 10.0          # °C

    # --- Step 2: Deep Ensemble for Uncertainty Quantification ---
    ENSEMBLE_SIZE = 5         # Number of independently trained ensemble members

    # --- Step 3: MQS Active Learning ---
    # Candidate pool — instrument-operable temperature range
    TEMP_MIN  = 20.0          # °C
    TEMP_MAX  = 60.0          # °C
    TEMP_STEP =  2.0          # °C   → 21 discrete candidate temperatures

    # Reference substrate concentrations used when scoring candidate temperatures
    # Uncertainty is averaged over these [S] values to give a temperature-level score
    QUERY_SUBSTRATE_CONCS = [10.0, 50.0, 100.0, 250.0, 500.0]  # µM

    # --- Training ---
    BATCH_SIZE     = 16
    LEARNING_RATE  = 1e-3
    NUM_EPOCHS     = 300
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # --- Physics stability ---
    EPSILON = 1e-8            # Guards against division-by-zero in Hill equation
