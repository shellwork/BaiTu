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
