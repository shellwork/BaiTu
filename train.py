import os
import torch
import torch.nn as nn
import torch.optim as optim
from config import Config, TrypsinConfig
from model import KineticsPredictor, TrypsinEnsemble
from dataset import get_dataloader, get_trypsin_dataloader
from utils import generate_dummy_data

def train():
    # 0. Check if preprocessed data exists, if not, prompt to run preprocessing
    if not os.path.exists(Config.PREPROCESSED_DATA_PATH):
        print(f"Preprocessed data not found at {Config.PREPROCESSED_DATA_PATH}.")
        print("Please run 'python preprocess_esm.py' first.")
        return

    # 1. Prepare data
    dataloader = get_dataloader(batch_size=Config.BATCH_SIZE, shuffle=True)
    
    # 2. Initialize model
    model = KineticsPredictor(
        enzyme_dim=Config.ENZYME_DIM,
        substrate_dim=Config.SUBSTRATE_DIM,
        condition_dim=Config.CONDITION_DIM,
        hidden_dim=Config.HIDDEN_DIM,
        dropout=Config.DROPOUT
    ).to(Config.DEVICE)

    # 3. Define loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)
    
    # 4. Training loop
    print(f"Starting training on {Config.DEVICE}...")
    for epoch in range(Config.NUM_EPOCHS):
        model.train()
        running_loss = 0.0
        
        for i, batch in enumerate(dataloader):
            # Get input data
            enzyme_embed = batch['enzyme_embed'].to(Config.DEVICE)
            substrate_fp = batch['substrate_fp'].to(Config.DEVICE)
            conditions = batch['conditions'].to(Config.DEVICE)
            enzyme_conc = batch['enzyme_conc'].to(Config.DEVICE)
            substrate_conc = batch['substrate_conc'].to(Config.DEVICE)
            target_v0 = batch['v0'].to(Config.DEVICE)

            # Zero gradients
            optimizer.zero_grad()

            # Forward pass
            # Note: model returns (v0_pred, k_cat, K_m)
            v0_pred, k_cat, K_m = model(
                enzyme_embed, 
                substrate_fp, 
                conditions, 
                enzyme_conc, 
                substrate_conc
            )

            # Calculate loss
            loss = criterion(v0_pred, target_v0)
            
            # Optional: Add regularization (L2 penalty on parameters to prevent explosion)
            # loss += 0.01 * (torch.mean(k_cat**2) + torch.mean(K_m**2))

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if i % 10 == 9:    # Print every 10 batches
                print(f'[Epoch {epoch + 1}, Batch {i + 1}] loss: {running_loss / 10:.4f}')
                running_loss = 0.0
        
        # Save checkpoint after each epoch
        if (epoch + 1) % 10 == 0:
            if not os.path.exists(Config.CHECKPOINT_DIR):
                os.makedirs(Config.CHECKPOINT_DIR)
            checkpoint_path = os.path.join(Config.CHECKPOINT_DIR, f'model_epoch_{epoch+1}.pth')
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Saved checkpoint to {checkpoint_path}")

    print('Finished Training')

if __name__ == '__main__':
    train()


# =============================================================================
# Trypsin Ensemble Training  (Step 1 + 2 of the active learning workflow)
# =============================================================================

def train_trypsin_ensemble(
        csv_path:   str   = TrypsinConfig.TRYPSIN_DATA_PATH,
        n_members:  int   = TrypsinConfig.ENSEMBLE_SIZE,
        hidden_dim: int   = TrypsinConfig.HIDDEN_DIM,
        num_epochs: int   = TrypsinConfig.NUM_EPOCHS,
        lr:         float = TrypsinConfig.LEARNING_RATE,
        batch_size: int   = TrypsinConfig.BATCH_SIZE,
        device:     str   = TrypsinConfig.DEVICE,
        save_dir:   str   = TrypsinConfig.TRYPSIN_CHECKPOINT_DIR,
) -> TrypsinEnsemble:
    """
    Train each ensemble member independently on the trypsin kinetics dataset.

    Deep Ensemble training protocol:
        - All members share the same architecture and the same training data.
        - Different random weight initialisation creates diversity, which
          translates to predictive variance in data-sparse regions.
        - Each member gets its own Adam optimiser (no shared state).
        - MSE loss on reaction velocity v0 [µM/s].

    Args:
        csv_path   : path to trypsin CSV (temperature, substrate_conc, v0)
        n_members  : number of ensemble members (default 5)
        hidden_dim : MLP hidden layer width
        num_epochs : training epochs per member
        lr         : Adam learning rate
        batch_size : mini-batch size
        device     : 'cpu' or 'cuda'
        save_dir   : directory for member checkpoints

    Returns:
        Fully trained TrypsinEnsemble (on CPU/device, in eval mode)
    """
    if not os.path.exists(csv_path):
        print(f"Trypsin data not found at '{csv_path}'.")
        print("Run 'python data/generate_trypsin_data.py' first.")
        return None

    os.makedirs(save_dir, exist_ok=True)
    dataloader = get_trypsin_dataloader(csv_path=csv_path, batch_size=batch_size)

    ensemble = TrypsinEnsemble(n_members=n_members, hidden_dim=hidden_dim).to(device)
    criterion = nn.MSELoss()

    print(f"\n{'='*60}")
    print(f"  Training TrypsinEnsemble  ({n_members} members x {num_epochs} epochs)")
    print(f"  Device : {device}  |  LR : {lr}  |  Batch : {batch_size}")
    print(f"{'='*60}")

    for member_idx, member in enumerate(ensemble.members):
        print(f"\n--- Member {member_idx + 1}/{n_members} ---")
        optimizer = optim.Adam(member.parameters(), lr=lr)

        for epoch in range(num_epochs):
            member.train()
            epoch_loss = 0.0

            for batch in dataloader:
                T       = batch['temperature'].to(device)
                S       = batch['substrate_conc'].to(device)
                v_target = batch['v0'].to(device)

                optimizer.zero_grad()
                v_pred, _, _, _ = member(T, S)
                loss = criterion(v_pred, v_target)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            if (epoch + 1) % 50 == 0:
                avg = epoch_loss / len(dataloader)
                print(f"  Epoch [{epoch+1:>3}/{num_epochs}]  loss: {avg:.6f}")

        # Save individual member checkpoint
        ckpt_path = os.path.join(save_dir, f"member_{member_idx}.pth")
        torch.save(member.state_dict(), ckpt_path)
        print(f"  Saved -> {ckpt_path}")

    ensemble.eval()
    print(f"\n[train_trypsin_ensemble] All members trained and saved to '{save_dir}'")
    return ensemble


if __name__ == '__main__':
    # Entry point for trypsin ensemble training
    trained_ensemble = train_trypsin_ensemble()
    if trained_ensemble is not None:
        print("\nEnsemble ready. Run active_learning.py to query the next experiment.")
