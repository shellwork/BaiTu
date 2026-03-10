import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.metrics import r2_score, mean_absolute_error
from torch.utils.data import random_split
from config import Config, TrypsinConfig
from model import KineticsPredictor, TrypsinEnsemble
from dataset import get_dataloader, get_trypsin_dataloader

def evaluate(model, dataloader, device):
    model.eval()
    all_targets_v0 = []
    all_preds_v0 = []
    all_targets_kcat = []
    all_preds_kcat = []
    all_targets_km = []
    all_preds_km = []
    
    with torch.no_grad():
        for batch in dataloader:
            enzyme_embed = batch['enzyme_embed'].to(device)
            substrate_fp = batch['substrate_fp'].to(device)
            substrate_conc = batch['substrate_conc'].to(device)
            enzyme_conc = batch['enzyme_conc'].to(device)
            
            outputs = model(
                enzyme_embed,
                substrate_fp,
                substrate_conc=substrate_conc,
                enzyme_conc=enzyme_conc,
            )
            
            # v0 metrics
            if batch['has_rate_label'].any():
                mask = batch['has_rate_label'].squeeze(-1) > 0
                all_targets_v0.append(batch['v0'][mask].cpu().numpy())
                all_preds_v0.append(outputs['v0_pred'][mask].cpu().numpy())
            
            # kcat/km metrics
            if batch['has_param_label'].any():
                mask = batch['has_param_label'].squeeze(-1) > 0
                all_targets_kcat.append(torch.exp(batch['log_kcat'][mask]).cpu().numpy())
                all_preds_kcat.append(outputs['kcat'][mask].cpu().numpy())
                all_targets_km.append(torch.exp(batch['log_km'][mask]).cpu().numpy())
                all_preds_km.append(outputs['km'][mask].cpu().numpy())

    metrics = {}
    
    def calc_reg_metrics(targets, preds, prefix):
        if len(targets) == 0: return
        t = np.concatenate(targets).flatten()
        p = np.concatenate(preds).flatten()
        metrics[f'{prefix}_r2'] = r2_score(t, p)
        metrics[f'{prefix}_mae'] = mean_absolute_error(t, p)
        # Approximate accuracy: fraction of predictions within 50%~150% of the ground truth
        metrics[f'{prefix}_acc'] = np.mean((p > 0.5 * t) & (p < 1.5 * t))

    calc_reg_metrics(all_targets_v0, all_preds_v0, 'v0')
    calc_reg_metrics(all_targets_kcat, all_preds_kcat, 'kcat')
    calc_reg_metrics(all_targets_km, all_preds_km, 'km')
    
    return metrics

def train():
    pt_path = Config.PREPROCESSED_DATA_PATH

    if not os.path.exists(pt_path):
        print(f"Preprocessed data not found at {pt_path}.")
        print(f"Run: python preprocess_esm.py --csv_path {Config.DATA_PATH} --output_path {pt_path}")
        return

    # Split into train/validation sets
    full_dataset = get_dataloader(batch_size=Config.BATCH_SIZE, shuffle=False).dataset
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=Config.BATCH_SIZE, shuffle=False)

    model = KineticsPredictor(
        enzyme_dim=Config.ENZYME_DIM,
        substrate_dim=Config.SUBSTRATE_DIM,
        hidden_dim=Config.HIDDEN_DIM,
        dropout=Config.DROPOUT
    ).to(Config.DEVICE)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)

    print(f"Starting training on {Config.DEVICE} (Task: parameter prediction + rate supervision)...")

    for epoch in range(Config.NUM_EPOCHS):
        model.train()
        running_loss = 0.0

        for i, batch in enumerate(train_loader):
            enzyme_embed = batch['enzyme_embed'].to(Config.DEVICE)
            substrate_fp = batch['substrate_fp'].to(Config.DEVICE)
            substrate_conc = batch['substrate_conc'].to(Config.DEVICE)
            enzyme_conc = batch['enzyme_conc'].to(Config.DEVICE)
            has_param_label = batch['has_param_label'].to(Config.DEVICE).squeeze(-1) > 0
            has_rate_label = batch['has_rate_label'].to(Config.DEVICE).squeeze(-1) > 0

            if not (has_param_label.any() or has_rate_label.any()):
                continue

            optimizer.zero_grad()
            outputs = model(
                enzyme_embed,
                substrate_fp,
                substrate_conc=substrate_conc,
                enzyme_conc=enzyme_conc,
            )

            loss = torch.tensor(0.0, device=Config.DEVICE)
            
            if has_param_label.any():
                target_log_kcat = batch['log_kcat'].to(Config.DEVICE)[has_param_label]
                target_log_km = batch['log_km'].to(Config.DEVICE)[has_param_label]
                pred_log_kcat = outputs['log_kcat'][has_param_label]
                pred_log_km = outputs['log_km'][has_param_label]
                param_loss = criterion(pred_log_kcat, target_log_kcat) + criterion(pred_log_km, target_log_km)
                loss = loss + Config.PARAM_LOSS_WEIGHT * param_loss

            if has_rate_label.any():
                target_v0 = batch['v0'].to(Config.DEVICE)[has_rate_label]
                pred_v0 = outputs['v0_pred'][has_rate_label]
                rate_loss = criterion(pred_v0, target_v0)
                loss = loss + Config.RATE_LOSS_WEIGHT * rate_loss

            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        # Evaluate at the end of each epoch
        val_metrics = evaluate(model, val_loader, Config.DEVICE)
        
        log_str = f'Epoch {epoch + 1}/{Config.NUM_EPOCHS} | Loss: {running_loss/len(train_loader):.4f}'
        if 'v0_r2' in val_metrics:
            log_str += f" | v0_R2: {val_metrics['v0_r2']:.4f} | v0_Acc: {val_metrics['v0_acc']:.2%}"
        if 'km_r2' in val_metrics:
            log_str += f" | Km_R2: {val_metrics['km_r2']:.4f} | kcat_R2: {val_metrics['kcat_r2']:.4f}"
        print(log_str)

        if (epoch + 1) % 10 == 0:
            if not os.path.exists(Config.CHECKPOINT_DIR):
                os.makedirs(Config.CHECKPOINT_DIR)
            checkpoint_path = os.path.join(Config.CHECKPOINT_DIR, f'model_epoch_{epoch+1}_params.pth')
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Saved checkpoint to {checkpoint_path}")

    print('Finished Training')

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
    # Default entrypoint runs direct Km prediction training (PREDICT_KM_DIRECT=True)
    # Trypsin active learning: from train import train_trypsin_ensemble; train_trypsin_ensemble()
    train()
