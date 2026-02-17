import torch
import torch.nn as nn
import torch.optim as optim
from config import Config
from model import KineticsPredictor
from dataset import get_dataloader
from utils import generate_dummy_data
import os

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
