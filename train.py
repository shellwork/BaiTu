import torch
import torch.nn as nn
import torch.optim as optim
from config import Config
from model import KineticsPredictor
from dataset import get_dataloader
from utils import generate_dummy_data
import os

def train():
    # 0. 检查预处理数据是否存在，不存在则提示运行预处理
    if not os.path.exists(Config.PREPROCESSED_DATA_PATH):
        print(f"Preprocessed data not found at {Config.PREPROCESSED_DATA_PATH}.")
        print("Please run 'python preprocess_esm.py' first.")
        return

    # 1. 准备数据
    dataloader = get_dataloader(batch_size=Config.BATCH_SIZE, shuffle=True)
    
    # 2. 初始化模型
    model = KineticsPredictor(
        enzyme_dim=Config.ENZYME_DIM,
        substrate_dim=Config.SUBSTRATE_DIM,
        condition_dim=Config.CONDITION_DIM,
        hidden_dim=Config.HIDDEN_DIM,
        dropout=Config.DROPOUT
    ).to(Config.DEVICE)

    # 3. 定义损失函数和优化器
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)
    
    # 4. 训练循环
    print(f"Starting training on {Config.DEVICE}...")
    for epoch in range(Config.NUM_EPOCHS):
        model.train()
        running_loss = 0.0
        
        for i, batch in enumerate(dataloader):
            # 获取输入数据
            enzyme_embed = batch['enzyme_embed'].to(Config.DEVICE)
            substrate_fp = batch['substrate_fp'].to(Config.DEVICE)
            conditions = batch['conditions'].to(Config.DEVICE)
            enzyme_conc = batch['enzyme_conc'].to(Config.DEVICE)
            substrate_conc = batch['substrate_conc'].to(Config.DEVICE)
            target_v0 = batch['v0'].to(Config.DEVICE)

            # 梯度清零
            optimizer.zero_grad()

            # 前向传播
            # 注意: model 返回 (v0_pred, k_cat, K_m)
            v0_pred, k_cat, K_m = model(
                enzyme_embed, 
                substrate_fp, 
                conditions, 
                enzyme_conc, 
                substrate_conc
            )

            # 计算损失
            loss = criterion(v0_pred, target_v0)
            
            # 可选: 添加正则化项 (L2 penalty on parameters to prevent explosion)
            # loss += 0.01 * (torch.mean(k_cat**2) + torch.mean(K_m**2))

            # 反向传播和优化
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if i % 10 == 9:    # 每 10 个 batch 打印一次
                print(f'[Epoch {epoch + 1}, Batch {i + 1}] loss: {running_loss / 10:.4f}')
                running_loss = 0.0
        
        # 每个 epoch 结束后保存检查点
        if (epoch + 1) % 10 == 0:
            if not os.path.exists(Config.CHECKPOINT_DIR):
                os.makedirs(Config.CHECKPOINT_DIR)
            checkpoint_path = os.path.join(Config.CHECKPOINT_DIR, f'model_epoch_{epoch+1}.pth')
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Saved checkpoint to {checkpoint_path}")

    print('Finished Training')

if __name__ == '__main__':
    train()
