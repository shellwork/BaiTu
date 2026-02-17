import torch
import torch.nn as nn
import torch.nn.functional as F
from config import Config

class MichaelisMentenLayer(nn.Module):
    """
    模块三：物理机制映射层 (Physics-Informed Layer)
    无训练参数，纯物理方程计算。
    """
    def __init__(self):
        super().__init__()

    def forward(self, k_cat, K_m, enzyme_conc, substrate_conc):
        """
        输入:
            k_cat: (Batch, 1) 催化常数 [s^-1]
            K_m:   (Batch, 1) 米氏常数 [uM]
            enzyme_conc: (Batch, 1) 酶浓度 [E_total] [uM]
            substrate_conc: (Batch, 1) 底物浓度 [S] [uM]
        输出:
            v0: (Batch, 1) 初始反应速率 [uM/s]
        公式:
            V_max = k_cat * [E]
            v0 = (V_max * [S]) / (K_m + [S])
        """
        V_max = k_cat * enzyme_conc
        # 为了数值稳定性，分母加一个极小值 epsilon 防止除零
        v0 = (V_max * substrate_conc) / (K_m + substrate_conc + Config.EPSILON)
        return v0

class KineticsPredictor(nn.Module):
    """
    主模型：包含编码器融合、参数预测 MLP 和 物理层
    """
    def __init__(self, 
                 enzyme_dim=Config.ENZYME_DIM,    
                 substrate_dim=Config.SUBSTRATE_DIM, 
                 condition_dim=Config.CONDITION_DIM,    
                 hidden_dim=Config.HIDDEN_DIM,
                 dropout=Config.DROPOUT):
        super().__init__()
        
        # --- 1. 特征融合前的预处理 (解决模态失衡) ---
        # 将不同模态映射到同一维度，利于特征对齐
        self.enzyme_projector = nn.Sequential(
            nn.Linear(enzyme_dim, hidden_dim),
            nn.LayerNorm(hidden_dim), # LayerNorm 有助于平衡不同模态的方差
            nn.GELU()
        )
        self.substrate_projector = nn.Sequential(
            nn.Linear(substrate_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        )
        self.condition_projector = nn.Sequential(
            nn.Linear(condition_dim, hidden_dim // 4), # 条件维度通常较小
            nn.LayerNorm(hidden_dim // 4),
            nn.GELU()
        )

        # 计算融合后的总维度
        fusion_dim = hidden_dim + hidden_dim + (hidden_dim // 4)

        # --- 2. 动力学参数预测网络 (MLP) ---
        self.mlp = nn.Sequential(
            nn.Linear(fusion_dim, 512),
            nn.BatchNorm1d(512), # 加上 BN 防止过拟合
            nn.GELU(),
            nn.Dropout(dropout),
            
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            
            nn.Linear(256, 2) # 输出两个值: raw_k_cat, raw_K_m
        )

        # --- 3. 物理层 ---
        self.physics_layer = MichaelisMentenLayer()
        self.predict_log_params = Config.PREDICT_LOG_PARAMS

    def forward(self, enzyme_embed, substrate_fp, conditions, enzyme_conc, substrate_conc):
        """
        前向传播流程
        """
        # Step 1: 编码与投影
        e_feat = self.enzyme_projector(enzyme_embed)
        s_feat = self.substrate_projector(substrate_fp)
        c_feat = self.condition_projector(conditions)

        # Step 2: 特征融合 (Concatenation)
        # 形状: (Batch, hidden*2 + hidden//4)
        x_fused = torch.cat([e_feat, s_feat, c_feat], dim=1)

        # Step 3: 预测动力学参数
        raw_params = self.mlp(x_fused)
        
        # 关键约束：解决量纲差异与数值稳定性
        if self.predict_log_params:
            # 如果预测的是 log 值，直接取 exp 还原到正数空间
            # 这种方式天然保证输出为正，且梯度在 log 空间更平滑
            k_cat = torch.exp(raw_params[:, 0:1]) 
            K_m = torch.exp(raw_params[:, 1:2])
        else:
            # 否则使用 Softplus 确保参数为正
            k_cat = F.softplus(raw_params[:, 0:1]) 
            K_m = F.softplus(raw_params[:, 1:2])

        # Step 4: 物理层计算 (无参数)
        # 注意：这里需要传入酶浓度和底物浓度，它们是实验条件，不是模型参数
        v0_pred = self.physics_layer(k_cat, K_m, enzyme_conc, substrate_conc)

        return v0_pred, k_cat, K_m
