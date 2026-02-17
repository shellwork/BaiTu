import torch

class Config:
    # 数据路径
    DATA_PATH = "data/kinetics_data.csv"
    PREPROCESSED_DATA_PATH = "data/kinetics_data_with_embeddings.pt"
    CHECKPOINT_DIR = "checkpoints"
    
    # 模型超参数
    # 如果用 larger model, 需要修改这里:
    # t12_35M: 480
    # t30_150M: 640
    # t33_650M: 1280
    ENZYME_DIM = 320        # ESM-2 8M (t6_8M_UR50D) 的维度
    SUBSTRATE_DIM = 2048    # Morgan 指纹维度 (radius=3, nBits=2048)
    CONDITION_DIM = 3       # 温度(T), pH, 盐浓度(Salt)
    HIDDEN_DIM = 256        # 投影层维度
    DROPOUT = 0.2
    
    # 训练超参数
    BATCH_SIZE = 32
    LEARNING_RATE = 1e-4
    NUM_EPOCHS = 100
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 物理约束相关
    PREDICT_LOG_PARAMS = True  # 是否预测 log(k_cat) 和 log(Km) 以保证数值稳定性
    EPSILON = 1e-8             # 防止除零的小数
