import torch
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
import os
import argparse

# 配置
# 使用 ESM-2 8M 模型作为默认，因为它小且快 (dim=320)
# 如果你有更多资源，可以换成 "facebook/esm2_t12_35M_UR50D" (dim=480) 
# 或 "facebook/esm2_t33_650M_UR50D" (dim=1280)
DEFAULT_MODEL_NAME = "facebook/esm2_t6_8M_UR50D"
DATA_PATH = "data/kinetics_data.csv"
OUTPUT_PATH = "data/kinetics_data_with_embeddings.pt"
BATCH_SIZE = 32

def get_device():
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    return "cpu"

def preprocess_esm_embeddings(csv_path, output_path, model_name=DEFAULT_MODEL_NAME, batch_size=BATCH_SIZE):
    print(f"Loading data from {csv_path}...")
    if not os.path.exists(csv_path):
        # 如果数据不存在，先生成假数据
        print("Data file not found. Generating dummy data first...")
        from src.utils import generate_dummy_data
        generate_dummy_data(num_samples=100, output_path=csv_path)

    df = pd.read_csv(csv_path)
    sequences = df['enzyme_seq'].tolist()
    
    print(f"Loading ESM-2 model: {model_name}...")
    device = get_device()
    print(f"Using device: {device}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    model.eval()

    embeddings = []
    
    print("Computing embeddings...")
    with torch.no_grad():
        for i in tqdm(range(0, len(sequences), batch_size)):
            batch_seqs = sequences[i:i + batch_size]
            
            # Tokenize
            inputs = tokenizer(batch_seqs, return_tensors="pt", padding=True, truncation=True, max_length=1024)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            # Inference
            outputs = model(**inputs)
            
            # Get Last Hidden State: (Batch, Seq_Len, Dim)
            last_hidden_state = outputs.last_hidden_state
            
            # Mean Pooling (注意要忽略 padding token)
            attention_mask = inputs['attention_mask'] # (Batch, Seq_Len)
            
            # 将 mask 扩展维度以匹配 hidden state
            mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
            
            # sum(hidden * mask) / sum(mask)
            sum_embeddings = torch.sum(last_hidden_state * mask_expanded, dim=1)
            sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
            
            batch_embeddings = sum_embeddings / sum_mask
            
            embeddings.append(batch_embeddings.cpu())

    # Concatenate all batches
    all_embeddings = torch.cat(embeddings, dim=0) # (Total_Samples, Dim)
    
    print(f"Embeddings computed. Shape: {all_embeddings.shape}")
    
    # 保存数据
    # 我们保存一个字典，包含原始 DataFrame 和 计算好的 Embeddings
    data_dict = {
        'dataframe': df,
        'enzyme_embeddings': all_embeddings,
        'model_name': model_name,
        'embedding_dim': all_embeddings.shape[1]
    }
    
    torch.save(data_dict, output_path)
    print(f"Saved preprocessed data to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Precompute ESM embeddings for enzyme sequences")
    parser.add_argument("--csv_path", type=str, default=DATA_PATH, help="Path to input CSV")
    parser.add_argument("--output_path", type=str, default=OUTPUT_PATH, help="Path to output .pt file")
    parser.add_argument("--model_name", type=str, default=DEFAULT_MODEL_NAME, help="ESM model name")
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE, help="Batch size for inference")
    
    args = parser.parse_args()
    
    preprocess_esm_embeddings(
        csv_path=args.csv_path,
        output_path=args.output_path,
        model_name=args.model_name,
        batch_size=args.batch_size
    )
