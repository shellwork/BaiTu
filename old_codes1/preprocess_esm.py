import torch
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
import os
import argparse

# Configuration
# Using ESM-2 8M model as default for speed and low resource usage (dim=320)
DEFAULT_MODEL_NAME = "facebook/esm2_t6_8M_UR50D"
DATA_PATH = "data/kinetics_data.csv"
OUTPUT_PATH = "data/kinetics_data_with_embeddings.pt"
BATCH_SIZE = 32

# Local model cache directory
MODEL_CACHE_DIR = "./esm_model_cache"

def get_device():
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    return "cpu"

def preprocess_esm_embeddings(csv_path, output_path, model_name=DEFAULT_MODEL_NAME, batch_size=BATCH_SIZE):
    print(f"Loading data from {csv_path}...")
    if not os.path.exists(csv_path):
        # If data doesn't exist, generate dummy data first
        print("Data file not found. Generating dummy data first...")
        from utils import generate_dummy_data
        generate_dummy_data(num_samples=100, output_path=csv_path)

    df = pd.read_csv(csv_path)
    # Supports both kinetics format (enzyme_seq) and Km format (Sequence)
    seq_col = 'Sequence' if 'Sequence' in df.columns else 'enzyme_seq'
    sequences = df[seq_col].tolist()
    
    print(f"Loading ESM-2 model: {model_name}...")
    print(f"Model will be cached in: {MODEL_CACHE_DIR}")
    device = get_device()
    print(f"Using device: {device}")
    
    # Load tokenizer and model with local cache support
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=MODEL_CACHE_DIR)
    model = AutoModel.from_pretrained(model_name, cache_dir=MODEL_CACHE_DIR).to(device)
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
            
            # Mean Pooling (ignoring padding tokens)
            attention_mask = inputs['attention_mask'] # (Batch, Seq_Len)
            
            # Expand mask to match hidden state dimensions
            mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
            
            # Compute: sum(hidden * mask) / sum(mask)
            sum_embeddings = torch.sum(last_hidden_state * mask_expanded, dim=1)
            sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
            
            batch_embeddings = sum_embeddings / sum_mask
            
            embeddings.append(batch_embeddings.cpu())

    # Concatenate all batches
    all_embeddings = torch.cat(embeddings, dim=0) # (Total_Samples, Dim)
    
    print(f"Embeddings computed. Shape: {all_embeddings.shape}")
    
    # Save processed data
    # We save a dictionary containing the original DataFrame and the precomputed Embeddings
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
