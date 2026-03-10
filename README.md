# BaiTu Project: End-to-End Enzyme Kinetics Prediction

This project implements a data-driven neural network for predicting enzyme kinetics parameters ($k_{cat}$, $K_m$) and reaction rates ($v_0$), utilizing a hybrid supervision approach and an active learning framework.

## Roadmap: Two-Stage Training & Active Learning

We follow a structured training logic to bridge the gap between simulated data and real-world experiments:

### Phase 1: Pre-training on Synthetic Data
- **Objective**: Establish a baseline "common sense" of enzyme-substrate-kinetics mapping.
- **Data**: Large-scale simulated kinetics data generated from aligned $k_{cat}$ and $K_m$ databases.
- **Focus**: Learning the underlying physics (Michaelis-Menten) through direct parameter prediction and rate regression.

### Phase 2: Fine-tuning & Active Learning (Real Data)
- **Objective**: Adapt the model to real-world experimental distributions and minimize synthetic bias.
- **Active Learning Strategy**:
  - **Uncertainty Estimation**: Uses Deep Ensembles to quantify predictive confidence.
  - **Contribution Scoring**: Evaluates the potential of new experimental data to improve performance on a fixed real validation set.
  - **Experiment Recommendation**: Identifies the most valuable enzyme-substrate-concentration points for laboratory testing (High Uncertainty + High Novelty).
- **Refinement**: Gradually shift training weights from synthetic data to high-contribution real data.

## Architecture

The model uses a hybrid supervision architecture:

1. **Feature Extraction**: 
   - **Enzyme**: ESM-2 embeddings for sequence-level features.
   - **Substrate**: Morgan Fingerprints (ECFP4) for molecular structure.
2. **Predictor Heads**:
   - **Parameter Heads**: Directly predicts $log(k_{cat})$ and $log(K_m)$.
   - **Physics Layer**: A non-trainable Michaelis-Menten layer that derives reaction rate $v_0$ from predicted parameters.
3. **Hybrid Loss**: Combined loss on kinetic parameters (database ground truth) and reaction rates (experimental observations).

## Directory Structure

```text
BaiTu/
├── preprocess_esm.py   # ESM-2 embedding generation script
├── model.py            # Core model (Hybrid Heads + MM Layer)
├── dataset.py          # Dataset loader (supports synthetic/real sourcing)
├── train.py            # Training loop with regression metrics (R2, MAE, Acc)
├── active_learning.py  # [WIP] Uncertainty and sample selection logic
├── config.py           # Configuration parameters
├── utils.py            # Utility functions (RDKit Morgan Generators)
├── data/               # Data directory
│   ├── kinetics_simulated_from_km_kcat.csv
│   └── kinetics_simulated_with_embeddings.pt
├── checkpoints/        # Model checkpoints
├── requirements.txt    # Dependencies
└── README.md           # Project documentation
```

## Setup

1. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

2. **Generate simulated kinetics data from Km + kcat:**

   ```bash
   python data/generate_kinetics_from_km_kcat.py
   ```

3. **Preprocess Data (Crucial Step):**
   Generate ESM-2 embeddings from the simulated table.

   ```bash
   python preprocess_esm.py --csv_path data/kinetics_simulated_from_km_kcat.csv --output_path data/kinetics_simulated_with_embeddings.pt
   ```

## Usage

To train the model:

```bash
python train.py
```

## Key Features & Risk Mitigation

- **Hybrid Supervision**: Jointly optimizes for intrinsic parameters and observed rates.
- **Uncertainty-Aware**: Designed for Deep Ensemble integration to guide experiments.
- **Real-World Alignment**: Explicitly designed to transition from synthetic "pre-training" to real "fine-tuning".
- **Numerical Stability**: Predicts in log-space to handle wide dynamic ranges of kinetic constants.
- **Synthetic Supervision**: Aligns `Km` and `kcat` to generate dense $v_0$ samples over substrate concentration grids.
- **Modality Balance**: Uses LayerNorm and projection layers to balance high-dimensional enzyme embeddings and sparse molecular fingerprints.
- **Direct Regression**: Removes the in-model physics layer and predicts $v_0$ directly from fused features.
- **Metadata Preserved**: Keeps EC / Organism / UniProtID for downstream active-learning analysis.


## Demo Inference & Streamlit UI

1. **Single-sample inference (same I/O schema as training):**

   ```bash
   python inference_demo.py \
     --checkpoint checkpoints/model_epoch_100_params.pth \
     --sequence MKT... \
     --smiles CCO \
     --substrate_conc_m 1e-4 \
     --enzyme_conc_m 1e-6
   ```

   Output keys are aligned with training targets: `log_kcat`, `log_km`, `kcat_s-1`, `km_m`, `v0`.

2. **Launch Streamlit app for interactive prediction + curve visualization:**

   ```bash
   streamlit run streamlit_demo.py
   ```

   In the UI, users can input sequence/SMILES/concentrations, run prediction, and view the Michaelis-Menten rate curve.
