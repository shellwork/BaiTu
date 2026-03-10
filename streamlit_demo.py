import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from config import Config
from inference_demo import InferenceEngine, InferenceInput

st.set_page_config(page_title="Enzyme Kinetics Demo", layout="wide")
st.title("BaiTu Enzyme Kinetics Inference Demo")
st.caption("Input enzyme sequence, substrate SMILES, and assay concentrations to predict kcat / Km / v0 and visualize the rate curve.")

if "engine" not in st.session_state:
    st.session_state.engine = None

with st.sidebar:
    st.header("Model Setup")
    ckpt_path = st.text_input("Checkpoint Path", value="checkpoints/model_epoch_100_params.pth")
    load_btn = st.button("Load Model")

    if load_btn:
        try:
            st.session_state.engine = InferenceEngine(checkpoint_path=ckpt_path)
            st.success("Model loaded successfully")
        except Exception as exc:
            st.session_state.engine = None
            st.error(f"Model loading failed: {exc}")

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Inputs")
    sequence = st.text_area("Enzyme Amino-acid Sequence", height=180)
    smiles = st.text_input("Substrate SMILES", value="CCO")
    substrate_conc_m = st.number_input("Substrate Concentration [M]", min_value=1e-12, value=1e-4, format="%.6e")
    enzyme_conc_m = st.number_input("Enzyme Concentration [M]", min_value=1e-12, value=Config.ENZYME_CONC_M, format="%.6e")

    st.markdown("#### Rate Curve Range")
    conc_min_m = st.number_input("[S] Minimum [M]", min_value=1e-12, value=1e-8, format="%.6e")
    conc_max_m = st.number_input("[S] Maximum [M]", min_value=1e-12, value=1e-2, format="%.6e")
    n_points = st.slider("Curve Sampling Points", min_value=20, max_value=400, value=120)

    run_infer = st.button("Run Inference")

with col2:
    st.subheader("Inference Results")
    if run_infer:
        if st.session_state.engine is None:
            st.warning("Please load a model checkpoint from the left panel first.")
        elif not sequence.strip():
            st.warning("Please input an enzyme sequence first.")
        elif conc_max_m <= conc_min_m:
            st.warning("Please ensure [S] maximum is greater than [S] minimum.")
        else:
            sample = InferenceInput(
                sequence=sequence.strip(),
                smiles=smiles.strip(),
                substrate_conc_m=substrate_conc_m,
                enzyme_conc_m=enzyme_conc_m,
            )
            try:
                substrate_grid, rates, outputs = st.session_state.engine.predict_rate_curve(
                    sample=sample,
                    conc_min_m=conc_min_m,
                    conc_max_m=conc_max_m,
                    n_points=n_points,
                )

                result_df = pd.DataFrame([outputs])
                st.dataframe(result_df, use_container_width=True)

                fig, ax = plt.subplots(figsize=(7, 4.5))
                ax.plot(substrate_grid, rates, linewidth=2)
                ax.scatter([substrate_conc_m], [outputs["v0"]], color="red", label="Prediction at Input Point", zorder=3)
                ax.set_xscale("log")
                ax.set_xlabel("Substrate concentration [M]")
                ax.set_ylabel("Predicted v0")
                ax.set_title("Predicted Michaelis-Menten Rate Curve")
                ax.grid(alpha=0.3)
                ax.legend()
                st.pyplot(fig, clear_figure=True)
            except Exception as exc:
                st.error(f"Inference failed: {exc}")
