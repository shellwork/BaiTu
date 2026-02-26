"""
Standalone script to generate synthetic trypsin kinetics data.

Usage:
    python data/generate_trypsin_data.py

Writes data/trypsin_data.csv with columns:
    temperature    [°C]
    substrate_conc [µM]
    v0             [µM/s]
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import generate_trypsin_data
from config import TrypsinConfig

if __name__ == "__main__":
    generate_trypsin_data(
        num_temps=20,
        num_substrate_concs=12,
        noise_level=0.05,
        seed=42,
        output_path=TrypsinConfig.TRYPSIN_DATA_PATH,
    )
    print("Done. You can now run: python train.py")
