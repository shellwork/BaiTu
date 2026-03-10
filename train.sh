python data/generate_kinetics_from_km_kcat.py
python preprocess_esm.py --csv_path data/kinetics_simulated_from_km_kcat.csv --output_path data/kinetics_simulated_with_embeddings.pt
python train.py