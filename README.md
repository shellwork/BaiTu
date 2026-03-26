# Battleship Lab Simulator

This project simulates a Battleship game as played on an Opentrons OT-2 liquid handler, combining robotics, computer vision, and active learning for experimental design and quality control.

## Overview

- **Setup:** The OT-2 robot dispenses NaOH into ship positions and water into empty positions on a virtual 96-well plate.
- **Play:** In each round, cabbage juice is dispensed into a selected well. The well turns purple if it contains NaOH (ship) and blue if it contains water (miss).
- **Analysis:** A computer vision algorithm analyzes the plate and returns a value from 0 (hit) to 1 (miss) for each well.
- **Learning:** An active learning algorithm selects the next well to query. Four strategies are available: max entropy, max probability, hunt-target heuristic, and random.

## Features

- Visual dashboard for monitoring the game, liquid usage, and quality control metrics.
- Four query selection strategies, each tracked independently.
- Human-in-the-loop verification for unclear readings in the hunt-target strategy.
- Tracks variance of readings, unclear readings, and remaining liquid.

## How to Run

1. **Install dependencies**

    Create and activate a conda environment (recommended):
     ```sh
     conda create -n bioinfo python=3.11
     conda activate bioinfo
     pip install -r requirements.txt
     ```

2. **Start the dashboard**
    
    Run the Streamlit dashboard:
     ```sh
     streamlit run battleship_dashboard.py
     ```
   The dashboard will open in your browser. If not, visit [http://localhost:8501](http://localhost:8501).

3. **How to Play**
- Use the strategy buttons to view and control each query strategy.
- Click **Next Shot** or **Play 5 Shots** to advance, or **Play All** to run to completion.
- For the **Hunt-Target** strategy, you may be prompted to verify unclear readings.
- Monitor liquid usage, QC metrics, and progress for each strategy.
- Start a new game anytime with the **New Board** button.

## File Structure

- `battleship_dashboard.py` — Main Streamlit dashboard
- `battleship_env.py`, `battleship_model.py`, `battleship_synthetic.py` — Core logic and simulation
- `requirements.txt` — Python dependencies
- `battleship_results/` — Saved simulation results

## Requirements
- Python 3.11+
- Streamlit
- numpy, matplotlib, etc. (see requirements.txt)

## License
MIT License
