# BaiTu — Battleship Active Learning on an OT-2 Liquid Handler

BaiTu treats a Battleship game as a proxy for an automated wet-lab experiment.
The Opentrons OT-2 dispenses NaOH into "ship" wells and water into "miss"
wells on a 96-well plate. Each round, cabbage-juice indicator is added to a
queried well — the well turns purple (hit, NaOH) or blue (miss, water). A
computer-vision readout feeds the result back into a Bayesian belief model,
which picks the next well to probe. The goal is to sink every ship in as few
queries as possible under realistic CV noise, liquid budgets, and quality
control constraints.

The same belief model, oracle, and plate simulator drive four parallel
deliverables:

1. **Hardware loop** — closed-loop run on a real OT-2 robot with a camera.
2. **Synthetic experiment suite** — batch comparison of acquisition strategies
   on an in-memory board.
3. **Plate-level active learning** — image-based experiments on a simulated
   96-well plate with a CV detector.
4. **Campaign** — outer-loop policy search that fits a surrogate over
   strategies and tracks stopping criteria.
5. **Interactive dashboard** — Streamlit UI that visualises the game and
   handles unclear CV readings with a human-in-the-loop.

---

## What's implemented

### Core library (`core/`)

- `battleship_env.py` — ground-truth Battleship board, ship placement, oracle
  queries.
- `battleship_model.py` — unified Bayesian belief model with two modes:
  - **Ship mode** — posterior over fleet placements.
  - **Plate mode** — per-cell Beta posteriors with Gaussian spatial spreading.
  - Shared acquisition API: `random`, `prob`, `entropy`, `hunt_target`,
    `pro_solver`, `grid`.
- `battleship_matrix_oracle.py` — adapter that normalises both board queries
  and plate-image readouts into the same active-area matrix.

### Plate simulation & computer vision (`plate/`)

- `plate_simulator.py` — generic 96-well plate photo simulator (meniscus,
  reflections, illumination gradient, plastic background).
- `battleship_plate_simulation.py` — embeds an 8×10 Battleship board into the
  full 8×12 plate (two reserved control columns) and renders a photo.
- `battleship_plate_readout.py` — fixed-ROI RGB sampling and colour
  classification (ship / water / unknown) against prototype colours.
- `plate_detector.py` — end-to-end CV pipeline that converts a plate image
  into an 8×12 label matrix. Grid-based and Hough-circle modes, HSV
  thresholds.
- `plate_analysis.py` — Hough-circle utilities.
- `plate_active_learning.py` — plate-mode active-learning runner
  (`random`, `grid`, `prob`, `entropy`).
- `plate_run.py` — CLI entry point: demo, full experiment, or CV accuracy
  check.
- `generate_battleship_plate_dataset.py` — generates synthetic
  image+label datasets for training / validation.

### OT-2 hardware loop (`hardware/`)

- `battleship_ot2_loop.py` — production closed-loop driver. Setup phase
  dispenses NaOH / H₂O to the whole plate, then runs a query loop:
  model picks a well → robot dispenses indicator → camera captures →
  `battleship_plate_readout` classifies → model updates. Handles dry runs,
  resets, calibration, skip-setup, and checkpointing. The canonical deck
  layout (pipette, slot assignments, labware, source wells, per-well
  volumes) lives in `DEFAULT_DECK` at the top of this file — see
  [Deck layout](#deck-layout) below.
- `calibrate_geometry.py` — interactive OpenCV tool. Click the four corner
  wells for geometry, then sample hit / miss prototype colours. Writes
  `hardware/calibration.json`.
- `test_rgb.py` — sanity check for RGB colour recognition on a saved or
  freshly captured plate photo.
- `calibration.json` — current geometry + RGB prototypes for the rig.
- `samples/` — two reference plate photos used in the docs.

### Synthetic experiments (`experiment/`)

- `battleship_experiment.py` — episode runner, learning / entropy curves,
  comparison plots.
- `battleship_run.py` — CLI entry point with three modes: `demo`, `compare`,
  `experiment`.
- `battleship_synthetic.py` — Gaussian-noise model for CV readouts used by
  the dashboard and batch experiments.

### Campaign / policy search (`campaign/`)

- `battleship_campaign.py` — proposes acquisition policies each cycle, runs
  them against the simulator, fits an ensemble surrogate, and tracks
  learning / QC / stopping metrics.
- `battleship_campaign_dashboard.py` — Streamlit view over the saved
  campaign history.

### Interactive dashboard (`dashboard/`)

- `battleship_dashboard.py` — multi-page Streamlit app (`st.navigation`):
  - **Home** — project overview and links to the other pages.
  - **Simulator** — runs all four strategies in the browser, shows a
    hits-vs-round chart plus the live hunt-target plate, and pauses on
    unclear readings (score in 0.4 – 0.6) for a human hit / miss call.
  - **OT-2 Hardware** — configure, launch, pause, resume, stop, and
    reset the real `hardware.battleship_ot2_loop`. Live view of the plate
    state, belief map, latest saved camera frame, an optional **~1 Hz
    live webcam preview** for sanity-checking the rig, last step, step
    history, and log stream. Dry-run mode rehearses the pipeline without
    any robot.
- `ot2_controller.py` — subprocess-based controller. The hardware loop
  runs as its own OS process so **Stop** sends `SIGTERM` (identical to a
  manual Ctrl-C), **Pause / Resume** use `SIGSTOP / SIGCONT`, and **Reset
  robot** shells out to `python -m hardware.battleship_ot2_loop reset
  --robot_ip …`. State is reconstructed from `checkpoint.json` that the
  loop writes after every step.

### Shared

- `config.py` — global constants (board / plate geometry, ship sizes,
  strategy colours). Deliberately light so every module can import it.
- `utils/plotting.py` — learning-curve interpolation helpers.

---

## Directory layout

```text
BaiTu/
├── README.md
├── requirements.txt
├── config.py                         # shared constants
│
├── core/                             # game engine + belief model
│   ├── battleship_env.py
│   ├── battleship_model.py
│   └── battleship_matrix_oracle.py
│
├── plate/                            # 96-well plate simulation & CV
│   ├── plate_simulator.py
│   ├── battleship_plate_simulation.py
│   ├── battleship_plate_readout.py
│   ├── plate_detector.py
│   ├── plate_analysis.py
│   ├── plate_active_learning.py
│   ├── plate_run.py                  # entry point
│   └── generate_battleship_plate_dataset.py
│
├── hardware/                         # OT-2 closed-loop & calibration
│   ├── battleship_ot2_loop.py        # entry point
│   ├── calibrate_geometry.py         # entry point
│   ├── test_rgb.py                   # entry point
│   ├── calibration.json
│   └── samples/                      # reference plate photos
│
├── experiment/                       # synthetic comparison runs
│   ├── battleship_experiment.py
│   ├── battleship_run.py             # entry point
│   └── battleship_synthetic.py
│
├── campaign/                         # policy search + surrogate
│   ├── battleship_campaign.py        # entry point
│   └── battleship_campaign_dashboard.py   # Streamlit
│
├── dashboard/                        # Streamlit demo
│   └── battleship_dashboard.py
│
├── utils/
│   └── plotting.py
│
└── archive/                          # legacy / superseded — not imported
```

Runtime output directories (`battleship_results/`,
`experiment/battleship_results/`, `plate/plate_results/`,
`plate/simulated_battleship_plate_dataset/`,
`campaign/battleship_campaign_results/`, `checkpoints/`) are gitignored.

---

## Installation

```sh
conda create -n baitu python=3.11
conda activate baitu
pip install -r requirements.txt
```

---

## Running each pipeline

All commands below are executed from the repository root. Most of commands can be run on the Streamlit Dashboard.

### Streamlit dashboard

```sh
streamlit run dashboard/battleship_dashboard.py
```

Opens on <http://localhost:8501>.

### Synthetic experiment suite

```sh
# Watch one episode step-by-step
python -m experiment.battleship_run --mode demo --strategy prob --seed 42

# Compare all strategies on the same board
python -m experiment.battleship_run --mode compare --seed 7

# Full batch (n boards × 5 strategies)
python -m experiment.battleship_run --mode experiment --n_episodes 200
```

### Plate active learning

```sh
# Demo one plate image + CV detection
python -m plate.plate_run --mode demo --seed 42

# Full experiment
python -m plate.plate_run --mode experiment --n_episodes 100

# Evaluate CV detector accuracy
python -m plate.plate_run --mode cv_test --n_plates 30

# Generate a synthetic image dataset
python -m plate.generate_battleship_plate_dataset \
    --output_dir plate/simulated_battleship_plate_dataset
```

### OT-2 hardware loop

```sh
# One-off calibration (click 4 corners + 2 sample wells per colour)
python -m hardware.calibrate_geometry capture
python -m hardware.calibrate_geometry annotate plate_photo_<timestamp>.jpg
# → writes hardware/calibration.json

# Sanity-check RGB recognition on a saved photo
python -m hardware.test_rgb <image_path> --calibration hardware/calibration.json

# Dry run (synthetic images, no hardware required)
python -m hardware.battleship_ot2_loop --dry_run --strategy prob --seed 42

# Real experiment
python -m hardware.battleship_ot2_loop --strategy prob --seed 42 \
    --robot_ip 169.254.200.128 --geometry_path hardware/calibration.json

# Skip board-setup if the plate is already prepared
python -m hardware.battleship_ot2_loop --strategy prob --seed 42 \
    --robot_ip 169.254.200.128 --geometry_path hardware/calibration.json --skip_setup

# Reset (robot returns tips, empties wells)
python -m hardware.battleship_ot2_loop reset --robot_ip 169.254.200.128
```

### Policy-search campaign

```sh
python -m campaign.battleship_campaign --max_cycles 5 --query_size 8
streamlit run campaign/battleship_campaign_dashboard.py
```

---

## Deck layout

The OT-2 physical deck (pipette, slot assignments, labware names, reservoir
source wells, per-well volumes) is defined by a single dict named
`DEFAULT_DECK` at the top of
[`hardware/battleship_ot2_loop.py`](hardware/battleship_ot2_loop.py). Current
defaults:

| Slot | Purpose            | Labware                           | Source well |
| ---- | ------------------ | --------------------------------- | ----------- |
| 1    | 96-well plate      | `corning_96_wellplate_360ul_flat` | —           |
| 2    | Tiprack            | `opentrons_96_tiprack_1000ul`     | —           |
| 4    | NaOH reservoir     | `nest_12_reservoir_15ml`          | A1          |
| 5    | H₂O reservoir      | `nest_12_reservoir_15ml`          | A1          |
| 6    | Indicator reservoir| `nest_12_reservoir_15ml`          | A1          |

Pipette: `p1000_single` on the **right** mount. Fill volume: 100 µL.
Indicator volume: 100 µL.

### Viewing the live deck

The **OT-2 Hardware** page of the Streamlit dashboard renders the deck at
the top of the page as a 4 × 3 OT-2 slot grid (slots 10 / 11 / trash on the
top row, slots 1 / 2 / 3 on the bottom). Each populated slot shows its role
(Plate / Tiprack / NaOH / H₂O / Indicator), the labware identifier, and the
reservoir source well. A caption above the grid reports the pipette, mount,
and the per-well fill / indicator volumes. When any field has been
overridden for the current session the caption shows *(overrides active)*.

### Changing the deck

You have two options, depending on how permanent the change should be:

1. **Temporary override (single run) via the dashboard.**
   On the OT-2 Hardware page, open **Edit deck layout (applied to the next
   run)**, update the fields, and click **Apply deck changes**. The form
   validates that no two roles share a slot. On the next **Start**, the
   merged deck is written to `<output_dir>/deck.json` and the subprocess
   launches with `--deck_path <output_dir>/deck.json`. A **Reset to
   defaults** button clears all overrides.

2. **Permanent change (new default).**
   Edit `DEFAULT_DECK` in
   [`hardware/battleship_ot2_loop.py`](hardware/battleship_ot2_loop.py)
   directly. Everything (CLI, subprocess, dashboard) reads from this dict,
   so a single source edit propagates everywhere.

### Running the CLI with a custom deck

You can also bypass the dashboard and pass a deck JSON to the CLI:

```sh
# deck.json may contain any subset of the DEFAULT_DECK keys — missing keys
# fall back to defaults.
python -m hardware.battleship_ot2_loop \
    --strategy prob --seed 42 \
    --robot_ip 169.254.200.128 \
    --geometry_path hardware/calibration.json \
    --deck_path path/to/deck.json
```

Example `deck.json` (moves the indicator reservoir to slot 9 and raises the
indicator volume):

```json
{
  "indicator_slot": "9",
  "indicator_volume": 150.0
}
```

---

## License

MIT
