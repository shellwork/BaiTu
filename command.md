# Quick command reference

## OT-2 hardware loop

```sh
# Built-in geometry calibration helper (no colour samples)
python hardware/battleship_ot2_loop.py calibrate

# Run with default settings
python hardware/battleship_ot2_loop.py --strategy prob --seed 42 --robot_ip 169.254.200.128

# Reset (home + drop tip)
python hardware/battleship_ot2_loop.py reset --robot_ip 169.254.200.128

# Run with a pre-built calibration (geometry + RGB + Lab discriminant)
python hardware/battleship_ot2_loop.py --strategy prob --seed 42 \
    --robot_ip 169.254.200.128 --geometry_path hardware/calibration.json

# Skip Phase-1 board setup (plate already prepared)
python hardware/battleship_ot2_loop.py --strategy prob --seed 42 \
    --robot_ip 169.254.200.128 --geometry_path hardware/calibration.json --skip_setup

# Tune / disable the human-check trigger
#   --human_check_margin 0.0   → disable
#   --human_check_margin 0.08  → wider band, more prompts (more conservative)
python hardware/battleship_ot2_loop.py --strategy prob --seed 42 \
    --robot_ip 169.254.200.128 --geometry_path hardware/calibration.json \
    --human_check_margin 0.05
```

## Calibration (geometry + RGB + Lab discriminant)

```sh
# 1) Take a photo of the current plate
python hardware/calibrate_geometry.py capture

# 2) Annotate: 4 corner clicks → 2-6 HIT clicks → 2-6 MISS clicks (press 'd' when done)
python hardware/calibrate_geometry.py annotate plate_photo_20260428_135456.jpg
# Writes hardware/calibration.json with both ship_rgb/water_rgb AND
# lab_ship/lab_water + lab_fisher_score (the Lab discriminant is preferred at runtime)
```

## Operator IPC files (written under <output_dir>)

```text
human_check_request.json   ← loop writes when |conf - 0.5| < margin (paused)
human_check_response.json  ← dashboard writes {"label": "ship"|"water"} to resume
corrections.json           ← dashboard queues {"corrections":[{row,col,label},...]}
                              loop reads + applies + deletes before the next step
```

The Streamlit dashboard exposes both flows as buttons; the JSON paths are
documented for headless / CLI override.
