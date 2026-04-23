# Archive

Legacy / superseded code kept only for historical reference. **Nothing in the
active codebase imports from this directory.**

| Path | Why it's here |
| --- | --- |
| `BaituOT2Battleship/` | Early OT-2 control scaffolding (Flask server + protocol stubs). Superseded by `hardware/battleship_ot2_loop.py`, which contains the production closed-loop driver. |
| `battleship_next_matrix.py` | Thin wrapper around `BeliefModel.select_query()` that converted an 8×12 result matrix to a next-query matrix. No longer imported by any entry point. |
| `battleship_plate_no_touch.py` | Standalone CSV ship-placement generator from an earlier prototype. Functionality now lives in `core/battleship_env.py` + `plate/battleship_plate_simulation.py`. |
| `command.md` | Loose CLI cheat-sheet. Contents merged into the top-level `README.md`. |
| `old_codes/`, `old_codes1/` | Pre-BaiTu project folders retained from earlier commits. |

Feel free to delete this directory entirely once you're confident nothing here
is needed.
