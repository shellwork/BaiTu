import random
import csv
import sys

ROWS = 8
COLS = 10
ROW_LABELS = [chr(ord('A') + i) for i in range(ROWS)]
COL_LABELS = list(range(1, 13))  # 96-well plate: 1-12

SHIPS = [
    ("Carrier", 5),
    ("Battleship", 4),
    ("Cruiser", 3),
    ("Submarine", 3),
    ("Destroyer", 2),
]


def _ship_cells(r, c, size, horizontal):
    """Return the list of (row, col) cells a ship would occupy."""
    if horizontal:
        return [(r, c + i) for i in range(size)]
    return [(r + i, c) for i in range(size)]


def _has_neighbor(board, cells):
    """Check whether any orthogonally adjacent cell to the ship already
    contains another ship ('o')."""
    occupied = set(cells)
    for r, c in cells:
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < ROWS and 0 <= nc < COLS and (nr, nc) not in occupied:
                if board[nr][nc] == 'o':
                    return True
    return False


def _try_place(board, size, require_gap):
    """Attempt a single random placement. Returns the cells placed, or None."""
    horizontal = random.choice([True, False])
    if horizontal:
        r = random.randint(0, ROWS - 1)
        c = random.randint(0, COLS - size)
    else:
        r = random.randint(0, ROWS - size)
        c = random.randint(0, COLS - 1)

    cells = _ship_cells(r, c, size, horizontal)

    # Basic overlap check
    if any(board[r][c] != 'x' for r, c in cells):
        return None
    # Optional adjacency check
    if require_gap and _has_neighbor(board, cells):
        return None

    for r, c in cells:
        board[r][c] = 'o'
    return cells


def make_board():
    board = [['x'] * COLS for _ in range(ROWS)]
    for _name, size in SHIPS:
        # Phase 1: try hard to place with a gap around existing ships
        placed = False
        for _ in range(500):
            if _try_place(board, size, require_gap=True):
                placed = True
                break
        # Phase 2: fall back to allowing touching
        if not placed:
            for attempt in range(500):
                if _try_place(board, size, require_gap=False):
                    placed = True
                    break
        if not placed:
            # Extremely unlikely; restart from scratch
            return make_board()
    return board


def write_csv(board, filename, keep=None):
    """Write board to CSV. If keep is 'x' or 'o', only that marker is shown;
    cells with the other marker are left blank."""
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([''] + COL_LABELS)
        for i, row_label in enumerate(ROW_LABELS):
            row_data = [
                cell if (keep is None or cell == keep) else ''
                for cell in board[i]
            ]
            row_data += ['', '']               # columns 11-12 empty
            writer.writerow([row_label] + row_data)


FILL_VOLUME = 250  # uL per well

API_LEVEL = "2.16"


def _board_to_wells(board):
    """Split board into NaOH well list and H2O well list.

    Liquid mapping:
      - 'x' cells (water on board) -> NaOH (slot 7)
      - 'o' cells (ship on board)  -> H2O  (slot 8)
    """
    naoh_wells = []
    h2o_wells = []
    for r in range(ROWS):
        for c in range(COLS):
            well_name = f"{ROW_LABELS[r]}{c + 1}"
            if board[r][c] == 'x':
                naoh_wells.append(well_name)
            elif board[r][c] == 'o':
                h2o_wells.append(well_name)
    return naoh_wells, h2o_wells


def run_ot2(board):
    """Connect directly to the OT-2 and build the plate from the board array."""
    from opentrons import execute

    naoh_wells, h2o_wells = _board_to_wells(board)

    protocol = execute.get_protocol_api(API_LEVEL)
    protocol.home()

    # ── Labware ──────────────────────────────────────────────
    plate    = protocol.load_labware("corning_96_wellplate_360ul_flat", 1)
    tiprack  = protocol.load_labware("opentrons_96_filtertiprack_1000ul", 3)
    naoh_res = protocol.load_labware("nest_12_reservoir_15ml", 7)
    h2o_res  = protocol.load_labware("nest_12_reservoir_15ml", 8)

    # ── Pipette ──────────────────────────────────────────────
    pipette = protocol.load_instrument("p1000_single_gen2", "right",
                                       tip_racks=[tiprack])

    # ── Dispense NaOH (slot 7) into water wells ─────────────
    print(f"Dispensing NaOH into {len(naoh_wells)} wells...")
    pipette.pick_up_tip()
    for well in naoh_wells:
        pipette.aspirate(FILL_VOLUME, naoh_res["A1"])
        pipette.dispense(FILL_VOLUME, plate[well])
    pipette.drop_tip()

    # ── Dispense H2O (slot 8) into ship wells ───────────────
    print(f"Dispensing H2O into {len(h2o_wells)} wells...")
    pipette.pick_up_tip()
    for well in h2o_wells:
        pipette.aspirate(FILL_VOLUME, h2o_res["A1"])
        pipette.dispense(FILL_VOLUME, plate[well])
    pipette.drop_tip()

    protocol.home()
    print("Plate build complete.")


def write_ot2_protocol(board, filename):
    """Write a standalone OT-2 protocol .py file from the board array."""
    naoh_wells, h2o_wells = _board_to_wells(board)

    protocol_text = f'''\
from opentrons import protocol_api

metadata = {{
    "protocolName": "Battleship Plate Setup",
    "author": "BaiTu",
    "description": "Dispense NaOH into water wells and H2O into ship wells",
}}
requirements = {{"robotType": "OT-2", "apiLevel": "{API_LEVEL}"}}

FILL_VOLUME = {FILL_VOLUME}  # uL per well

NAOH_WELLS = {naoh_wells}
H2O_WELLS  = {h2o_wells}


def run(protocol: protocol_api.ProtocolContext):
    plate    = protocol.load_labware("corning_96_wellplate_360ul_flat", 1)
    tiprack  = protocol.load_labware("opentrons_96_filtertiprack_1000ul", 3)
    naoh_res = protocol.load_labware("nest_12_reservoir_15ml", 7)
    h2o_res  = protocol.load_labware("nest_12_reservoir_15ml", 8)

    pipette = protocol.load_instrument("p1000_single_gen2", "right",
                                       tip_racks=[tiprack])

    pipette.pick_up_tip()
    for well in NAOH_WELLS:
        pipette.aspirate(FILL_VOLUME, naoh_res["A1"])
        pipette.dispense(FILL_VOLUME, plate[well])
    pipette.drop_tip()

    pipette.pick_up_tip()
    for well in H2O_WELLS:
        pipette.aspirate(FILL_VOLUME, h2o_res["A1"])
        pipette.dispense(FILL_VOLUME, plate[well])
    pipette.drop_tip()
'''

    with open(filename, 'w', newline='') as f:
        f.write(protocol_text)
    print(f"Saved protocol: {filename}")


def main():
    out_dir = sys.argv[1] if len(sys.argv) > 1 else "."
    board = make_board()

    combined = f"{out_dir}/combined_bs_plate.csv"
    water    = f"{out_dir}/water_placement.csv"
    ships    = f"{out_dir}/ship_placement.csv"

    write_csv(board, combined)
    write_csv(board, water,  keep='x')
    write_csv(board, ships,  keep='o')

    print(f"Saved:\n  {combined}\n  {water}\n  {ships}\n")

    # Comment out whichever you don't need:
    run_ot2(board)                                          # direct OT-2 control
    # write_ot2_protocol(board, f"{out_dir}/battleship_ot2_protocol.py")  # save protocol file

    header = "   " + "  ".join(str(c) for c in range(1, 11))
    print(header)
    for i, label in enumerate(ROW_LABELS):
        print(f"{label}  " + "  ".join(board[i]))


if __name__ == "__main__":
    main()
