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
    header = "   " + "  ".join(str(c) for c in range(1, 11))
    print(header)
    for i, label in enumerate(ROW_LABELS):
        print(f"{label}  " + "  ".join(board[i]))


if __name__ == "__main__":
    main()
