"""
Shared constants for the BaiTu Battleship / Plate active-learning project.
"""

# ── Board geometry ────────────────────────────────────────────────────────
BOARD_ROWS = 8
BOARD_COLS = 10          # active experiment area
PLATE_ROWS = 8
PLATE_COLS = 12          # full 96-well plate
ACTIVE_COLS = BOARD_COLS

# ── Ship definitions ─────────────────────────────────────────────────────
DEFAULT_SHIP_SIZES = [5, 4, 3, 3, 2]
SHIP_NAMES = ["Carrier", "Battleship", "Cruiser", "Submarine", "Destroyer"]

# ── Strategy colours (shared palette) ────────────────────────────────────
STRATEGY_COLORS = {
    "random":      "#e74c3c",
    "prob":        "#2ecc71",
    "entropy":     "#3498db",
    "hunt_target": "#f39c12",
    "pro_solver":  "#9b59b6",
    "grid":        "#9b59b6",
}
