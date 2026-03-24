"""
Battleship Active Learning – Entry Point
=========================================
Usage
-----
  # Interactive demo: watch one episode step by step
  python battleship_run.py --mode demo --strategy prob --seed 42

  # Single comparison episode (all strategies, same board)
  python battleship_run.py --mode compare --seed 7

  # Full experiment (n boards × 4 strategies)
  python battleship_run.py --mode experiment --n_episodes 200

Modes
-----
  demo       : ASCII step-by-step walkthrough of one game
  compare    : show all 4 strategies on the same board, print query counts
  experiment : run batch, save figures, print summary table
"""

import argparse
import os
import time

import numpy as np

from battleship_env import BattleshipBoard
from battleship_model import Game
from battleship_experiment import (
    LABELS,
    STRATEGIES,
    plot_comparison,
    plot_episode_detail,
    plot_learning_curves,
    print_summary,
    run_episode,
    run_experiment,
)

# ------------------------------------------------------------------
# Demo mode
# ------------------------------------------------------------------

def demo(strategy: str, seed: int, pause: float = 0.0):
    """Step-through a single episode in the terminal."""
    print(f"\n{'='*55}")
    print(f" BATTLESHIP  |  strategy={LABELS[strategy]}  |  seed={seed}")
    print(f"{'='*55}")

    board = BattleshipBoard(rows=8, cols=10, seed=seed)
    model = Game(board_rows=8, board_cols=10)

    print("\n[Ground truth – hidden from learner]")
    gt_sym = {0: "·", 1: "■"}
    header = "   " + " ".join(str(c) for c in range(board.cols))
    print(header)
    for r in range(board.rows):
        print(f"{r:2d} " + " ".join(gt_sym[board.grid[r, c]] for c in range(board.cols)))

    input("\nPress Enter to start...\n")

    while not board.is_game_over():
        pos = model.select_query(strategy)
        if pos is None:
            break
        row, col = pos

        is_hit, sunk_ship = board.query(row, col)
        model.update(row, col, is_hit, sunk_ship)

        result_str = "HIT 🎯" if is_hit else "miss"
        if sunk_ship:
            result_str += f"  ← SUNK size-{sunk_ship.size} ship!"

        print(f"  Step {board.n_queries:3d} | query ({row},{col}) → {result_str}")
        print(board)
        print(f"  [entropy={model.get_entropy_map().sum():.2f}  max_prob={model.prob_map.max():.3f}]")

        if pause > 0:
            time.sleep(pause)
        elif pause == 0:
            input("  (Enter for next step)")
        print()

    print(f"\nGame over in {board.n_queries} queries.")
    print(f"Ships: {[s.size for s in board.ships]}")


# ------------------------------------------------------------------
# Compare mode
# ------------------------------------------------------------------

def compare(seed: int):
    """Run all strategies on the same board and show query counts."""
    print(f"\n[Compare mode – seed={seed}]\n")
    results = {}
    for strategy in STRATEGIES:
        ep = run_episode(strategy, seed=seed)
        results[strategy] = ep
        print(f"  {LABELS[strategy]:<35} → {ep['n_queries']:3d} queries")
    return results


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Battleship Active Learning")
    parser.add_argument("--mode",       choices=["demo", "compare", "experiment"],
                        default="experiment")
    parser.add_argument("--strategy",   choices=STRATEGIES, default="prob",
                        help="Acquisition strategy (demo mode only)")
    parser.add_argument("--seed",       type=int, default=42)
    parser.add_argument("--n_episodes", type=int, default=200,
                        help="Number of episodes per strategy (experiment mode)")
    parser.add_argument("--pause",      type=float, default=0.0,
                        help="Seconds between steps in demo mode (0=manual)")
    parser.add_argument("--out_dir",    default="battleship_results",
                        help="Directory for saved figures (experiment mode)")
    args = parser.parse_args()

    if args.mode == "demo":
        demo(args.strategy, seed=args.seed, pause=args.pause)

    elif args.mode == "compare":
        compare(seed=args.seed)

    elif args.mode == "experiment":
        os.makedirs(args.out_dir, exist_ok=True)
        print(f"\nRunning {args.n_episodes} episodes × {len(STRATEGIES)} strategies …")
        t0 = time.time()
        results = run_experiment(n_episodes=args.n_episodes, verbose=True)
        elapsed = time.time() - t0
        print(f"\nFinished in {elapsed:.1f}s")

        print_summary(results)

        print("\nGenerating figures …")
        plot_comparison(results, save_dir=args.out_dir)
        plot_learning_curves(results, save_dir=args.out_dir)

        # Episode detail for multiple strategies
        for strategy in ["prob", "entropy", "hunt_target"]:
            # Pick episode with median query count for a representative example
            eps = results[strategy]
            counts = np.array([ep["n_queries"] for ep in eps])
            median_seed = eps[int(np.argsort(counts)[len(counts) // 2])]["seed"]
            ep = run_episode(strategy, seed=median_seed)
            plot_episode_detail(ep, save_dir=args.out_dir)

        print(f"\nAll figures saved to '{args.out_dir}/'")


if __name__ == "__main__":
    main()
