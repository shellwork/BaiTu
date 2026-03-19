"""
Plate Active Learning – Entry Point
=====================================
Usage
-----
  # Show simulated plate image + CV detection result
  python plate_run.py --mode demo --seed 42

  # Run full experiment and save all figures
  python plate_run.py --mode experiment --n_episodes 100

  # Quick accuracy check of the CV detector
  python plate_run.py --mode cv_test --n_plates 30
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np

from plate_active_learning import (
    LABELS_STR,
    STRATEGIES,
    plot_belief_evolution,
    plot_plate_comparison,
    plot_plate_learning_curves,
    print_summary,
    run_plate_episode,
    run_plate_experiment,
)
from plate_detector import PlateDetector
from plate_simulator import PlateSimulator, clustered_plate_labels, random_plate_labels


# ──────────────────────────────────────────────────────────────────────
# Demo: visualise one plate image + CV detection
# ──────────────────────────────────────────────────────────────────────

def demo(seed: int, save_dir: str):
    os.makedirs(save_dir, exist_ok=True)
    gt  = clustered_plate_labels(positive_fraction=0.25, seed=seed)
    sim = PlateSimulator(seed=seed)
    img = sim.generate_image(gt)

    detector = PlateDetector(geometry=sim.get_geometry())
    detected = detector.process(img)
    acc      = detector.accuracy(detected, gt)

    print(f"\n[Demo – seed={seed}]")
    print(f"  Ground truth  : {gt.sum()} purple / {(1-gt).sum()} blue")
    print(f"  CV accuracy   : {acc['accuracy']:.3f}")
    print(f"  CV precision  : {acc['precision']:.3f}  recall={acc['recall']:.3f}")
    print(f"  Unknown wells : {acc['n_unknown']}")

    fig, axes = plt.subplots(1, 3, figsize=(20, 7))

    # Ground truth
    ax = axes[0]
    ax.imshow(np.where(gt == 1, 1, 0),
              cmap="PRGn", vmin=-1, vmax=2, aspect="auto")
    ax.set_title("Ground Truth (purple=1, blue=0)", fontsize=10)
    ax.set_xlabel("Column"); ax.set_ylabel("Row")
    ax.set_xticks(range(12)); ax.set_xticklabels(range(1, 13), fontsize=7)
    ax.set_yticks(range(8));  ax.set_yticklabels(list("ABCDEFGH"), fontsize=7)

    # Simulated photograph
    import cv2
    axes[1].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    axes[1].set_title("Simulated Plate Photograph", fontsize=10)
    axes[1].axis("off")

    # CV detection result
    detector.visualise(img, detected, title="CV Detection Result", ax=axes[2])

    plt.tight_layout()
    path = os.path.join(save_dir, f"plate_demo_seed{seed}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"\nSaved → {path}")
    plt.show()


# ──────────────────────────────────────────────────────────────────────
# CV accuracy test
# ──────────────────────────────────────────────────────────────────────

def cv_test(n_plates: int):
    accs, precs, recs = [], [], []

    for seed in range(n_plates):
        gt  = clustered_plate_labels(positive_fraction=0.25, seed=seed)
        sim = PlateSimulator(seed=seed)
        img = sim.generate_image(gt)
        det = PlateDetector(geometry=sim.get_geometry())
        lab = det.process(img)
        m   = det.accuracy(lab, gt)
        accs.append(m["accuracy"])
        precs.append(m["precision"])
        recs.append(m["recall"])

    print(f"\n[CV Accuracy Test – {n_plates} plates]")
    print(f"  Accuracy  : {np.mean(accs):.4f} ± {np.std(accs):.4f}")
    print(f"  Precision : {np.mean(precs):.4f} ± {np.std(precs):.4f}")
    print(f"  Recall    : {np.mean(recs):.4f} ± {np.std(recs):.4f}")


# ──────────────────────────────────────────────────────────────────────
# Full experiment
# ──────────────────────────────────────────────────────────────────────

def experiment(n_episodes: int, save_dir: str):
    os.makedirs(save_dir, exist_ok=True)

    print(f"\nRunning {n_episodes} episodes × {len(STRATEGIES)} strategies …")
    results = run_plate_experiment(n_episodes=n_episodes, verbose=True)

    print_summary(results)

    print("\nGenerating figures …")
    plot_plate_comparison(results,     save_dir=save_dir)
    plot_plate_learning_curves(results, save_dir=save_dir)

    # Belief evolution for two strategies using a representative episode
    for strategy in ["prob", "entropy"]:
        eps    = results[strategy]
        counts = np.array([ep["n_queries"] for ep in eps])
        med_ep = eps[int(np.argsort(counts)[len(counts) // 2])]
        # Re-run that episode to get oracle attached
        gt = clustered_plate_labels(seed=med_ep["seed"])
        ep = run_plate_episode(strategy, gt, seed=med_ep["seed"])
        plot_belief_evolution(ep, save_dir=save_dir)

    print(f"\nAll figures saved to '{save_dir}/'")


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Plate Active Learning")
    parser.add_argument("--mode",       choices=["demo", "experiment", "cv_test"],
                        default="demo")
    parser.add_argument("--seed",       type=int,   default=42)
    parser.add_argument("--n_episodes", type=int,   default=100)
    parser.add_argument("--n_plates",   type=int,   default=30)
    parser.add_argument("--out_dir",    default="plate_results")
    args = parser.parse_args()

    if args.mode == "demo":
        demo(args.seed, save_dir=args.out_dir)
    elif args.mode == "cv_test":
        cv_test(args.n_plates)
    elif args.mode == "experiment":
        experiment(args.n_episodes, save_dir=args.out_dir)


if __name__ == "__main__":
    main()
