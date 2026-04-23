"""Subprocess-based controller for the OT-2 hardware loop.

The loop is a blocking, monolithic pipeline with synchronous sleeps, camera
captures, and HTTP calls to the robot — an in-process stop flag cannot
interrupt those calls. We instead spawn ``hardware.battleship_ot2_loop`` as
its own process and use OS signals, so the dashboard's Stop button behaves
the same as a manual ``Ctrl-C`` on the CLI:

- **Pause**   → ``SIGSTOP`` on the process group (suspend)
- **Resume**  → ``SIGCONT``
- **Stop**    → ``SIGTERM`` with a grace period, then ``SIGKILL``
- **Reset**   → spawn ``battleship_ot2_loop reset --robot_ip ...``
                synchronously; no long-running state to track.

State is reconstructed by polling ``output_dir/checkpoint.json`` that the
loop writes after every step, plus a read of the latest image under
``output_dir/images/``. Ground truth and the belief probability map are
reproduced deterministically from the saved seed + history so the UI
doesn't depend on internal objects of the subprocess.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import threading
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional

import numpy as np

from config import BOARD_COLS, BOARD_ROWS, PLATE_COLS, PLATE_ROWS
from core.battleship_env import BattleshipBoard
from core.battleship_model import Game
from hardware.battleship_ot2_loop import DEFAULT_DECK, LoopConfig


REPO_ROOT = Path(__file__).resolve().parent.parent
_POSIX = os.name == "posix"


# ─────────────────────────────────────────────────────────────────────────
# Snapshot — pure-data view, safe to hand to the UI thread
# ─────────────────────────────────────────────────────────────────────────

@dataclass
class LoopSnapshot:
    phase: str = "idle"                 # idle / setup / loop / paused / stopped / done / error
    phase_message: str = ""
    step: int = 0
    total_ship_cells: int = 0
    ships_sunk: int = 0
    ships_total: int = 0
    hits: int = 0
    misses: int = 0
    last_step: Optional[Dict[str, Any]] = None
    ground_truth: Optional[List[List[int]]] = None
    results_matrix: Optional[List[List[int]]] = None
    prob_map: Optional[List[List[float]]] = None
    last_image_path: Optional[str] = None
    history: List[Dict[str, Any]] = field(default_factory=list)
    output_dir: Optional[str] = None


# ─────────────────────────────────────────────────────────────────────────
# Controller
# ─────────────────────────────────────────────────────────────────────────

class OT2Controller:
    """One controller instance per Streamlit session."""

    MAX_LOG_LINES = 4000
    STOP_GRACE_SECONDS = 5.0

    def __init__(self) -> None:
        self._proc: Optional[subprocess.Popen] = None
        self._cfg: Optional[LoopConfig] = None
        self._log_buffer: Deque[str] = deque(maxlen=self.MAX_LOG_LINES)
        self._reader_thread: Optional[threading.Thread] = None
        self._paused: bool = False
        self._stopped: bool = False  # sticky: set when stop() is invoked
        self._last_reset_output: Optional[str] = None

    # ── Lifecycle ───────────────────────────────────────────────────────

    def is_running(self) -> bool:
        p = self._proc
        return p is not None and p.poll() is None

    def is_paused(self) -> bool:
        return self.is_running() and self._paused

    def start(self, cfg: LoopConfig) -> None:
        if self.is_running():
            raise RuntimeError("An OT-2 run is already in progress.")

        self._cfg = cfg
        self._paused = False
        self._stopped = False
        self._log_buffer.clear()
        self._last_reset_output = None

        out_dir = Path(cfg.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        # If deck overrides were supplied, serialise them next to the run so
        # the subprocess picks them up via --deck_path. We write the merged
        # deck (defaults + overrides) so the file is self-contained.
        deck_path_arg: Optional[str] = cfg.deck_path
        if cfg.deck_overrides:
            merged = {**DEFAULT_DECK, **cfg.deck_overrides}
            deck_file = out_dir / "deck.json"
            deck_file.write_text(json.dumps(merged, indent=2), encoding="utf-8")
            deck_path_arg = str(deck_file)

        args = self._build_args(cfg, deck_path_override=deck_path_arg)
        self._log_buffer.append(f"$ {' '.join(args)}")

        popen_kwargs: Dict[str, Any] = dict(
            cwd=str(REPO_ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=1,
            text=True,
        )
        # Put the child in its own process group so SIGTERM / SIGSTOP target
        # the whole subprocess tree, not this Streamlit server.
        if _POSIX:
            popen_kwargs["start_new_session"] = True

        self._proc = subprocess.Popen(args, **popen_kwargs)
        self._reader_thread = threading.Thread(
            target=self._read_stdout, name="ot2-stdout", daemon=True,
        )
        self._reader_thread.start()

    def pause(self) -> None:
        if not self.is_running():
            return
        if not _POSIX:
            self._log_buffer.append("[controller] Pause is only supported on POSIX systems.")
            return
        try:
            os.killpg(os.getpgid(self._proc.pid), signal.SIGSTOP)
            self._paused = True
            self._log_buffer.append("[controller] SIGSTOP sent — process suspended.")
        except ProcessLookupError:
            pass
        except Exception as e:  # pragma: no cover
            self._log_buffer.append(f"[controller] pause failed: {e}")

    def resume(self) -> None:
        if not self.is_running():
            return
        if not _POSIX:
            return
        try:
            os.killpg(os.getpgid(self._proc.pid), signal.SIGCONT)
            self._paused = False
            self._log_buffer.append("[controller] SIGCONT sent — process resumed.")
        except ProcessLookupError:
            pass
        except Exception as e:  # pragma: no cover
            self._log_buffer.append(f"[controller] resume failed: {e}")

    def stop(self, timeout: Optional[float] = None) -> None:
        """Terminate the subprocess (SIGTERM, then SIGKILL after timeout)."""
        if not self.is_running():
            return
        grace = timeout if timeout is not None else self.STOP_GRACE_SECONDS
        self._stopped = True

        # A paused process can't observe SIGTERM — un-pause first.
        if self._paused:
            self.resume()

        try:
            if _POSIX:
                os.killpg(os.getpgid(self._proc.pid), signal.SIGTERM)
            else:
                self._proc.terminate()
            self._log_buffer.append("[controller] SIGTERM sent — waiting for exit.")
        except ProcessLookupError:
            pass
        except Exception as e:  # pragma: no cover
            self._log_buffer.append(f"[controller] terminate failed: {e}")

        def _escalate() -> None:
            try:
                self._proc.wait(timeout=grace)
            except subprocess.TimeoutExpired:
                try:
                    if _POSIX:
                        os.killpg(os.getpgid(self._proc.pid), signal.SIGKILL)
                    else:
                        self._proc.kill()
                    self._log_buffer.append(
                        "[controller] SIGKILL sent — process forced to exit."
                    )
                except Exception as e:  # pragma: no cover
                    self._log_buffer.append(f"[controller] kill failed: {e}")

        threading.Thread(target=_escalate, name="ot2-kill", daemon=True).start()

    def reset_robot(self, robot_ip: str, timeout: float = 90.0) -> Dict[str, Any]:
        """Run ``battleship_ot2_loop reset`` synchronously.

        Returns a dict with ``returncode``, ``stdout``, ``stderr``. Safe to
        call while idle; refuses while a run is in progress.
        """
        if self.is_running():
            raise RuntimeError("Stop the current run before resetting the robot.")

        args = [
            sys.executable, "-u", "-m", "hardware.battleship_ot2_loop",
            "reset", "--robot_ip", robot_ip,
        ]
        try:
            proc = subprocess.run(
                args,
                cwd=str(REPO_ROOT),
                capture_output=True,
                text=True,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired as exc:
            self._last_reset_output = f"TIMEOUT after {timeout:.0f}s"
            return {"returncode": -1, "stdout": exc.stdout or "", "stderr": "timeout"}

        output = proc.stdout + (proc.stderr or "")
        self._last_reset_output = output.strip()
        # Also surface reset output in the main log tail so the user sees it.
        for line in output.splitlines():
            self._log_buffer.append(f"[reset] {line}")
        return {
            "returncode": proc.returncode,
            "stdout": proc.stdout,
            "stderr": proc.stderr,
        }

    def last_reset_output(self) -> Optional[str]:
        return self._last_reset_output

    # ── Observers ───────────────────────────────────────────────────────

    def snapshot(self) -> LoopSnapshot:
        cfg = self._cfg
        if cfg is None:
            return LoopSnapshot()

        snap = LoopSnapshot(output_dir=cfg.output_dir)

        # Phase — inferred from process + checkpoint state
        running = self.is_running()
        out_dir = Path(cfg.output_dir)
        cp_path = out_dir / "checkpoint.json"
        if running and self._paused:
            snap.phase = "paused"
            snap.phase_message = "Process suspended (SIGSTOP)"
        elif running:
            if cp_path.exists():
                snap.phase = "loop"
                snap.phase_message = f"strategy={cfg.strategy}"
            else:
                snap.phase = "setup"
                snap.phase_message = "Dispensing NaOH / H₂O on the plate"
        else:
            rc = self._proc.returncode if self._proc else None
            if self._stopped:
                snap.phase = "stopped"
                snap.phase_message = f"Stopped by operator (rc={rc})"
            elif rc in (0, None):
                snap.phase = "done"
                snap.phase_message = "Run finished"
            else:
                snap.phase = "error"
                snap.phase_message = f"Subprocess exited with rc={rc}"

        # Load checkpoint (may not exist yet during setup)
        cp = self._read_checkpoint(cp_path)
        if cp is not None:
            history = cp.get("history", [])
            snap.step = int(cp.get("step", len(history)))
            snap.results_matrix = cp.get("results_matrix")
            snap.history = history
            snap.hits = sum(1 for r in history if r.get("is_hit"))
            snap.misses = sum(1 for r in history if not r.get("is_hit") and r.get("label"))
            if history:
                snap.last_step = history[-1]
                img = history[-1].get("image_path")
                if img:
                    img_p = Path(img)
                    if not img_p.is_absolute():
                        img_p = REPO_ROOT / img_p
                    if img_p.exists():
                        snap.last_image_path = str(img_p)

            seed = cp.get("seed", cfg.seed)
            gt, sunk, prob = self._replay_ground_truth(seed, history)
            if gt is not None:
                snap.ground_truth = gt.tolist()
                snap.total_ship_cells = int(gt.sum())
                snap.ships_total = 5  # default fleet
                snap.ships_sunk = sunk
                snap.prob_map = prob

        # Fallback for the latest image during setup (no checkpoint yet)
        if snap.last_image_path is None:
            img_dir = out_dir / "images"
            if img_dir.exists():
                imgs = sorted(img_dir.glob("*.*"))
                if imgs:
                    snap.last_image_path = str(imgs[-1])

        return snap

    def logs(self, tail: int = 500) -> List[str]:
        return list(self._log_buffer)[-tail:]

    # ── Internals ───────────────────────────────────────────────────────

    @staticmethod
    def _build_args(cfg: LoopConfig, deck_path_override: Optional[str] = None) -> List[str]:
        args = [
            sys.executable, "-u", "-m", "hardware.battleship_ot2_loop",
            "--strategy", cfg.strategy,
            "--output_dir", cfg.output_dir,
            "--robot_ip", cfg.robot_ip,
            "--color_develop_seconds", str(cfg.color_develop_seconds),
        ]
        if cfg.seed is not None:
            args += ["--seed", str(cfg.seed)]
        if cfg.geometry_path:
            args += ["--geometry_path", cfg.geometry_path]
        if cfg.dry_run:
            args += ["--dry_run"]
        if cfg.skip_setup:
            args += ["--skip_setup"]
        if cfg.checkpoint_path:
            args += ["--resume", cfg.checkpoint_path]
        deck_path = deck_path_override or cfg.deck_path
        if deck_path:
            args += ["--deck_path", deck_path]
        return args

    def _read_stdout(self) -> None:
        proc = self._proc
        if proc is None or proc.stdout is None:
            return
        try:
            for line in proc.stdout:
                self._log_buffer.append(line.rstrip("\n"))
        except Exception:  # pragma: no cover
            pass
        finally:
            rc = proc.wait() if proc else None
            self._log_buffer.append(f"[controller] Process exited with rc={rc}")

    @staticmethod
    def _read_checkpoint(path: Path) -> Optional[Dict[str, Any]]:
        if not path.exists():
            return None
        try:
            with open(path) as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            # The subprocess may be mid-write. Skip this snapshot.
            return None

    @staticmethod
    def _replay_ground_truth(
        seed: Optional[int], history: List[Dict[str, Any]],
    ) -> tuple[Optional[np.ndarray], int, Optional[List[List[float]]]]:
        """Rebuild ground-truth grid and belief prob_map from seed + history."""
        try:
            board = BattleshipBoard(rows=BOARD_ROWS, cols=BOARD_COLS, seed=seed)
        except Exception:
            return None, 0, None

        model = Game(board_rows=BOARD_ROWS, board_cols=BOARD_COLS)
        for r in history:
            try:
                row = int(r["row"])
                col = int(r["col"])
                _, sunk = board.query(row, col)
                model.update(row, col, is_hit=bool(r["is_hit"]), sunk_ship=sunk)
            except Exception:
                continue

        prob_full: Optional[List[List[float]]] = None
        try:
            p = np.asarray(model.prob_map, dtype=float)
            full = np.zeros((PLATE_ROWS, PLATE_COLS), dtype=float)
            full[: p.shape[0], : p.shape[1]] = p
            prob_full = full.tolist()
        except Exception:
            pass

        return board.grid, len(board.get_sunk_ships()), prob_full


__all__ = ["OT2Controller", "LoopSnapshot", "LoopConfig"]
