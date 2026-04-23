"""Thread-safe wrapper around a single ``cv2.VideoCapture`` device.

The dashboard's live-preview widget grabs frames from this object on every
Streamlit rerun. We keep the capture open across reruns (Streamlit's
``cache_resource`` gives us a single shared instance) because opening a
webcam on macOS takes ~1–2 s and would make the preview feel laggy.

Note on contention with ``hardware.battleship_ot2_loop``: webcams are
exclusive. While an experiment is running, the subprocess opens the same
device briefly for each step's photo. If this preview is also open, the
subprocess may fail a grab and retry. The UI therefore defaults the preview
OFF while a run is in progress, and surfaces a warning if enabled anyway.
"""

from __future__ import annotations

import threading
import time
from typing import Optional

import cv2
import numpy as np


class LivePreview:
    """Persistent, thread-safe handle to a single webcam."""

    def __init__(self, device_index: int = 0) -> None:
        self.device_index = device_index
        self._cap: Optional[cv2.VideoCapture] = None
        self._lock = threading.Lock()
        self._last_error: Optional[str] = None
        self._last_frame_ts: float = 0.0

    def open(self) -> bool:
        """Open the webcam if it isn't already. Returns True on success."""
        with self._lock:
            if self._cap is not None and self._cap.isOpened():
                return True
            cap = cv2.VideoCapture(self.device_index)
            if not cap.isOpened():
                try:
                    cap.release()
                except Exception:
                    pass
                self._cap = None
                self._last_error = (
                    f"Failed to open camera device {self.device_index}. "
                    "Another process may be using it."
                )
                return False
            self._cap = cap
            self._last_error = None
            return True

    def grab(self) -> Optional[np.ndarray]:
        """Return the most recent frame as an RGB uint8 array, or None."""
        with self._lock:
            if self._cap is None or not self._cap.isOpened():
                return None
            # Drain the driver buffer so we show a fresh frame rather than a
            # stale one from the last rerun. 3 reads is plenty on macOS.
            frame = None
            for _ in range(3):
                ret, f = self._cap.read()
                if ret and f is not None:
                    frame = f
            if frame is None:
                self._last_error = "Camera read returned no frame."
                return None
            self._last_error = None
            self._last_frame_ts = time.time()
            return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    def release(self) -> None:
        with self._lock:
            if self._cap is not None:
                try:
                    self._cap.release()
                except Exception:
                    pass
                self._cap = None

    def last_error(self) -> Optional[str]:
        return self._last_error

    def last_frame_ts(self) -> float:
        return self._last_frame_ts


__all__ = ["LivePreview"]
