from __future__ import annotations
import os
import time
from collections import deque
from dataclasses import dataclass
from typing import Optional
import numpy as np


@dataclass
class Snapshot:
    """Purpose: carry one-second metrics snapshot for console/CSV emission."""
    wall_time_ns: int
    ttfa_ms: Optional[float]
    lag_p50_ms: Optional[float]
    lag_p95_ms: Optional[float]
    buf_ms: float
    underruns: int
    overruns: int


class Metrics:
    """Purpose: collect TTFA once, rolling lag percentiles, buffer occupancy, and XRUNs; emit every second."""

    def __init__(self, sr: int, csv_path: str):
        self.sr = sr
        self.csv_path = csv_path
        self._ensure_csv_header()
        self._lag_window = deque()  # (now_ns, lag_ms)
        self._window_ns = 5_000_000_000  # 5 seconds
        self._last_emit_sec = None
        self._ttfa_ms: Optional[float] = None

    def set_ttfa_ms(self, ttfa_ms: float) -> None:
        """Purpose: store TTFA once when first real chunk plays."""
        if self._ttfa_ms is None:
            self._ttfa_ms = float(ttfa_ms)

    def record_play_event(self, origin_ts_ns: int, playback_now_ns: int) -> None:
        """Purpose: record a single chunk’s lag sample for rolling stats."""
        lag_ms = (playback_now_ns - origin_ts_ns) / 1e6
        self._lag_window.append((playback_now_ns, lag_ms))
        cutoff = playback_now_ns - self._window_ns
        while self._lag_window and self._lag_window[0][0] < cutoff:
            self._lag_window.popleft()

    def _compute_percentiles(self) -> tuple[Optional[float], Optional[float]]:
        """Purpose: compute P50 and P95 over the 5s lag window."""
        if not self._lag_window:
            return None, None
        arr = np.array([x[1] for x in self._lag_window], dtype=np.float64)
        return float(np.percentile(arr, 50)), float(np.percentile(arr, 95))

    def maybe_emit(self, ring, now_ns: Optional[int] = None) -> Optional[Snapshot]:
        """Purpose: emit one snapshot per wall-clock second and write to CSV."""
        now_ns = now_ns or time.perf_counter_ns()
        now_sec = now_ns // 1_000_000_000
        if self._last_emit_sec is not None and now_sec == self._last_emit_sec:
            return None
        self._last_emit_sec = now_sec

        p50, p95 = self._compute_percentiles()
        buf_ms = (ring.size_frames / self.sr) * 1000.0
        snap = Snapshot(
            wall_time_ns=now_ns,
            ttfa_ms=self._ttfa_ms,
            lag_p50_ms=p50,
            lag_p95_ms=p95,
            buf_ms=buf_ms,
            underruns=ring.underruns,
            overruns=ring.overruns,
        )
        self._append_csv(snap)
        return snap

    def _ensure_csv_header(self) -> None:
        """Purpose: create CSV with header if it does not exist."""
        os.makedirs(os.path.dirname(self.csv_path), exist_ok=True)
        if not os.path.exists(self.csv_path):
            with open(self.csv_path, "w", encoding="utf-8") as f:
                f.write("wall_time_ns,ttfa_ms,lag_p50_ms,lag_p95_ms,buffer_ms,underruns,overruns\n")

    def _append_csv(self, snap: Snapshot) -> None:
        """Purpose: append one snapshot row to CSV."""
        def fmt(x):
            return "" if x is None else f"{x:.3f}" if isinstance(x, float) else str(x)

        with open(self.csv_path, "a", encoding="utf-8") as f:
            f.write(
                f"{snap.wall_time_ns},{fmt(snap.ttfa_ms)},{fmt(snap.lag_p50_ms)},{fmt(snap.lag_p95_ms)},"
                f"{snap.buf_ms:.3f},{snap.underruns},{snap.overruns}\n"
            )
