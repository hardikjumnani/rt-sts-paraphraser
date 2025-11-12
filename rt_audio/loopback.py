from __future__ import annotations
import argparse
import time
import threading
from loguru import logger
import numpy as np

from rt_audio.ring import RingBuffer
from rt_audio.capture import MicSource
from rt_audio.playback import SpeakerSink
from tests.test_metrics import Metrics


def _monitor_thread_fn(metrics: Metrics, ring: RingBuffer, stop_event: threading.Event):
    """Purpose: emit one-line console status once per second until stopped."""
    while not stop_event.is_set():
        snap = metrics.maybe_emit(ring)
        if snap:
            # Compact single-line status
            ttfa = "NA" if snap.ttfa_ms is None else f"{snap.ttfa_ms:.1f}"
            p50 = "NA" if snap.lag_p50_ms is None else f"{snap.lag_p50_ms:.1f}"
            p95 = "NA" if snap.lag_p95_ms is None else f"{snap.lag_p95_ms:.1f}"
            logger.info(
                f"TTFA(ms)={ttfa} | Lag P50/P95(ms)={p50}/{p95} | Buf(ms)={snap.buf_ms:.1f} | "
                f"XRUNs U/O={snap.underruns}/{snap.overruns}"
            )
        stop_event.wait(1.0)


def main():
    """Purpose: run capture→ring→playback loopback with once-per-second metrics and CSV."""
    def _parse_device(value: str | None):
        if value is None:
            return None
        try:
            return int(value)
        except (TypeError, ValueError):
            return value

    parser = argparse.ArgumentParser()
    parser.add_argument("--sr", type=int, default=48000)
    parser.add_argument("--chunk-ms", type=int, default=20)
    parser.add_argument("--channels", type=int, default=1)
    parser.add_argument("--dtype", type=str, default="float32")
    parser.add_argument("--input-dev", type=_parse_device, default=None)
    parser.add_argument("--output-dev", type=_parse_device, default=None)
    parser.add_argument("--duration-s", type=int, default=60)
    parser.add_argument("--capacity-ms", type=int, default=1000, help="ring capacity in ms")
    args = parser.parse_args()

    sr = args.sr
    blocksize = int(sr * args.chunk_ms / 1000)
    capacity_frames = int(sr * (args.capacity_ms / 1000.0))
    ring = RingBuffer(capacity_frames)

    metrics = Metrics(sr=sr, csv_path="artifacts/loopback_metrics.csv")

    # Hook for per-chunk playback timing to compute rolling lag and TTFA
    def on_play(origin_ts_ns: int, playback_now_ns: int, ttfa_ms_first: float | None):
        if ttfa_ms_first is not None:
            metrics.set_ttfa_ms(ttfa_ms_first)
        metrics.record_play_event(origin_ts_ns, playback_now_ns)

    mic = MicSource(
        ring=ring,
        sr=sr,
        chunk_ms=args.chunk_ms,
        channels=args.channels,
        dtype=args.dtype,
        input_device=args.input_dev,
    )

    spk = SpeakerSink(
        ring=ring,
        sr=sr,
        chunk_ms=args.chunk_ms,
        channels=args.channels,
        dtype=args.dtype,
        output_device=args.output_dev,
        on_play=on_play,  # new optional hook
    )

    stop_event = threading.Event()
    mon = threading.Thread(target=_monitor_thread_fn, args=(metrics, ring, stop_event), daemon=True)

    try:
        mon.start()
        mic.start()
        spk.start()

        t0 = time.perf_counter()
        while time.perf_counter() - t0 < args.duration_s:
            time.sleep(0.05)
    except KeyboardInterrupt:
        logger.info("Interrupted by user.")
    finally:
        stop_event.set()
        spk.stop()
        mic.stop()
        mon.join(timeout=2.0)


if __name__ == "__main__":
    main()
