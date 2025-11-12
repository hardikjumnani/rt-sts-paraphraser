import sounddevice as sd
import time
from loguru import logger
import numpy as np


class SpeakerSink:
    """Purpose: output audio from ring buffer and optionally report per-chunk playback timing."""
    def __init__(self, ring, sr=48000, chunk_ms=20, channels=1, dtype='float32', output_device=None, on_play=None):
        """Purpose: configure output stream and optional on_play hook."""
        self.ring = ring
        self.sr = sr
        self.chunk_ms = chunk_ms
        self.channels = channels
        self.dtype = dtype
        self.output_device = output_device
        self.blocksize = int(sr * chunk_ms / 1000)
        self.stream = None
        self.ttfa_ms = None
        self._on_play = on_play  # callable(origin_ts_ns, playback_now_ns, ttfa_ms_first)

    def _callback(self, outdata, frames, time_info, status):
        """Purpose: fill output buffer from ring; compute TTFA and report lag samples."""
        if status:
            logger.warning(f"Output status: {status}")
        chunk, ts_ns = self.ring.pop(frames)
        if ts_ns is None:
            outdata[:] = np.zeros((frames, self.channels), dtype=self.dtype)
            return

        now_ns = time.perf_counter_ns()
        if self.ttfa_ms is None:
            self.ttfa_ms = (now_ns - ts_ns) / 1e6
            logger.info(f"TTFA (Time To First Audio): {self.ttfa_ms:.2f} ms")

        if self._on_play is not None:
            # Provide TTFA only on the first real chunk; None afterwards.
            ttfa_once = self.ttfa_ms if abs((now_ns - ts_ns) / 1e6 - self.ttfa_ms) < 1e-6 else None
            self._on_play(ts_ns, now_ns, ttfa_once)

        out_block = np.expand_dims(chunk, axis=1) if self.channels == 1 else chunk
        outdata[:] = out_block

    def start(self):
        """Purpose: start playback stream."""
        settings = sd.WasapiSettings(exclusive=False)
        self.stream = sd.OutputStream(
            device=self.output_device,
            samplerate=self.sr,
            channels=self.channels,
            dtype=self.dtype,
            blocksize=self.blocksize,
            callback=self._callback,
            extra_settings=settings,
            # latency='low',
        )
        self.stream.start()

    def stop(self):
        """Purpose: stop playback stream."""
        if self.stream:
            self.stream.stop()
            self.stream.close()
