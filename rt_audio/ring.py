import numpy as np
from typing import Tuple, Optional
from loguru import logger


class RingBuffer:
    """
        A fixed-capacity single-producer/single-consumer ring buffer for PCM (Pulse Code Modulation)
        audio frames stored as float32 samples. Designed for real-time audio capture and playback,
        where small timing differences between input and output streams must be absorbed smoothly.
    """
    def __init__(self, capacity_frames: int):
        assert capacity_frames > 0
        self.capacity_frames = capacity_frames
        self.buffer = np.zeros(capacity_frames, dtype=np.float32)
        self.ts_ring = np.zeros(capacity_frames, dtype=np.int64)  # parallel timestamps
        self.write_idx = 0
        self.read_idx = 0
        self.size_frames = 0
        self.underruns = 0
        self.overruns = 0

    def push(self, frames: np.ndarray, ts_ns: int) -> int:
        """Push PCM frames into the ring buffer, tagging with timestamp."""
        n = len(frames)
        available = self.capacity_frames - self.size_frames
        if n > available:
            # Overrun case — accept only what fits
            n = available
            self.overruns += 1
            if n == 0:
                return 0

        end_idx = (self.write_idx + n) % self.capacity_frames

        if end_idx > self.write_idx:
            self.buffer[self.write_idx:end_idx] = frames[:n]
            self.ts_ring[self.write_idx:end_idx] = ts_ns
        else:
            # Wrap around
            split = self.capacity_frames - self.write_idx
            self.buffer[self.write_idx:] = frames[:split]
            self.buffer[:end_idx] = frames[split:n]
            self.ts_ring[self.write_idx:] = ts_ns
            self.ts_ring[:end_idx] = ts_ns

        self.write_idx = end_idx
        self.size_frames += n
        return n

    def pop(self, n_frames: int) -> Tuple[np.ndarray, Optional[int]]:
        """Pop up to n_frames from the buffer. Returns (frames, origin_ts_ns)."""
        if self.size_frames == 0:
            # Underrun case
            self.underruns += 1
            return np.zeros(n_frames, dtype=np.float32), None

        n = min(n_frames, self.size_frames)
        end_idx = (self.read_idx + n) % self.capacity_frames

        if end_idx > self.read_idx:
            chunk = self.buffer[self.read_idx:end_idx].copy()
            ts_ns = int(self.ts_ring[self.read_idx])
        else:
            # Wrap around
            chunk = np.concatenate(
                (self.buffer[self.read_idx:], self.buffer[:end_idx])
            )
            ts_ns = int(self.ts_ring[self.read_idx])

        self.read_idx = end_idx
        self.size_frames -= n
        return chunk, ts_ns
