from pathlib import Path
import sys

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rt_audio.ring import RingBuffer


def test_basic_push_pop():
    """Test basic push and pop without wraparound."""
    rb = RingBuffer(8)
    frames = np.arange(4, dtype=np.float32)
    ts = 1000

    accepted = rb.push(frames, ts)
    assert accepted == 4
    assert rb.size_frames == 4

    out, ts_out = rb.pop(4)
    np.testing.assert_array_equal(out, frames)
    assert ts_out == ts
    assert rb.size_frames == 0


def test_wraparound_push_pop():
    """Test wraparound behavior when buffer indices roll over."""
    rb = RingBuffer(8)

    # Fill up partially, then pop to shift the read index
    rb.push(np.arange(6, dtype=np.float32), 111)
    rb.pop(4)  # advance read index to 4
    assert rb.read_idx == 4
    assert rb.write_idx == 6

    # Push 4 more frames, which should wrap around
    rb.push(np.arange(4, dtype=np.float32), 222)

    # Now pop all remaining frames and verify continuity
    out, _ = rb.pop(6)
    assert len(out) == 6
    assert rb.size_frames == 0


def test_overrun_handling():
    """Test that overrun counter increases when buffer is full."""
    rb = RingBuffer(8)
    data1 = np.arange(8, dtype=np.float32)
    data2 = np.arange(4, dtype=np.float32)

    rb.push(data1, 111)
    assert rb.size_frames == 8

    # This should trigger an overrun
    accepted = rb.push(data2, 222)
    assert accepted == 0
    assert rb.overruns == 1


def test_underrun_handling():
    """Test that underrun counter increases when popping empty buffer."""
    rb = RingBuffer(8)

    # Popping when empty should produce zeros
    out, ts = rb.pop(4)
    assert np.allclose(out, np.zeros(4, dtype=np.float32))
    assert ts is None
    assert rb.underruns == 1


def test_partial_overrun_accepts_available_space():
    """Test partial acceptance when not all frames fit."""
    rb = RingBuffer(8)
    rb.push(np.arange(6, dtype=np.float32), 100)
    accepted = rb.push(np.arange(4, dtype=np.float32), 200)
    # Only 2 frames can fit
    assert accepted == 2
    assert rb.overruns == 1
    assert rb.size_frames == 8


def test_multiple_wraparound_cycles():
    """Test several push-pop cycles across wraparounds."""
    rb = RingBuffer(8)
    ts = 1234

    for i in range(3):
        frames = np.ones(4, dtype=np.float32) * i
        rb.push(frames, ts)
        out, _ = rb.pop(4)
        np.testing.assert_array_equal(out, frames)

    assert rb.underruns == 0
    assert rb.overruns == 0
