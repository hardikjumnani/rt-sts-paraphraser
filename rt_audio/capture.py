import sounddevice as sd
import time
from loguru import logger


class MicSource:
    """Callback-based microphone audio source for real-time capture."""

    def __init__(self, ring, sr=48000, chunk_ms=20, channels=1, dtype='float32', input_device=None):
        """Initialize the microphone input stream."""
        self.ring = ring
        self.sr = sr
        self.chunk_ms = chunk_ms
        self.channels = channels
        self.dtype = dtype
        self.input_device = input_device
        self.blocksize = int(sr * chunk_ms / 1000)
        self.stream = None

    def _callback(self, indata, frames, time_info, status):
        """Audio callback executed whenever a new audio block is available."""
        if status:
            logger.warning(f"Input status: {status}")
        ts_ns = time.perf_counter_ns()
        frames_1d = indata[:, 0].copy() if self.channels > 1 else indata.copy().flatten()
        self.ring.push(frames_1d, ts_ns)

    def start(self):
        """Start capturing from the microphone."""
        logger.info("Starting microphone capture stream...")
        settings = sd.WasapiSettings(exclusive=False)
        self.stream = sd.InputStream(
            device=self.input_device,
            samplerate=self.sr,
            channels=self.channels,
            dtype=self.dtype,
            blocksize=self.blocksize,
            callback=self._callback,
            extra_settings=settings,
            latency='high',
        )
        self.stream.start()
        logger.info("Microphone stream started.")

    def stop(self):
        """Stop capturing from the microphone."""
        if self.stream:
            self.stream.stop()
            self.stream.close()
            logger.info("Microphone stream stopped.")
