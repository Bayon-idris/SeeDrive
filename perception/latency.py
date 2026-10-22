from collections import deque
import threading


class LatencyStats:
    """Keeps the last N measurements (in ms) for each stage."""

    def __init__(self, maxlen=100):
        self.maxlen = maxlen
        self.values = {}
        self.lock = threading.Lock()

    def add(self, stage, ms):
        with self.lock:
            self.values.setdefault(stage, deque(maxlen=self.maxlen)).append(ms)

    def means(self):
        with self.lock:
            return {k: sum(v) / len(v) for k, v in self.values.items() if v}

