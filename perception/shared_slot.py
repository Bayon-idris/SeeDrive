import threading

class SharedSlot:
    """Latest item written by one thread and read by another (protected by a lock)."""

    def __init__(self):
        self.lock = threading.Lock()
        self.data = None
        self.version = 0  # increases at each put(): lets the reader know if it's new

    def put(self, data):
        with self.lock:
            self.data = data
            self.version += 1

    def get(self):
        with self.lock:
            return self.data, self.version

