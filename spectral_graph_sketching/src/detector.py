"""Streaming anomaly detector."""

from collections import deque
from typing import Deque

import numpy as np

from .sketch import LaplacianSketch


class StreamingAnomalyDetector:
    def __init__(self, num_nodes: int, window: int = 256, z_threshold: float = 3.0):
        self.window = window
        self.z_threshold = z_threshold
        self.sketch = LaplacianSketch(num_nodes)
        self.history: Deque[float] = deque(maxlen=window)

    def push(self, node_id: int, value: float) -> bool:
        self.sketch.update(node_id, value)
        energy = float(np.linalg.norm(self.sketch.spectrum()))
        self.history.append(energy)
        if len(self.history) < 16:
            return False
        baseline = float(np.mean(self.history))
        spread = float(np.std(self.history)) or 1e-6
        return (energy - baseline) / spread > self.z_threshold
