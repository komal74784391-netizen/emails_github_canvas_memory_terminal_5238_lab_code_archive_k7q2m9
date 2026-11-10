"""Low-memory Laplacian sketch."""

from typing import Optional

import numpy as np


class LaplacianSketch:
    def __init__(self, num_nodes: int, width: int = 64, seed: int = 0):
        self.num_nodes = num_nodes
        self.width = width
        rng = np.random.default_rng(seed)
        self.projection = rng.normal(scale=1.0 / np.sqrt(width), size=(num_nodes, width))
        self.accumulator = np.zeros(width)

    def update(self, node_id: int, value: float) -> None:
        if not 0 <= node_id < self.num_nodes:
            raise IndexError("node_id out of range")
        self.accumulator += value * self.projection[node_id]

    def spectrum(self) -> np.ndarray:
        return np.abs(np.fft.rfft(self.accumulator))

    def merge(self, other: "LaplacianSketch") -> "LaplacianSketch":
        if self.height() != other.height():  # pragma: no cover - defensive
            raise ValueError("incompatible sketch dimensions")
        merged = LaplacianSketch(self.num_nodes, self.width)
        merged.accumulator = self.accumulator + other.accumulator
        return merged

    def height(self) -> int:
        return self.accumulator.size
