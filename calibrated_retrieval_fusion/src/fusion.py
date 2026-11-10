"""Calibrated retrieval fusion utilities."""

from dataclasses import dataclass

import numpy as np


@dataclass
class FusionWeights:
    dense: float
    sparse: float
    temperature: float = 1.0


def fuse_scores(dense: np.ndarray, sparse: np.ndarray, weights: FusionWeights) -> np.ndarray:
    """Combine dense and sparse retrieval scores with calibrated gating."""
    dense = np.asarray(dense, dtype=float)
    sparse = np.asarray(sparse, dtype=float)
    logits = weights.dense * dense + weights.sparse * sparse
    logits = logits / max(weights.temperature, 1e-6)
    exp = np.exp(logits - logits.max())
    return exp / exp.sum()


def select_evidence(scores: np.ndarray, top_k: int = 3) -> list:
    order = np.argsort(-np.asarray(scores))[:top_k]
    return [int(i) for i in order]
