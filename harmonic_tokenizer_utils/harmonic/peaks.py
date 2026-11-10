"""Peak picking for audio spectra."""

import numpy as np


def top_peaks(spectrum: np.ndarray, count: int = 5):
    order = np.argsort(-spectrum)[:count]
    return [(int(i), float(spectrum[i])) for i in order]
