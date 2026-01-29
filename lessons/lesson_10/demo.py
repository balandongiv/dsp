"""Week 10 demonstration: time‑domain feature extraction.

This script computes basic statistical features—RMS, peak, crest factor,
kurtosis and skewness—for a synthetic vibration signal containing impulsive
events.  Learners can observe how these features capture energy and
impulsiveness in the signal.
"""

from typing import Dict

import numpy as np
from scipy.stats import kurtosis, skew

import os
import sys
# Append package root to sys.path so dsp_utils resolves during direct execution
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from dsp_utils import generate_week_data


def run_demo() -> Dict[str, float]:
    # Seed RNG for reproducible impulsive event generation
    import numpy as np
    np.random.seed(0)
    fs = 2000.0
    data = generate_week_data('week_10', {'fs': fs, 'duration': 5.0})
    x = data['x']
    # Compute features
    rms = np.sqrt(np.mean(x ** 2))
    peak = np.max(np.abs(x))
    crest_factor = peak / rms if rms > 0 else np.inf
    kurt = kurtosis(x, fisher=False)  # Use Pearson definition (3 for Gaussian)
    skewness = skew(x)
    print(f"RMS: {rms:.4f}")
    print(f"Peak: {peak:.4f}")
    print(f"Crest factor: {crest_factor:.2f}")
    print(f"Kurtosis: {kurt:.2f}")
    print(f"Skewness: {skewness:.2f}")
    return {
        'RMS': float(rms),
        'peak': float(peak),
        'crest_factor': float(crest_factor),
        'kurtosis': float(kurt),
        'skewness': float(skewness),
    }


if __name__ == '__main__':
    run_demo()