"""Week 1 demonstration: sampling and quantisation effects.

This script synthesises a multi‑tone signal with a transient event and Gaussian noise,
then computes the signal‑to‑noise ratio for a sweep of sampling rates.  It serves
as a minimal example of how sampling frequency influences the quality of the
digitised measurement.  Learners are encouraged to modify the `FS_VALUES` list
to explore additional rates or bit‑depth proxies.
"""

from typing import List, Dict

import os
import sys
# Ensure imports work when running this script directly by adding the package root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from dsp_utils import generate_week_data, snr_db

# Sampling rates to compare (in Hz)
FS_VALUES: List[int] = [1000, 5000]


def run_demo() -> List[Dict[str, float]]:
    # Use a fixed seed for reproducibility of synthetic data and noise
    import numpy as np
    np.random.seed(0)
    """Run the Week 1 demonstration.

    Returns a list of dictionaries containing the sampling rate and the
    computed signal‑to‑noise ratio in dB for each experiment.
    """
    results = []
    for fs in FS_VALUES:
        # Generate synthetic data for week 1 at the specified sampling rate
        data = generate_week_data('week_01', {'fs': fs, 'duration': 2.0})
        x_clean = data['x_clean']
        x = data['x']
        # Compute noise by subtracting the clean signal from the noisy measurement
        noise = x - x_clean
        snr = snr_db(x_clean, noise)
        print(f"Sampling rate {fs} Hz → SNR = {snr:.2f} dB")
        results.append({'fs': fs, 'snr_db': snr})
    return results


if __name__ == '__main__':
    run_demo()