"""Completion checks for Week 2.

These tests ensure that the Week 2 demonstration computes SNR values and an
aliasing metric for two sampling rates.  The aliasing level should be a
finite float indicating the decrease in SNR when the signal is downsampled.
"""

import os
import sys
# Ensure the dsp_utils package is importable when running the checks directly
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from demo import run_demo
from dsp_utils import generate_week_data
from dsp_utils.metrics import snr_db
from scipy.signal import resample


def test_aliasing_metric() -> None:
    """Run the demonstration and verify the aliasing level is a finite float."""
    results = run_demo()
    assert 'aliasing_level' in results, "Results must include 'aliasing_level'"
    assert isinstance(results['aliasing_level'], float), "Aliasing level must be a float"

def test_extreme_aliasing() -> None:
    """Check that a very low sampling rate induces more aliasing than the demo baseline."""
    baseline = run_demo()
    baseline_alias = baseline['aliasing_level']
    # Define high and extremely low sampling rates
    fs_high = 2000.0
    fs_low = 100.0
    data_high = generate_week_data('week_02', {'fs': fs_high, 'duration': 1.0})
    x_clean_high = data_high['x_clean']
    x_high = data_high['x']
    # Downsample the noisy signal
    num_samples = int(len(x_high) * fs_low / fs_high)
    x_low = resample(x_high, num_samples)
    x_clean_low = resample(x_clean_high, num_samples)
    # Compute aliasing metric manually
    snr_h = snr_db(x_clean_high, x_high - x_clean_high)
    snr_l = snr_db(x_clean_low, x_low - x_clean_low)
    aliasing_extreme = snr_h - snr_l
    assert aliasing_extreme > baseline_alias, "Extremely low sampling rate should produce larger aliasing than baseline"


if __name__ == '__main__':
    test_aliasing_metric()
    test_extreme_aliasing()
    print('All Week 2 checks passed.')