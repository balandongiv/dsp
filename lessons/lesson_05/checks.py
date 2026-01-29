"""Completion checks for Week 5.

The check ensures that the Chebyshev IIR filter is stable and that the
demonstration reports SNR values before and after filtering.  It also checks
that the SNR does not degrade significantly when the filter is applied.
"""

import os
import sys
# Ensure dsp_utils is importable for direct execution
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from demo import run_demo
from dsp_utils import generate_week_data
import numpy as np
from scipy.signal import cheby1, tf2zpk


def test_iir_filter() -> None:
    """Verify that the Chebyshev IIR filter in the demo is stable and improves SNR."""
    results = run_demo()
    assert results['stable'] is True, "Designed IIR filter must be stable (all poles inside unit circle)"
    assert results['snr_after'] >= results['snr_before'] - 1e-3, \
        "Filtering should not worsen the SNR"

def test_high_order_unstable() -> None:
    """Design a high-order, high-ripple Chebyshev filter that should be unstable."""
    fs = 100.0
    order = 10
    ripple = 10.0
    nyq = fs / 2.0
    wp = [3.0 / nyq, 7.0 / nyq]
    b, a = cheby1(order, ripple, wp, btype='bandpass')
    _, p, _ = tf2zpk(b, a)
    # At least one pole magnitude should be ≥1.0 for this aggressive design
    assert np.any(np.abs(p) >= 1.0), \
        "A high-order Chebyshev filter with large ripple should have poles at or outside the unit circle"


if __name__ == '__main__':
    test_iir_filter()
    test_high_order_unstable()
    print('All Week 5 checks passed.')