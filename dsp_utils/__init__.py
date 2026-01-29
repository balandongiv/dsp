"""DSP Utilities
This package contains helper functions for generating synthetic datasets,
computing common metrics (SNR, RMSE, bandpower), and building basic DSP
demonstrations.  Functions here are intentionally simple and free of
deep mathematical derivations so that learners can focus on the
operational aspects of each method.
"""

from .data_gen import (
    generate_week_data,
    generate_sine_wave,
    add_gaussian_noise,
)
from .metrics import (
    snr_db,
    rmse,
    bandpower,
)

__all__ = [
    "generate_week_data",
    "generate_sine_wave",
    "add_gaussian_noise",
    "snr_db",
    "rmse",
    "bandpower",
]