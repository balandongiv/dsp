## Week 6 – Power Spectral Density and Welch’s Method

This lesson introduces spectral estimation techniques, including the periodogram and Welch’s method.  You will compute power spectral density (PSD) for a synthetic gearbox vibration signal, tune the segment length and overlap and calculate bandpower in specific frequency bands.  The demonstration reports PSD variance and bandpower values.

### How to Run

```
python demo.py
```

### Expected Output

Running the demonstration prints the PSD variance and bandpower values for several frequency bands.  For example:

```
PSD variance: 2.2e-06
Bandpower 50–70 Hz: 0.32
Bandpower 140–160 Hz: 0.086
Bandpower 240–260 Hz: 0.053
```

Your numbers may differ but should be positive and show decreasing power at higher frequencies.

### Completion Criteria

1. Implement PSD estimation using periodogram or Welch’s method.
2. Adjust segment length and overlap to reduce variance while preserving resolution.
3. Compute bandpower in at least one frequency band and report the results.