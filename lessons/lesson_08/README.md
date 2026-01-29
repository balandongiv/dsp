## Week 8 – Wavelet Transform and Multiresolution Analysis

This lesson introduces continuous wavelet transforms (CWT) for analysing transient events with multi‑resolution capabilities.  You will apply Morlet and Ricker wavelets to a signal containing impulsive events and interpret the resulting scalograms.  The demonstration estimates an impulse detection rate based on CWT coefficients.

### How to Run

```
python demo.py
```

### Expected Output

The script detects impulsive events using the continuous wavelet transform and reports the detection rate and false‑alarm rate.  A typical run might print:

```
Detected impulses: 166
Detection rate (clipped): 1.00
False alarm rate: 32.2
```

Values depend on the chosen wavelet, width and threshold.

### Completion Criteria

1. Compute a CWT using at least one wavelet (e.g., Morlet or Ricker).
2. Visualise or inspect the scalogram and identify transient events.
3. Report an impulse detection rate and comment on false alarms.