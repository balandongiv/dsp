## Week 3 – FFT, Windowing and Spectral Leakage

This lesson explores the Fast Fourier Transform (FFT) as a black‑box tool for analysing the frequency content of signals.  You will compare different window functions and observe how spectral leakage spreads energy into neighbouring bins.  The demonstration computes a simple leakage ratio for a multi‑tone signal and reports it.

### How to Run

```
python demo.py
```

### Expected Output

The demonstration prints the leakage ratio for a Hann window and optionally other windows.  An example output is:

```
Leakage ratio with hann window: 0.187
```

Values will vary depending on the signal and window choice.  Lower values indicate less spectral leakage.

### Completion Criteria

1. Compute the FFT of a sampled signal and plot or inspect the magnitude spectrum.
2. Compare at least two window functions and quantify leakage (e.g., as the fraction of energy outside the main bins).
3. Report the leakage ratio and comment on how the window choice affects it.