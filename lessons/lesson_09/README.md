## Week 9 – Cross‑Correlation and Coherence

This lesson introduces multichannel analysis tools such as cross‑correlation and coherence to detect coupling between sensor signals.  You will compute the cross‑correlation of two signals to estimate time delays, and the magnitude‑squared coherence to quantify frequency‑dependent relationships.

### How to Run

```
python demo.py
```

### Expected Output

Running the demonstration prints the peak lag of the cross‑correlation and the maximum coherence value.  For example:

```
Peak cross‑correlation lag: 0.000000 s
Maximum coherence: 1.000
```

These indicate the two signals are synchronised and perfectly coherent at the dominant frequency in the synthetic dataset.

### Completion Criteria

1. Compute and plot (or print) the cross‑correlation function of two signals and identify the lag at which it peaks.
2. Compute magnitude‑squared coherence using SciPy and identify frequency bands with high coherence.
3. Report the peak lag and maximum coherence value.