## Week 11 – Spectral Features From Frequency Domain

This lesson introduces frequency‑domain feature extraction techniques such as bandpower and envelope detection.  You will compute bandpower for a synthetic bearing signal, perform envelope detection to demodulate high‑frequency vibrations and identify characteristic defect frequencies.

### How to Run

```
python demo.py
```

### Expected Output

The demonstration prints bandpower values in selected bands and the amplitude of the defect frequency in the demodulated envelope.  Example output:

```
Bandpower 40–60 Hz: 0.0000e+00
Envelope defect frequency amplitude (50 Hz): 2.41e+03
```

Bandpower may be zero if the band contains no power.  The envelope amplitude indicates the strength of the defect component.

### Completion Criteria

1. Compute bandpower in at least one frequency band using PSD integration.
2. Implement envelope detection (band‑pass → rectification → low‑pass) and compute the spectrum of the envelope.
3. Report the amplitude of defect frequencies in the envelope spectrum and discuss parameter choices.