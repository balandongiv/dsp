## Week 13 – Unsupervised Data Driven Signal Analysis and Anomaly Detection

This lesson covers unsupervised and one‑class methods for detecting anomalies in sensor streams.  You will train anomaly detectors on healthy segments only, compute anomaly scores on mixed healthy and faulty data, and select a threshold to meet a false‑alarm budget.  A simple regime‑aware strategy is demonstrated by including a regime feature.

### How to Run

```
python demo.py
```

### Expected Output

The demonstration prints the false‑alarm rates for the PCA‑based anomaly detector and the Isolation Forest.  Typical output:

```
PCA false‑alarm rate: 0.05
Isolation Forest false‑alarm rate: 0.05
```

These values correspond to a 5\% false‑alarm budget.  The numbers may vary slightly.

### Completion Criteria

1. Build a healthy baseline dataset and fit at least two unsupervised models (e.g., PCA and Isolation Forest).
2. Compute anomaly scores on a validation set containing anomalies and set a threshold to meet a false‑alarm target.
3. Implement a regime‑aware mitigation (e.g., adding a regime feature) and comment on improvements.