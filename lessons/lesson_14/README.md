## Week 14 – Deep Learning for Time‑Frequency Signal Representations

This capstone lesson builds an end‑to‑end pipeline that converts sensor data into a time‑frequency representation and feeds it into a lightweight neural classifier.  You will compute spectrogram features using STFT, pool or flatten them, train a multi‑layer perceptron (MLP) classifier, tune representation and model parameters and evaluate accuracy and false‑alarm rate under latency constraints.

### How to Run

```
python demo.py
```

### Expected Output

The script trains a multi‑layer perceptron on spectrogram features and prints the classification accuracy, false‑alarm rate and inference latency.  Example output:

```
MLP hidden layers (50, 50), accuracy: 0.86, false‑alarm rate: 0.33, latency: 0.9 ms
```

Latency may vary depending on hardware and input dimensionality.

### Completion Criteria

1. Construct a time‑frequency representation (e.g., log‑scaled STFT) for each segment.
2. Train a neural classifier (using `sklearn.neural_network.MLPClassifier`) and perform a small parameter sweep.
3. Report accuracy, false‑alarm rate and approximate per‑segment inference latency; provide a brief deployment plan (threshold, monitoring) in your report.