# Gravitational-Wave-Noise-Filter
Deep learning denoiser for LIGO gravitational-wave strain data. A 1-D Conv autoencoder trained to suppress non-Gaussian noise artefacts and boost matched-filter SNR on real GWTC events : before the universe's loudest signals get lost in the noise. PyTorch · SciPy · GWpy.

# 02 · Gravitational Wave Noise Filter

> **Tags:** Deep Learning · Signal Processing · Astrophysics  
> **Level:** Intermediate  
> **Verification targets:** SNR ≥ +3 dB · Reconstruction MSE < 0.05 (normalised) · Noise Power Reduction ≥ 40 %

---

## Table of Contents
1. [Background](#background)
2. [Problem Statement](#problem-statement)
3. [Dataset Guide](#dataset-guide)
4. [Architecture](#architecture)
5. [Repository Structure](#repository-structure)
6. [Deliverables](#deliverables)
7. [Evaluation Methodology](#evaluation-methodology)
8. [Quick Start](#quick-start)
9. [Verification Metrics](#verification-metrics)
10. [Tech Stack](#tech-stack)
11. [References](#references)

---

## Background

LIGO's interferometers measure mirror displacements **a thousand times smaller than a proton**, making them among the most sensitive instruments ever built. That extreme sensitivity comes at a cost: the same apparatus continuously picks up seismic, thermal, and electronic noise. Separating genuine astrophysical signals from detector artefacts is one of the hardest open problems in observational physics.

This project trains a **1-D Convolutional Autoencoder** on real LIGO strain data to:
- Suppress non-Gaussian noise artefacts
- Measurably improve SNR before matched-filter searches run on the cleaned signal

---

## Problem Statement

Build a deep-learning denoising pipeline for LIGO gravitational-wave strain data. Train a 1-D convolutional autoencoder to reconstruct clean strain signals from noisy input windows, and demonstrate SNR improvement on held-out events from the **GWTC-3 / GWTC-5 catalogue**.

---

## Dataset Guide

### Source
**LIGO Open Science Center (GWOSC) — GWTC Strain Data**  
<https://gwosc.org/eventapi/html/GWTC/>

The included `data/events.csv` contains **392 confirmed gravitational-wave events** from GWTC-1 through GWTC-5.0, including:

| Column | Description |
|---|---|
| `name` | Event identifier (e.g. `GW150914`) |
| `gps` | GPS time of the event |
| `catalog` | Source catalogue version |
| `network_matched_filter_snr` | Network SNR from matched-filter search |
| `mass_1_source`, `mass_2_source` | Component masses in M☉ |
| `luminosity_distance` | Distance in Mpc |
| `chirp_mass_source` | Chirp mass in M☉ |
| `chi_eff` | Effective spin parameter |
| `p_astro` | Probability of astrophysical origin |

### Downloading Strain Data
```python
from gwosc.datasets import event_gps
from gwpy.timeseries import TimeSeries

gps   = event_gps("GW150914")
data  = TimeSeries.fetch_open_data("H1", gps - 16, gps + 16, cache=True)
strain = data.value          # numpy array, 4096 Hz
```

### Held-Out Test Events (5 events)
Events selected for evaluation (highest-SNR, fully characterised):

| Event | Catalogue SNR | M₁ (M☉) | M₂ (M☉) | Distance (Mpc) |
|---|---|---|---|---|
| GW250114_082203 | 78.6 | 33.8 | 32.3 | 405 |
| GW241225_082815 | 19.5 | 55.7 | 42.2 | 1880 |
| GW250119_190238 | 21.3 | 11.5 | 10.0 | 311 |
| GW241231_054133 | 17.4 | 12.4 | 7.1  | 910 |
| GW241225_042553 | 16.5 | 12.4 | 8.1  | 660 |

---

## Architecture

```
INPUT  h(t) noisy strain segment  [1 × 4096]
│
│  ┌─────────────────── ENCODER ──────────────────────┐
│  │                                                   │
│  │  Conv1d(1→32,  k=9) → BN → LeakyReLU → MaxPool  │  L → L/2
│  │  Conv1d(32→64, k=7) → BN → LeakyReLU → MaxPool  │  L/2 → L/4
│  │  Conv1d(64→128,k=5) → BN → LeakyReLU → MaxPool  │  L/4 → L/8
│  │                                                   │
│  │  ┌─── BOTTLENECK ───┐                            │
│  │  │  Conv1d(128→64)  │                            │
│  │  │  BN → Tanh       │  ← latent representation  │
│  │  └──────────────────┘                            │
│  └───────────────────────────────────────────────────┘
│
│  ┌─────────────────── DECODER ──────────────────────┐
│  │                                                   │
│  │  Upsample(×2) → Conv1d(64→128, k=5) → BN → LReLU│  L/8 → L/4
│  │  Upsample(×2) → Conv1d(128→64, k=7) → BN → LReLU│  L/4 → L/2
│  │  Upsample(×2) → Conv1d(64→32,  k=9) → BN → LReLU│  L/2 → L
│  │  Conv1d(32→1, k=1)                               │  output head
│  └───────────────────────────────────────────────────┘
│
OUTPUT ĥ(t) reconstructed strain  [1 × 4096]
```

**Parameters:** ~1.2 M trainable  
**Loss:** `α·MSE(time) + β·MSE(|FFT|)` where α=0.7, β=0.3  
**Optimiser:** Adam + OneCycleLR (max_lr = 3×10⁻³)

---

## Repository Structure

```
gw_noise_filter/
├── data/
│   └── events.csv              ← GWTC catalogue (392 events, from GWOSC)
│
├── src/
│   ├── preprocessing.py        ← Deliverable 1: bandpass, whiten, window
│   ├── autoencoder.py          ← Deliverable 2: model + training loop
│   ├── snr_evaluation.py       ← Deliverable 3: SNR comparison plots
│   └── spectrogram_viz.py      ← Deliverable 4: spectrogram visualisations
│
├── outputs/
│   ├── gw_autoencoder.pt       ← trained model weights (after training)
│   ├── training_history.json   ← per-epoch train/val loss
│   ├── snr_results.json        ← SNR results per held-out event
│   ├── snr_comparison.png      ← Deliverable 3 plot
│   └── spectrograms.png        ← Deliverable 4 plot
│
└── README.md                   ← this file (Deliverable 5)
```

---

## Deliverables

### ✅ Deliverable 1 — Preprocessing Pipeline (`src/preprocessing.py`)

Three-stage pipeline applied to every raw strain segment:

| Stage | Method | Purpose |
|---|---|---|
| **Bandpass** | Butterworth order-8 (20–2048 Hz) | Remove low-freq seismic & high-freq aliasing |
| **Whitening** | Welch PSD → divide FFT bins by √PSD | Flatten noise floor to unit variance |
| **Windowing** | Hann window, L=4096, 50% overlap | Reduce spectral leakage between segments |

```python
from src.preprocessing import preprocess, generate_synthetic_strain

noisy, clean = generate_synthetic_strain(duration=32.0, snr=5.0)
segments, meta = preprocess(noisy)
# segments.shape → (n_windows, 4096)
```

---

### ✅ Deliverable 2 — 1-D Conv Autoencoder (`src/autoencoder.py`)

```python
from src.autoencoder import GWAutoencoder, build_dataset, train

model   = GWAutoencoder(input_len=4096, base_channels=32, latent_channels=64)
dataset = build_dataset(n_events=500, snr_range=(3, 20))
history = train(model, dataset, epochs=50, batch_size=64, lr=3e-4)
# history keys: "train_loss", "val_loss"
```

The training loop saves:
- Best model checkpoint → `outputs/gw_autoencoder.pt`
- Loss curve data → `outputs/training_history.json`

---

### ✅ Deliverable 3 — SNR Comparison Plots (`src/snr_evaluation.py`)

```python
from src.snr_evaluation import load_gwtc_events, evaluate_events, plot_snr_comparison

events  = load_gwtc_events("data/events.csv", min_snr=9.0, n_holdout=5)
results = evaluate_events(events, model=model)
plot_snr_comparison(results, "outputs/snr_comparison.png")
```

Output: grouped bar chart + delta-SNR panel for each held-out event.

---

### ✅ Deliverable 4 — Spectrogram Visualisations (`src/spectrogram_viz.py`)

```python
from src.spectrogram_viz import plot_spectrograms

plot_spectrograms(noisy_raw, clean_raw, recon_raw,
                  event_name="GW250114_082203")
```

Four-panel layout:
1. Raw noisy spectrogram  
2. Bandpass + whitened spectrogram  
3. Autoencoder-reconstructed spectrogram  
4. Residual noise spectrogram  
Plus: PSD comparison overlay + noise power reduction by frequency band.

---

## Evaluation Methodology

### SNR (Signal-to-Noise Ratio)

```
SNR [dB] = 10 · log₁₀ ( P_signal / P_noise )

where:
  P_signal = mean( clean(t)² )
  P_noise  = mean( (noisy(t) − clean(t))² )
```

**Improvement criterion:** `SNR_reconstructed − SNR_noisy ≥ 3 dB`

### Reconstruction MSE (Normalised)

```
MSE_norm = MSE(ĥ, h_clean) / MSE(h_noisy, h_clean)
```

**Target:** MSE_norm < 0.05 (the autoencoder recovers ≥ 95% of the noise contribution)

### Noise Power Reduction

Computed per frequency band (20–60, 60–150, 150–350, 350–750, 750–2048 Hz):

```
NPR_band [dB] = mean_PSD_noisy_band − mean_PSD_recon_band
NPR_pct       = (1 − 10^(−NPR_dB / 10)) × 100 %
```

**Target:** NPR_pct ≥ 40 % averaged across all bands.

### Matched-Filter SNR

The cleaned output is also tested by re-running a simplified matched-filter
search using cross-correlation against the CBC template waveform, comparing
the pre- and post-denoising peak SNR.

---

## Quick Start

```bash
# 1. Install dependencies
pip install torch numpy scipy matplotlib gwpy gwosc pycbc

# 2. Run preprocessing self-test
python src/preprocessing.py

# 3. Train the autoencoder (synthetic data; no internet required)
python src/autoencoder.py

# 4. Evaluate SNR on held-out GWTC events
python src/snr_evaluation.py

# 5. Generate spectrogram visualisations
python src/spectrogram_viz.py
```

All scripts fall back to **NumPy/SciPy baselines** when PyTorch is not installed,
so you can verify the pipeline logic without a GPU or deep-learning stack.

---

## Verification Metrics

| Metric | Target | Measured |
|---|---|---|
| SNR Improvement | ≥ +3 dB on held-out events | See `outputs/snr_results.json` |
| Reconstruction MSE (norm.) | < 0.05 | See `outputs/snr_results.json` |
| Noise Power Reduction | ≥ 40 % | See `outputs/spectrograms.png` |

---

## Tech Stack

| Library | Role |
|---|---|
| **PyTorch / Keras** | 1-D Conv Autoencoder training |
| **PyCBC** | Matched-filter template bank & whitening reference |
| **GWpy** | LIGO strain data I/O and time-series utilities |
| **NumPy** | Numerical computation |
| **SciPy** | Bandpass filter, Welch PSD, spectrogram |
| **Matplotlib** | All visualisations |
| **GWOSC** | Programmatic access to Open Science Center data |

---

## References

1. LIGO Open Science Center: <https://gwosc.org>
2. GWTC-3 Catalogue: Abbott et al. (2023), *Phys. Rev. X* 13, 041039
3. GWTC-5.0 Catalogue: <https://doi.org/10.7935/bk00-6a89>
4. GWpy documentation: <https://gwpy.github.io>
5. PyCBC tutorials: <https://pycbc.org/pycbc/latest/html/tutorials.html>
6. Torres-Forné et al. (2016) — *Denoising of gravitational wave signals via dictionary learning*

---

*Project 02 — Gravitational Wave Noise Filter | GWTC-5 catalogue | LIGO Open Science Center*

