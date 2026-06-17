"""
Deliverable 2: 1-D Convolutional Autoencoder for GW Strain Denoising
=====================================================================
Architecture:
  Encoder: Conv1d → LeakyReLU → MaxPool  (×3 blocks)
  Bottleneck: Conv1d latent representation
  Decoder:  ConvTranspose1d → LeakyReLU (×3 blocks) → Conv1d output

Training:
  Loss = α·MSE + β·Spectral Loss  (frequency-domain component)
  Optimizer: Adam with OneCycleLR scheduler

Usage:
  python autoencoder.py          # trains on synthetic data and saves model
"""

import numpy as np
import json
import os
import sys
from pathlib import Path

# ─── PyTorch ──────────────────────────────────────────────────────────────────
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset, random_split
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("PyTorch not installed – running in NUMPY-ONLY simulation mode.")

# ─── Local ────────────────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))
from preprocessing import generate_synthetic_strain, preprocess, SAMPLE_RATE, NPERSEG


# ══════════════════════════════════════════════════════════════════════════════
#  Architecture
# ══════════════════════════════════════════════════════════════════════════════
if TORCH_AVAILABLE:

    class ConvBlock(nn.Module):
        """Conv1d → BatchNorm → LeakyReLU"""
        def __init__(self, in_ch, out_ch, kernel=9, stride=1, padding=4):
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv1d(in_ch, out_ch, kernel, stride=stride, padding=padding),
                nn.BatchNorm1d(out_ch),
                nn.LeakyReLU(0.1, inplace=True),
            )
        def forward(self, x):
            return self.net(x)


    class GWAutoencoder(nn.Module):
        """
        1-D Convolutional Autoencoder for gravitational-wave strain denoising.

        Input  : (B, 1, L)  – single-channel strain segment, length L=NPERSEG
        Output : (B, 1, L)  – reconstructed (denoised) strain segment
        """
        def __init__(self, input_len: int = NPERSEG,
                     base_channels: int = 32,
                     latent_channels: int = 64):
            super().__init__()
            self.input_len = input_len

            # ── Encoder ──────────────────────────────────────────────────
            self.enc1 = ConvBlock(1,               base_channels,     kernel=9)
            self.enc2 = ConvBlock(base_channels,   base_channels * 2, kernel=7)
            self.enc3 = ConvBlock(base_channels*2, base_channels * 4, kernel=5)
            self.pool = nn.MaxPool1d(2, return_indices=True)

            # ── Bottleneck ────────────────────────────────────────────────
            self.bottleneck = nn.Sequential(
                nn.Conv1d(base_channels * 4, latent_channels, kernel_size=3, padding=1),
                nn.BatchNorm1d(latent_channels),
                nn.Tanh(),
            )

            # ── Decoder ───────────────────────────────────────────────────
            self.unpool = nn.MaxUnpool1d(2)
            self.dec1 = ConvBlock(latent_channels, base_channels * 4, kernel=5)
            self.dec2 = ConvBlock(base_channels*4, base_channels * 2, kernel=7)
            self.dec3 = ConvBlock(base_channels*2, base_channels,     kernel=9)
            self.out_conv = nn.Conv1d(base_channels, 1, kernel_size=1)

            # Up-sampling (replaces unpool when indices not available)
            self.up = nn.Upsample(scale_factor=2, mode="linear",
                                  align_corners=False)

        def encode(self, x):
            x = self.enc1(x)
            x, idx1 = self.pool(x)
            x = self.enc2(x)
            x, idx2 = self.pool(x)
            x = self.enc3(x)
            x, idx3 = self.pool(x)
            z = self.bottleneck(x)
            return z, (idx1, idx2, idx3)

        def decode(self, z, original_len: int):
            x = self.dec1(z)
            x = self.up(x)
            x = self.dec2(x)
            x = self.up(x)
            x = self.dec3(x)
            x = self.up(x)
            x = self.out_conv(x)
            # Crop / pad to exactly original_len
            if x.size(-1) > original_len:
                x = x[..., :original_len]
            elif x.size(-1) < original_len:
                x = torch.nn.functional.pad(x, (0, original_len - x.size(-1)))
            return x

        def forward(self, x):
            L  = x.size(-1)
            z, _ = self.encode(x)
            return self.decode(z, L)

        def count_params(self) -> int:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)


    # ══════════════════════════════════════════════════════════════════════════
    #  Loss Function
    # ══════════════════════════════════════════════════════════════════════════
    class CombinedLoss(nn.Module):
        """
        MSE in time-domain  +  MSE in frequency-domain (spectral loss).
        alpha·MSE(x, x̂) + beta·MSE(|FFT(x)|, |FFT(x̂)|)
        """
        def __init__(self, alpha: float = 0.7, beta: float = 0.3):
            super().__init__()
            self.alpha = alpha
            self.beta  = beta
            self.mse   = nn.MSELoss()

        def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
            time_loss     = self.mse(pred, target)
            pred_fft      = torch.abs(torch.fft.rfft(pred,   dim=-1))
            target_fft    = torch.abs(torch.fft.rfft(target, dim=-1))
            spectral_loss = self.mse(pred_fft, target_fft)
            return self.alpha * time_loss + self.beta * spectral_loss


    # ══════════════════════════════════════════════════════════════════════════
    #  Dataset Builder
    # ══════════════════════════════════════════════════════════════════════════
    def build_dataset(n_events: int = 500, snr_range=(3, 20),
                      duration: float = 32.0, seed: int = 0):
        """
        Build a synthetic paired dataset: (noisy_segment, clean_segment).
        Each event uses a different random seed and SNR.
        """
        noisy_list, clean_list = [], []
        rng = np.random.default_rng(seed)

        for i in range(n_events):
            snr   = float(rng.uniform(*snr_range))
            s     = int(rng.integers(0, 10000))
            noisy_raw, clean_raw = generate_synthetic_strain(duration, snr=snr, seed=s)

            noisy_segs, _ = preprocess(noisy_raw, return_segments=True)
            clean_segs, _ = preprocess(clean_raw, return_segments=True)

            n = min(len(noisy_segs), len(clean_segs))
            noisy_list.append(noisy_segs[:n])
            clean_list.append(clean_segs[:n])

        noisy_arr = np.concatenate(noisy_list, axis=0)   # (N, L)
        clean_arr = np.concatenate(clean_list, axis=0)

        noisy_t = torch.from_numpy(noisy_arr).unsqueeze(1)   # (N,1,L)
        clean_t = torch.from_numpy(clean_arr).unsqueeze(1)
        return TensorDataset(noisy_t, clean_t)


    # ══════════════════════════════════════════════════════════════════════════
    #  Training Loop
    # ══════════════════════════════════════════════════════════════════════════
    def train(model: GWAutoencoder,
              dataset,
              epochs:    int   = 50,
              batch_size:int   = 64,
              lr:        float = 3e-4,
              val_split: float = 0.1,
              device:    str   = "cpu",
              save_path: str   = "outputs/gw_autoencoder.pt") -> dict:

        model = model.to(device)
        criterion = CombinedLoss(alpha=0.7, beta=0.3)
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)

        # Train / val split
        n_val   = max(1, int(len(dataset) * val_split))
        n_train = len(dataset) - n_val
        train_ds, val_ds = random_split(dataset, [n_train, n_val],
                                        generator=torch.Generator().manual_seed(42))

        train_loader = DataLoader(train_ds, batch_size=batch_size,
                                  shuffle=True,  num_workers=0, pin_memory=False)
        val_loader   = DataLoader(val_ds,   batch_size=batch_size,
                                  shuffle=False, num_workers=0, pin_memory=False)

        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=lr * 10,
            steps_per_epoch=len(train_loader), epochs=epochs)

        history  = {"train_loss": [], "val_loss": []}
        best_val = float("inf")

        for epoch in range(1, epochs + 1):
            # ── Train ──
            model.train()
            train_losses = []
            for noisy, clean in train_loader:
                noisy, clean = noisy.to(device), clean.to(device)
                optimizer.zero_grad()
                recon = model(noisy)
                loss  = criterion(recon, clean)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                train_losses.append(loss.item())

            # ── Validate ──
            model.eval()
            val_losses = []
            with torch.no_grad():
                for noisy, clean in val_loader:
                    noisy, clean = noisy.to(device), clean.to(device)
                    recon = model(noisy)
                    val_losses.append(criterion(recon, clean).item())

            t_loss = np.mean(train_losses)
            v_loss = np.mean(val_losses)
            history["train_loss"].append(t_loss)
            history["val_loss"].append(v_loss)

            if v_loss < best_val:
                best_val = v_loss
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                torch.save(model.state_dict(), save_path)

            if epoch % 5 == 0 or epoch == 1:
                print(f"  Epoch {epoch:3d}/{epochs} │ "
                      f"Train Loss: {t_loss:.5f} │ Val Loss: {v_loss:.5f}")

        print(f"\n  Best val loss: {best_val:.5f} → saved to {save_path}")
        return history


# ══════════════════════════════════════════════════════════════════════════════
#  NumPy-only fallback (no PyTorch)
# ══════════════════════════════════════════════════════════════════════════════
else:
    def _numpy_autoencoder_demo():
        """Minimal numpy autoencoder demo when PyTorch is unavailable."""
        print("\n[NumPy Demo Mode]")
        noisy, clean = generate_synthetic_strain(32.0, snr=5.0)
        segs, meta   = preprocess(noisy)
        print(f"  Preprocessed {segs.shape[0]} segments of length {segs.shape[1]}")

        # Trivial 'denoising' via low-pass smoothing as placeholder
        from scipy.ndimage import uniform_filter1d
        denoised = uniform_filter1d(segs, size=5, axis=1)

        mse = np.mean((denoised - segs) ** 2)
        print(f"  Reconstruction MSE (smoothing baseline): {mse:.6f}")
        return {"train_loss": [mse], "val_loss": [mse]}


# ══════════════════════════════════════════════════════════════════════════════
#  Entry Point
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    if not TORCH_AVAILABLE:
        history = _numpy_autoencoder_demo()
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"=== GW Autoencoder Training  [device: {device}] ===\n")

        model = GWAutoencoder(input_len=NPERSEG, base_channels=32, latent_channels=64)
        print(f"  Model parameters : {model.count_params():,}")

        print("  Building synthetic dataset …")
        dataset = build_dataset(n_events=200, snr_range=(3, 20))
        print(f"  Dataset size      : {len(dataset)} segment pairs")

        print("\n  Training …")
        history = train(model, dataset, epochs=50, batch_size=64,
                        lr=3e-4, device=device,
                        save_path="outputs/gw_autoencoder.pt")

    # Save history for plotting
    os.makedirs("outputs", exist_ok=True)
    with open("outputs/training_history.json", "w") as f:
        json.dump(history, f, indent=2)
    print("\n  Training history saved → outputs/training_history.json")
    print("\n✓  Autoencoder training complete.")
