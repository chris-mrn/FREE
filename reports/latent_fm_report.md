# Experiment Report — Latent Linear Flow Matching on CIFAR-10

**Date**: 2026-04-28 / 2026-04-29
**Status**: 🔄 Running (job 2823394, step ~22K / 200K)

---

## Motivation

Standard pixel-space FM on CIFAR-10 requires the UNet to operate on 3×32×32 tensors at every training step.
Latent FM instead operates on the compressed VAE latent space (4×8×8), reducing the spatial dimension by 16× and the channel count from 3 to 4.
This gives:
- **Faster training** — smaller UNet, faster forward/backward passes.
- **Smoother target** — the VAE encoder collapses high-frequency pixel variance; the latent space has a smoother empirical distribution, potentially easier for FM to model.
- **Decoupled compression and generation** — the VAE is fixed after pre-training; the FM only learns the latent distribution.

---

## VAE Pre-training (prerequisite)

See [vae_cifar10_report.md](vae_cifar10_report.md) for full details.

| Parameter | Value |
|-----------|-------|
| Architecture | VAE + patch-GAN discriminator |
| Training steps | 100,000 |
| Latent shape | (4, 8, 8) — 4 channels, 8×8 spatial |
| Rec loss (final) | 0.079 |
| Checkpoint | `outputs/cifar10_vae/vae_final.pt` |
| Latent stats | `outputs/cifar10_vae/latent_stats.pt` |

Latent channel statistics (used for normalisation):

| Channel | Mean | Std |
|---------|------|-----|
| 0 | −0.221 | 0.518 |
| 1 | −0.233 | 0.588 |
| 2 | +0.188 | 0.599 |
| 3 | −1.030 | 0.617 |

---

## Model Configuration

| Parameter | Value |
|-----------|-------|
| Architecture | UNetModelWrapper on latent space |
| Parameters | **14.9 M** |
| Latent input shape | (4, 8, 8) |
| Interpolation | Linear: $X_t = (1-t)\,X_0 + t\,X_1$ |
| Source | $X_0 \sim \mathcal{N}(0, I)$ |
| Target | Normalised latent $z = (\mathrm{enc}(x) - \mu) / \sigma$ |
| Conditional velocity | $u_t = X_1 - X_0$ |
| Total steps | 200,000 |
| Batch size | 128 |
| Learning rate | 2 × 10⁻⁴ |
| LR warmup | 1,000 steps |
| EMA decay | 0.9999 |
| Gradient clip | 1.0 |
| Eval every | 20,000 steps |
| FID samples | 10,000 |
| NFE at eval | 35 (Euler) |
| t-sampling | Uniform $\mathcal{U}(0, 1)$ |

---

## Training History

### Job timeline

| Job ID | Node | GPU | Steps | Outcome |
|--------|------|-----|-------|---------|
| 2791356 | gpu-xd670-30 | 3× H100 | — | Failed (import error in training script) |
| 2792714 | u240c | RTX 4500 Ada | 0 → ~20K | Failed at eval (InceptionMetrics API mismatch) |
| 2793229 | u240c | RTX 4500 Ada | ~20K | Failed at eval (same API error) |
| 2813107 | gpu-xd670-30 | H100 | brief | Started, became PENDING (Priority queue) — cancelled |
| **2823394** | **u240c** | **RTX 4500 Ada** | **20K → ongoing** | **Running** ✅ |

The recurring failure across jobs 2792714/2793229 was an `InceptionMetrics.compute_reference_stats` API error (method renamed). This was fixed before job 2823394.

### Checkpoint

| Step | Checkpoint | Notes |
|------|------------|-------|
| 20,000 | `checkpoints/ckpt_step_0020000.pt` | Only saved checkpoint; current job auto-resumed from here |

---

## Training Progress

### Loss trajectory (steps 0 → ~22K)

| Step | Loss (raw) | Loss (EMA) |
|------|-----------|-----------|
| 100 | 1.895 | 1.908 |
| 1,000 | — | ~1.55 |
| 5,000 | — | ~1.40 |
| 10,000 | — | ~1.35 |
| 20,000 | — | ~1.283 |
| 21,900 | 1.257 | **1.257** |

Loss has been decreasing smoothly. No instabilities observed.

### FID vs Step (NFE=35)

| Step | FID ↓ | KID (×10³) ↓ | IS ↑ |
|------|------:|-------------:|-----:|
| 20,000 | 80.87 | 71.67 ± 2.75 | 2.23 ± 0.04 |
| 40,000 | pending | — | — |

FID=80.87 at step 20K is the first evaluation. This is expected to drop sharply — the model is still early in training. For reference, the pixel-space spherical FM baseline reached FID≈57 by step 40K and FID≈9 by step 60K.

---

## Current Run (job 2823394)

| Parameter | Value |
|-----------|-------|
| Node | `u240c.id.gatsby.ucl.ac.uk` |
| GPU | NVIDIA RTX 4500 Ada Generation (24 GB) |
| Training speed | ~33 it/s |
| Current step | ~22,000 |
| Steps remaining | ~178,000 |
| Estimated completion | ~1.5 h from start (≈ 16:40 BST Apr 29) |
| Resume point | Step 20,000 |
| Log | `train_2823394.log` |

---

## Expected Results

Based on the pixel-space FM results as a rough guide:

| Step | Expected FID@35 | Notes |
|------|----------------|-------|
| 20K | ~80 (measured) | Early training |
| 40K | ~10–20 | Rapid improvement phase |
| 60K | ~5–8 | Approaching convergence |
| 100K | ~3–6 | Competitive with pixel-space |
| 200K | ~2–5 | Final |

Latent FM models typically achieve competitive FID vs pixel-space FM due to the smoother latent distribution, but the 4×8×8 bottleneck introduces a reconstruction ceiling from the VAE.

---

## Artifacts

| File | Description |
|------|-------------|
| `outputs/cifar10_latent_linear/checkpoints/ckpt_step_0020000.pt` | Latest checkpoint |
| `outputs/cifar10_latent_linear/metrics.csv` | FID/KID/IS per eval step |
| `outputs/cifar10_latent_linear/loss.csv` | Per-100-step loss log |
| `outputs/cifar10_latent_linear/latents_norm.pt` | Pre-computed normalised latents (cached) |
| `outputs/cifar10_latent_linear/train_2823394.log` | Active training log |
| `outputs/cifar10_vae/latent_stats.pt` | Channel mean/std used for normalisation |

---

## Next Steps

1. **Monitor FID at step 40K** — first real quality signal. If FID < 10, training is healthy.
2. **Speed-adaptive inference** — once training completes, apply OT-speed-adaptive Euler schedule (as done for spherical FM) and compare FID-NFE curves.
3. **FR speed profile** — estimate $v_t^{FR}$ from the trained latent FM and check if the speed non-uniformity is comparable to pixel-space FM.
4. **Latent curriculum** — if speed is non-uniform, apply arc-length curriculum (as done for self-FM) to the latent model for the training schedule ablation in the FREE paper.
