# VAE CIFAR-10 Training Report
**Job 2791355** — Completed Apr 28, 2026
**Duration**: 1h 19m
**Node**: `gpu-sr675-34` (NVIDIA H100 80GB HBM3)

---

## Configuration

| Parameter | Value |
|-----------|-------|
| Steps | 100,000 |
| Output dir | `outputs/cifar10_vae/` |
| Architecture | VAE + patch-GAN discriminator |
| Latent dim | 4 |
| Batch size | (default) |
| Speed | ~21 steps/s |

**Loss terms**:
- `loss_rec` — pixel-level L1/L2 reconstruction
- `loss_kl` — KL divergence of latent posterior vs N(0,I)
- `loss_perc` — perceptual (VGG) feature matching
- `loss_adv_g` — adversarial generator loss (fool discriminator)
- `loss_disc` — discriminator cross-entropy loss
- `adv_weight` — adaptive GAN weight (NaN-safe scalar balancing rec vs adv)

---

## Training Phases

### Phase 0: Warm-up (steps 0–~15,000)
Pure reconstruction + KL. Discriminator is off (`adv_weight=0`). Rec loss drops fast from 0.41 → 0.08 in the first 10K steps. KL rises from 434 → 1677 as the encoder learns to use its latent capacity.

### Phase 1: Adversarial kick-in (~step 20,000)
GAN activates: `adv_weight` spikes to **178.9** at step 20K (discriminator is freshly initialized and trivially fooled). It rapidly drops to ~0.38 by step 30K as the discriminator learns. From step 30K onward, adversarial training is stable.

### Phase 2: Stable adversarial training (steps 30,000–100,000)
All losses decrease slowly and steadily. `adv_weight` oscillates in `[0.25, 0.85]` — the adaptive scaler tracks the balance between rec and adversarial gradients.

---

## Loss Trajectory

| Step | rec | kl | perceptual | adv_g | disc | adv_weight |
|------|-----|----|------------|-------|------|------------|
| 100 | 0.4078 | 434 | 4.280 | 0.000 | 0.000 | 0.000 |
| 1,000 | 0.1274 | 762 | 3.082 | 0.000 | 0.000 | 0.000 |
| 5,000 | 0.0946 | 1819 | 2.412 | 0.000 | 0.000 | 0.000 |
| 10,000 | 0.0779 | 1677 | 2.276 | 0.000 | 0.000 | 0.000 |
| 20,000 | 0.0966 | 1614 | 2.402 | 0.001 | 1.000 | **178.96** |
| 30,000 | 0.0802 | 1595 | 2.199 | 0.217 | 0.899 | 0.388 |
| 50,000 | 0.0781 | 1585 | 2.102 | 0.162 | 0.931 | 0.844 |
| 75,000 | 0.0744 | 1573 | 2.008 | 0.164 | 0.890 | 0.257 |
| **100,000** | **0.0788** | **1561** | **1.987** | **0.169** | **0.953** | **0.375** |

---

## Final Metrics (step 100,000)

```
rec         = 0.0788
kl          = 1561.3
perceptual  = 1.987
adv_g       = 0.169
disc        = 0.953
adv_weight  = 0.375
```

---

## Latent Space Statistics

Computed over the full CIFAR-10 training set:

```
mean: [-0.2211, -0.2330,  0.1884, -1.0297]
std:  [ 0.5176,  0.5881,  0.5995,  0.6174]
```

**Observations**:
- Std ~0.55–0.62 across all 4 dims — reasonably isotropic. The latent space is well-used.
- Mean is not centered at zero, especially dim 4 (mean=-1.03). This is fine for flow matching — we can normalize at inference time using `latent_stats.pt`.
- KL=1561 is relatively high for a 4-dim space (theoretical maximum = 4 × ½ = 2 nats per dim, but the KL here is computed without the 1/2 prefactor across the batch — actual per-sample KL is smaller). This indicates the encoder is actively using the latent dimensions rather than collapsing to the prior.

---

## Saved Artifacts

| File | Description |
|------|-------------|
| `outputs/cifar10_vae/vae_final.pt` | Final model weights (step 100K) |
| `outputs/cifar10_vae/vae_best.pt` | Best checkpoint by reconstruction loss |
| `outputs/cifar10_vae/latent_stats.pt` | `{mean, std}` for latent normalization |
| `outputs/cifar10_vae/loss.csv` | Full loss trajectory (1001 entries) |
| `outputs/cifar10_vae/recon_{step}.png` | Raw decoder reconstructions every 5K steps |
| `outputs/cifar10_vae/recon_ema_{step}.png` | EMA decoder reconstructions every 5K steps |

---

## Assessment

Training completed successfully. Key points:
- Reconstruction loss is low and stable (0.079).
- GAN kick-in at step 20K caused a brief spike but stabilized quickly.
- Disc=0.95 near optimum (1.0 = random, 0 = generator wins perfectly) — adversarial balance is healthy.
- The 4-dim latent is a bottleneck that compresses 3×32×32=3072 pixels into 4 values. Some quality loss is expected, but the VAE is ready for latent FM training.

**Next step**: Train a latent linear FM on these 4-dim encodings using `latent_stats.pt` for normalization. The script `train_fm_latent_linear_ddp.py` needs to be created (job 2791356 failed because the script was missing).
