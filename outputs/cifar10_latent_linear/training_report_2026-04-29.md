# Training Report — Latent Linear FM — 2026-04-29

## Configuration

| Parameter | Value |
|-----------|-------|
| Script | `train_fm_latent_linear_ddp.py` |
| VAE checkpoint | `/nfs/ghome/live/cmarouani/FREE/outputs/cifar10_vae/vae_final.pt` |
| Latent stats | `/nfs/ghome/live/cmarouani/FREE/outputs/cifar10_vae/latent_stats.pt` |
| Interpolation | $X_t = (1-t)X_0 + tX_1$ (linear) |
| Source | $X_0 \sim \mathcal{N}(0,I)$ (latent space) |
| GPUs | 1× NVIDIA RTX 4500 Ada Generation |
| Per-GPU batch | 128 |
| Effective batch | 128 |
| Total steps | 200000 |
| LR | 0.0002 × 1 = 2.00e-04 |
| Total time | 2.10 h |

## FID / KID / IS Progression

| Step | FID ↓ | KID mean | IS mean |
|------|-------|----------|---------|
| 20000 | 80.872 | 71.671±2.750 | 2.23±0.04 |
| 40000 | 42.795 | 34.108±1.593 | 2.75±0.09 |
| 60000 | 32.587 | 25.011±1.257 | 3.00±0.06 |
| 80000 | 30.438 | 22.926±1.160 | 3.09±0.08 |
| 100000 | 29.536 | 22.069±1.441 | 3.10±0.07 |
| 120000 | 29.499 | 22.369±1.390 | 3.10±0.04 |
| 140000 | 28.942 ← **best** | 21.689±1.513 | 3.13±0.10 |
| 160000 | 29.142 | 21.788±1.397 | 3.13±0.06 |
| 180000 | 29.644 | 22.379±1.479 | 3.15±0.05 |
| 200000 | 29.018 | 21.405±1.084 | 3.11±0.08 |

**Best FID**: 28.942 @ step 140000
**Final FID**: 29.018 @ step 200000

## Notes

- Latent space: 4×8×8 (4× spatial downsampling from 32×32)
- Latents normalised per-channel to zero mean / unit std before FM training
- Velocity field: small UNet on 8×8 spatial latents
- Decoded via trained VAE for FID evaluation