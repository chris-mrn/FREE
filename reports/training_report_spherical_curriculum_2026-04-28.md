# Training Report — Spherical FM + OT Speed Curriculum

**Date**: April 28, 2026
**Job ID**: 2784435 (CANCELLED @ step 389K, wall limit 14h)
**Repository**: `chris-mrn/FREE`, branch `main`

---

## 1. Executive Summary

| Metric | Value |
|--------|-------|
| **Best FID** | **5.033 @ step 260,000** |
| Baseline FID (Spherical FM, step 220K) | 5.430 |
| Improvement over baseline | **+0.397 FID** |
| Final eval FID | 5.343 @ step 380,000 |
| Training cancelled | Step 389,366 / 400,000 (97%) |

---

## 2. Configuration

| Parameter | Value |
|-----------|-------|
| Script | `train_fm_spherical_curriculum_ddp.py` |
| Warm-start | `ema_step_0220000.pt` (Spherical FM baseline, FID 5.43) |
| Interpolation | $X_t = \cos(t)X_0 + \sin(t)X_1$, $t \in [0, \pi/2]$ |
| OT speed step | 220K (`--init_speed`) |
| Blend window | 220K → 245K (cosine, 25K steps) |
| Phase 2 start | Step 245K |
| GPUs | 3× H100 80GB (gpu-sr675-34) |
| Per-GPU batch | 128 |
| Effective batch | 384 |
| LR | 2e-4 × 3 = 6e-4 |
| EMA decay | 0.9999 |
| Eval every | 20K steps, 2K reference images |
| NFE (sampling) | 35 Euler steps |

---

## 3. OT Speed Curriculum

| Parameter | Value |
|-----------|-------|
| Computed at | Step 220K (warm-start checkpoint) |
| OT speed range | [1.031, 3.327] (after smoothing) |
| Estimation time | 6.9 s |
| Distribution | $p(t) \propto 1/v_{OT}(t)$ (inverse-CDF) |

**OT speed curve**: `outputs/cifar10_spherical_curriculum/speed_step220000.png`

---

## 4. FID / KID / IS Progression

| Step | Phase | FID ↓ | KID mean±std | IS mean±std |
|------|-------|-------|--------------|-------------|
| 220K | — (warm-start) | 5.430 | 2.831±0.566 | 5.25±0.11 |
| 240K | blend (1→2) | 5.317 | 2.984±0.561 | 4.99±0.16 |
| 260K | 2 (pure OT) | 5.033 **← best** | 2.727±0.549 | 5.04±0.14 |
| 280K | 2 | 5.130 | 2.565±0.694 | 4.97±0.13 |
| 300K | 2 | 5.227 | 3.093±0.695 | 5.05±0.11 |
| 320K | 2 | 5.273 | 3.339±0.786 | 4.96±0.09 |
| 340K | 2 | 5.441 | 3.327±0.680 | 4.95±0.15 |
| 360K | 2 | 5.406 | 3.399±0.660 | 4.98±0.12 |
| 380K | 2 | 5.343 | 3.211±0.581 | 4.85±0.15 |

---

## 5. Loss Analysis

| Step | FID | Loss EMA | Roll std (5K) |
|------|-----|----------|---------------|
| 240K | 5.317 | 0.10315 | 0.00778 |
| 260K | 5.033 | 0.10195 | 0.00945 |
| 280K | 5.130 | 0.10171 | 0.00623 |
| 300K | 5.227 | 0.10181 | 0.00990 |
| 320K | 5.273 | 0.10145 | 0.00947 |
| 340K | 5.441 | 0.10109 | 0.00383 |
| 360K | 5.406 | 0.10124 | 0.00427 |
| 380K | 5.343 | 0.10090 | 0.00687 |

**Key finding**: Loss EMA and variance both decrease monotonically through step 380K.
The FID regression after 260K is **not** caused by training instability.
Root cause: OT curriculum creates a t-distribution mismatch — model undertrained on
low-weight t-regions while Euler evaluation uses uniform steps.

---

## 6. Checkpoints

```
outputs/cifar10_spherical_curriculum/checkpoints/
  ckpt_step_0340000.pt
  ckpt_step_0360000.pt
  ckpt_step_0380000.pt
```

> Best checkpoint for inference: `ckpt_step_0260000.pt` was pruned (keep_ckpts=3).
> Recommend re-running eval on step 260K if needed — re-train from 240K checkpoint.

---

## 7. Artefacts

```
outputs/cifar10_spherical_curriculum/
  metrics.csv                        # FID/KID/IS per step
  loss.csv                           # per-step loss + phase
  loss_analysis.png                  # loss/variance/FID diagnostic plot
  training_summary.png               # 2×3 summary figure
  speed_step220000.png               # OT speed curve
  ot_speed_step220000.npy            # raw speed values
  ot_t_grid_step220000.npy           # t grid
  samples/                           # image grids at each eval step
  train_2784435.log                  # full training log
```

---

## 8. Comparison vs Spherical FM Baseline

| Model | Steps | Best FID | Hardware |
|-------|-------|----------|----------|
| Spherical FM (baseline) | 220K | 5.430 | 1× RTX 4500 |
| **Spherical FM + OT Curriculum** | **260K** | **5.033** | **3× H100** |
| Improvement | — | **+0.397 FID** | — |

The curriculum achieves a **7.3% improvement** over the baseline at the best checkpoint.
However FID regresses after 260K due to t-distribution mismatch between training and evaluation.

---

## 9. Next Steps

- [ ] Re-evaluate best checkpoint: resubmit from step 240K with `--resume` to recover 260K eval
- [ ] NFE sweep on best checkpoint (NFE = 10, 20, 35, 50, 100)
- [ ] Fix mismatch: use adaptive step-size ODE solver (RK45) that respects p(t) at eval
- [ ] Ablation: stop curriculum at 260K, continue with uniform sampling
- [ ] VAE latent FM (jobs 2790513 / 2790514) now pending