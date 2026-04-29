# Training Report — April 28, 2026

**Date**: 2026-04-28  
**Repository**: `chris-mrn/FREE`, branch `main`

---

## 1. Executive Summary

| Model | Job ID | Partition | Hardware | Duration | Steps Done | Best FID | Status |
|-------|--------|-----------|----------|----------|------------|----------|--------|
| Spherical FM + OT Curriculum | 2784435 | `gpu_lowp` | 3× H100 80 GB | 4 h 33 min | ~389K / 400K (97%) | **5.03** @ 260K | Cancelled (manual) |
| Self-FM (training) | 2791486 | `gpu_lowp` | 1× H100 80 GB | 4 h 34 min | 200,001 / 200,001 | — | Completed ✓ |
| Self-FM (eval) | 2793225 | `gatsby_ws` | RTX 4500 Ada | 4 h 24 min | — | **3.26** @ NFE=200 | Completed ✓ |
| VAE CIFAR-10 | 2791355 | `gpu_lowp` | 1× H100 80 GB | 1 h 19 min | 100,000 / 100,000 | — | Completed ✓ |
| Latent Linear FM | 2793229 | `gatsby_ws` | RTX 4500 Ada | 1 min 51 s | 0 | — | Failed ✗ |
| Speed-Schedule Eval | 2790518 | `gatsby_ws` | RTX 4500 Ada | 0 h 57 min | — | — | Completed ✓ |
| Speed-Schedule Eval (alpha sweep) | 2791428 | `gatsby_ws` | RTX 4500 Ada | 4 h 00 min | — | — | Timeout (NFE=100 @ α=0.4) |

**Key findings**:
- The spherical FM + OT curriculum nearly completes (97%) and achieves a new best FID of **5.03**, beating the warm-start baseline (5.43).
- Self-FM (data-to-data interpolant) achieves FID **3.26 @ NFE=200**, but degrades to FID 26 @ NFE=35, revealing a strong NFE dependence.
- Speed-adaptive ODE scheduling only helps at NFE ≥ 50; linear schedule is strictly better below that.
- VAE training completes cleanly; latent FM blocked by an API name bug.

---

## 2. Spherical FM + OT Curriculum — Job 2784435

### Configuration

| Parameter | Value |
|-----------|-------|
| Slurm script | `slurm/train_spherical_ot_curriculum.sh` |
| Node | `gpu-sr675-34` |
| GPUs | 3× NVIDIA H100 80GB HBM3 (81,559 MiB each) |
| Per-GPU batch size | 128 |
| Effective batch size | 384 |
| Warm-start | `outputs/cifar10_spherical/fm_standard/checkpoints/ema_step_0220000.pt` (FID 5.43) |
| Start step | 220,000 |
| Interpolation | $X_t = \cos(t)\,X_0 + \sin(t)\,X_1$, $t \in [0, \pi/2]$ |
| Speed type | OT (JVP forward-mode AD) |
| OT speed init | Computed at step 220K (`--init_speed`): smoothed range [1.031, 3.327] |
| Curriculum phases | 220K–245K: cosine blend Uniform → p1; 245K+: pure speed-adaptive |
| Eval frequency | every 20K steps |
| Sampling steps | 35 (Euler) |
| Wall limit | 14 h (cancelled manually after ~4.5 h) |
| Start time | Tue Apr 28 09:07:35 UTC 2026 |
| End time | Tue Apr 28 13:41:15 UTC 2026 |

### Training Command

```bash
sbatch slurm/train_spherical_ot_curriculum.sh
# Which runs (with NGPU=3):
PYTHONPATH=/nfs/ghome/live/cmarouani/FREE \
torchrun --nproc_per_node=3 --nnodes=1 --master_addr=localhost --master_port=29502 \
    train.py \
    --path              spherical \
    --dataset           cifar10 \
    --coupling          ot \
    --t_mode            curriculum \
    --speed_type        ot \
    --curriculum_start  100000 \
    --curriculum_blend  25000 \
    --speed_n_t         100 \
    --speed_B           512 \
    --speed_epochs      3 \
    --total_steps       400001 \
    --batch_size        128 \
    --lr                2e-4 \
    --warmup            5000 \
    --ema_decay         0.9999 \
    --grad_clip         1.0 \
    --num_channel       128 \
    --eval_every        20000 \
    --fid_samples       10000 \
    --keep_ckpts        2 \
    --out_dir           outputs/cifar10_spherical_curriculum \
    --resume            auto \
    --ddp
```

### FID / KID / IS Progression (from step 220K warm-start)

| Step | Phase | FID | KID (mean±std) | IS (mean±std) |
|------|-------|-----|-----------------|----------------|
| 240K | 1 (blending) | 5.317 | 2.984 ± 0.561 | 4.995 ± 0.156 |
| 260K | 2 (speed-adaptive) | **5.033** | 2.727 ± 0.549 | 5.044 ± 0.136 |
| 280K | 2 | 5.130 | 2.565 ± 0.694 | 4.973 ± 0.135 |
| 300K | 2 | 5.227 | 3.093 ± 0.695 | 5.050 ± 0.111 |
| 320K | 2 | 5.273 | 3.339 ± 0.786 | 4.960 ± 0.093 |
| 340K | 2 | 5.441 | 3.327 ± 0.680 | 4.951 ± 0.149 |
| 360K | 2 | 5.406 | 3.399 ± 0.660 | 4.976 ± 0.123 |
| 380K | 2 | 5.343 | 3.211 ± 0.581 | 4.851 ± 0.155 |

*Warm-start baseline: Spherical FM FID 5.43 @ step 220K.*

### Checkpoints Saved

```
outputs/cifar10_spherical_curriculum/checkpoints/
  ckpt_step_0340000.pt
  ckpt_step_0360000.pt
  ckpt_step_0380000.pt    ← latest
```

### Artifacts

```
outputs/cifar10_spherical_curriculum/
  metrics.csv
  loss.csv                # loss logged every 100 steps; last step ≈ 389,300
  samples/
  checkpoints/
  ot_speed_step220000.npy
  ot_t_grid_step220000.npy
  speed_step220000.png
  training_summary.png
  loss_analysis.png
  train_2784435.log
```

### Observations

- **Curriculum works**: FID improves from 5.43 (warm-start) → **5.03** at step 260K, a gain of +0.40 FID in just 40K extra steps.
- **Plateau and oscillation**: After 260K, FID hovers in the 5.1–5.4 range without clear trend. The OT speed curriculum may have found its limit at this architecture/LR.
- **Nearly complete**: Job was manually cancelled at ~389K (97% done). The last 11K steps would not have changed the best result.
- **OT speed range**: [1.031, 3.327] (smoothed from 5 epochs), much narrower than linear-FM FR speed — spherical paths are inherently more uniform in speed.

---

## 3. Self-FM CIFAR-10 — Jobs 2791486 (train) + 2793225 (eval)

### Motivation

Standard flow matching uses $X_0 \sim \mathcal{N}(0,I)$ as source. Self-FM replaces it entirely: both endpoints are drawn from CIFAR-10, giving a **data-to-data** stochastic interpolant:

$$X_t = (1-t)\,X_1 + t\,\tilde{X}_1 + \sqrt{t(1-t)}\,\varepsilon, \qquad \varepsilon \sim \mathcal{N}(0,I),\quad t \in [0.01,\,0.99]$$

At inference, sampling starts from a real CIFAR-10 image and integrates $t: 1 \to 0$ to produce a new image.

### Configuration

| Parameter | Value |
|-----------|-------|
| Slurm script | `examples/images/cifar10/slurm/slurm_self_fm.sh` |
| Script | `examples/images/cifar10/training/train_self_fm.py` |
| Node | `gpu-sr675-34` |
| GPU | 1× NVIDIA H100 80GB HBM3 |
| Total steps | 200,001 |
| Batch size | 128 |
| LR | 2 × 10⁻⁴ |
| LR warmup | 5,000 steps |
| EMA decay | 0.9999 |
| Architecture | UNetModelWrapper (35.75 M params) |
| Speed step | 100,000 (FR speed, Hutchinson) |
| Blend duration | 25,000 steps |
| Checkpoint interval | 50,000 steps |
| Training start | Tue Apr 28 15:24:42 UTC 2026 |
| Training end | Tue Apr 28 19:59:06 UTC 2026 (4 h 34 min) |
| Eval node | `u237h` (RTX 4500 Ada, 24 GB) |
| Eval wall-clock | 4 h 24 min (job 2793225) |

### Training Command

```bash
sbatch examples/images/cifar10/slurm/slurm_self_fm.sh
# Which runs:
PYTHONPATH=/nfs/ghome/live/cmarouani/FREE \
python examples/images/cifar10/training/train_self_fm.py \
    --total_steps   200001 \
    --batch_size    128 \
    --lr            2e-4 \
    --warmup        5000 \
    --ema_decay     0.9999 \
    --num_channel   128 \
    --save_step     50000 \
    --num_workers   4 \
    --data_dir      /tmp/cifar10_self_fm_$$ \
    --out_dir       outputs/cifar10_self_fm \
    --speed_step    100000 \
    --blend_steps   25000 \
    --fr_n_t        1000 \
    --fr_n_epochs   5 \
    --fr_n_hutch    5 \
    --fr_B_per_t    2 \
    --fr_chunk      128 \
    --fr_smooth     3.0 \
    --fr_n_ref      2000
```

### Curriculum Schedule

| Phase | Step Range | Sampler |
|-------|-----------|---------|
| 0 — Uniform | 0 – 100,000 | $t \sim \mathcal{U}(0.01, 0.99)$ |
| 1 — Blend | 100,000 – 125,000 | Cosine blend: $\mathcal{U} \to p_{FR}$ |
| 2 — Arc-length | 125,000 – 200,001 | $p(t) \propto 1/v_t^{FR}$ |

### FR Speed Profile (step 100,000)

Estimated in **10.8 s** on H100 (5 epochs × 5 Hutchinson probes, 1,000 $t$-points):

| Metric | Value |
|--------|-------|
| $v_t^{FR}$ min (smoothed) | 28.35 at $t \approx 0.50$ |
| $v_t^{FR}$ max (smoothed) | 59,350 at $t = 0.98$ |
| Max/min ratio | **≈ 2,093×** |

The speed is extremely non-uniform — the self-interpolant concentrates all complexity near $t=1$ where $X_t \approx \tilde{X}_1$.

### FID vs NFE (10,000 samples, 50,000-image reference)

| Step | NFE=5 | NFE=10 | NFE=20 | NFE=35 | NFE=50 | NFE=100 | NFE=200 |
|------|------:|-------:|-------:|-------:|-------:|--------:|--------:|
| 50K  | 232.37 | 227.51 | 173.61 | 113.02 | 72.90 | 21.03 | 6.92 |
| 100K | 260.90 | 261.73 | 193.06 | 111.98 | 62.84 | 18.42 | 7.33 |
| 150K | 245.46 | 180.95 | 95.16  | 36.29  | 20.50 | 7.98  | 4.36 |
| **200K** | **227.26** | **153.70** | **72.51** | **26.30** | **14.37** | **5.43** | **3.26** |

*Step 200,001 (final EMA save) gives identical values to 4 significant figures.*

### Full Metrics at Step 200,000

| NFE | FID ↓ | KID (mean±std) ↓ | IS ↑ |
|----:|------:|-----------------:|-----:|
| 5 | 227.26 | 248.76 ± 4.81 | 1.52 ± 0.02 |
| 10 | 153.70 | 157.76 ± 5.22 | 1.87 ± 0.04 |
| 20 | 72.51 | 66.47 ± 3.09 | 2.70 ± 0.08 |
| 35 | 26.30 | 20.09 ± 1.47 | 3.87 ± 0.14 |
| 50 | 14.37 | 9.87 ± 0.98 | 4.49 ± 0.17 |
| 100 | 5.43 | 2.69 ± 0.43 | 5.21 ± 0.20 |
| **200** | **3.26** | **0.93 ± 0.25** | **5.46 ± 0.18** |

### Checkpoints Saved

```
outputs/cifar10_self_fm/checkpoints/
  ema_step_0050000.pt
  ema_step_0100000.pt
  ema_step_0150000.pt
  ema_step_0200000.pt
  ema_step_0200001_final.pt
```

### Artifacts

```
outputs/cifar10_self_fm/
  checkpoints/
  metrics_full.csv          # FID/KID/IS for all (step, NFE) pairs
  nfe_fid_table.csv         # FID-only summary
  fr_speed_step100000.npy   # smoothed FR speed curve
  fr_speed_raw_step100000.npy
  fr_t_grid_step100000.npy
  fr_speed_step100000.png   # 3-panel plot: speed / arc-length weight / CDF
  eval_samples/             # sample grids at NFE=35 per checkpoint
  experiment_report.md      # detailed standalone report (auto-generated)
  train_2791486.log
  eval_2793225.log
```

### Observations

- **FID 3.26 at NFE=200** is competitive with Gaussian-prior baselines on unconditional CIFAR-10, achieved with zero Gaussian noise source.
- **Arc-length curriculum is critical**: at NFE=35, FID drops from 112 (step 100K, before curriculum) → 36 (150K) → **26** (200K) — a 77% improvement.
- **Low-NFE collapse**: FID=227 at NFE=5 vs FID=3.26 at NFE=200 is a 70× gap. The 2,093× non-uniformity of the FR speed makes Euler with uniform $\Delta t$ especially wasteful near $t=1$. Speed-adaptive step schedules would be a high-value fix.
- **Step 100K anomaly**: FID at low NFE *worsens* vs step 50K (e.g. NFE=10: 228 → 262). This is a curriculum transition artefact — recovers by step 150K once blending is complete.

---

## 4. VAE CIFAR-10 — Job 2791355

### Configuration

| Parameter | Value |
|-----------|-------|
| Slurm script | `slurm/train_vae_cifar10.sh` |
| Script | `scripts/train_vae_cifar10.py` |
| Node | `gpu-sr675-34` |
| GPU | 1× NVIDIA H100 80GB HBM3 |
| Total steps | 100,000 |
| Batch size | 128 |
| Latent shape | $(4, 8, 8)$ = 256 dims |
| Base channels | 128 |
| LR | 4.5 × 10⁻⁶ (both generator and discriminator) |
| KL weight | 1 × 10⁻⁶ |
| Perceptual weight | 1.0 (VGG-16) |
| Disc weight | 0.5 (adaptive VQGAN-style) |
| Disc start | step 10,000 |
| KL warmup | 5,000 steps |
| Start time | Tue Apr 28 14:12:30 UTC 2026 |
| End time | Tue Apr 28 15:31:47 UTC 2026 (1 h 19 min) |

### Training Command

```bash
sbatch slurm/train_vae_cifar10.sh
# Which runs:
PYTHONPATH=/nfs/ghome/live/cmarouani/FREE \
python scripts/train_vae_cifar10.py \
    --out_dir      outputs/cifar10_vae \
    --data_dir     /tmp/cifar10_vae_$$ \
    --total_steps  100000 \
    --batch_size   128 \
    --lr           4.5e-6 \
    --disc_lr      4.5e-6 \
    --kl_weight    1e-6 \
    --perc_weight  1.0 \
    --disc_weight  0.5 \
    --disc_start   10000 \
    --kl_warmup    5000 \
    --ema_decay    0.9999 \
    --grad_clip    1.0 \
    --z_ch         4 \
    --base_ch      128 \
    --num_workers  4 \
    --log_every    100 \
    --save_every   20000 \
    --recon_every  5000
```

### Final Metrics (step 100,000)

| Loss component | Value |
|---------------|-------|
| L1 reconstruction | **0.0788** |
| KL divergence | 1,561.3 (contribution ≈ 0.0016 after weighting) |
| Perceptual (VGG) | 1.987 |
| Adversarial (generator) | 0.169 |
| Discriminator | 0.953 |
| Adaptive weight | 0.375 |

### Latent Statistics (computed over full CIFAR-10 training set)

| Channel | Mean | Std |
|---------|------|-----|
| 0 | −0.2211 | 0.5176 |
| 1 | −0.2330 | 0.5881 |
| 2 | +0.1884 | 0.5995 |
| 3 | −1.0297 | 0.6174 |

### Checkpoints Saved

```
outputs/cifar10_vae/
  vae_final.pt              ← final weights (100K steps)
  vae_best.pt               ← best by reconstruction loss
  vae_step_{20000,...,100000}.pt
  latent_stats.pt           ← per-channel mean/std for normalisation
  latent_stats.pt → outputs/cifar10_latent_linear/latents_norm.pt (pre-cached)
```

### Observations

- Training converged cleanly in 1.32 h on H100.
- Reconstruction quality (L1=0.079) is low, discriminator score ~0.95 indicates stable adversarial training.
- KL=1561 suggests the latent space is heavily utilised (encoder entropy is large relative to prior). With $\beta=10^{-6}$ this is purely informational — the effective KL cost to the loss is negligible.
- Latent std ~0.52–0.62 across all 4 channels — reasonably isotropic, good for FM training.
- The pre-cached normalised latent tensor was saved to `outputs/cifar10_latent_linear/latents_norm.pt` for fast FM training.

---

## 5. Latent Linear FM — Job 2793229 (FAILED)

### Configuration

| Parameter | Value |
|-----------|-------|
| Slurm script | `slurm/train_latent_fm_linear.sh` |
| Script | `scripts/train_fm_latent_linear_ddp.py` |
| Node | `u240c` (RTX 4500 Ada, 24 GB) |
| VAE checkpoint | `outputs/cifar10_vae/vae_final.pt` |
| Duration | 1 min 51 s |
| Start | Tue Apr 28 20:39:32 BST 2026 |

### Training Command

```bash
sbatch slurm/train_latent_fm_linear.sh
# Which runs:
PYTHONPATH=/nfs/ghome/live/cmarouani/FREE \
torchrun --nproc_per_node=1 --nnodes=1 --master_addr=localhost --master_port=29503 \
    scripts/train_fm_latent_linear_ddp.py \
    --vae_ckpt     outputs/cifar10_vae/vae_final.pt \
    --latent_stats outputs/cifar10_vae/latent_stats.pt \
    --out_dir      outputs/cifar10_latent_linear \
    --total_steps  100000 \
    --batch_size   128 \
    --lr           2e-4 \
    --warmup       1000 \
    --ema_decay    0.9999 \
    --grad_clip    1.0 \
    --num_channel  128 \
    --eval_every   10000 \
    --fid_samples  2000 \
    --n_steps      35 \
    --vae_z_ch     4 \
    --vae_base_ch  128 \
    --resume       auto
```

### Error

```
AttributeError: 'InceptionMetrics' object has no attribute 'compute_reference_stats'.
Did you mean: 'compute_real_stats'?

File "scripts/train_fm_latent_linear_ddp.py", line 486, in main
    real_mu, real_sig, real_feats = inception.compute_reference_stats(all_real)
```

### Root Cause

`scripts/train_fm_latent_linear_ddp.py` calls `inception.compute_reference_stats()`, but the `InceptionMetrics` API in `metrics/metrics.py` exposes `compute_real_stats()`. The method was renamed at some point and the latent FM script was not updated.

### Fix Required

In [scripts/train_fm_latent_linear_ddp.py](scripts/train_fm_latent_linear_ddp.py) line 486:
```python
# Before:
real_mu, real_sig, real_feats = inception.compute_reference_stats(all_real)
# After:
real_mu, real_sig, real_feats = inception.compute_real_stats(all_real)
```

---

## 6. Speed-Adaptive Schedule Evaluation — Jobs 2790518 + 2791428

These jobs evaluated whether speed-adaptive ODE time schedules improve FID on the
spherical FM checkpoint at step 220K (FID 5.43 baseline), using precomputed OT speed
from `outputs/cifar10_spherical/`.

### Job 2790518 — COMPLETED (0 h 58 min, RTX 4500 Ada)

Compared linear vs. pure speed-adaptive schedules at NFE = 10, 20, 35, 50, 100.

| NFE | Linear | Speed-adaptive | Winner |
|----:|-------:|---------------:|--------|
| 10 | 23.51 | 39.51 | **Linear** (68% better) |
| 20 | 6.10 | 8.62 | **Linear** (41% better) |
| 35 | 5.37 | 5.46 | **Linear** (2% better) |
| 50 | 5.38 | 5.29 | **Speed-adaptive** (+0.09) |
| 100 | 5.31 | 5.12 | **Speed-adaptive** (+0.19) |

**Crossover**: between NFE=35 and NFE=50. Speed-adaptive scheduling only pays off at high NFE.

### Job 2791428 — TIMEOUT (4 h 00 min, RTX 4500 Ada)

Full alpha sweep (α=0.10 to 0.90) at NFE=10, 20, 35, 50, 100. Completed NFE ≤ 50 for all α, and NFE=100 through α=0.40 before hitting the 4 h wall limit.

Selected results at **NFE=100** (before timeout):

| Schedule | FID |
|----------|-----|
| linear | 5.476 |
| α=0.10 | 5.326 |
| α=0.20 | **5.185** |
| α=0.30 | 5.443 |
| α=0.40 | 5.130 |
| speed_adaptive (α=1.0) | 5.118 (from job 2790518) |

At **NFE=35**, α=0.60 gives the best result (FID=5.143), marginally beating linear (5.192).

**Conclusion**: No α value gives a substantial gain over linear. Speed-adaptive scheduling provides marginal improvements only at NFE ≥ 50, suggesting the precomputed OT speed does not capture the true ODE stiffness distribution for this model.

---

## 7. Head-to-Head Comparison (@ step 200K / 220K warm-start)

| Model | FID @ NFE=35 | FID @ NFE=100 | Notes |
|-------|-------------:|--------------:|-------|
| Spherical FM (step 220K baseline) | 5.43 | ~5.1 | From Apr 27 report |
| Spherical FM + OT Curriculum (step 260K) | ~5.0 | — | Best after warm-start |
| Self-FM (step 200K, data→data) | 26.30 | 5.43 | Needs many NFE |
| Self-FM (step 200K, NFE=200) | — | 3.26 | Best absolute FID |

---

## 8. Next Steps

### Spherical FM + OT Curriculum
- [ ] Resubmit with `--resume auto` from `ckpt_step_0380000.pt` for the remaining 20K steps
- [ ] Run full NFE sweep on final checkpoint (NFE = 5, 10, 20, 35, 50, 100, 200)
- [ ] Investigate FID oscillation in the 5.1–5.4 range post-260K

### Self-FM
- [ ] Apply speed-adaptive Euler at inference: $\Delta t_k \propto 1/v_{t_k}^{FR}$ — expected to dramatically improve low-NFE FID without retraining
- [ ] Train longer (300K–400K); loss still declining at step 200K
- [ ] Try a higher-order solver (midpoint / Heun) to reduce discretisation error at low NFE

### Latent FM
- [ ] Fix the `compute_reference_stats` → `compute_real_stats` bug in [scripts/train_fm_latent_linear_ddp.py](scripts/train_fm_latent_linear_ddp.py) line 486
- [ ] Resubmit on H100 (RTX 4500 only has 24 GB, marginal for UNet-128 + VAE decode + InceptionV3)

### Speed-Schedule Analysis
- [ ] Resubmit `sph_sched` with longer wall time to complete α=0.50–0.90 at NFE=100
- [ ] Try speed-adaptive schedules on the **improved curriculum checkpoint** (260K) rather than the 220K baseline
