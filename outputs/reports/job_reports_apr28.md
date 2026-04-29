# Job Reports — April 28, 2026

Report covering all SLURM jobs submitted around April 28, 2026.

---

## Summary Table

| Job ID  | Name         | State     | Duration | Node       | Result |
|---------|--------------|-----------|----------|------------|--------|
| 2790518 | sph_sched    | COMPLETED | 0:57:58  | u240c      | Schedule eval done — linear wins at low NFE |
| 2791355 | vae_cifar10  | COMPLETED | 1:19:17  | H100       | VAE 100K training done — rec=0.079 |
| 2791356 | latent_lfm   | FAILED    | 0:00:20  | H100       | Script file missing |
| 2791428 | sph_sched    | RUNNING   | 3h+      | u240c      | Extended alpha sweep in progress |
| 2791473 | sph_yparam   | FAILED    | 0:01:33  | RTX 4500   | CUDA OOM during JVP speed estimation |
| 2791486 | self_fm      | RUNNING   | 2h34min  | H100       | 56% done, phase=1 (curriculum active) |

---

## Job 2790518 — sph_sched — COMPLETED

**Script**: Schedule evaluation on spherical FM checkpoint (step 220K)
**Duration**: 0:57:58 on `u240c`
**Purpose**: Compare speed-adaptive ODE schedule vs. linear schedule and alpha-blended variants

### Key Results

| NFE | linear | speed_adaptive | Best alpha |
|-----|--------|---------------|------------|
|  10 | 23.508 | 39.513        | —          |
|  20 |  6.097 |  8.616        | —          |
|  35 |  5.374 |  5.455        | —          |
|  50 |  5.381 |  5.285        | —          |
| 100 |  5.309 |  5.118        | —          |

### Conclusions
- **Linear dominates at low NFE** (≤35 steps). At NFE=10, linear is 40% better than speed-adaptive (23.5 vs 39.5 FID).
- **Speed-adaptive overtakes at high NFE** (≥50 steps), gaining ~0.2 FID improvement at NFE=100.
- Crossover occurs between NFE=35 and NFE=50.
- Saved to: `outputs/cifar10_spherical/comparison/schedule_comparison.csv`

---

## Job 2791355 — vae_cifar10 — COMPLETED

**Script**: VAE training on CIFAR-10 with adversarial discriminator
**Duration**: 1:19:17 (1.32h) on H100
**Node**: `gpu-sr675-34`
**Steps**: 100,000

### Final Metrics (step 100,000)
```
rec  = 0.0788
kl   = 1561.285
disc = 0.953
```

### Latent Statistics (computed over full training set)
```
mean: [-0.2211, -0.2330,  0.1884, -1.0297]
std:  [ 0.5176,  0.5881,  0.5995,  0.6174]
```

### Saved Artifacts
- `outputs/cifar10_vae/vae_final.pt` — final weights
- `outputs/cifar10_vae/vae_best.pt` — best checkpoint (by rec loss)
- `outputs/cifar10_vae/vae_step_{20k,40k,...,100k}.pt` — interval checkpoints
- `outputs/cifar10_vae/latent_stats.pt` — mean/std for normalisation
- `outputs/cifar10_vae/vae_loss.png` — loss curves

### Assessment
Training converged well. Reconstruction loss is low (0.079). The KL=1561 is high, suggesting the 4-dim latent space is using its full capacity. Discriminator score ~0.95 indicates stable adversarial training. Latent std ~0.55–0.62 across all 4 dims — reasonably isotropic.

---

## Job 2791356 — latent_lfm — FAILED

**Script**: `examples/images/cifar10/training/train_fm_latent_linear_ddp.py`
**Duration**: 0:00:20 on `gpu-xd670-30` (H100)
**Failure time**: Apr 28 16:02:55

### Error
```
/nfs/ghome/live/cmarouani/.conda/envs/code-drifting/bin/python3.11:
  can't open file '.../train_fm_latent_linear_ddp.py': [Errno 2] No such file or directory
```

### Root Cause
The training script `train_fm_latent_linear_ddp.py` does not exist at the expected path. The SLURM submission script hard-coded a path that was never created.

### Fix Required
Create `examples/images/cifar10/training/train_fm_latent_linear_ddp.py` (or update the submission script to point to the correct existing script).

---

## Job 2791428 — sph_sched — RUNNING

**Script**: Extended schedule evaluation on spherical FM
**Started**: Apr 28 14:46
**Duration so far**: ~3h10min on `u240c`
**Purpose**: Full alpha sweep (0.10–0.90) + speed-adaptive at NFE=10,20,35,50,100

### Results So Far (complete for NFE=10,20,35,50; NFE=100 in progress)

#### NFE=10
| Schedule       | FID   |
|----------------|-------|
| linear         | 24.154|
| alpha=0.10     | 24.986|
| alpha=0.20     | 25.313|
| alpha=0.30     | 27.243|
| alpha=0.40     | 28.461|
| alpha=0.50     | 29.316|
| alpha=0.60     | 31.155|
| alpha=0.70     | 33.030|
| alpha=0.80     | 35.219|
| alpha=0.90     | 37.383|
| speed_adaptive | 39.338|

**Winner at NFE=10**: linear (24.15) — speed-adaptive 63% worse.

#### NFE=20
| Schedule       | FID  |
|----------------|------|
| linear         | 6.238|
| alpha=0.10     | 6.268|
| alpha=0.20     | 6.501|
| alpha=0.30     | 6.809|
| alpha=0.40     | 6.957|
| alpha=0.50     | 6.949|
| alpha=0.60     | 7.166|
| alpha=0.70     | 7.468|
| alpha=0.80     | 7.908|
| alpha=0.90     | 8.390|
| speed_adaptive | 8.601|

**Winner at NFE=20**: linear (6.24), alpha=0.10 essentially tied (6.27).

#### NFE=35
| Schedule       | FID  |
|----------------|------|
| linear         | 5.192|
| alpha=0.10     | 5.457|
| alpha=0.20     | 5.302|
| alpha=0.30     | 5.236|
| alpha=0.40     | 5.277|
| alpha=0.50     | 5.295|
| **alpha=0.60** |**5.143**|
| alpha=0.70     | 5.389|
| alpha=0.80     | 5.454|
| alpha=0.90     | 5.309|
| speed_adaptive | 5.556|

**Winner at NFE=35**: alpha=0.60 (5.143) — slightly better than linear (5.192).

#### NFE=50
| Schedule       | FID  |
|----------------|------|
| linear         | 5.412|
| alpha=0.10     | 5.323|
| alpha=0.20     | 5.372|
| alpha=0.30     | 5.305|
| alpha=0.40     | 5.165|
| alpha=0.50     | 5.381|
| alpha=0.60     | 5.150|
| alpha=0.70     | 5.262|
| alpha=0.80     | 5.314|
| alpha=0.90     | 5.249|
| **speed_adaptive** | **5.117**|

**Winner at NFE=50**: speed_adaptive (5.117).

#### NFE=100 (still running)
- `linear` completed: FID=5.476
- Currently evaluating `alpha=0.10...`

### Emerging Pattern
- Linear is best at NFE ≤ 20.
- At NFE=35, alpha=0.60 slightly edges out linear — transition zone.
- At NFE=50, speed_adaptive (5.117) wins — consistent with job 2790518.
- **Practical recommendation**: use linear for real-time/fast inference (NFE≤35), switch to speed-adaptive for quality-focused runs (NFE≥50).

---

## Job 2791473 — sph_yparam — FAILED

**Script**: `train_fm_spherical_y_param_ddp.py`
**Duration**: 0:01:33 on `u434a` (RTX 4500, 24GB)
**Init checkpoint**: `outputs/cifar10_spherical/fm_standard/checkpoints/ema_step_0220000.pt`

### Error
```
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 256.00 MiB.
GPU 0 has 23.53 GiB total, only 152.19 MiB free.
Process 39933 has 17.94 GiB in use (another process occupying the GPU).
```

The error occurs during `estimate_speed_grid()` which uses `jvp()` (JVP-based Fisher-Rao speed estimation) — this is called before training even starts. The JVP computation is significantly more memory-intensive than a standard forward pass.

### Stack Trace
```
speed.py:152 → _ot_speed_at_t → jvp(f, (t,), ...)
  → functional_call(model, ...)
  → unet forward → ResBlock → SiLU activation
  → OutOfMemoryError
```

### Root Causes
1. RTX 4500 (24GB) is too small for JVP-based speed estimation over the full UNet.
2. Another process was already occupying 17.94 GB of the GPU, leaving only ~5.6 GB free.

### Fixes
- Submit to H100 (`gpu_lowp` or `gatsby_ws`) instead of `gpu_lowp` (which can land on RTX 4500).
- Or reduce `B_per_t` / `chunk_size` in `estimate_speed_grid()` to lower JVP memory footprint.
- Or add `--constraint=h100` to the SLURM submission.

---

## Job 2791486 — self_fm — RUNNING

**Script**: Self-FM CIFAR-10 (standard 100k phase + FR speed arc-length curriculum)
**Started**: Apr 28 15:24
**Node**: `gpu-sr675-34` (H100 80GB)
**Total steps**: 200,001
**Current**: ~111,930 / 200,001 (56%), phase=1, ~1h54min remaining

### Interpolant
```
X_t = (1-t)*X1 + t*X1_tilde + sqrt(t(1-t))*eps,   t ∈ [0.01, 0.99]
```
Where `X1_tilde` is the model's self-prediction of the data — a self-referential interpolant.

### Curriculum Schedule
- Steps 0–100,000: standard training (phase=0)
- Step 100,000: FR speed computed (took 10.8s)
  - Speed range: `[28.35, 59350.59]` (highly non-uniform — most mass at high-speed regions)
- Steps 100,000–200,001: arc-length curriculum training (phase=1)

### FR Speed at Curriculum Start (step 100,000)
```
Epoch ranges (unnormalised speed):
  [3.1, 66749.9]  [7.3, 65745.5]  [6.6, 67587.9]  [10.4, 66484.8]  [6.6, 67032.0]
Smoothed range: [28.35, 59350.59]
```

### Loss Trajectory
| Step    | Loss  | Phase |
|---------|-------|-------|
| ~0      | 1.82  | 0     |
| 50,000  | 0.531 | 0     |
| 100,000 | 0.518 | 0     |
| 111,930 | ~0.49 | 1     |

Loss decreased from 0.518 → 0.49 after the curriculum switch — the FR-weighted sampling is providing a harder, more informative distribution.

### Checkpoints Saved
- `outputs/cifar10_self_fm/checkpoints/ema_step_0050000.pt`
- More at 100K and 200K (pending)

### Assessment
Healthy training. The loss drop after curriculum activation suggests the self-FM model is benefiting from speed-weighted sampling. Results will be available after step 200K.

---

## Cross-Job Summary

### What Completed Successfully
1. **VAE training** is done — latent space is ready for latent FM experiments.
2. **Schedule comparison** (2790518) gives a clear picture: linear for low NFE, adaptive for high NFE.

### What Failed and Why
1. **latent_lfm**: Simple script path error — easy fix.
2. **sph_yparam**: OOM on RTX 4500 during JVP speed estimation — needs H100.

### What is Still Running
1. **sph_sched (2791428)**: Extended NFE=100 alpha sweep — will give full picture of alpha interpolation.
2. **self_fm (2791486)**: Novel self-referential interpolant + curriculum — interesting experiment, ~2h left.

### Next Steps
1. Fix `latent_lfm` script path and resubmit.
2. Resubmit `sph_yparam` on H100 with `--constraint=h100` or on `gpu_lowp` with explicit H100 request.
3. Wait for `self_fm` to finish, then evaluate FID.
4. Evaluate VAE-latent FM pipeline once the latent training script is fixed.
