# Latest Progress — April 28, 2026

## Active SLURM Jobs

| Job ID | Name | Partition | State | Runtime | Time Limit | Node |
|--------|------|-----------|-------|---------|------------|------|
| 2784435 | `sph_ot_curr` | gpu_lowp | RUNNING | ~4h 10min | 14h | gpu-sr675-34 |
| 2790518 | `sph_sched` | gatsby_ws | RUNNING | ~18 min | 4h | u240c |
| 2790513 | `vae_cifar10` | gpu_lowp | PENDING | — | 10h | (Resources) |
| 2790514 | `latent_lfm` | gpu_lowp | PENDING | — | 6h | (Dependency on 2790513) |

---

## Job Details & Current Progress

---

### [RUNNING] 2784435 — `sph_ot_curr`: Spherical FM + OT Curriculum (DDP × 3)

**Goal:** Train spherical FM with an adaptive OT-speed curriculum starting from the pre-trained step-220K spherical baseline.

**Setup:**
- Path: $X_t = \cos(t) X_0 + \sin(t) X_1$, $t \in [0, \pi/2]$
- DDP × 3 GPUs (H100), effective batch = 384
- Curriculum: `speed_step=100K`, `blend=25K steps`
- Resumed from `outputs/cifar10_spherical/fm_standard/checkpoints/ema_step_0220000.pt`

**Current state (as of ~13:17):**
- At step **~238,000 / 400,000** (≈ 60% complete)
- Currently in **Phase 1** (blending Uniform → p1 over steps 220K–245K)
- Current training loss: **~0.1038**
- Speed: ~9–11 steps/s, estimated total ~12.4h

**OT speed estimated at step 220K:**
- Speed range (raw): [0.831, 8.344] (across 5 epochs)
- Speed range (smoothed): [1.031, 3.327]
- Estimation took 6.9s

**FID trajectory (at step 35 NFE, evaluated every 20K steps):**

| Step | FID | KID | IS |
|------|-----|-----|----|
| 240K | 5.317 | 2.984 ± 0.561 | 4.995 ± 0.156 |
| 260K | **5.033** | 2.727 ± 0.549 | 5.044 ± 0.136 |
| 280K | 5.130 | 2.565 ± 0.694 | 4.973 ± 0.135 |
| 300K | 5.227 | 3.093 ± 0.695 | 5.050 ± 0.111 |
| 320K | 5.273 | 3.339 ± 0.786 | 4.960 ± 0.093 |
| 340K | 5.441 | 3.327 ± 0.680 | 4.951 ± 0.149 |
| 360K | 5.406 | 3.398 ± 0.660 | 4.976 ± 0.123 |

Latest checkpoint: `ckpt_step_0360000.pt`

**Observation:** Best FID is at 260K (5.033), slightly before curriculum kicks in. FID slightly degraded after 280K — need to monitor whether it recovers in phase 3 (pure p1 sampling, after step 245K).

---

### [RUNNING] 2790518 — `sph_sched`: Spherical FM Speed-Adaptive Schedule Evaluation

**Goal:** Compare linear vs speed-adaptive ODE integration schedule on the pre-trained step-220K spherical baseline.

**Checkpoint:** `ema_step_0220000.pt` (FID@35 ≈ 5.37 on this eval run)

**Setup:**
- Loaded pre-computed OT speed: $v_t \in [1.909, 5.663]$ over 201 time points
- Speed-adaptive schedule: time steps allocated proportional to $1/v_t$ (more steps where speed is high)

**Results so far (from sched_eval_2790518.log):**

| NFE | Linear FID | Speed-Adaptive FID |
|-----|------------|-------------------|
| 10 | 23.508 | 39.513 |
| 20 | 6.097 | 8.616 |
| 35 | 5.374 | still running… |

**Observation:** Speed-adaptive schedule is **worse** than linear so far at low NFE (10, 20). This is noteworthy — the speed-adaptive schedule concentrates steps at high-speed $t$ regions, but the spherical Euler integrator may not benefit from this. Still waiting for NFE=35 and above.

Job was previously cancelled at step 2790486 (ran on too-small GPU, NVIDIA T1000 4GB). Current run is on RTX 4500 Ada (24GB).

---

### [PENDING] 2790513 — `vae_cifar10`: Train VAE for Latent FM

**Goal:** Train a full convolutional VAE on CIFAR-10 for use in latent FM training.

**Status:** Waiting for GPU resources (gpu_lowp partition).

**Architecture:** Encoder $3\times32\times32 \to 4\times8\times8$, Decoder symmetric.
Loss: L1 + KL ($\beta=10^{-6}$) + VGG perceptual + PatchGAN adversarial (starts at 10K steps).

---

### [PENDING] 2790514 — `latent_lfm`: Latent FM Training

**Goal:** Train the full flow matching model in VAE latent space.

**Status:** Pending — depends on job 2790513 (VAE must finish first).

**Prior test run (job 2774515, Apr 27):**
- Used Stability AI's pre-trained VAE (`stabilityai/sd-vae-ft-mse`) as a proxy
- Latent shape: $(4, 4, 4) = 64$ dims (from $32\times32$ CIFAR-10)
- 100 steps test: FID@35 = **141.9** (as expected, too few steps)
- FM model: 14.9M params, LR=2e-4
- Latents pre-cached to disk: 50K × (4,4,4)

---

## Completed Experiments

---

### OT-CFM Linear FM Training — COMPLETED ✅ (job 2771684)

Trained OT-CFM with linear interpolation ($X_t = (1-t)X_0 + tX_1$, $t \in [0,1]$) for 400K steps on 4× H100. Finished Mon Apr 27 at 08:27 UTC after 9h 42min.

**FID progression (35 steps, 10K samples):**

| Step | FID@35 | IS |
|------|--------|----|
| 20K | 343.2 | 1.20 |
| 40K | 56.9 | 2.89 |
| 60K | 10.9 | 4.51 |
| 100K | 9.0 | 4.94 |
| 200K | 7.65 | 4.99 |
| 300K | **7.44** | 4.91 |
| 400K | 8.02 | 4.91 |

Best FID: **7.44 at step 300K** (slight regression to 8.02 by 400K).

**Final NFE sweep (step 400K, EMA weights):**

| NFE | FID | IS |
|-----|-----|----|
| 10 | 26.2 | 3.49 |
| 20 | 12.4 | 4.32 |
| 35 | 7.85 | 4.79 |
| 50 | 6.46 | 5.03 |
| 100 | **5.58** | 5.31 |

Latest checkpoint: `outputs/cifar10_ot_linear/fm_ot/checkpoints/ema_step_0400000.pt`
Summary plot: `outputs/cifar10_ot_linear/fm_ot/training_summary.png`

---

### Spherical FM Baseline — COMPLETED ✅ (job 2768198 / 2774706)

Trained standard spherical FM ($X_t = \cos(t)X_0 + \sin(t)X_1$, $t \in [0, \pi/2]$) for 220K steps on RTX 4500 Ada (24GB). Single GPU.

**FID vs Steps (NFE=35, 10K samples):**

| Step | FID@35 | IS |
|------|--------|----|
| 20K | 322.6 | 1.19 |
| 40K | 57.9 | 2.93 |
| 60K | 9.21 | 4.30 |
| 80K | 6.69 | 4.77 |
| 100K | 6.04 | 4.98 |
| 120K | 5.84 | 5.03 |
| 140K | 5.56 | 5.10 |
| 160K | 5.46 | 5.19 |
| 180K | 5.45 | 5.21 |
| 200K | 5.51 | 5.12 |
| 220K | **5.43** | 5.25 |

Best checkpoint: `ema_step_0220000.pt` (FID@35 = 5.43). Used as baseline for curriculum and schedule experiments.

**FID vs NFE at step 220K:**

| NFE | FID |
|-----|-----|
| 35 | 5.43 |
| 50 | ~5.27 |
| 100 | ~5.46 |

**Speed curves (Apr 26):**
- Analytical OT / Score / FR speeds computed and saved: `ot_speed_sph.npy`, `score_speed_sph.npy`, `fr_speed_sph.npy`
- OT speed range: [1.909, 5.663] over 201 pts, $t \in [0.01, 0.90]$

---

## Next Steps / Open Questions

1. **sph_ot_curr (2784435):** Will FID recover below 5.033 in Phase 1+ (after 245K)? Need to wait for 380K–400K evaluation.
2. **sph_sched (2790518):** Does speed-adaptive schedule help at high NFE (100+)? Currently looks negative at low NFE.
3. **latent_lfm (2790514):** Full training run pending VAE completion. Will use custom VAE (not stability AI proxy).
4. **Consider:** Whether speed-adaptive schedule should use $1/v_t$ (focus on slow regions) vs $v_t$ (focus on fast regions) — current result suggests the formulation may need revisiting for spherical geometry.
