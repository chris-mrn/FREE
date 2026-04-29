# Experiment Report — Spherical FM + OT-Speed Curriculum (Job 2784435)

**Date:** April 28, 2026
**Status:** CANCELLED (manually, after 380K steps reached)
**Node:** gpu-sr675-34 (NVIDIA H100 80GB HBM3 × 3)
**Wall time used:** ~4h 20min out of 14h limit

---

## Setup

### Goal
Test whether an OT-speed-adaptive curriculum — shifting the time-sampling distribution toward slow regions of the flow — improves generation quality over the spherical FM baseline trained with uniform $t$.

### Starting Point
Resumed from the **pre-trained spherical FM baseline** at step 220K:
- Checkpoint: `outputs/cifar10_spherical/fm_standard/checkpoints/ema_step_0220000.pt`
- Baseline FID@35 at step 220K: **5.43**

### Flow Matching Setup
- **Path:** $X_t = \cos(t) \cdot X_0 + \sin(t) \cdot X_1$, $t \in [0, \pi/2]$
- **Target velocity:** $u_t = -\sin(t) X_0 + \cos(t) X_1$
- **Loss:** standard unweighted MSE $\mathbb{E}[\|v_\theta(t, X_t) - u_t\|^2]$
- **Model:** UNet, 35.7M params
- **DDP:** 3 × H100, effective batch size = 384
- **LR:** $2\times10^{-4} \times 3 = 6\times10^{-4}$, EMA decay = 0.9999
- **Total planned steps:** 400K

---

## Curriculum Algorithm

### OT Speed Estimation
Computed at the start (from the step-220K checkpoint, before any new training):

- Grid: 1000 time points, $t \in [0.01, 0.99]$
- 5 independent epochs, $B_{\text{per\_t}} = 2$ samples/point, `chunk=128`
- Method: finite-differences on the marginal velocity field (JVP, `eps_fd=0.001`)
- Estimation time: **6.9 seconds**

Raw speed range across 5 epochs:

| Epoch | Min | Max |
|-------|-----|-----|
| 1 | 0.880 | 4.351 |
| 2 | 0.928 | 6.343 |
| 3 | 0.831 | 4.894 |
| 4 | 0.979 | 5.823 |
| 5 | 0.853 | 8.344 |

After Gaussian smoothing (σ=3): **[1.031, 3.327]**

Speed saved to: `ot_speed_step220000.npy`, `ot_t_grid_step220000.npy`

### Sampling Distribution
From smoothed OT speed $v_t$, derived:
$$p_1(t) \propto \frac{1}{v_t}$$
(focus more training on slow $t$ regions)

Implemented via inverse-CDF sampling (`InverseCDFSampler`).

### Phase Schedule

| Phase | Steps | Sampling | Details |
|-------|-------|----------|---------|
| 1 | 220K → 245K | Cosine blend: $(1-\alpha)U + \alpha p_1(t)$ | $\alpha = \frac{1-\cos(\pi\cdot\text{progress})}{2}$ |
| 2 | 245K → 400K | Pure $p_1(t)$ | Full OT-speed-adaptive sampling |

(Note: the baseline was already trained for 220K steps with uniform $t$, so the "Phase 0 uniform" pre-dates this run.)

---

## Training Progress

### Loss Evolution

| Step | Loss | Phase | Elapsed |
|------|------|-------|---------|
| 220K (start) | 0.1095 | 1 (blend) | 0 min |
| 240K | 0.1027 | 1 (blend) | 30.7 min |
| 260K | 0.1024 | 2 (pure p1) | 62.7 min |
| 280K | 0.1017 | 2 | 94.6 min |
| 300K | 0.1017 | 2 | 126.6 min |
| 320K | 0.1014 | 2 | 158.6 min |
| 340K | 0.1021 | 2 | 191.2 min |
| 360K | 0.1016 | 2 | 223.3 min |
| 380K | **0.0998** | 2 | 256.3 min |

Loss decreased slowly and consistently throughout (~8.8% total reduction from 0.1095 to 0.0998 over 160K steps). No instability or spikes on curriculum transition at 245K.

Training speed: ~9–11 steps/s (3 × H100).

---

## FID Results (NFE=35, 10K samples)

Evaluated every 20K steps.

| Step | Curriculum FID | Baseline FID | Δ vs baseline |
|------|----------------|--------------|--------------|
| 220K | 5.43 (baseline start) | 5.43 | — |
| 240K | 5.317 | — | — |
| 260K | **5.033** | — | — |
| 280K | 5.130 | — | +0.10 vs 260K |
| 300K | 5.227 | — | — |
| 320K | 5.273 | — | — |
| 340K | 5.441 | — | — |
| 360K | 5.406 | — | — |
| 380K | 5.343 | — | — |

**Best achieved:** FID = **5.033** at step 260K (just 15K into phase 2).

For reference, the spherical baseline plateaued around **5.43–5.55** between steps 180K–220K without the curriculum.

---

## KID and IS Results

| Step | KID mean | KID std | IS mean | IS std |
|------|----------|---------|---------|--------|
| 240K | 2.984 | ±0.561 | 4.995 | ±0.156 |
| 260K | **2.727** | ±0.549 | **5.044** | ±0.136 |
| 280K | 2.565 | ±0.694 | 4.973 | ±0.135 |
| 300K | 3.093 | ±0.695 | 5.050 | ±0.111 |
| 320K | 3.339 | ±0.786 | 4.960 | ±0.093 |
| 340K | 3.327 | ±0.680 | 4.951 | ±0.149 |
| 360K | 3.398 | ±0.660 | 4.976 | ±0.123 |
| 380K | 3.211 | ±0.581 | 4.851 | ±0.155 |

KID lowest (best) at 280K (2.565) despite FID not being best there. IS relatively stable throughout (~5.0).

---

## Analysis & Interpretation

### What Worked
- The curriculum did produce a clear initial improvement: FID dropped from 5.43 (baseline) to **5.033 at step 260K**, a **-7.4%** improvement.
- Training was numerically stable throughout — no loss spikes on curriculum transitions.
- Speed estimation at step 220K was fast (6.9s) and gave a smooth, plausible OT speed curve.

### What Didn't Work
- After the initial improvement at 260K, FID degraded progressively to 5.44 at step 340K — **worse than the baseline** the curriculum started from.
- FID partially recovered at 360K (5.406) and 380K (5.343), suggesting oscillation rather than monotone decay.
- The best result (5.033 at 260K) was only 15K steps into phase 2 (pure $p_1$ sampling). This suggests the curriculum has a **short beneficial window** before overfitting to the biased distribution.

### Hypothesis for FID Degradation
1. **Distribution mismatch accumulation:** Sampling heavily from slow-$t$ regions biases gradient updates. Over many steps, this distorts the velocity field at under-sampled $t$ values (fast regions), degrading the overall flow quality.
2. **Speed computed once, gets stale:** The OT speed was computed at step 220K and used for all subsequent steps. As the model improves, the speed profile changes, but $p_1(t)$ does not adapt — creating an increasingly incorrect prior.
3. **$p(t) \propto 1/v_t$ may be wrong for spherical geometry:** In spherical FM, the OT speed $v_t$ is high near $t = \pi/2$ (the data end). Focusing training away from this region might hurt the model's ability to cleanly arrive at the data distribution.

### Comparison with Baseline

| | Spherical FM Baseline | Spherical FM + OT Curriculum |
|--|----------------------|------------------------------|
| Best FID@35 | ~5.27 (step 180K, NFE=50) | **5.033** (step 260K) |
| FID@35 at 200K | 5.51 | — |
| FID@35 at 220K | 5.43 | — (start) |
| FID@35 plateau | ~5.43 | ~5.3–5.4 (later steps) |
| Training stability | Stable | Stable |

---

## Artifacts

| File | Description |
|------|-------------|
| `outputs/cifar10_spherical_curriculum/metrics.csv` | FID/KID/IS per 20K steps |
| `outputs/cifar10_spherical_curriculum/loss.csv` | Per-100-step loss log |
| `outputs/cifar10_spherical_curriculum/train_2784435.log` | Full training log |
| `outputs/cifar10_spherical_curriculum/ot_speed_step220000.npy` | OT speed curve used for curriculum |
| `outputs/cifar10_spherical_curriculum/ot_t_grid_step220000.npy` | Time grid for speed curve |
| `outputs/cifar10_spherical_curriculum/speed_step220000.png` | Speed curve plot |
| `outputs/cifar10_spherical_curriculum/checkpoints/` | Checkpoints at 320K, 340K, 360K |
| `outputs/cifar10_spherical_curriculum/samples/` | Sample grids at 240K–360K |
| `outputs/cifar10_spherical_curriculum/loss_analysis.png` | Loss curve visualization |

---

## Conclusions & Next Steps

**Conclusion:** The OT-speed curriculum offers a short-term boost (~7% FID improvement in the first 40K steps after curriculum start) but leads to long-term degradation when $p_1(t)$ is fixed. The spherical + curriculum approach is promising but needs refinement.

**Suggested next steps:**
1. **Re-estimate speed periodically** (e.g., every 50K steps) instead of once — prevents the distribution from going stale.
2. **Weaker curriculum (lower blend fraction):** Instead of blending fully to $p_1$, cap at e.g. 50% blend to preserve coverage of all $t$.
3. **Use $p(t) \propto v_t$** (focus on fast regions) instead of $1/v_t$ — test the opposite hypothesis for spherical geometry.
4. **Evaluate with more NFE** (50, 100) at step 260K checkpoint — the best FID checkpoint may show more benefit at higher NFE.
5. **Train from scratch** with curriculum from step 0 rather than resuming at step 220K — less risk of prior model being entrenched.
