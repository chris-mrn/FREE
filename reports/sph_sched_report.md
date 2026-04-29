# Spherical FM Schedule Evaluation Report
**Job 2791428** — Running as of Apr 28, 2026 (3h17m elapsed)
**Node**: `u240c.id.gatsby.ucl.ac.uk`
**GPU**: NVIDIA RTX 4500 Ada Generation

---

## Experiment Description

Full alpha-sweep schedule evaluation on the **spherical FM** checkpoint at step 220K.

The experiment asks: *can we do better than a linear ODE timestep schedule by blending it with a speed-adaptive one?*

**Schedule family**: For each alpha ∈ {0.0, 0.1, …, 0.9, 1.0}:
```
t_schedule = (1 - alpha) * linear + alpha * speed_adaptive
```
- `alpha=0.0` = pure linear (uniform steps in t)
- `alpha=1.0` = pure speed-adaptive (steps proportional to OT speed v_t)
- Intermediate values blend both

**Checkpoint**: `outputs/cifar10_spherical/fm_standard/checkpoints/ema_step_0220000.pt`
**Speed source**: Pre-computed OT speed, t_grid=[0.01, 0.90], v_t ∈ [1.91, 5.66]
**Output dir**: `outputs/cifar10_spherical/schedule_comparison_mixed/`
**Evaluated NFE**: 10, 20, 35, 50, 100
**Sample count**: 10,000 per config

---

## Results

### NFE = 10

| Schedule | FID ↓ | KID (×10⁻⁴) | IS |
|----------|-------|-------------|-----|
| **linear (α=0)** | **24.154** | 20.65 | 3.977 |
| alpha=0.10 | 24.986 | 22.26 | 3.787 |
| alpha=0.20 | 25.313 | 21.91 | 3.798 |
| alpha=0.30 | 27.243 | 24.43 | 3.782 |
| alpha=0.40 | 28.461 | 25.23 | 3.760 |
| alpha=0.50 | 29.316 | 25.96 | 3.723 |
| alpha=0.60 | 31.155 | 28.39 | 3.656 |
| alpha=0.70 | 33.030 | 30.67 | 3.598 |
| alpha=0.80 | 35.219 | 32.49 | 3.497 |
| alpha=0.90 | 37.383 | 34.30 | 3.485 |
| speed_adaptive (α=1) | 39.338 | 36.72 | 3.422 |

Winner: **linear** — speed-adaptive is **63% worse** at NFE=10.

### NFE = 20

| Schedule | FID ↓ | KID (×10⁻⁴) | IS |
|----------|-------|-------------|-----|
| **linear (α=0)** | **6.238** | 2.63 | 4.757 |
| alpha=0.10 | 6.268 | 2.54 | 4.783 |
| alpha=0.20 | 6.501 | 2.56 | 4.738 |
| alpha=0.30 | 6.809 | 2.92 | 4.785 |
| alpha=0.40 | 6.957 | 3.27 | 4.722 |
| alpha=0.50 | 6.949 | 3.13 | 4.691 |
| alpha=0.60 | 7.166 | 3.19 | 4.745 |
| alpha=0.70 | 7.468 | 3.58 | 4.691 |
| alpha=0.80 | 7.908 | 3.82 | 4.670 |
| alpha=0.90 | 8.390 | 4.40 | 4.585 |
| speed_adaptive (α=1) | 8.601 | 4.55 | 4.573 |

Winner: **linear** (6.24). alpha=0.10 is essentially tied (6.27 — within noise).

### NFE = 35

| Schedule | FID ↓ | KID (×10⁻⁴) | IS |
|----------|-------|-------------|-----|
| linear (α=0) | 5.192 | 2.46 | 4.981 |
| alpha=0.10 | 5.457 | 2.70 | 5.035 |
| alpha=0.20 | 5.302 | 2.49 | 5.010 |
| alpha=0.30 | 5.236 | 2.50 | 5.013 |
| alpha=0.40 | 5.277 | 2.34 | 4.903 |
| alpha=0.50 | 5.295 | 2.45 | 5.031 |
| **alpha=0.60** | **5.143** | **2.08** | 5.050 |
| alpha=0.70 | 5.389 | 2.44 | 4.970 |
| alpha=0.80 | 5.454 | 2.38 | 4.878 |
| alpha=0.90 | 5.309 | 2.26 | 4.986 |
| speed_adaptive (α=1) | 5.556 | 2.47 | 4.918 |

Winner: **alpha=0.60** (5.143) — slightly better than linear (5.192). Transition zone.

### NFE = 50

| Schedule | FID ↓ | KID (×10⁻⁴) | IS |
|----------|-------|-------------|-----|
| linear (α=0) | 5.412 | 2.95 | 5.089 |
| alpha=0.10 | 5.323 | 2.72 | 5.102 |
| alpha=0.20 | 5.372 | 2.74 | 5.151 |
| alpha=0.30 | 5.305 | 2.78 | 5.153 |
| alpha=0.40 | 5.165 | 2.57 | 5.121 |
| alpha=0.50 | 5.381 | 2.78 | 5.053 |
| alpha=0.60 | 5.150 | 2.62 | 5.014 |
| alpha=0.70 | 5.262 | 2.65 | 5.022 |
| alpha=0.80 | 5.314 | 2.48 | 5.081 |
| alpha=0.90 | 5.249 | 2.51 | 5.039 |
| **speed_adaptive (α=1)** | **5.117** | **2.40** | 5.053 |

Winner: **speed_adaptive** (5.117). Also alpha=0.40 (5.165) and alpha=0.60 (5.150) are competitive.

### NFE = 100 (in progress)

| Schedule | FID ↓ | Status |
|----------|-------|--------|
| linear (α=0) | 5.476 | done |
| alpha=0.10 | — | running |
| ... | — | pending |

---

## Summary: Best FID Per NFE

| NFE | Best Schedule | Best FID | Linear FID | Δ FID |
|-----|--------------|----------|------------|-------|
| 10  | linear       | 24.154   | 24.154     | 0.000 |
| 20  | linear       | 6.238    | 6.238      | 0.000 |
| 35  | alpha=0.60   | 5.143    | 5.192      | -0.049 |
| 50  | speed_adaptive | 5.117 | 5.412      | -0.295 |
| 100 | TBD          | —        | 5.476      | — |

---

## Analysis

### Why linear wins at low NFE

At low NFE (≤20), the ODE integrator takes large steps. Linear spacing distributes function evaluations uniformly in t, which maximizes step-level accuracy for a simple Euler solver. The spherical OT speed profile for this checkpoint is relatively flat (v_t ∈ [1.91, 5.66] — only 3× dynamic range), which means the speed-adaptive schedule does not offer a meaningful advantage but does hurt by concentrating steps in a narrow region.

### Why speed-adaptive wins at high NFE

At NFE=50+, the integrator has enough budget to follow the velocity field faithfully. Speed-adaptive scheduling concentrates evaluation points where the velocity changes fastest (near t~0 for spherical FM), squeezing more accuracy from the available budget. At NFE=100 this effect is expected to be larger.

### The alpha=0.60 sweet spot (NFE=35)

At NFE=35, the best schedule is alpha=0.60 — a 40% linear / 60% speed-adaptive blend. This is the crossover regime where neither pure schedule dominates and a hybrid works best. The improvement over linear is small (0.05 FID) but consistent with the trend.

### Practical recommendations

| Use case | Recommended schedule | Rationale |
|----------|---------------------|-----------|
| Fast inference (NFE ≤ 20) | **linear** | Clearly best; simple and fast |
| Quality with moderate budget (NFE ~35) | **alpha=0.60** | Marginal gain; may not be worth tuning |
| Quality-focused (NFE ≥ 50) | **speed_adaptive** | Measurable improvement (~0.3 FID at NFE=50) |

---

## Comparison with Job 2790518

Job 2790518 (same experiment, earlier run on `u240c`) produced slightly different absolute numbers (linear NFE=10: 23.508 vs 24.154 here). This is expected — both used the same checkpoint but different random seeds for sample generation. The **qualitative conclusions are identical**: linear wins at low NFE, speed-adaptive at high NFE.

---

## Status

Job still running (NFE=100 alpha sweep in progress). Estimated completion: ~30-45 minutes from now. Full results will be in `outputs/cifar10_spherical/schedule_comparison_mixed/`.
