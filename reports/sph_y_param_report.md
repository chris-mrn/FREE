# Experiment Report — Spherical FM Y-Parametrization

**Started**: 2026-04-28  
**Status**: 🟡 Running

---

## Motivation

Standard CFM on the spherical path trains the model to predict the conditional
velocity $\partial_t X_t$.  At inference with a non-uniform (speed-adaptive)
schedule, the step sizes $\Delta t_k$ vary but the model output has the same
scale everywhere in $t$.

The Y-parametrization rewrites the target so that the model is consistent with
the dynamics of the **time-reparametrised process** $Y_s = X_{\alpha(s)}$, where
$\alpha(s) = F^{-1}(s)$ and $F$ is the CDF of $q(t) \propto v_t$.

---

## Theory

### Speed measure

$$v_t = \sqrt{\mathbb{E}\!\left[\left\|\partial_t u_t^\theta(X_t)\right\|^2\right]}$$

computed once from the EMA checkpoint at step 220K via **JVP** (forward-mode AD):

```python
from torch.func import jvp, functional_call
_, du_dt = jvp(f, (t,), (torch.ones_like(t),))
```

### Training objective

$$\mathcal{L}^Y(\theta) = \mathbb{E}\!\left[\left\|u_t^\theta(X_t) -
\frac{1}{\sqrt{\tilde{v}_t}}\,\partial_t X_t\right\|^2\right]$$

where $\tilde{v}_t = v_t / \bar{v}$ is normalised so that $\text{mean}(1/\sqrt{\tilde{v}_t}) = 1$,
keeping the loss scale stable relative to the standard objective.

### Annealing schedule

| Step range | $\beta$ | Target |
|---|---|---|
| 0 – 5K | 0 | Standard: $\partial_t X_t$ |
| 5K – 55K | $0 \to 1$ (linear) | Blend: $\partial_t X_t \cdot [(1-\beta) + \beta/\sqrt{\tilde{v}_t}]$ |
| 55K – 200K | 1 | Full Y-param: $\partial_t X_t / \sqrt{\tilde{v}_t}$ |

### Inference modes

Once trained, the model predicts $u^\theta \approx u_t / \sqrt{\tilde{v}_t}$.
Three evaluation modes are tracked at every checkpoint:

| Mode | Integration rule | When correct |
|---|---|---|
| `linear_raw` | $x \mathrel{+}= u^\theta \cdot dt$ | Only at $\beta=0$ |
| `linear_rescaled` | $x \mathrel{+}= u^\theta \cdot \sqrt{\tilde{v}_t} \cdot dt$ | $\beta>0$, compensates the scale |
| `speed_adaptive` | non-uniform $\Delta t_k \propto 1/v_{t_k}$, raw $u^\theta$ | $\beta>0$, schedule absorbs the scale |

---

## Configuration

| Parameter | Value |
|---|---|
| Init checkpoint | `ema_step_0220000.pt` (FID@35 = 5.43) |
| Path | Spherical: $X_t = \cos(t) X_0 + \sin(t) X_1$,  $t \in [0, \pi/2]$ |
| Coupling | Independent |
| t-sampling | Uniform $U[0, \pi/2]$ |
| Total steps | 200 001 (from step 0, i.e. 220K+200K effective) |
| Batch size | 128 × 3 GPUs = 384 |
| LR | 2e-4 × 3 = 6e-4 (linear scaling) |
| Anneal start / steps | 5 000 / 50 000 |
| NFE at eval | 35 (all 3 modes) |
| Eval every | 20 000 steps |

---

## Speed Profile (estimated at step 0 of this run, from step-220K EMA)

<!-- Filled in after job starts -->

| Metric | Value |
|---|---|
| t range | TBD |
| v_t min | TBD |
| v_t max | TBD |
| v_t mean | TBD |
| Estimation time | TBD |

---

## FID Trajectory

<!-- Filled in as training progresses -->

| Step | β | FID (linear-raw) | FID (linear-rescaled) | FID (speed-adaptive) |
|------|---|---|----|---|
| 0 (init, 220K) | — | — | — | 5.43 |
| 20 000 | 0.30 | TBD | TBD | TBD |
| 40 000 | 0.70 | TBD | TBD | TBD |
| 60 000 | 1.00 | TBD | TBD | TBD |
| 80 000 | 1.00 | TBD | TBD | TBD |
| 100 000 | 1.00 | TBD | TBD | TBD |
| 120 000 | 1.00 | TBD | TBD | TBD |
| 140 000 | 1.00 | TBD | TBD | TBD |
| 160 000 | 1.00 | TBD | TBD | TBD |
| 180 000 | 1.00 | TBD | TBD | TBD |
| 200 000 | 1.00 | TBD | TBD | TBD |

**Baseline (standard spherical FM, step 220K):**
- FID@35 (linear, 10K samples) = **5.43**
- FID@50 = ~5.27,  FID@100 = ~5.46

---

## Hypothesis

- `linear_raw` FID should **degrade** as β→1 (model no longer predicts $u_t$ directly)
- `linear_rescaled` and `speed_adaptive` FID should **improve or match** the baseline
  if the Y-parametrization provides better conditioning of the velocity field
- The key question: does normalising the velocity by $1/\sqrt{v_t}$ reduce the
  numerical integration error at low NFE?

---

## Results (to be filled in)

### Final FID at step 200K

| Mode | NFE=10 | NFE=20 | NFE=35 | NFE=50 | NFE=100 |
|---|---|---|---|---|---|
| linear_raw | TBD | TBD | TBD | TBD | TBD |
| linear_rescaled | TBD | TBD | TBD | TBD | TBD |
| speed_adaptive | TBD | TBD | TBD | TBD | TBD |
| **baseline** | 23.51 | 6.10 | 5.37 | 5.38 | 5.31 |

### Analysis

TBD after training completes.

---

## Files

| File | Description |
|---|---|
| `speed_t_grid.npy` | t-grid used for v_t estimation |
| `speed_v_t.npy` | Estimated OT speed from step-220K EMA |
| `loss.csv` | step, loss_raw, loss_ema, beta |
| `metrics.csv` | step, mode, fid, kid_mean, kid_std, is_mean, is_std |
| `checkpoints/` | EMA checkpoints every 20K steps |
| `samples/` | Image grids (one per mode per eval step) |
| `train_*.log` | SLURM log |
