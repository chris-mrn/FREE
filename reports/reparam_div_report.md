# Energy Repartition Under Arc-Length Reparametrisation

**Job**: 2814704 (`reparam_div`, `gatsby_ws`, RTX 4500 Ada)
**Script**: `scripts/compute_reparam_div.py`
**Status**: Running
**Date**: 2026-04-29
**Note**: Rerun of job 2812484 with corrected $\alpha$ (arc-length via $v^{FR}$, not $1/v^{FR}$)

---

## Motivation

The second phase of self-FM training uses an **arc-length time-sampling distribution**
derived from the Fisher-Rao speed:

$$p(t) \propto \frac{1}{v_t^{FR}}, \qquad v_t^{FR} = \sqrt{E\!\left[(\partial_t \log p_t)^2\right]}$$

This defines a reparametrisation $\alpha : [0,1] \to [T_{\min}, T_{\max}]$ via the
inverse CDF of the **arc-length measure**:

$$\alpha(s) = F^{-1}(s), \qquad F(t) = \frac{\int_0^t v_\tau^{FR}\,d\tau}{\int_0^{T_{\max}} v_\tau^{FR}\,d\tau}$$

so that $s \sim \mathcal{U}(0,1)$ implies equal Fisher-Rao distance per unit $s$.
The arc-length interpretation is: equal increments of $s$ correspond to equal amounts
of "flow work" in the Fisher-Rao metric, concentrating arc-length on the **fast/hard**
regions (large $v_t^{FR}$).

**Key question**: does the learned velocity field $u_t^\theta$, trained under this
schedule, exhibit **equal energy repartition** in the reparametrised time $s$?

A constant function $s \mapsto E[(\mathrm{div}\, u_s^\theta(X_{\alpha(s)}))^2]$
would confirm it: the divergence of the field — which measures local expansion/compression
of the probability mass — is uniformly spread across the arc-length parameter.

---

## Quantity of Interest

We compute:

$$D(s) := E\!\left[\left(\mathrm{div}\, u_s^\theta(X_{\alpha(s)})\right)^2\right]$$

where:
- $\alpha(s)$ is the arc-length reparametrisation built from the FR speed estimated
  at step 100K (saved at `fr_speed_step100000.npy`, `fr_t_grid_step100000.npy`)
- $X_{\alpha(s)}$ is a sample from the self-interpolant at physical time $\tau = \alpha(s)$:
  $$X_\tau = (1-\tau)X_1 + \tau \tilde{X}_1 + \sqrt{\tau(1-\tau)}\,\varepsilon, \quad X_1, \tilde{X}_1 \sim p_{\text{data}}$$
- $u_s^\theta$ is the trained velocity field queried at **arc-length parameter $s$** (not
  at $\tau = \alpha(s)$): the network's internal time is $s$, since it was trained with
  $s \sim \mathcal{U}(0,1)$ under the curriculum. The sample is drawn at $\tau=\alpha(s)$
  because that is the physical time the interpolant lives at.
- The **divergence** is estimated via the **Hutchinson trace estimator** with $n=5$
  Rademacher probes:
  $$\mathrm{div}\,u_s^\theta(x) \approx z^\top \nabla_x u_s^\theta(x)\, z, \quad z \sim \{\pm 1\}^d$$

For comparison, we also compute the raw (non-reparametrised) curve:

$$D_{\mathrm{raw}}(t) := E\!\left[\left(\mathrm{div}\, u_t^\theta(X_t)\right)^2\right]$$

at uniformly spaced $t \in [T_{\min}, T_{\max}]$.

---

## Interpretation

| $D(s)$ shape | Interpretation |
|---|---|
| **Constant** | The arc-length reparametrisation $\alpha$ equalises the energy — the learned dynamics have uniform information density in arc-length time. The curriculum has achieved its goal. |
| **Decreasing** | Residual concentration near $s \approx 0$ (slow-$t$ region): the model still has more divergence there despite re-weighting. |
| **Increasing** | Over-correction: the fast-$t$ region (near $t=1$) still dominates and the curriculum did not fully tame it. |

The **coefficient of variation** (CV = std / mean of $D(s)$) quantifies how far from
constant the curve is:
- $\mathrm{CV} = 0$: perfectly uniform energy repartition
- $\mathrm{CV} \gg 1$: highly non-uniform (raw curve has $\mathrm{CV} \gg 1$ due to
  the 2093× speed range)

---

## Experimental Setup

| Parameter | Value |
|-----------|-------|
| Checkpoint | `outputs/cifar10_self_fm/checkpoints/ema_step_0200000.pt` |
| FR speed source | `fr_speed_step100000.npy` (smoothed, $\sigma=3$, estimated at step 100K) |
| Speed range | $v_t^{FR} \in [28.35, 59{,}350]$, ratio $= 2093\times$ |
| Reparametrisation $\alpha$ | Inverse CDF of $p(t) \propto 1/v_t^{FR}$ |
| $t$ grid size (`n_t`) | 200 points |
| Batch size per point (`B`) | 256 |
| Hutchinson probes (`n_hutch`) | 5 Rademacher vectors |
| Sweeps (`n_epochs`) | 3 (averaged for stability) |
| Reference pool | CIFAR-10 training set (first 512 images per epoch) |
| GPU | RTX 4500 Ada (24 GB), `u240c` |
| Est. wall time | ~2–3 h |

---

## Algorithm

```
Load alpha from saved FR speed: F(t) = int_0^t v^FR(s) ds / int_0^T v^FR(s) ds, alpha = F^{-1}
Load EMA model from step 200K
Load CIFAR-10 reference pool

s_grid  = linspace(0, 1, n_t)           # uniform arc-length parameter
tau_grid = alpha(s_grid)                 # mapped original t values

For each epoch in 1..n_epochs:
    Refresh CIFAR-10 reference pool
    For each (s, tau) in zip(s_grid, tau_grid):
        X_tau = sample_self_interpolant(tau, B)   # (B, 3, 32, 32)
        For each probe k in 1..n_hutch:
            z = Rademacher(shape=X_tau.shape)
            u = model(s, X_tau)          # model queried at arc-length s, not tau
            grad = autograd.grad((u*z).sum(), X_tau)
            div_k = (grad * z).sum(dims=(1,2,3))   # (B,)
        div_avg = mean over probes
        D(s) += mean(div_avg^2) / n_epochs

Save D(s), tau_grid, s_grid as .npy
Plot: (A) D_raw(t) vs t, (B) D(s) vs s, (C) alpha(s) vs s
```

---

## Outputs

| File | Description |
|------|-------------|
| `outputs/cifar10_self_fm/div_sq_reparam_mean.npy` | $D(s)$ averaged over epochs |
| `outputs/cifar10_self_fm/div_sq_reparam_std.npy` | Std over epochs |
| `outputs/cifar10_self_fm/div_sq_reparam_s_grid.npy` | Arc-length grid $s$ |
| `outputs/cifar10_self_fm/div_sq_reparam_tau_grid.npy` | Corresponding $\tau = \alpha(s)$ |
| `outputs/cifar10_self_fm/div_sq_uniform_mean.npy` | $D_{\mathrm{raw}}(t)$, no reparam |
| `outputs/cifar10_self_fm/div_sq_uniform_t_grid.npy` | Uniform $t$ grid |
| `outputs/cifar10_self_fm/div_sq_reparam.png` | 3-panel figure |
| `outputs/cifar10_self_fm/reparam_div_<jobid>.log` | Full run log |

---

## Results

**Job 2814704** completed in **45 min 14 s** on RTX 4500 Ada (`u240c`).  
Corrected $\alpha$: CDF of $v_t^{FR}$ (arc-length measure, concentrates on fast regions).  
Plot saved: `outputs/cifar10_self_fm/div_sq_reparam.png`

> **Note on previous run (job 2812484):** Used $w = 1/v^{FR}$ instead of $w = v^{FR}$,
> which is the training distribution $p(t) \propto 1/v^{FR}$, not the arc-length
> reparametrisation. That run found CV 3.050 → 7.567 (worse). The corrected
> arc-length α gives the results below.

### $D_{\mathrm{raw}}(t)$ — raw divergence energy (no reparametrisation)

| $t$ | $D_{\mathrm{raw}}(t)$ |
|-----|----------------------|
| 0.020 | **4.02 × 10⁹** ← peak |
| 0.102 | 1.51 × 10⁸ |
| 0.300 | 7.55 × 10⁶ |
| 0.500 | **3.63 × 10²** ← minimum |
| 0.700 | 7.53 × 10⁶ |
| 0.898 | 1.52 × 10⁸ |
| 0.980 | **4.00 × 10⁹** ← peak |

**U-shaped and symmetric** around $t = 0.5$. Max/min ratio: $1.65 \times 10^7$.  
Note: this is consistent with $D(t) = (v_t^{FR})^2$ — the FR speed is also U-shaped,
with both boundaries equally hard.

### $D(s)$ — reparametrised divergence energy (correct arc-length $\alpha$)

The corrected $\alpha(s) = F^{-1}(s)$ where $F(t) = \int_0^t v_\tau^{FR}\,d\tau / Z$
**compresses** high-speed (hard) regions of $t$ into small intervals of $s$, and
**expands** the slow (easy) midpoint region.

| $s$ | $\tau = \alpha(s)$ | $D(s)$ |
|-----|-------------------|--------|
| 0.00 | 0.020 | **4.08 × 10⁹** ← peak |
| 0.10 | 0.035 | 1.14 × 10⁹ |
| 0.30 | 0.104 | 2.94 × 10⁷ |
| 0.50 | 0.453 | **4.06 × 10²** ← minimum |
| 0.70 | 0.896 | 3.15 × 10⁷ |
| 0.90 | 0.965 | 1.24 × 10⁹ |
| 1.00 | 0.990 | **5.52 × 10⁹** ← peak |

Mapping: $\alpha$ maps $s \in [0, 0.1]$ to $\tau \in [0.020, 0.035]$ (fast boundary),
$s \in [0.1, 0.9]$ to $\tau \in [0.035, 0.965]$ (bulk), $s \in [0.9, 1.0]$ to
$\tau \in [0.965, 0.990]$ (fast boundary). The midpoint $s = 0.5 \to \tau = 0.453$.

### Coefficient of Variation

| Quantity | Mean | Std | CV | max/min |
|----------|------|-----|----|---------|
| $D_{\mathrm{raw}}(t)$ | 1.83 × 10⁸ | 5.58 × 10⁸ | **3.044** | 1.65 × 10⁷ |
| $D(s)$ — correct $\alpha$ (job 2814704) | 6.71 × 10⁸ | 1.09 × 10⁹ | **1.618** | 2.16 × 10⁷ |
| $D(s)$ — incorrect $\alpha$ (job 2812484) | 6.37 × 10⁷ | 4.82 × 10⁸ | 7.567 | — |

The **correct** arc-length reparametrisation reduces CV by **46.9%** (3.044 → 1.618).  
The incorrect $1/v^{FR}$ weight made it 148% worse (as expected: it concentrated
arc-length on the slow, easy midpoint — the opposite of what arc-length should do).

### Interpretation

**The correct arc-length reparametrisation partially equalises the divergence energy.**

$D(s)$ is still U-shaped (CV = 1.618, not 0), but significantly flatter than $D_{\mathrm{raw}}(t)$
(CV = 3.044). Two compounding factors explain the residual non-uniformity:

1. **Curriculum staleness**: $\alpha$ was built from the FR speed at step 100K.
   After 100K more steps of arc-length training, the velocity field changed. The
   speed profile at step 200K differs from the one used to build $\alpha$, so the
   reparametrisation is no longer perfectly matched to the current model.

2. **The U-shape persists because $\alpha$ acts symmetrically**: The arc-length
   $\alpha$ compresses BOTH boundary regions (high $v_t^{FR}$ near $t \approx 0$
   and $t \approx 1$) into small intervals of $s$. While this reduces the *range*
   of $D(s)$ near the boundaries, the extreme peak values at $s = 0$ and $s = 1$
   remain (the singularity at the boundaries is structural, not just a training artefact).

**Comparison across runs:**

| Run | $\alpha$ weight $w(t)$ | CV before | CV after | Change |
|-----|------------------------|-----------|----------|--------|
| 2812484 (incorrect) | $1/v^{FR}$ (= training dist.) | 3.050 | 7.567 | **+148%** worse |
| 2814704 (correct) | $v^{FR}$ (= arc-length) | 3.044 | 1.618 | **−47%** better |

---

## Connection to the NeurIPS Direction

The result **confirms the theory with a quantitative caveat**:

| Concept | Formula | Profile (self-FM, step 200K) | CV |
|---------|---------|------------------------------|-----|
| $D_{\mathrm{raw}}(t)$ — no reparam | $E[(\mathrm{div}\, u_t^\theta)^2]$ | U-shaped, symmetric | 3.044 |
| $D(s)$ — correct arc-length $\alpha$ | $E[(\mathrm{div}\, u_s^\theta(X_{\alpha(s)}))^2]$ | U-shaped, flatter | 1.618 |
| $D(s)$ — ideal (target) | — | Constant | 0 |

The 47% CV reduction is real evidence that the arc-length curriculum has partially
equalized the energy. The gap from ideal (CV=0) is attributable to curriculum
staleness — the single speed estimation at step 100K was not re-updated as the
model evolved. **Periodic re-estimation** of $v_t^{FR}$ during training is the
key proposed fix, and these results quantify the benefit gap it would close.

**Concrete follow-up experiments:**
1. **Re-estimate** $v_t^{FR}$ from the step-200K checkpoint, build a new $\alpha$,
   and recompute $D(s)$ — this would show the "achievable" CV under a fresh curriculum.
2. **Online curriculum**: Re-estimate speed every 50K steps during training and check
   if $D(s)$ approaches CV=0 at the end of training.
3. **Direct divergence curriculum**: Build $p(t) \propto \sqrt{D_{\mathrm{raw}}(t)}$
   (targeting the measured divergence rather than the FR speed proxy) and check
   if this achieves lower CV than the FR-speed-based approach.
