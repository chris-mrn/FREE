# NeurIPS Paper Direction

**Based on**: experiments in `reports/sph_sched_report.md`, `reports/self_fm_report.md`,
`reports/sph_ot_curriculum_report.md`  
**Date**: 2026-04-29

---

## Core Observation

Across all experiments, one quantity keeps surfacing as the key variable:

$$v_t = \sqrt{\mathbb{E}\!\left[\left(\partial_t \log p_t(X_t)\right)^2\right]}$$

the **Fisher-Rao (FR) speed** of the probability path. It governs:

- **How hard step $t$ is to learn during training** — gradient variance at time $t$
  scales with $v_t^2$, so uniform $t$-sampling wastes budget on easy regions.
- **How stiff the ODE is at step $t$ during inference** — Euler error at step $k$
  scales with $v_{t_k} \cdot \Delta t_k$, so uniform $\Delta t$ wastes budget on
  smooth regions.

The experiments reveal these two problems are **dual**: the same quantity $v_t$ should
drive both the training sampling distribution and the inference step schedule, but no
existing work treats them jointly.

---

## Three Empirical Pillars

### 1. The NFE Crossover (sph_sched_report)

Evaluating a spherical FM model with a family of blended schedules
$t_k = (1-\alpha) t_k^{\text{linear}} + \alpha t_k^{\text{speed}}$:

| NFE | Best α | Linear FID | Best FID | Winner |
|-----|--------|------------|----------|--------|
| 10  | 0.0    | 24.15      | 24.15    | linear |
| 20  | 0.0    | 6.24       | 6.24     | linear |
| 35  | 0.60   | 5.19       | **5.14** | blend  |
| 50  | 1.0    | 5.41       | **5.12** | speed-adaptive |
| 100 | 1.0    | 5.48       | **5.12** | speed-adaptive |

There is a sharp **crossover between NFE=20 and NFE=50**. At low NFE, the Euler step
is so large that discretisation dominates everywhere — uniform spacing is optimal.
At high NFE, the error is localised to where $v_t$ peaks — concentrating steps there
gives a measurable gain. This crossover is an unexplained empirical phenomenon that
calls for a theoretical account.

### 2. The Self-FM Speed Non-Uniformity (self_fm_report)

The self-interpolant $X_t = (1-t)X_1 + t\tilde{X}_1 + \sqrt{t(1-t)}\,\varepsilon$
produces a FR speed with **2,093× dynamic range** ($v_t^{FR}$: 28 at $t=0.5$, 59,350
at $t=0.98$). This translates directly into a **70× FID gap**:

| NFE | FID (uniform Euler) |
|-----|---------------------|
| 5   | 227.26              |
| 20  | 72.51               |
| 50  | 14.37               |
| 100 | 5.43                |
| 200 | **3.26**            |

Speed-adaptive Euler ($\Delta t_k \propto 1/v_{t_k}^{FR}$) has not been applied here
yet, but the extreme non-uniformity makes it the most promising single experiment for
the paper: the prediction is that it recovers most of the gap between NFE=35 and
NFE=200 without retraining.

### 3. Curriculum Staleness (sph_ot_curriculum_report)

The OT-speed curriculum achieves a 7.4% FID improvement at step 260K (FID 5.43 →
5.03) but then degrades back toward baseline as training continues. The speed was
estimated once at step 220K and never updated. As the model improves, $v_t$ changes,
but $p_1(t)$ remains fixed — creating a distribution mismatch that accumulates.

This establishes a strong negative result: **one-shot speed estimation is insufficient**.
Periodic re-estimation (every 50K steps) is needed, and the re-estimation is cheap
(6.9 s on H100 for 1,000 $t$-points).

---

## Proposed Paper

### Title

**"Speed-Adaptive Flow Matching: A Unified Framework for Training and Inference via the Fisher-Rao Metric"**

### Core Claim

The Fisher-Rao speed $v_t^{FR}$ of a probability path is the single quantity that
optimally governs both:

1. **Training**: the time-sampling distribution $p^*(t) \propto v_t^{FR}$, which
   minimises gradient variance at fixed compute.
2. **Inference**: the ODE step schedule $\Delta t_k \propto 1/v_{t_k}^{FR}$, which
   minimises global discretisation error at fixed NFE.

These two policies are **dual** in a precise sense: the training distribution
concentrates mass where the flow is fast; the inference schedule concentrates steps
where the flow is fast. We propose a practical algorithm that estimates $v_t^{FR}$
online during training (cheap: Hutchinson trace estimator, ~7 s per call on H100)
and uses it to update both policies jointly.

### Theoretical Contributions

**Theorem 1 (Optimal training distribution).** Under squared-loss FM training with a
fixed model class, the gradient variance at time $t$ is proportional to $v_t^2$.
The variance-minimising sampling distribution under the constraint $\int p(t)dt = 1$
and fixed total steps is:
$$p^*(t) \propto v_t^{FR}(t)$$
(analogous to importance sampling optimal proposal).

**Theorem 2 (Optimal inference schedule).** For Euler integration of a flow ODE with
global truncation error budget $\epsilon$, the step-size sequence minimising
$\max_k \|\text{local error}_k\|$ is:
$$\Delta t_k \propto \frac{1}{v_{t_k}^{FR}}$$
This schedule allocates more steps to regions where the velocity field changes fastest.

**Proposition 3 (NFE crossover).** There exists a threshold $N^*(v_t)$ depending on
the dynamic range $\max v_t / \min v_t$ such that:
- For NFE $< N^*$: uniform schedule minimises FID (discretisation error dominates
  everywhere).
- For NFE $\geq N^*$: speed-adaptive schedule is optimal (error localises to
  high-$v_t$ regions).

For the spherical FM model ($\max/\min \approx 3\times$), $N^*$ falls between 20 and 35,
consistent with the observed crossover. For self-FM ($\max/\min \approx 2000\times$),
$N^*$ is very low, explaining why FID collapses dramatically even at NFE=100 with
uniform steps.

### Algorithm: Joint Speed-Adaptive FM

```
Input: dataset D, path type (linear/spherical/self), total_steps T

Phase 0 (0 → T_warm):
    Train with uniform t ~ U(0, T_MAX)

Phase k (T_{k-1} → T_k):  [periodic, e.g. every 50K steps]
    1. Estimate v_t^FR from current EMA model (7s, Hutchinson)
    2. Update training sampler: p(t) ∝ v_t^FR
    3. Cosine-blend from old p to new p over B_blend steps
    4. Continue training

Inference:
    Given v_t^FR, solve Δt_k ∝ 1/v_t^FR to allocate NFE budget
    Use Euler (or Heun) with the derived schedule
```

Key properties:
- **Cheap**: speed estimation costs ~7 s vs hours of training.
- **Adaptive**: re-estimation every 50K steps prevents staleness.
- **Training-inference consistent**: same $v_t^{FR}$ drives both policies.
- **Path-agnostic**: applies to any interpolant.

### Experiments

| Experiment | Baseline | + Speed-Adaptive | Expected Δ FID |
|------------|----------|-----------------|---------------|
| Spherical FM, NFE=50 | 5.41 | ~5.10 | −0.31 |
| Spherical FM, NFE=100 | 5.48 | ~5.10 | −0.38 |
| Self-FM, NFE=35 | 26.30 | ~8–12 (predicted) | −55%+ |
| Self-FM, NFE=100 | 5.43 | ~3.5 (predicted) | −35%+ |
| Self-FM + periodic curriculum | ~5.43@100K | <5.43 (no degradation) | TBD |
| Latent FM (pending) | TBD | TBD | TBD |

The **self-FM + speed-adaptive inference** experiment is the highest-stakes prediction:
a 2093× speed non-uniformity implies that speed-adaptive Euler should dramatically
compress the FID-NFE curve, potentially achieving FID < 10 at NFE=35 (vs current 26.30).
This is a clean, falsifiable prediction from theory.

### Ablations

1. **$p^*(t) \propto v_t$ vs $\propto 1/v_t$ vs $\propto v_t^{1/2}$** — validate theoretical
   optimum against alternatives (including the common heuristic of focusing on *slow* regions).
2. **One-shot vs periodic re-estimation** — directly tests the staleness hypothesis from
   the spherical curriculum failure.
3. **Training schedule only, inference schedule only, both** — decompose the dual benefit.
4. **FR vs OT speed measure** — test whether the two speed definitions give the same
   training/inference recommendations in practice.
5. **Linear vs spherical vs self-FM path** — show universality.

### Baselines / Related Work

- **Training**: Kingma & Gao (2023) loss-weighting by DDPM SNR; Esser et al. (2024)
  RF rectified flow; Albergo & Vanden-Eijnden (2023) stochastic interpolants.
  None use $v_t^{FR}$ adaptively during training.
- **Inference**: DPM-Solver, DEIS, UniPC — higher-order solvers but fixed linear
  schedules; do not use model-specific speed profiles.
- **Curriculum**: existing CFM curricula (this work's own sph_ot_curriculum) use
  one-shot estimation and don't connect training to inference.

### Why This is a NeurIPS Paper

1. **Theoretical novelty**: the duality between training distribution and inference
   schedule — both governed by $v_t^{FR}$ — is not in the literature.
2. **Unexplained empirical phenomena are explained**: the NFE crossover gets an
   analytic account; self-FM's NFE collapse gets both a diagnosis (2093× non-uniformity)
   and a predicted cure (speed-adaptive inference).
3. **Practical algorithm** with a single hyperparameter (re-estimation period) and
   negligible overhead (~7 s per call).
4. **Strong predicted result**: self-FM FID@35 ~8–12 (vs current 26.30) would be
   a striking demonstration of the theory. If confirmed it becomes the headline result.
5. **Clean story**: one quantity, two applications, one algorithm.

---

## What to Run Next (Priority Order)

### Experiment 1 — Speed-adaptive inference on self-FM (highest priority)

Apply $\Delta t_k \propto 1/v_{t_k}^{FR}$ at inference using the existing step-200K
checkpoint. No retraining needed. Directly tests the core theoretical prediction.

```bash
# pseudo-code
t_grid = load("fr_t_grid_step100000.npy")
v_t    = load("fr_speed_step100000.npy")
# derive step schedule from CDF of v_t
t_schedule = cdf_inverse_schedule(t_grid, v_t, n_steps=35)
# evaluate FID with this schedule on 10K samples
fid = eval_fid(ema_model, t_schedule, n_samples=10000)
```

Expected FID@35: 8–12 (vs current 26.30). If correct, this is the paper's headline.

### Experiment 2 — Periodic curriculum on spherical FM (medium priority)

Re-run the spherical + curriculum experiment but re-estimate OT speed every 50K steps
(not once). Tests the staleness fix. Expected: no degradation after 260K, continued
slow improvement toward step 400K.

### Experiment 3 — Joint training+inference on latent FM (lower priority, pending job)

Once the latent FM training completes, apply speed-adaptive inference and compare
FID-NFE curves against linear schedule. Smaller model, faster turnaround for ablations.

### Experiment 4 — $p^*(t) \propto v_t$ ablation

Replace the existing $p(t) \propto 1/v_t$ (slow-focus) with $p(t) \propto v_t$
(fast-focus, the theoretically optimal) on a fresh run of spherical FM from step 0.
Tests whether the theory's prediction ($p^* \propto v_t$, not $1/v_t$) holds.

---

## Summary

The experiments already in hand provide:

- **One strong unexplained empirical regularity** (NFE crossover)
- **One dramatic predicted fix** (self-FM low-NFE collapse via speed-adaptive inference)
- **One clear negative result with a hypothesis** (curriculum staleness → periodic re-estimation)

These three together, unified by the FR speed, form a coherent NeurIPS submission.
The highest-risk/highest-reward experiment (Experiment 1 above) requires no new training —
just running inference with a different step schedule on an existing checkpoint.
It can be done in under an hour and either confirms or refutes the central theoretical
prediction.
