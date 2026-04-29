# Project Description — What I've Implemented

## 1. Speed Computation

Three speed measures characterizing how fast the probability path $p_t$ evolves:

### Fisher-Rao Speed
$$v_t^{FR} = \sqrt{\mathbb{E}[(\partial_t \log p_t(X_t))^2]}$$
- Computed via the **Hutchinson trace estimator** with Rademacher probes ($z \sim \pm1$): estimate $\text{div}(v_\theta)$ via VJPs, then aggregate over probes.
- Used at training time (curriculum) and as a post-hoc analysis tool.
- Key function: `fr_speed_hutchinson()` in `examples/images/cifar10/speed/speed_analysis.py`.

### OT Speed
$$v_t^{OT} = \sqrt{\mathbb{E}[\|u_t(X_t)\|^2]}$$
- Computed via JVP (`torch.func.jvp`) of the marginal velocity field.
- The marginal velocity is approximated as a softmax mixture over a reference batch.
- Key file: `examples/images/cifar10/speed/cifar10_speed.py`.

### Score Speed
$$v_t^{Score} = \sqrt{\mathbb{E}[\|\partial_t s_t(X_t)\|^2]}$$
- Closed-form for Gaussian paths, Hutchinson otherwise.
- `examples/images/cifar10/speed/compare_speed_methods.py` compares all three.

All speed curves are saved as `.npy` files in `outputs/cifar10/`: `{key}_speed.npy`, `{key}_weighting.npy`, `{key}_schedule.npy`, alongside `t_grid.npy`.

---

## 2. Batch Approximation Study

`examples/2D_tutorials/batch_approx_report.py` — a controlled study on 2D toy distributions (moons, circles, 8-Gaussians, checkerboard) quantifying how well mini-batch speed estimates approximate the full-batch ground truth. Sweeps over batch sizes $B \in \{32, \ldots, 1024\}$, 15 trials each. Reports MAE, RMSE, relative RMSE, Pearson $r$. Validates whether online speed estimation (during training) is reliable.

---

## 3. Training with Speed-Based Weighting

`examples/images/cifar10/training/train_fm_weighted.py` — modifies the FM loss to:

$$L(\theta) = \mathbb{E}_{t \sim U[0,1]}\left[w(t) \cdot \|v_\theta(t, X_t) - u_t\|^2\right]$$

where $w(t) = L / v_t^{OT}$ is precomputed and normalized so $\text{mean}(w)=1$. The `WeightInterp` class loads the `.npy` weight file and interpolates online during training.

An equivalent importance-sampling formulation (sample $t \sim p(t) \propto w(t)$, use unweighted loss) is implemented via `TimeSampler` (inverse-CDF) in `train_comparison.py`.

---

## 4. Curriculum Training (Adaptive Schedule from Live Model)

`examples/images/cifar10/training/train_fm_curriculum.py` and its DDP variant — compute the FR speed from the **EMA model during training** at milestone steps, then gradually shift the time-sampling distribution to focus on slow regions.

### Algorithm

| Phase | Steps | Sampling |
|---|---|---|
| 0 | 0 → 50K | Uniform $t \sim U[0,1]$ |
| 1 | 50K → 75K | Cosine blend: $(1-\alpha)U + \alpha p_1(t)$ |
| 2 | 75K → 100K | Cosine blend: $(1-\alpha)p_1 + \alpha p_2(t)$ |
| 3 | 100K+ | Pure $p_2(t)$ |

$p_i(t) \propto 1/v_t^{FR}$ derived from the EMA model at step $i$. Cosine blend: $\alpha = \frac{1 - \cos(\pi \cdot \text{progress})}{2}$.

### FR Estimation Details
`compute_fr_speed_curve` uses:
- 5 independent sweeps over a $t$-grid of 1000 points
- Hutchinson trace with 5 probes per point
- Gaussian smoothing ($\sigma=3$) and clipping

`InverseCDFSampler` stores the CDF for cheap online inverse sampling. Curriculum state (speed curves, phase) is checkpointed and resumed.

### Key Hyperparameters
- `--speed_step1=50000`, `--speed_step2=75000` (milestone steps)
- `--blend_steps=25000` (cosine blend duration)
- `--fr_n_t=1000`, `--fr_n_epochs=5`, `--fr_n_hutch=5`
- `--fr_B_per_t=2`, `--fr_chunk=128`, `--fr_smooth=3.0`, `--fr_n_ref=2000`

---

## 5. Latent Space FM (VAE + FM)

`examples/images/cifar10/training/train_vae_cifar10.py` — trains a convolutional VAE to compress CIFAR-10 ($3\times32\times32$) down to $4\times8\times8 = 256$ dims. The VAE uses:
- **L1 reconstruction loss** (primary)
- **KL divergence** with $\beta = 10^{-6}$ (near-deterministic encoder)
- **VGG-16 perceptual loss**
- **PatchGAN adversarial loss** (VQGAN-style adaptive weighting, starts at step 10K)
- KL warmup over 5K steps

Saves `latent_stats.pt` (per-channel mean/std for normalization). The FM trains in this latent space; generation decodes back through the VAE decoder. Artifacts in `outputs/cifar10_latent/`.

---

## 6. Spherical Flow Matching

`examples/images/cifar10/training/train_fm_spherical.py` — replaces the linear interpolation path with a **spherical (great-circle) path**:

$$X_t = \cos(t) \cdot X_0 + \sin(t) \cdot X_1, \quad t \in [0, \pi/2]$$

Conditional velocity: $u_t(X_t | X_0, X_1) = -\sin(t) X_0 + \cos(t) X_1$.

Sampling uses Euler integration from $t=0$ to $t=\pi/2$. Speed analysis for this geometry is in `examples/images/cifar10/speed/cifar10_speed_spherical.py` and `examples/2D_tutorials/speed_weighting_schedule_spherical.py`, with outputs in `outputs/cifar10_spherical/`.

The DDP+curriculum variant `train_fm_spherical_curriculum_ddp.py` uses **OT speed** (finite differences of $u_\theta$) as the speed measure, with a single-phase cosine blend curriculum at step 100K.

---

## Files Map

```
torchcfm/                        ← core library (mostly upstream)
examples/images/cifar10/
  speed/
    speed_analysis.py            ← FR + Score speed (Hutchinson)
    cifar10_speed.py             ← OT speed (JVP)
    cifar10_speed_all.py         ← all three speeds
    cifar10_speed_spherical.py   ← spherical OT speed
    compare_speed_methods.py     ← comparison plots
  training/
    train_fm_weighted.py         ← weighted loss training
    train_comparison.py          ← IS-sampled t training
    train_fm_curriculum.py       ← adaptive FR curriculum (single GPU)
    train_fm_curriculum_ddp.py   ← adaptive FR curriculum (DDP)
    train_fm_spherical.py        ← spherical FM
    train_fm_spherical_curriculum_ddp.py  ← spherical + OT curriculum
    train_vae_cifar10.py         ← VAE for latent FM
examples/2D_tutorials/
  batch_approx_report.py         ← mini-batch speed bias study
  speed_weighting_schedule.py    ← 2D weighting analysis
  speed_weighting_schedule_spherical.py   ← spherical 2D
outputs/
  cifar10/                       ← speed .npy files (linear)
  cifar10_spherical/             ← speed .npy files (spherical)
  cifar10_curriculum/            ← curriculum checkpoints + speed curves
  cifar10_latent/                ← VAE + latent FM outputs
```
