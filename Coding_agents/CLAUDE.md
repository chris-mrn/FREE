# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## About

TorchCFM is a PyTorch library for **Conditional Flow Matching (CFM)** — simulation-free training of continuous normalizing flows (CNFs). The upstream `torchcfm/` package provides loss functions and OT plan samplers. This repo extends it with a unified training framework (`train.py`) for studying speed-adaptive t-sampling on CIFAR-10 and 2D toy datasets.

## Commands

```bash
# Install (editable)
pip install -e .

# Run all tests
pytest tests/

# Run a single test file
pytest tests/test_conditional_flow_matcher.py -v

# Run tests excluding slow ones
pytest tests/ -m "not slow"

# Lint / format
ruff check torchcfm/
ruff format torchcfm/

# Pre-commit
pre-commit run --all-files
```

**Line length**: 99 (configured in `pyproject.toml`).

**Note**: `pyproject.toml` passes `--doctest-modules` to pytest, so all public docstrings with `>>>` examples are automatically tested.

### Unified training entry point (`train.py`)

```bash
# Linear FM on CIFAR-10, uniform t, single GPU
PYTHONPATH=. python train.py --path linear --dataset cifar10 --out_dir outputs/myrun

# Spherical FM, OT coupling, precomputed speed-weighted t-sampling
PYTHONPATH=. python train.py --path spherical --dataset cifar10 --coupling ot \
    --t_mode weighted --speed_type ot --speed_dir outputs/cifar10_spherical \
    --out_dir outputs/sph_weighted

# Spherical FM + curriculum, 3-GPU DDP
PYTHONPATH=. torchrun --nproc_per_node=3 train.py \
    --path spherical --dataset cifar10 --coupling ot \
    --t_mode curriculum --speed_type ot --ddp \
    --out_dir outputs/sph_curriculum

# 2D toy experiment
PYTHONPATH=. python train.py --path linear --dataset 8gaussians --out_dir outputs/2d_test
```

Training always auto-resumes (`--resume auto`) from the latest checkpoint in `out_dir/checkpoints/`. Pass `--resume disabled` to start fresh.

Output directories follow the naming convention: `{dataset}_{path}_{coupling}_{t_mode}`.

**Outputs per run (`{out_dir}/`):**

| File | Contents |
|------|----------|
| `checkpoints/ckpt_step_*.pt` | Model, EMA, optimizer, scheduler, curriculum state |
| `samples/step_*.png` | 8×8 sample grids (image datasets) |
| `loss.csv` | step, loss_raw, loss_ema, t_phase |
| `metrics.csv` | step, fid, kid_mean, kid_std, is_mean, is_std |
| `nfe_fid.csv` | NFE sweep results at end of training |
| `speed_t_grid_step*.npy` / `speed_v_t_step*.npy` | Speed profiles estimated during curriculum |

### Key train.py arguments

| Flag | Choices / type | Default | Notes |
|------|----------------|---------|-------|
| `--path` | `linear`, `spherical` | `linear` | Interpolation geometry |
| `--dataset` | `cifar10`, `8gaussians`, `40gaussians`, `moons`, `circles`, `checkerboard` | — | Required |
| `--coupling` | `independent`, `ot` | `independent` | Minibatch coupling |
| `--t_mode` | `uniform`, `weighted`, `curriculum` | `uniform` | Time-sampling strategy |
| `--speed_type` | `ot`, `fr`, `score` | `ot` | Speed measure; `score` is precomputed-only |
| `--speed_dir` | path | `None` | Dir with `.npy` files for weighted mode |
| `--curriculum_start` | int | `100_000` | Step to estimate speed and begin blending |
| `--curriculum_blend` | int | `25_000` | Steps to blend uniform → speed-adaptive |
| `--curriculum_restarts` | int | `0` | Extra speed re-estimations after first (0=none) |
| `--curriculum_restart_every` | int | `50_000` | Steps between restarts |
| `--total_steps` | int | `400_001` | Training steps |
| `--batch_size` | int | `128` | Per-GPU |
| `--ddp` | flag | off | Multi-GPU via `torchrun` |

Speed estimation hyperparameters (curriculum + 2D weighted):

| Flag | Default | Notes |
|------|---------|-------|
| `--speed_n_t` | `100` | Grid points for speed estimation |
| `--speed_B` | `512` | Batch size per speed epoch |
| `--speed_epochs` | `3` | Epochs over reference set |
| `--speed_hutch` | `4` | Hutchinson probes (FR speed only) |
| `--speed_smooth` | `0.05` | Gaussian smoothing bandwidth in t-units for CdfSampler |

### Slurm

Ready-made scripts live in `slurm/`. Submit with `sbatch slurm/<script>.sh`. The conda env is `code-drifting`; the repo root must be on `PYTHONPATH`.

### Pre-trained weights

`weights/fm_cifar10_weights_step_80000.pt` — early CIFAR-10 checkpoint (step 80K). Uses:
- Speed estimation via `speed/cifar10_speed.py --ckpt weights/fm_cifar10_weights_step_80000.pt`
- Fine-tuning: copy to `out_dir/checkpoints/` and auto-resume
- Inference statistics: evaluate with `eval_fid_nfe.py --ckpt_dir weights`

See **User_guide.md** for detailed usage examples (section "Using Pre-Trained Weights").

### Standalone scripts (`scripts/`)

- `eval_fid_nfe.py` — sweeps NFE list over saved checkpoints, merges with `metrics.csv`, writes `nfe_fid_table.csv`.
- `eval_self_fm_nfe.py` — evaluates self-FM NFE sweep (data-to-data, integrates backward from real images).
- `compute_reparam_div.py` / `compute_reparam_div_2d.py` — computes reparameterised divergence under arc-length schedule.
- `train_vae_cifar10.py` / `train_fm_latent_linear_ddp.py` — latent FM experiments over a trained CIFAR-10 VAE.
- `compute_latent_fr_speed.py` — estimates Fisher-Rao speed on a trained latent FM model using Hutchinson divergence estimator (with configurable `--speed_hutch` probes, default 5). Outputs `fr_speed.npy`, `fr_weighting.npy`, `t_grid.npy`.
- `plot_latent_speed.py` — visualizes speed profile: 3-panel plot of speed v_t^FR, weighting w(t), and arc-length schedule α(t). Includes summary statistics (CV, density ratio, deviation from uniform).

Scripts in `scripts/` and `kaggle/` use `sys.path.insert(0, repo_root)` for cluster/Kaggle portability — this is intentional, do not remove.

### Self-FM (`examples/images/cifar10/training/train_self_fm.py`)

A distinct experiment where **both source and target are data** (no Gaussian noise):

```
X_t = (1-t)*X1 + t*X1_tilde + sqrt(t*(1-t)) * eps
u_t = X1_tilde - X1 + dsigma_dt * eps
```

`X1` and `X1_tilde` are two independent CIFAR-10 samples; `eps ~ N(0,I)`. `t` is clipped to `[T_MIN=0.01, T_MAX=0.99]` to avoid the `dsigma_dt` singularity. Uses FR-speed curriculum (same blend logic as `train.py`). Inference integrates **backward** from `t=T_MAX` (≈ `X1_tilde`) to `t=T_MIN` starting from real images. Outputs land in `outputs/cifar10_self_fm/`.

## Architecture

### Core library (`torchcfm/`)

**`conditional_flow_matching.py`** — All CFM variants share:

```python
t, xt, ut = fm.sample_location_and_conditional_flow(x0, x1)
# Pass return_noise=True to get (t, xt, ut, epsilon)
```

Class hierarchy:
- `ConditionalFlowMatcher` — base: independent coupling, Gaussian path `N(t·x1 + (1-t)·x0, σ)`
  - `ExactOptimalTransportConditionalFlowMatcher` — re-pairs `(x0, x1)` via exact OT
  - `TargetConditionalFlowMatcher` — Lipman et al. 2023; source fixed to `N(0, I)`
  - `SchrodingerBridgeConditionalFlowMatcher` — entropically regularised OT
  - `VariancePreservingConditionalFlowMatcher` — trigonometric interpolation

To add a new CFM variant: subclass `ConditionalFlowMatcher` and override `compute_mu_t`, `compute_sigma_t`, and/or `compute_conditional_flow`.

**`optimal_transport.py`** — `OTPlanSampler` wraps [POT](https://pythonot.github.io/). Methods: `"exact"`, `"sinkhorn"`, `"unbalanced"`, `"partial"`.

**`utils.py`** — `pad_t_like_x(t, x)`: always use before multiplying `t` (shape `(bs,)`) with data `x` (shape `(bs, *dim)`). Also provides `torch_wrapper` for torchdyn ODE solver compatibility.

**`torchcfm/models/`** — Reusable UNet and MLP building blocks (referenced by `models/models.py`).

### Unified training framework (top-level modules)

All modules are imported via `PYTHONPATH=.` from the repo root; `train.py` is the single entry point.

**`path/path.py`** — Interpolation path definitions, path-agnostic interface:

| Class | `xt(t, x0, x1)` | `ut(t, x0, x1)` | `T_MAX` |
|---|---|---|---|
| `LinearPath` | `(1-t)x0 + t·x1` | `x1 - x0` | `1.0` |
| `SphericalPath` | `cos(t)x0 + sin(t)x1` | `-sin(t)x0 + cos(t)x1` | `π/2` |

- `euler_sample(model, path, n_samples, n_steps, x0_shape, device)` — uniform-step Euler; default `n_steps=35` in `run_image_eval`.
- `euler_sample_schedule(model, path, t_schedule, x0_shape, device)` — speed-adaptive Euler using a custom `t_schedule` array (steps proportional to `1/v_t`).

**`speed/speed.py`** — Speed estimation and t-samplers:

- **`UniformSampler`** — samples `t ~ Uniform(0, T_MAX)`.
- **`CdfSampler`** — samples `t ~ q(t) ∝ v_t` via inverse-CDF; built from `(t_grid, v_t)` numpy arrays with Gaussian smoothing.
- **`BlendedSampler(a, b)`** — mixes two samplers; call `.sample(n, device, mix)` where `mix=0` → all `a`, `mix=1` → all `b` (used during curriculum blending via `cosine_blend`).
- **`make_cdf_sampler(t_grid, v_t, T_MAX, smooth_sigma)`** — convenience wrapper around `CdfSampler`.
- **`estimate_speed_grid(model, path, t_grid, x1_ref, B, n_epochs, speed_type, device)`** — estimates speed on a grid from a trained model. `speed_type='ot'` uses `torch.func.jvp` (forward-mode AD); `speed_type='fr'` uses the Hutchinson divergence estimator. `speed_type='score'` raises `ValueError` — precomputed only.
- **`load_precomputed(speed_dir, path_name, speed_type)`** — loads `.npy` files from `outputs/cifar10/` or `outputs/cifar10_spherical/`.

**Time derivative convention**: always use `torch.func.jvp` for ∂_t of model outputs; never finite differences.

**`datasets/datasets.py`** — `get_cifar10_loaders`, `collect_images`, `sample_2d`. Images: `cifar10`. 2D: `8gaussians`, `40gaussians`, `moons`, `circles`, `checkerboard`.

**`models/models.py`** — Root-level `build_model(args, device)` factory used by `train.py`: dispatches to `MLP2D` (2D datasets) or `UNetModelWrapper` (image datasets, imported from `torchcfm/models/unet/`).

**`metrics/metrics.py`** — `InceptionMetrics`: FID, KID, Inception Score via pretrained InceptionV3. Input images must be in `[-1, 1]`. KID is multiplied by 1000.

**`utils/helpers.py`** — `ema_update`, `warmup_lr`, `cosine_blend`, `save_ckpt`/`find_last_ckpt`, `setup_ddp`, `run_image_eval`, `run_2d_eval`.

### Training modes

`train.py` supports three t-sampling modes (`--t_mode`):

| Mode | Behaviour |
|---|---|
| `uniform` | `t ~ Uniform(0, T_MAX)` throughout |
| `weighted` | Load precomputed speed from `--speed_dir`, then sample `t ~ CdfSampler` |
| `curriculum` | Uniform until `--curriculum_start`; estimate speed online from EMA model; cosine blend over `--curriculum_blend` steps; optional restarts |

Curriculum state (phase, speed curves, last estimation step) is checkpointed and resumed automatically.

### Precomputed speed files

Located in `outputs/cifar10/` (linear) and `outputs/cifar10_spherical/` (spherical). File-name conventions: `t_grid.npy`, `ot_weighting.npy`, `fr_weighting.npy`, `score_weighting.npy` (linear); `t_grid_sph.npy`, `ot_speed_sph.npy`, etc. (spherical). Generate them with the scripts in `speed/`.

### Key conventions

- **Time `t`** ∈ `[0, T_MAX]`, shape `(bs,)`. `T_MAX=1.0` for linear, `π/2` for spherical.
- **`x0`** is the source (Gaussian noise); **`x1`** is the data.
- Model interface: `model(t, x) → velocity`, where both `t` and `x` have a batch dimension.
- OT coupling is applied in flat space: `x0/x1` are flattened, paired, then reshaped.
- DDP: `torchrun --nproc_per_node=N train.py --ddp ...`; speed is estimated on rank 0 and broadcast via `dist.broadcast`.
