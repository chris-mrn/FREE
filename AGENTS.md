# TorchCFM — Agent Instructions

TorchCFM is a PyTorch library for **Conditional Flow Matching (CFM)** — simulation-free training of continuous normalizing flows. See [README.md](README.md) for the full project overview and [CLAUDE.md](CLAUDE.md) for additional Claude-specific guidance.

## Build & Test

```bash
pip install -e .                        # install from source (editable)
pytest tests/                           # run all tests
pytest tests/ -m "not slow"             # skip slow tests
pytest tests/test_conditional_flow_matcher.py -v  # single file
ruff check torchcfm/                    # lint
ruff format torchcfm/                   # format
pre-commit run --all-files              # full pre-commit suite
```

**Critical**: `pyproject.toml` passes `--doctest-modules` to pytest. All public docstrings with `>>>` examples are automatically executed as tests — keep them runnable. Line length limit: 99.

## Architecture

| Layer | Directory | Purpose |
|-------|-----------|---------|
| Core library | `torchcfm/` | CFM losses, OT samplers, utilities |
| Unified trainer | `train.py` | Main entrypoint for all experiments |
| Path abstractions | `path/` | `LinearPath`, `SphericalPath`, Euler solvers |
| Speed estimation | `speed/` | t-samplers, FR/OT speed measures |
| Datasets | `datasets/` | CIFAR-10 loaders, 2D toy data |
| Models (root) | `models/` | `build_model()` factory + `MLP2D` for `train.py` |
| Architectures | `torchcfm/models/` | Reusable UNet and MLP building blocks |
| Metrics | `metrics/` | InceptionV3 FID / KID / IS |
| Runner (legacy) | `runner/` | Hydra + PyTorch Lightning framework |
| Examples | `examples/` | Notebooks and standalone scripts |
| Tests | `tests/` | pytest unit tests |
| Slurm jobs | `slurm/` | H100 cluster batch scripts |

## Unified Trainer (`train.py`)

The root-level `train.py` is the **primary entry point** for all experiments. It supports multiple paths, datasets, couplings, and time-sampling strategies in one CLI.

### Key arguments

| Flag | Choices | Default | Notes |
|------|---------|---------|-------|
| `--path` | `linear`, `spherical` | `linear` | Interpolation geometry |
| `--dataset` | `cifar10`, `8gaussians`, `40gaussians`, `moons`, `circles`, `checkerboard` | — | Required |
| `--coupling` | `independent`, `ot` | `independent` | Minibatch coupling |
| `--t_mode` | `uniform`, `weighted`, `curriculum` | `uniform` | Time-sampling strategy |
| `--speed_type` | `ot`, `fr`, `score` | `ot` | Speed measure for weighted/curriculum |
| `--out_dir` | path | — | Output directory |
| `--resume` | `auto`, path, `disabled` | `auto` | Checkpoint resumption |
| `--ddp` | flag | off | Enable multi-GPU via `torchrun` |

### Example commands

```bash
# Linear FM, CIFAR-10, single GPU
python train.py --path linear --dataset cifar10 --coupling independent \
    --t_mode uniform --total_steps 400000 --out_dir outputs/cifar10

# Spherical + OT + curriculum, multi-GPU
torchrun --nproc_per_node=3 train.py --path spherical --dataset cifar10 \
    --coupling ot --t_mode curriculum --speed_type fr \
    --total_steps 400000 --ddp --out_dir outputs/cifar10_sph_ot_curriculum
```

### Outputs per run (`{out_dir}/`)

| File | Contents |
|------|----------|
| `checkpoints/ckpt_step_*.pt` | Model, EMA, optimizer, scheduler, curriculum state |
| `samples/step_*.png` | 8×8 sample grids |
| `loss.csv` | step, loss_raw, loss_ema, t_phase |
| `metrics.csv` | step, fid, kid_mean, kid_std, is_mean, is_std |
| `nfe_fid.csv` | NFE sweep results |
| `speed_t_grid_step*.npy` / `speed_v_t_step*.npy` | Speed profiles |

Output dirs follow naming convention: `{dataset}_{path}_{coupling}_{t_mode}`.

## Core Library Abstractions (`torchcfm/`)

- **`ConditionalFlowMatcher`** and subclasses in [`torchcfm/conditional_flow_matching.py`](torchcfm/conditional_flow_matching.py) — each exposes `sample_location_and_conditional_flow(x0, x1, t=None)` returning `(t, xt, ut)`. Subclasses: `ExactOptimalTransportConditionalFlowMatcher`, `SchrodingerBridgeConditionalFlowMatcher`, `VariancePreservingConditionalFlowMatcher`.
- **`OTPlanSampler`** in [`torchcfm/optimal_transport.py`](torchcfm/optimal_transport.py) — wraps POT library; methods: `"exact"`, `"sinkhorn"`, `"unbalanced"`, `"partial"`.
- **`CFMLitModule`** in [`runner/src/models/cfm_module.py`](runner/src/models/cfm_module.py) — PyTorch Lightning module (legacy runner).

To add a new CFM variant: subclass `ConditionalFlowMatcher` and override `compute_mu_t`, `compute_sigma_t`, and/or `compute_conditional_flow`.

## Path Module (`path/path.py`)

Two interpolation paths with identical interfaces, enabling path-agnostic training:

| Class | Interpolation | Velocity | `T_MAX` |
|-------|--------------|----------|---------|
| `LinearPath` | $x_t = (1-t)x_0 + tx_1$ | $u_t = x_1 - x_0$ | `1.0` |
| `SphericalPath` | $x_t = \cos(t)x_0 + \sin(t)x_1$ | $u_t = -\sin(t)x_0 + \cos(t)x_1$ | `π/2` |

Key helpers: `get_path(name)`, `euler_sample(model, path, ...)`, `euler_sample_schedule(model, t_schedule, ...)`.

## Speed Module (`speed/speed.py`)

t-samplers and speed estimation for curriculum training:

- **`UniformSampler(T_MAX)`** — baseline uniform `t ~ U[0, T_MAX]`
- **`CdfSampler(t_grid, v_t, T_MAX, smooth_sigma)`** — inverse-CDF from smoothed speed profile
- **`BlendedSampler(a, b)`** — curriculum mixer; set `.mix` weight to blend between two samplers
- **`estimate_speed_grid(ema_net, path, t_grid, x1_ref, B, n_epochs, speed_type, device, n_hutch)`** — compute speed via JVP (forward-mode AD), not finite differences
- **`load_precomputed(speed_dir, path, speed_type)`** — load `.npy` speed files from `outputs/`

Speed types: **OT** ($\|\partial_t u_t^\theta\|$), **FR** (Fisher-Rao via Hutchinson trace, 4 probes by default).

## Conventions

- **Time `t`**: scalar ∈ `[0, T_MAX]`, shape `(bs,)`. Always use `pad_t_like_x(t, x)` (from `torchcfm/utils.py` **or** `path/path.py`'s `_b()`) before multiplying `t` with a data tensor.
- **`x0`** = source (often `N(0, I)`); **`x1`** = data distribution.
- **Return noise**: pass `return_noise=True` to `sample_location_and_conditional_flow` → `(t, xt, ut, epsilon)`.
- **Models** follow `torch_wrapper` convention from `torchcfm/utils.py` for `torchdyn` ODE solvers.
- **`sys.path` injection**: scripts in `examples/` and `kaggle/` prepend the repo root to `sys.path` — intentional for cluster/Kaggle compatibility, do not remove.
- **Runner experiments**: Hydra configs in `runner/configs/`; CLI: `python runner/src/train.py trainer=gpu experiment=<name>`.

## Key Examples & Scripts

- **2D training tutorial**: [`examples/2D_tutorials/tutorial_training_8_gaussians_to_moons.ipynb`](examples/2D_tutorials/tutorial_training_8_gaussians_to_moons.ipynb)
- **CIFAR-10**: see [`examples/images/cifar10/README.md`](examples/images/cifar10/README.md)
- **Single-cell dynamics**: [`examples/single_cell/single-cell_example.ipynb`](examples/single_cell/single-cell_example.ipynb)
- **VAE training** (CIFAR-10 latent FM): [`scripts/train_vae_cifar10.py`](scripts/train_vae_cifar10.py), [`scripts/train_fm_latent_linear_ddp.py`](scripts/train_fm_latent_linear_ddp.py)
- **NFE/FID evaluation**: [`scripts/eval_fid_nfe.py`](scripts/eval_fid_nfe.py) — sweeps NFE list and writes `nfe_fid_table.csv`
- **Slurm batch jobs**: [`slurm/`](slurm/) — H100 scripts; use `torchrun --nproc_per_node=$NGPU` for DDP
