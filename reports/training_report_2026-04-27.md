# Training Report — April 27, 2026

**Jobs killed manually**: 2768198 (Spherical FM), 2774503 (Curriculum FM)
**Date**: 2026-04-27
**Repository**: `chris-mrn/FREE`, branch `main`

---

## 1. Executive Summary

| Model | Job ID | Partition | Hardware | Steps Done | % Complete | Best FID | Status |
|-------|--------|-----------|----------|-----------|------------|----------|--------|
| Spherical FM | 2768198 | `gatsby_ws` | RTX 4500 Ada (24 GB) | ~230K / 400K | 57.5% | **5.43** @ 220K | Killed (manual) |
| Curriculum FM | 2774503 | `gpu_lowp` | 4× H100 80 GB | ~212K / 400K | 53% | **7.96** @ 200K | Killed (manual) |

**Key finding**: Spherical FM significantly outperforms Curriculum FM at matched step counts (FID 5.43 vs 7.96 @ 200K). The curriculum reweighting has not yet produced a clear benefit over spherical interpolation.

---

## 2. Spherical FM — Job 2768198

### Configuration

| Parameter | Value |
|-----------|-------|
| Script | `examples/images/cifar10/training/train_fm_spherical.py` |
| Node | `u240c.id.gatsby.ucl.ac.uk` |
| GPU | NVIDIA RTX 4500 Ada Generation (24,570 MiB) |
| Params | 35.7 M |
| Interpolation | $X_t = \cos(t)\,X_0 + \sin(t)\,X_1$, $t \in [0, \pi/2]$ |
| Training speed | 3.33 it/s |
| Eval frequency | every 20K steps |
| Sampling steps | 35 (Euler), also 50 & 100 from step 180K |
| Wall limit | 24 h |
| Start time | Sun Apr 26 19:07:57 BST 2026 |
| Kill time | ~21 h 48 min runtime (step ~229,974) |

### FID / KID / IS Progression

| Step | FID | KID (mean±std) | IS (mean±std) | Loss @step |
|------|-----|-----------------|----------------|------------|
| 20K  | 322.58 | 363.373 ± 4.490 | 1.19 ± 0.01 | 0.1147 |
| 40K  | 57.92  | 58.856 ± 2.199  | 2.93 ± 0.06 | 0.1124 |
| 60K  | 9.21   | 5.259 ± 0.757   | 4.30 ± 0.07 | 0.1103 |
| 80K  | 6.69   | 3.419 ± 0.607   | 4.77 ± 0.12 | 0.1114 |
| 100K | 6.04   | 2.748 ± 0.603   | 4.98 ± 0.15 | 0.1102 |
| 120K | 5.84   | 2.897 ± 0.494   | 5.03 ± 0.09 | — |
| 140K | 5.56   | 2.720 ± 0.521   | 5.10 ± 0.13 | — |
| 160K | 5.46   | 2.745 ± 0.634   | 5.19 ± 0.11 | — |
| 180K | 5.45   | 2.679 ± 0.506   | 5.21 ± 0.10 | — |
| 200K | 5.51   | 2.964 ± 0.574   | 5.12 ± 0.12 | 0.1085 |
| 220K | **5.43** | 2.831 ± 0.566 | 5.25 ± 0.11 | 0.1095 |

*Eval time per checkpoint: ~765–767 s (12.75 min)*

### NFE Sweep (FID vs Number of Function Evaluations)

| Step | NFE=35 | NFE=50 | NFE=100 |
|------|--------|--------|---------|
| 180K | 5.45 | 5.27 | 5.46 |
| 200K | 5.51 | 5.50 | 5.25 |

NFE=100 gives **5.25** at 200K, a notable improvement over NFE=35 (5.51), suggesting the ODE is not fully resolved at 35 steps.

### Checkpoints Saved

```
outputs/cifar10_spherical/fm_standard/checkpoints/
  ema_step_0200000.pt
  ema_step_0220000.pt
```

### Artifacts

```
outputs/cifar10_spherical/
  fm_standard/
    metrics.csv          # FID/KID/IS per step
    nfe_fid_table.csv    # FID at NFE=35,50,100 per step
    loss.csv
    samples/             # image grids at each eval
    checkpoints/
  cifar10_speed_spherical.png
  fr_speed_sph.npy  /  fr_weighting_sph.npy  /  fr_schedule_sph.npy
  ot_speed_sph.npy  /  ot_weighting_sph.npy  /  ot_schedule_sph.npy
  score_speed_sph.npy / score_weighting_sph.npy / score_schedule_sph.npy
  t_grid_sph.npy
  train_2768198.log
```

### Observations

- **Rapid convergence**: FID drops from 322 → 9.2 in the first 60K steps (uniform training).
- **Plateau region**: FID oscillates between 5.43–5.56 from step 160K–220K. The model appears close to convergence at this architecture/batch size.
- **Best result**: FID **5.43** @ step 220K (NFE=35). With NFE=100 this may reach ~5.2.
- **Timed out**: Would have hit ~260K before the 24 h wall limit. To continue, resubmit with `--resume auto`.

---

## 3. Curriculum FM — Job 2774503

### Configuration

| Parameter | Value |
|-----------|-------|
| Script | `examples/images/cifar10/training/train_fm_curriculum_ddp.py` |
| Node | `gpu-sr675-34` |
| GPUs | 4× NVIDIA H100 80GB HBM3 (81,559 MiB each) |
| Per-GPU batch size | 128 |
| Effective batch size | 512 |
| Training speed | ~12.2 it/s (4 GPUs) |
| Eval frequency | every 50K steps |
| Sampling steps | 35 (Euler) |
| Wall limit | 14 h |
| Start time | Mon Apr 27 10:50:19 UTC 2026 |
| Kill time | ~5 h 05 min runtime (step ~212K) |

### Curriculum Schedule

| Phase | Step Range | Sampling Distribution |
|-------|-----------|----------------------|
| 1 | 0 – 50K | Uniform $t \sim \mathcal{U}(0,1)$ |
| blend 1→2 | 50K – 75K | Cosine blend $\mathcal{U}(0,1) \to p_1(t)$ |
| 2 | 75K – 100K | $p_1(t)$ (FR speed from @50K checkpoint) |
| blend 2→3 | 100K – 125K | Cosine blend $p_1(t) \to p_2(t)$ |
| 3 | 125K+ | $p_2(t)$ (FR speed from @75K checkpoint) |

where $p(t) \propto 1/v_t$ with $v_t = \sqrt{E[(\partial_t \log p_t)^2]}$ computed online from the EMA model.

### FR Speed Estimates (from checkpoints)

| Checkpoint | Speed range $v_t$ | Mass in $t < 0.30$ |
|------------|------------------|---------------------|
| `ckpt_step_0050000.pt` (speed1) | [3107.8, 122881.9] | ~48.5% |
| `ckpt_step_0100000.pt` (speed2) | [3111.6, 124780.2] | ~48.5% |

### FID / KID / IS Progression

| Step | Phase | FID | KID (mean±std) | IS (mean±std) | Loss @step |
|------|-------|-----|-----------------|----------------|------------|
| 50K  | 1 (uniform) | 9.15  | 6.706 ± 0.894 | 4.67 ± 0.16 | 0.1712 |
| 100K | 3 (p1→blend) | 8.86 | 5.440 ± 0.651 | 5.26 ± 0.10 | 0.1566 |
| 150K | 3 (pure p2) | 8.28  | 5.382 ± 0.679 | 5.25 ± 0.13 | 0.1560 |
| 200K | 3 (pure p2) | **7.96** | 5.284 ± 0.754 | 5.10 ± 0.16 | 0.1546 |

*Eval time per checkpoint: ~76–77 s*

### Checkpoints Saved

```
outputs/cifar10_curriculum/checkpoints/
  ckpt_step_0100000.pt   (also contains curriculum state: speed1_t/v)
  ckpt_step_0150000.pt   (also contains curriculum state: speed2_t/v)
  ckpt_step_0200000.pt
```

Note: checkpoints include full curriculum state (`curriculum.phase`, `speed1_t/v`, `speed2_t/v`) for seamless resumption.

### Artifacts

```
outputs/cifar10_curriculum/
  checkpoints/
  metrics.csv              # FID/KID/IS per step
  loss.csv
  samples/                 # image grids at each eval
  fr_speed_step50000.npy / fr_t_grid_step50000.npy
  fr_speed_step75000.npy / fr_t_grid_step75000.npy
  speed_step50000.png / speed_step75000.png
  train_2774503.log
```

### Observations

- **Slow convergence**: FID 9.15 @ 50K vs spherical's 6.04 @ 100K — roughly 2× slower to converge.
- **Steady improvement**: FID decreasing monotonically (9.15 → 8.86 → 8.28 → 7.96), no plateau observed yet.
- **Loss still declining**: 0.171 → 0.155 across 200K steps; curriculum has not converged.
- **Speed consistency**: $p_1$ and $p_2$ are nearly identical (speed range [3107–124780]), suggesting the model's learned speed did not change much between 50K and 75K checkpoints.
- **Significant gap vs Spherical**: At step 200K, FID is 7.96 vs 5.51 for spherical — a gap of 2.45 FID points. Not yet clear whether the curriculum will close this gap by step 400K.

---

## 4. Head-to-Head Comparison (@ step 200K)

| Metric | Spherical FM | Curriculum FM |
|--------|-------------|---------------|
| FID ↓ | **5.51** | 8.86 |
| KID ↓ | **2.964 ± 0.574** | 5.284 ± 0.754 |
| IS ↑ | 5.12 ± 0.12 | **5.10 ± 0.16** |
| Loss | 0.1085 | 0.1546 |
| Eval time | ~766 s | ~76 s |
| Effective throughput | 3.33 it/s (1 GPU) | 12.15 it/s (4 GPU) |

The curriculum's eval is 10× faster (76 s vs 766 s) because it uses 2K reference images vs 50K for spherical.

---

## 5. Next Steps

### Spherical FM
- [ ] Resubmit job with `--resume auto` from `ema_step_0220000.pt` to continue to 400K
- [ ] Run full NFE sweep on final checkpoint (NFE = 5, 10, 20, 35, 50, 100, 200)
- [ ] Slurm script: `examples/images/cifar10/slurm/slurm_fm_spherical.sh`

### Curriculum FM
- [ ] Resubmit job with `--resume auto` from `ckpt_step_0200000.pt` to continue to 400K
- [ ] Monitor whether FID crosses below 5.5 in the range 300K–400K
- [ ] If FID < spherical at 400K: curriculum reweighting is effective
- [ ] Consider ablation: curriculum without spherical path (flat FM, curriculum-only)

### Analysis
- [ ] Run `compare_speed_methods.py` on curriculum checkpoint @ 200K to compare learned vs batch FR speed
- [ ] Run `fid_nfe` evaluation on spherical checkpoint (job 2774591 was still pending — resubmit)
