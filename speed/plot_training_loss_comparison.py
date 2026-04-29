"""
Comparison plot of training loss across all CIFAR-10 experiments.
"""
import sys
sys.path.insert(0, '/nfs/ghome/live/cmarouani/FREE')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter1d

ROOT = '/nfs/ghome/live/cmarouani/FREE/outputs'

EXPERIMENTS = [
    dict(
        path=f'{ROOT}/cifar10_ot_linear/fm_ot/loss.csv',
        label='OT-CFM (linear, 400K)',
        color='tab:blue',
        linestyle='-',
    ),
    dict(
        path=f'{ROOT}/cifar10/comparison/ot/loss.csv',
        label='OT speed-weighted (comparison)',
        color='tab:cyan',
        linestyle='--',
    ),
    dict(
        path=f'{ROOT}/cifar10/comparison/fr/loss.csv',
        label='FR speed-weighted (comparison)',
        color='tab:orange',
        linestyle='--',
    ),
    dict(
        path=f'{ROOT}/cifar10/comparison/score/loss.csv',
        label='Score-weighted (comparison)',
        color='tab:purple',
        linestyle='--',
    ),
    dict(
        path=f'{ROOT}/cifar10_curriculum/loss.csv',
        label='Curriculum FM (FR, 4×H100 DDP)',
        color='tab:red',
        linestyle='-',
    ),
    dict(
        path=f'{ROOT}/cifar10_spherical/fm_standard/loss.csv',
        label='Spherical FM (standard)',
        color='tab:green',
        linestyle='-',
    ),
]

SMOOTH_WINDOW = 50   # steps (in units of 100-step rows)

fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=False)

for exp in EXPERIMENTS:
    df = pd.read_csv(exp['path'])
    steps = df['step'].values
    loss_ema = df['loss_ema'].values

    # light smoothing on top of the already-EMA'd loss for visual clarity
    loss_smooth = uniform_filter1d(loss_ema, size=SMOOTH_WINDOW)

    kw = dict(label=exp['label'], color=exp['color'], linestyle=exp['linestyle'], linewidth=1.5)

    # left: full range
    axes[0].plot(steps / 1e3, loss_smooth, **kw)

    # right: zoom into first 130K steps where most experiments overlap
    mask = steps <= 130_000
    if mask.sum() > 1:
        axes[1].plot(steps[mask] / 1e3, loss_smooth[mask], **kw)

# ── curriculum phase annotations (both panels) ─────────────────────────────
df_curric = pd.read_csv(f'{ROOT}/cifar10_curriculum/loss.csv')
phase_changes = df_curric.groupby('phase')['step'].min().to_dict()
phase_labels = {
    1: 'blend→p₁',
    2: 'p₁',
    3: 'p₂',
}
y_annot = 0.33  # fixed y position for annotation text (in loss units)
for phase, step in phase_changes.items():
    if phase == 0:
        continue
    for ax in axes:
        ax.axvline(step / 1e3, color='tab:red', linewidth=0.8, linestyle=':', alpha=0.5)
        ax.text(step / 1e3 + 0.5, y_annot, phase_labels.get(phase, ''),
                fontsize=6, color='tab:red', va='top')

for ax, title in zip(axes, ['All steps', 'First 130K steps (zoom)']):
    ax.set_xlabel('Training step (×10³)', fontsize=11)
    ax.set_ylabel('Loss (EMA, smoothed)', fontsize=11)
    ax.set_title(title, fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc='upper right')
    ax.set_xlim(left=0)

fig.suptitle('CIFAR-10 Training Loss — All Experiments', fontsize=14, fontweight='bold')
fig.tight_layout()

out = f'{ROOT}/cifar10/comparison/training_loss_comparison.png'
fig.savefig(out, dpi=150, bbox_inches='tight')
print(f'Saved → {out}')
