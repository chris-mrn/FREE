#!/usr/bin/env python3
"""
train.py — Unified flow matching training entry point.

Supports:
  --path        linear | spherical
  --dataset     cifar10 | 8gaussians | 40gaussians | moons | circles | checkerboard
  --coupling    independent | ot          (default: independent)
  --t_mode      uniform | weighted | curriculum
  --speed_type  ot | fr | score           (which speed measure; score = precomputed only)
  --ddp                                   (multi-GPU via torchrun)

Examples
--------
# Linear FM on CIFAR-10, uniform t, single GPU
python train.py --path linear --dataset cifar10 --out_dir outputs/myrun

# Spherical FM, OT coupling, speed-weighted t-sampling (precomputed)
python train.py --path spherical --dataset cifar10 --coupling ot \\
    --t_mode weighted --speed_type ot --speed_dir outputs/cifar10_spherical \\
    --out_dir outputs/sph_weighted

# Spherical FM + curriculum, DDP × 4
torchrun --nproc_per_node=4 train.py --path spherical --dataset cifar10 \\
    --coupling ot --t_mode curriculum --speed_type ot --ddp \\
    --curriculum_start 100000 --curriculum_blend 25000 \\
    --out_dir outputs/sph_curriculum

# 2D flow matching
python train.py --path linear --dataset 8gaussians --out_dir outputs/2d_test
"""

import copy, csv, glob, math, os, sys, time, argparse
import numpy as np
import torch
import torch.nn as nn

from path     import get_path, euler_sample, euler_sample_schedule
from speed    import (UniformSampler, CdfSampler, BlendedSampler,
                      estimate_speed_grid, load_precomputed, make_cdf_sampler)
from datasets import (is_image, get_x0_shape, get_cifar10_loaders,
                      collect_images, sample_2d)
from models   import MLP2D, build_model
from utils    import (ema_update, warmup_lr, cosine_blend,
                      find_last_ckpt, save_ckpt,
                      infinite_image_loop, infinite_2d_loop,
                      run_image_eval, run_2d_eval, setup_ddp)
from metrics  import InceptionMetrics


# ── OT coupling ────────────────────────────────────────────────────────────────

def maybe_ot_pair(x0, x1, ot_sampler):
    if ot_sampler is None:
        return x0, x1
    shape = x0.shape
    x0_flat = x0.view(len(x0), -1)
    x1_flat = x1.view(len(x1), -1)
    x0_paired, x1_paired = ot_sampler.sample_plan(x0_flat, x1_flat, replace=False)
    return x0_paired.view(shape), x1_paired.view(shape)


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Core axes
    parser.add_argument('--path',     default='linear',
                        choices=['linear', 'spherical'])
    parser.add_argument('--dataset',  default='cifar10',
                        help='cifar10 | 8gaussians | 40gaussians | moons | circles | checkerboard')
    parser.add_argument('--coupling', default='independent',
                        choices=['independent', 'ot'])
    parser.add_argument('--t_mode',   default='uniform',
                        choices=['uniform', 'weighted', 'curriculum'])
    parser.add_argument('--speed_type', default='ot',
                        choices=['ot', 'fr', 'score'],
                        help='Speed measure for weighted/curriculum t-sampling')

    # Speed source (weighted mode)
    parser.add_argument('--speed_dir', default=None,
                        help='Dir with precomputed *_speed*.npy files (weighted mode)')

    # Curriculum
    parser.add_argument('--curriculum_start', type=int, default=100_000,
                        help='Step at which to estimate speed and start blending')
    parser.add_argument('--curriculum_blend', type=int, default=25_000,
                        help='Steps to blend from uniform to speed-adaptive')
    parser.add_argument('--curriculum_restarts', type=int, default=0,
                        help='Number of speed re-estimations after the first (0=none)')
    parser.add_argument('--curriculum_restart_every', type=int, default=50_000)

    # Speed estimation hyperparams (curriculum + 2D weighted)
    parser.add_argument('--speed_n_t',     type=int,   default=100)
    parser.add_argument('--speed_B',       type=int,   default=512)
    parser.add_argument('--speed_epochs',  type=int,   default=3)
    parser.add_argument('--speed_hutch',   type=int,   default=4,
                        help='Hutchinson probes (fr speed only)')
    parser.add_argument('--speed_smooth',  type=float, default=0.05,
                        help='Gaussian smoothing bandwidth in t-units for CDF sampler')

    # Training
    parser.add_argument('--total_steps',  type=int,   default=400_001)
    parser.add_argument('--batch_size',   type=int,   default=128,
                        help='Per-GPU batch size')
    parser.add_argument('--lr',           type=float, default=2e-4)
    parser.add_argument('--warmup',       type=int,   default=5_000)
    parser.add_argument('--ema_decay',    type=float, default=0.9999)
    parser.add_argument('--grad_clip',    type=float, default=1.0)
    parser.add_argument('--eval_every',   type=int,   default=20_000)
    parser.add_argument('--fid_samples',  type=int,   default=10_000)
    parser.add_argument('--nfe_list',     type=int,   nargs='+',
                        default=[10, 20, 35, 50, 100])
    parser.add_argument('--keep_ckpts',   type=int,   default=2)
    parser.add_argument('--num_workers',  type=int,   default=4)
    parser.add_argument('--seed',         type=int,   default=42)

    # Model
    parser.add_argument('--num_channel',  type=int,   default=128,
                        help='UNet base channels (image datasets)')
    parser.add_argument('--hidden_2d',    type=int,   default=256,
                        help='MLP hidden size (2D datasets)')
    parser.add_argument('--depth_2d',     type=int,   default=4,
                        help='MLP depth (2D datasets)')

    # I/O
    parser.add_argument('--data_dir',  default='/tmp/fm_data')
    parser.add_argument('--out_dir',   required=True)
    parser.add_argument('--resume',    default='auto',
                        help='Checkpoint path, "auto" to find latest, or empty to disable')
    parser.add_argument('--ddp',       action='store_true',
                        help='Enable multi-GPU via torchrun')

    args = parser.parse_args()

    # ── DDP / device ──────────────────────────────────────────────────────────
    use_ddp = args.ddp and ('LOCAL_RANK' in os.environ)
    if use_ddp:
        local_rank, world_size, is_main = setup_ddp()
        device = torch.device(f'cuda:{local_rank}')
    else:
        local_rank, world_size, is_main = 0, 1, True
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    import torch.distributed as dist

    torch.manual_seed(args.seed + local_rank)
    np.random.seed(args.seed + local_rank)

    # ── Dirs ──────────────────────────────────────────────────────────────────
    if is_main:
        for d in [args.out_dir,
                  f'{args.out_dir}/checkpoints',
                  f'{args.out_dir}/samples']:
            os.makedirs(d, exist_ok=True)

    if use_ddp:
        dist.barrier()

    # ── Path and shape ────────────────────────────────────────────────────────
    path     = get_path(args.path)
    x0_shape = get_x0_shape(args.dataset)
    image_ds = is_image(args.dataset)

    if is_main:
        print(f'=== Flow Matching Training ===')
        print(f'path={args.path}  dataset={args.dataset}  coupling={args.coupling}')
        print(f't_mode={args.t_mode}  speed_type={args.speed_type}')
        print(f'device={device}' +
              (f'  GPU={torch.cuda.get_device_name(local_rank)}' if device.type == 'cuda' else ''))
        if use_ddp:
            print(f'DDP × {world_size}  effective_bs={args.batch_size * world_size}')

    # ── Data ──────────────────────────────────────────────────────────────────
    if image_ds:
        if is_main:
            # rank-0 downloads, others wait
            get_cifar10_loaders(args.data_dir, args.batch_size, args.num_workers)
        if use_ddp:
            dist.barrier()
        train_loader, eval_loader, train_sampler = get_cifar10_loaders(
            args.data_dir, args.batch_size, args.num_workers,
            distributed=use_ddp, rank=local_rank, world_size=world_size)
        looper = infinite_image_loop(train_loader, train_sampler)
    else:
        train_loader = eval_loader = train_sampler = None
        looper = infinite_2d_loop(args.dataset, args.batch_size, device)

    # ── Model ─────────────────────────────────────────────────────────────────
    net = build_model(args, device)
    if use_ddp:
        net = nn.parallel.DistributedDataParallel(net, device_ids=[local_rank])
    ema_net = copy.deepcopy(net.module if use_ddp else net)

    if is_main:
        n_params = sum(p.numel() for p in net.parameters())
        print(f'Params: {n_params / 1e6:.2f}M')

    # ── Optimizer ─────────────────────────────────────────────────────────────
    lr_eff   = args.lr * world_size if use_ddp else args.lr
    optim    = torch.optim.Adam(net.parameters(), lr=lr_eff)
    lr_sched = torch.optim.lr_scheduler.LambdaLR(
        optim, lr_lambda=lambda s: warmup_lr(s, args.warmup))

    # ── OT plan sampler ───────────────────────────────────────────────────────
    ot_sampler = None
    if args.coupling == 'ot':
        from torchcfm.optimal_transport import OTPlanSampler
        ot_sampler = OTPlanSampler(method='exact')

    # ── t-sampler setup ───────────────────────────────────────────────────────
    uniform_sampler = UniformSampler(path.T_MAX)

    def build_t_grid():
        return np.linspace(0.01, path.T_MAX - 0.01, args.speed_n_t)

    speed_sampler = None   # holds CdfSampler when ready

    if args.t_mode == 'weighted':
        if image_ds and args.speed_dir:
            t_grid, v_t = load_precomputed(args.speed_dir, args.path, args.speed_type)
            speed_sampler = make_cdf_sampler(t_grid, v_t, path.T_MAX,
                                              smooth_sigma=args.speed_smooth)
        else:
            print('Speed will be estimated from model after warmup.')

    # Curriculum state (serialised into checkpoint)
    curriculum = {
        'phase': 0,         # 0=uniform, 1=blending, 2=pure speed-adaptive
        'restart_count': 0,
        'last_speed_step': None,
        't_grid': None,
        'v_t': None,
    }

    # ── Reference images for speed estimation (image datasets, rank 0 only) ───
    x1_ref = None
    if image_ds and args.t_mode in ('curriculum', 'weighted') and not (
            args.t_mode == 'weighted' and args.speed_dir):
        if is_main:
            x1_ref = collect_images(eval_loader, max_n=2000).to(device)
            print(f'Reference images for speed: {x1_ref.shape}')

    # ── Inception metrics (image only, rank 0 only) ───────────────────────────
    # Only needed when we'll actually run evals (eval_every ≤ total_steps).
    inception = real_mu = real_sig = real_feats = None
    if image_ds and is_main and args.eval_every <= args.total_steps:
        print('Computing CIFAR-10 reference statistics...')
        inception = InceptionMetrics(device=device, batch_size=256)
        real_mu, real_sig, real_feats = inception.compute_real_stats(eval_loader)
        print(f'Reference: {len(real_feats):,} images')

    if use_ddp:
        dist.barrier()

    # ── Resume ────────────────────────────────────────────────────────────────
    start_step = 0
    loss_ema   = None
    resume_path = None
    if args.resume == 'auto':
        resume_path = find_last_ckpt(args.out_dir)
    elif args.resume:
        resume_path = args.resume

    if resume_path and os.path.isfile(resume_path):
        if is_main:
            print(f'Resuming from {resume_path}')
        ck = torch.load(resume_path, map_location=device)
        (net.module if use_ddp else net).load_state_dict(ck['net'])
        ema_net.load_state_dict(ck['ema'])
        optim.load_state_dict(ck['optim'])
        lr_sched.load_state_dict(ck['sched'])
        start_step = ck['step']
        loss_ema   = ck.get('loss_ema')
        if 'curriculum' in ck:
            curriculum = ck['curriculum']
            if curriculum['v_t'] is not None:
                speed_sampler = make_cdf_sampler(
                    np.array(curriculum['t_grid']),
                    np.array(curriculum['v_t']),
                    path.T_MAX, smooth_sigma=args.speed_smooth)
        if is_main:
            print(f'Resumed at step={start_step}  phase={curriculum["phase"]}')

    if use_ddp:
        dist.barrier()

    # ── CSV logs ──────────────────────────────────────────────────────────────
    loss_csv = metrics_csv = loss_w = metrics_w = None
    if is_main:
        csv_mode = 'a' if start_step > 0 else 'w'
        loss_csv    = open(f'{args.out_dir}/loss.csv',    csv_mode, newline='')
        metrics_csv = open(f'{args.out_dir}/metrics.csv', csv_mode, newline='')
        loss_w    = csv.writer(loss_csv)
        metrics_w = csv.writer(metrics_csv)
        if start_step == 0:
            loss_w.writerow(['step', 'loss_raw', 'loss_ema', 't_phase'])
            metrics_w.writerow(['step', 'fid', 'kid_mean', 'kid_std',
                                 'is_mean', 'is_std'])
        loss_csv.flush(); metrics_csv.flush()

    # ── Speed estimation helper (shared by curriculum + 2D weighted) ──────────

    def estimate_and_broadcast_speed(step_label):
        """Estimate speed on rank 0, broadcast to all ranks, return CdfSampler."""
        if use_ddp:
            dist.barrier()
        n_t = args.speed_n_t

        if is_main:
            print(f'\n[speed] Estimating {args.speed_type} speed at step {step_label}...',
                  flush=True)
            ref  = x1_ref if image_ds else sample_2d(args.dataset, 2000).to(device)
            t_g  = build_t_grid()
            v_t_ = estimate_speed_grid(
                ema_net, path, t_g, ref,
                B=args.speed_B, n_epochs=args.speed_epochs,
                speed_type=args.speed_type, device=device,
                n_hutch=args.speed_hutch)
            print(f'  v_t range: [{v_t_.min():.4f}, {v_t_.max():.4f}]')
            np.save(f'{args.out_dir}/speed_t_grid_step{step_label}.npy', t_g)
            np.save(f'{args.out_dir}/speed_v_t_step{step_label}.npy', v_t_)
            speed_buf = torch.tensor(
                np.stack([t_g, v_t_]), dtype=torch.float32, device=device)
        else:
            speed_buf = torch.zeros(2, n_t, dtype=torch.float32, device=device)

        if use_ddp:
            dist.broadcast(speed_buf, src=0)

        t_g_local = speed_buf[0].cpu().numpy()
        v_t_local = speed_buf[1].cpu().numpy()
        sampler   = make_cdf_sampler(t_g_local, v_t_local, path.T_MAX,
                                      smooth_sigma=args.speed_smooth)
        curriculum['t_grid']          = t_g_local.tolist()
        curriculum['v_t']             = v_t_local.tolist()
        curriculum['last_speed_step'] = step_label

        if use_ddp:
            dist.barrier()
        return sampler

    # ── Weighted mode: compute speed before loop if no precomputed ────────────
    speed_warmup_done = False
    WEIGHTED_WARMUP = 1000 if not image_ds else 0

    # ── Training loop ─────────────────────────────────────────────────────────
    from tqdm import tqdm
    t_start = time.time()
    pbar    = tqdm(range(start_step + 1, args.total_steps + 1),
                   dynamic_ncols=True, disable=not is_main,
                   initial=start_step, total=args.total_steps)

    net_module = net.module if use_ddp else net

    for step in pbar:

        # ── Weighted warmup: compute speed once after WEIGHTED_WARMUP steps ──
        if (args.t_mode == 'weighted' and not speed_warmup_done
                and speed_sampler is None and step == start_step + 1 + WEIGHTED_WARMUP):
            speed_sampler     = estimate_and_broadcast_speed(step)
            speed_warmup_done = True

        # ── Curriculum: first speed estimation ────────────────────────────────
        if (args.t_mode == 'curriculum'
                and step == args.curriculum_start
                and curriculum['phase'] == 0):
            speed_sampler = estimate_and_broadcast_speed(step)
            curriculum['phase'] = 1
            if is_main:
                print(f'[curriculum] Phase 0→1: blending over '
                      f'{args.curriculum_blend} steps', flush=True)

        # ── Curriculum: optional restarts ─────────────────────────────────────
        if (args.t_mode == 'curriculum'
                and curriculum['phase'] == 2
                and args.curriculum_restarts > 0
                and curriculum['restart_count'] < args.curriculum_restarts
                and step == (curriculum['last_speed_step']
                             + args.curriculum_restart_every)):
            speed_sampler = estimate_and_broadcast_speed(step)
            curriculum['restart_count'] += 1
            curriculum['phase'] = 1   # re-blend
            if is_main:
                print(f'[curriculum] Restart {curriculum["restart_count"]}: '
                      f're-blending', flush=True)

        # ── Detect end of blend ────────────────────────────────────────────────
        if (args.t_mode == 'curriculum'
                and curriculum['phase'] == 1
                and curriculum['last_speed_step'] is not None
                and step >= curriculum['last_speed_step'] + args.curriculum_blend):
            curriculum['phase'] = 2
            if is_main:
                print(f'[curriculum] Phase 2: pure speed-adaptive from step {step}',
                      flush=True)

        # ── Sample t ──────────────────────────────────────────────────────────
        B     = args.batch_size
        phase = curriculum['phase']

        if args.t_mode == 'uniform' or speed_sampler is None:
            t = uniform_sampler.sample(B, device)
        elif args.t_mode == 'weighted':
            t = speed_sampler.sample(B, device)
        else:  # curriculum
            if phase == 0:
                t = uniform_sampler.sample(B, device)
            elif phase == 1:
                mix = cosine_blend(step, curriculum['last_speed_step'],
                                   args.curriculum_blend)
                t   = BlendedSampler(uniform_sampler, speed_sampler).sample(B, device, mix)
            else:
                t = speed_sampler.sample(B, device)

        # ── Training step ─────────────────────────────────────────────────────
        x1 = next(looper)
        if image_ds:
            x1 = x1.to(device)
        x0 = torch.randn_like(x1)

        x0, x1 = maybe_ot_pair(x0, x1, ot_sampler)

        xt = path.xt(t, x0, x1)
        ut = path.ut(t, x0, x1)

        optim.zero_grad()
        loss = ((net(t, xt) - ut) ** 2).mean()
        loss.backward()
        nn.utils.clip_grad_norm_(net.parameters(), args.grad_clip)
        optim.step()
        lr_sched.step()
        ema_update(net_module, ema_net, args.ema_decay)

        loss_val = loss.item()
        loss_ema = loss_val if loss_ema is None else 0.99 * loss_ema + 0.01 * loss_val

        if is_main:
            pbar.set_postfix(loss=f'{loss_ema:.4f}', phase=phase)

        # ── Logging ───────────────────────────────────────────────────────────
        if is_main and step % 100 == 0:
            loss_w.writerow([step, loss_val, loss_ema, phase])
            loss_csv.flush()

        # ── Timing ────────────────────────────────────────────────────────────
        if is_main and step == start_step + 200:
            elapsed = time.time() - t_start
            rate    = 200 / elapsed
            print(f'\n[Timing] {rate:.1f} steps/s  '
                  f'est. {args.total_steps / rate / 3600:.1f}h total', flush=True)

        # ── Eval + checkpoint ─────────────────────────────────────────────────
        if step % args.eval_every == 0 or step == args.total_steps:
            if use_ddp:
                dist.barrier()
            if is_main:
                elapsed_min = (time.time() - t_start) / 60
                print(f'\n[step {step}] loss={loss_ema:.4f}  '
                      f'elapsed={elapsed_min:.1f}min', flush=True)

                ckpt_path = (f'{args.out_dir}/checkpoints/'
                             f'ckpt_step_{step:07d}.pt')
                save_ckpt(ckpt_path, net_module, ema_net, optim, lr_sched,
                          step, loss_ema, curriculum)
                if args.keep_ckpts > 0:
                    for old in sorted(glob.glob(
                            f'{args.out_dir}/checkpoints/ckpt_step_*.pt'))[:-args.keep_ckpts]:
                        os.remove(old)

                torch.cuda.empty_cache()
                if image_ds:
                    if inception is not None:
                        fid, kid_m, kid_s, is_m, is_s = run_image_eval(
                            step, ema_net, path, inception,
                            real_mu, real_sig, real_feats,
                            args.out_dir, args.fid_samples, device)
                        metrics_w.writerow([step, fid, kid_m, kid_s, is_m, is_s])
                        metrics_csv.flush()
                else:
                    run_2d_eval(step, ema_net, path, args.dataset,
                                args.out_dir, 2000, device)
                torch.cuda.empty_cache()
            if use_ddp:
                dist.barrier()

    # ── NFE sweep (image only, rank 0 only, when inception stats were computed) ─
    if image_ds and is_main and inception is not None:
        print('\n=== NFE sweep ===')
        nfe_csv = open(f'{args.out_dir}/nfe_fid.csv', 'w', newline='')
        nfe_w   = csv.writer(nfe_csv)
        nfe_w.writerow(['nfe', 'fid', 'kid_mean', 'is_mean', 'is_std'])
        for nfe in args.nfe_list:
            print(f'  NFE={nfe}...', flush=True)
            samples         = euler_sample(ema_net, path, args.fid_samples,
                                           nfe, x0_shape, device)
            feats_f, prob_f = inception.get_activations(samples)
            fid             = inception.compute_fid(real_mu, real_sig, feats_f)
            kid_m, _        = inception.compute_kid(real_feats, feats_f)
            is_m, is_s      = inception.compute_is(prob_f)
            print(f'    FID={fid:.3f}  KID={kid_m:.4f}  IS={is_m:.3f}±{is_s:.3f}')
            nfe_w.writerow([nfe, fid, kid_m, is_m, is_s])
            nfe_csv.flush()
        nfe_csv.close()

    # ── Cleanup ───────────────────────────────────────────────────────────────
    if is_main:
        if loss_csv:
            loss_csv.close()
        if metrics_csv:
            metrics_csv.close()
        total = time.time() - t_start
        print(f'\nDone in {total / 3600:.2f}h  →  {args.out_dir}')

    if use_ddp:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
