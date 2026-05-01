"""
trainer.py — Unified flow matching training loop.

Entry point: run_training(args)
"""
import copy, csv, glob, os, time
import numpy as np
import torch
import torch.nn as nn

from path      import get_path, euler_sample
from data.datasets import (
    is_image, get_x0_shape, get_cifar10_loaders, collect_images,
)
from models    import build_model
from utils.helpers    import ema_update, warmup_lr, cosine_blend, setup_ddp
from utils.checkpoint import find_last_ckpt, save_ckpt
from utils.logging    import infinite_image_loop, infinite_2d_loop
from utils.plotting   import run_image_eval, run_2d_eval
from evaluation.speed   import (
    UniformSampler, make_cdf_sampler, load_precomputed,
)
from evaluation.metrics import InceptionMetrics
from training.curriculum import CurriculumState, estimate_and_broadcast_speed
from training.losses     import maybe_ot_pair, training_step


def run_training(args):
    """Run the full training loop given a parsed args namespace."""
    import torch.distributed as dist

    # ── DDP / device ──────────────────────────────────────────────────────────
    use_ddp = args.ddp and ('LOCAL_RANK' in os.environ)
    if use_ddp:
        local_rank, world_size, is_main = setup_ddp()
        device = torch.device(f'cuda:{local_rank}')
    else:
        local_rank, world_size, is_main = 0, 1, True
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
    ema_net    = copy.deepcopy(net.module if use_ddp else net)
    net_module = net.module if use_ddp else net

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
    speed_sampler   = None

    if args.t_mode == 'weighted' and image_ds and args.speed_dir:
        t_grid, v_t = load_precomputed(args.speed_dir, args.path, args.speed_type)
        speed_sampler = make_cdf_sampler(t_grid, v_t, path.T_MAX,
                                          smooth_sigma=args.speed_smooth)
    elif args.t_mode == 'weighted':
        print('Speed will be estimated from model after warmup.')

    curriculum = CurriculumState()

    # ── Reference images for speed estimation (image datasets, rank 0 only) ───
    x1_ref = None
    if image_ds and args.t_mode in ('curriculum', 'weighted') and not (
            args.t_mode == 'weighted' and args.speed_dir):
        if is_main:
            x1_ref = collect_images(eval_loader, max_n=2000).to(device)
            print(f'Reference images for speed: {x1_ref.shape}')

    # ── Inception metrics ─────────────────────────────────────────────────────
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
        net_module.load_state_dict(ck['net'])
        ema_net.load_state_dict(ck['ema'])
        optim.load_state_dict(ck['optim'])
        lr_sched.load_state_dict(ck['sched'])
        start_step = ck['step']
        loss_ema   = ck.get('loss_ema')
        if 'curriculum' in ck:
            curriculum = CurriculumState.from_dict(ck['curriculum'])
            if curriculum.v_t is not None:
                speed_sampler = make_cdf_sampler(
                    np.array(curriculum.t_grid),
                    np.array(curriculum.v_t),
                    path.T_MAX, smooth_sigma=args.speed_smooth)
        if is_main:
            print(f'Resumed at step={start_step}  phase={curriculum.phase}')

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

    # ── Training loop ─────────────────────────────────────────────────────────
    from tqdm import tqdm
    t_start = time.time()
    pbar    = tqdm(range(start_step + 1, args.total_steps + 1),
                   dynamic_ncols=True, disable=not is_main,
                   initial=start_step, total=args.total_steps)

    speed_warmup_done = False
    WEIGHTED_WARMUP   = 1000 if not image_ds else 0

    for step in pbar:

        # ── Weighted warmup: estimate speed once if no precomputed ───────────
        if (args.t_mode == 'weighted' and not speed_warmup_done
                and speed_sampler is None
                and step == start_step + 1 + WEIGHTED_WARMUP):
            speed_sampler     = estimate_and_broadcast_speed(
                step, ema_net, path, x1_ref, args, curriculum,
                image_ds, is_main, use_ddp, device)
            speed_warmup_done = True

        # ── Curriculum: first speed estimation ────────────────────────────────
        if args.t_mode == 'curriculum' and curriculum.should_start_curriculum(step, args):
            speed_sampler = estimate_and_broadcast_speed(
                step, ema_net, path, x1_ref, args, curriculum,
                image_ds, is_main, use_ddp, device)
            curriculum.phase = 1
            if is_main:
                print(f'[curriculum] Phase 0→1: blending over '
                      f'{args.curriculum_blend} steps', flush=True)

        # ── Curriculum: optional restarts ─────────────────────────────────────
        if args.t_mode == 'curriculum' and curriculum.should_restart(step, args):
            speed_sampler = estimate_and_broadcast_speed(
                step, ema_net, path, x1_ref, args, curriculum,
                image_ds, is_main, use_ddp, device)
            curriculum.restart_count += 1
            curriculum.phase = 1
            if is_main:
                print(f'[curriculum] Restart {curriculum.restart_count}: '
                      f're-blending', flush=True)

        # ── Detect end of blend ────────────────────────────────────────────────
        if args.t_mode == 'curriculum' and curriculum.should_end_blend(step, args):
            curriculum.phase = 2
            if is_main:
                print(f'[curriculum] Phase 2: pure speed-adaptive from step {step}',
                      flush=True)

        # ── Sample t ──────────────────────────────────────────────────────────
        B = args.batch_size
        if args.t_mode == 'curriculum':
            t = curriculum.sample_t(B, step, device, uniform_sampler, speed_sampler, args)
        else:
            t = curriculum.sample_t(B, step, device, uniform_sampler, speed_sampler, args)

        # ── Training step ─────────────────────────────────────────────────────
        loss_val = training_step(
            net, net_module, ema_net, path, looper, ot_sampler,
            optim, lr_sched, t, args, image_ds, device)

        loss_ema = loss_val if loss_ema is None else 0.99 * loss_ema + 0.01 * loss_val

        if is_main:
            pbar.set_postfix(loss=f'{loss_ema:.4f}', phase=curriculum.phase)

        # ── Logging ───────────────────────────────────────────────────────────
        if is_main and step % 100 == 0:
            loss_w.writerow([step, loss_val, loss_ema, curriculum.phase])
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
                          step, loss_ema, curriculum.to_dict())
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

    # ── NFE sweep ─────────────────────────────────────────────────────────────
    if image_ds and is_main and inception is not None:
        _run_nfe_sweep(ema_net, path, inception, real_mu, real_sig, real_feats,
                       x0_shape, args, device)

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


def _run_nfe_sweep(ema_net, path, inception, real_mu, real_sig, real_feats,
                   x0_shape, args, device):
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
