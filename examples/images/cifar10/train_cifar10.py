# Inspired from https://github.com/w86763777/pytorch-ddpm/tree/master.

# Authors: Kilian Fatras
#          Alexander Tong

import copy
import os

import matplotlib
import numpy as np
import torch
from absl import app, flags
from torchvision import datasets, transforms
from tqdm import tqdm, trange
from utils_cifar import ema, generate_samples, infiniteloop

matplotlib.use("Agg")

from torchcfm.conditional_flow_matching import (
    ConditionalFlowMatcher,
    ExactOptimalTransportConditionalFlowMatcher,
    TargetConditionalFlowMatcher,
    VariancePreservingConditionalFlowMatcher,
)
from torchcfm.models.unet.unet import UNetModelWrapper

FLAGS = flags.FLAGS

flags.DEFINE_string("model", "otcfm", help="flow matching model type")
flags.DEFINE_string("output_dir", "./results/", help="output_directory")
# UNet
flags.DEFINE_integer("num_channel", 128, help="base channel of UNet")

# Training
flags.DEFINE_float("lr", 2e-4, help="target learning rate")  # TRY 2e-4
flags.DEFINE_float("grad_clip", 1.0, help="gradient norm clipping")
flags.DEFINE_integer(
    "total_steps", 400001, help="total training steps"
)  # Lipman et al uses 400k but double batch size
flags.DEFINE_integer("warmup", 5000, help="learning rate warmup")
flags.DEFINE_integer("batch_size", 128, help="batch size")  # Lipman et al uses 128
flags.DEFINE_integer("num_workers", 4, help="workers of Dataloader")
flags.DEFINE_float("ema_decay", 0.9999, help="ema decay rate")
flags.DEFINE_bool("parallel", False, help="multi gpu training")
flags.DEFINE_bool("use_weight", False, help="use time-dependent loss weighting w(t) = C/v(t)")

# Evaluation
flags.DEFINE_integer(
    "save_step",
    20000,
    help="frequency of saving checkpoints, 0 to disable during training",
)


use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")


def _score_chunked(t_val, xt_flat, x1_flat, chunk_size=16):
    """nabla log p_t(xt) via mixture-of-Gaussians closed form.

    Uses the identity:
        nabla log p_t(x) = softmax(-||t*x1^i - x||^2 / (2*(1-t)^2))_i  *  (t*x1^i - x) / (1-t)^2

    Chunked over xt to avoid a (B_xt, B_x1, D) tensor larger than GPU memory.
    """
    inv_var = 1.0 / (1.0 - t_val) ** 2
    mu = t_val * x1_flat  # (B_x1, D)
    scores = []
    with torch.no_grad():
        for start in range(0, xt_flat.shape[0], chunk_size):
            xt_chunk = xt_flat[start : start + chunk_size]  # (chunk, D)
            diff = mu.unsqueeze(0) - xt_chunk.unsqueeze(1)  # (chunk, B_x1, D)
            sq_dist = (diff ** 2).sum(-1)  # (chunk, B_x1)
            weights = torch.softmax(-sq_dist * inv_var / 2.0, dim=1)  # (chunk, B_x1)
            scores.append((weights.unsqueeze(-1) * diff).sum(1) * inv_var)
    return torch.cat(scores, dim=0)  # (B_xt, D)


def compute_weighting(dataset, savedir, n_times=100, batch_size=512, eps_fd=1e-3):
    """Pre-compute w(t) = C / v(t) where:
        v(t) = sqrt( E[ ||d/dt nabla log p_t(X_t)||^2 ] )
        C    = int_0^1 v(s) ds

    d/dt score is estimated by central finite differences in t, keeping x_t fixed.
    Returns (t_grid, w_values) as numpy arrays.
    """
    import matplotlib.pyplot as plt
    from torch.utils.data import DataLoader

    print("Pre-computing time-dependent weighting w(t) ...")
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True)
    data_iter = iter(loader)

    t_grid = np.linspace(0.01, 0.98, n_times)
    v_values = np.zeros(n_times)

    for idx, t_val in enumerate(tqdm(t_grid, desc="v(t)")):
        try:
            x1, _ = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            x1, _ = next(data_iter)

        x1 = x1.to(device)
        x0 = torch.randn_like(x1)
        x1_flat = x1.reshape(x1.shape[0], -1)
        x0_flat = x0.reshape(x0.shape[0], -1)

        # Sample x_t from p_t; kept fixed while we vary t in the FD stencil
        xt_flat = (t_val * x1_flat + (1.0 - t_val) * x0_flat).detach()

        t_plus = min(t_val + eps_fd, 0.9999)
        t_minus = max(t_val - eps_fd, 0.0001)

        score_plus = _score_chunked(t_plus, xt_flat, x1_flat)
        score_minus = _score_chunked(t_minus, xt_flat, x1_flat)
        dscore_dt = (score_plus - score_minus) / (t_plus - t_minus)  # (B, D)

        v_values[idx] = dscore_dt.pow(2).mean().sqrt().item()

    C = float(np.trapezoid(v_values, t_grid))
    w_values = C / (v_values + 1e-8)

    # Compute alpha^* by inverting (alpha^*)^{-1}(t) = (1/C) * int_0^t v(s) ds
    cum_v = np.array([float(np.trapezoid(v_values[:i+1], t_grid[:i+1])) for i in range(len(t_grid))])
    cum_v_norm = cum_v / (cum_v[-1] + 1e-8)  # (alpha^*)^{-1} on t_grid
    # alpha^*(t): for each t, find s s.t. cum_v_norm(s) = t
    t_uniform = np.linspace(0, 1, 1000)
    alpha_star = np.interp(t_uniform, cum_v_norm, t_grid)

    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].plot(t_grid, v_values)
    axes[0].set_xlabel("t")
    axes[0].set_ylabel("v(t)")
    axes[0].set_title(r"$v(t) = \sqrt{E[\|\partial_t \nabla \log p_t(X_t)\|^2]/d}$")
    axes[1].plot(t_grid, w_values)
    axes[1].set_xlabel("t")
    axes[1].set_ylabel("w(t)")
    axes[1].set_title(r"$w(t) = C \,/\, v(t)$")
    axes[2].plot(t_uniform, alpha_star)
    axes[2].plot([0, 1], [0, 1], "k--", linewidth=0.8, label="identity")
    axes[2].set_xlabel("t")
    axes[2].set_ylabel(r"$\alpha^*(t)$")
    axes[2].set_title(r"Noise schedule $\alpha^*(t)$")
    axes[2].legend()
    plt.tight_layout()
    plot_path = os.path.join(savedir, "weighting.png")
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"Weighting plot saved to {plot_path}")

    return t_grid, w_values


def warmup_lr(step):
    return min(step, FLAGS.warmup) / FLAGS.warmup


def train(argv):
    print(
        "lr, total_steps, ema decay, save_step:",
        FLAGS.lr,
        FLAGS.total_steps,
        FLAGS.ema_decay,
        FLAGS.save_step,
    )

    # DATASETS/DATALOADER
    dataset = datasets.CIFAR10(
        root="./data",
        train=True,
        download=True,
        transform=transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        ),
    )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=FLAGS.batch_size,
        shuffle=True,
        num_workers=FLAGS.num_workers,
        drop_last=True,
    )

    datalooper = infiniteloop(dataloader)

    # MODELS
    net_model = UNetModelWrapper(
        dim=(3, 32, 32),
        num_res_blocks=2,
        num_channels=FLAGS.num_channel,
        channel_mult=[1, 2, 2, 2],
        num_heads=4,
        num_head_channels=64,
        attention_resolutions="16",
        dropout=0.1,
    ).to(device)  # new dropout + bs of 128

    ema_model = copy.deepcopy(net_model)
    optim = torch.optim.Adam(net_model.parameters(), lr=FLAGS.lr)
    sched = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda=warmup_lr)
    if FLAGS.parallel:
        print(
            "Warning: parallel training is performing slightly worse than single GPU training due to statistics computation in dataparallel. We recommend to train over a single GPU, which requires around 8 Gb of GPU memory."
        )
        net_model = torch.nn.DataParallel(net_model)
        ema_model = torch.nn.DataParallel(ema_model)

    # show model size
    model_size = 0
    for param in net_model.parameters():
        model_size += param.data.nelement()
    print("Model params: %.2f M" % (model_size / 1024 / 1024))

    #################################
    #            OT-CFM
    #################################

    sigma = 0.0
    if FLAGS.model == "otcfm":
        FM = ExactOptimalTransportConditionalFlowMatcher(sigma=sigma)
    elif FLAGS.model == "icfm":
        FM = ConditionalFlowMatcher(sigma=sigma)
    elif FLAGS.model == "fm":
        FM = TargetConditionalFlowMatcher(sigma=sigma)
    elif FLAGS.model == "si":
        FM = VariancePreservingConditionalFlowMatcher(sigma=sigma)
    else:
        raise NotImplementedError(
            f"Unknown model {FLAGS.model}, must be one of ['otcfm', 'icfm', 'fm', 'si']"
        )

    savedir = FLAGS.output_dir + FLAGS.model + "/"
    os.makedirs(savedir, exist_ok=True)

    # ---- optional time-dependent weighting ----
    w_t_grid = w_t_values = None
    if FLAGS.use_weight:
        w_t_grid, w_t_values = compute_weighting(dataset, savedir)

    with trange(FLAGS.total_steps, dynamic_ncols=True) as pbar:
        for step in pbar:
            optim.zero_grad()
            x1 = next(datalooper).to(device)
            x0 = torch.randn_like(x1)
            t, xt, ut = FM.sample_location_and_conditional_flow(x0, x1)
            vt = net_model(t, xt)
            if FLAGS.use_weight:
                w = torch.from_numpy(
                    np.interp(t.cpu().numpy(), w_t_grid, w_t_values)
                ).float().to(device)  # (B,)
                per_sample = ((vt - ut) ** 2).mean(dim=[1, 2, 3])  # (B,)
                loss = (w * per_sample).mean()
            else:
                loss = torch.mean((vt - ut) ** 2)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net_model.parameters(), FLAGS.grad_clip)  # new
            optim.step()
            sched.step()
            ema(net_model, ema_model, FLAGS.ema_decay)  # new

            # sample and Saving the weights
            if FLAGS.save_step > 0 and step % FLAGS.save_step == 0:
                generate_samples(net_model, FLAGS.parallel, savedir, step, net_="normal")
                generate_samples(ema_model, FLAGS.parallel, savedir, step, net_="ema")
                torch.save(
                    {
                        "net_model": net_model.state_dict(),
                        "ema_model": ema_model.state_dict(),
                        "sched": sched.state_dict(),
                        "optim": optim.state_dict(),
                        "step": step,
                    },
                    savedir + f"{FLAGS.model}_cifar10_weights_step_{step}.pt",
                )


if __name__ == "__main__":
    app.run(train)
