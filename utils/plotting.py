"""Image and 2D sample generation + evaluation plotting."""
import time


def run_image_eval(step, ema_model, path, inception, real_mu, real_sig,
                   real_feats, out_dir, n_samples, device, n_steps=35):
    from torchvision.utils import make_grid, save_image
    from path import euler_sample
    t0 = time.time()
    x0_shape = (3, 32, 32)
    samples  = euler_sample(ema_model, path, n_samples, n_steps, x0_shape, device)
    grid     = make_grid(samples[:64].clamp(-1, 1), nrow=8,
                         normalize=True, value_range=(-1, 1))
    save_image(grid, f'{out_dir}/samples/step_{step:07d}.png')
    feats_f, probs_f = inception.get_activations(samples)
    fid            = inception.compute_fid(real_mu, real_sig, feats_f)
    kid_m, kid_s   = inception.compute_kid(real_feats, feats_f)
    is_m,  is_s    = inception.compute_is(probs_f)
    print(f'  [eval {step}] FID={fid:.2f}  KID={kid_m:.3f}±{kid_s:.3f}'
          f'  IS={is_m:.2f}±{is_s:.2f}  ({time.time()-t0:.0f}s)')
    return fid, kid_m, kid_s, is_m, is_s


def run_2d_eval(step, ema_model, path, dataset_name, out_dir, n_samples, device):
    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from path import euler_sample
    from data.datasets import sample_2d
    x0_shape = (2,)
    samples  = euler_sample(ema_model, path, n_samples, 200, x0_shape, device)
    ref      = sample_2d(dataset_name, n_samples)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    fig.suptitle(f'Step {step}')
    axes[0].scatter(ref[:, 0], ref[:, 1], s=1, alpha=0.3, label='data')
    axes[0].set_title('Data'); axes[0].set_aspect('equal')
    axes[1].scatter(samples[:, 0], samples[:, 1], s=1, alpha=0.3, label='model')
    axes[1].set_title('Generated'); axes[1].set_aspect('equal')
    plt.tight_layout()
    plt.savefig(f'{out_dir}/samples/step_{step:07d}.png', dpi=120)
    plt.close()
