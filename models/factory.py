"""Model factory: dispatches to MLP2D or UNetModelWrapper based on dataset."""
from .mlp import MLP2D
from .unet import UNetModelWrapper


def build_model(args, device):
    """
    Build and return the flow-matching model on `device`.

    Selects UNetModelWrapper for image datasets (cifar10, etc.) and
    MLP2D for 2D toy datasets.
    """
    from data.datasets import is_image
    if is_image(args.dataset):
        net = UNetModelWrapper(
            dim=(3, 32, 32), num_res_blocks=2, num_channels=args.num_channel,
            channel_mult=[1, 2, 2, 2], num_heads=4, num_head_channels=64,
            attention_resolutions='16', dropout=0.1,
        ).to(device)
    else:
        net = MLP2D(dim=2, hidden=args.hidden_2d, depth=args.depth_2d).to(device)
    return net
