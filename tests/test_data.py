"""Tests for data/ module."""
import torch
from data.datasets import is_image, get_x0_shape, sample_2d


def test_is_image():
    assert is_image('cifar10')
    assert not is_image('8gaussians')


def test_get_x0_shape():
    assert get_x0_shape('cifar10') == (3, 32, 32)
    assert get_x0_shape('8gaussians') == (2,)


def test_sample_2d():
    x = sample_2d('8gaussians', 64)
    assert x.shape == (64, 2)
    assert x.dtype == torch.float32
