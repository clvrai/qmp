import os
import sys
import torch

from garage.torch import as_torch_dict, as_torch, global_device
from torch.distributions.kl import register_kl
from garage.torch.distributions.tanh_normal import TanhNormal


class SuppressStdout:
    def __enter__(self):
        self._prev_stdout = sys.stdout
        sys.stdout = open(os.devnull, "w")

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._prev_stdout


def np_to_torch(array):
    """Numpy arrays to PyTorch tensors.
    Args:
        array (np.ndarray): Data in numpy array.
    Returns:
        torch.Tensor: float tensor on the global device.
    """
    tensor = torch.from_numpy(array)

    if tensor.dtype != torch.float32:
        tensor = tensor.float()

    return tensor.to(global_device())


def list_to_tensor(data):
    """Convert a list to a PyTorch tensor.
    Args:
        data (list): Data to convert to tensor
    Returns:
        torch.Tensor: A float tensor
    """
    return torch.as_tensor(data, dtype=torch.float32, device=global_device())


@register_kl(TanhNormal, TanhNormal)
def kl_tanh_normal_normal(p, q):
    ### merged implementation from _kl_normal_normal and _kl_independent_independent
    p, q = p._normal, q._normal
    if p.reinterpreted_batch_ndims != q.reinterpreted_batch_ndims:
        raise NotImplementedError
    p, q = p.base_dist, q.base_dist
    var_ratio = (p.scale / q.scale).pow(2)
    t1 = ((p.loc - q.loc) / q.scale).pow(2)
    return 0.5 * (var_ratio + t1 - 1 - var_ratio.log())
