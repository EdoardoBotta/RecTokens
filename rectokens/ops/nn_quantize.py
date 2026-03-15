import torch

IS_GPU_AVAILABLE = torch.cuda.is_available()

if IS_GPU_AVAILABLE:
    from rectokens.kernels.nn_quantize import quantize_fwd
    from rectokens.kernels.nn_quantize import quantize_fwd_mm


def _cuda_nearest_neighbor_quantize(x: torch.Tensor, codebook: torch.Tensor):
    if x.shape[-1] <= 128 and x.shape[0] > 2 * codebook.shape[0]:
        return quantize_fwd(x, codebook)
    return quantize_fwd_mm(x, codebook)[0]


def nearest_neighbor_quantize(x: torch.Tensor, codebook: torch.Tensor):
    assert x.device == codebook.device, (
        f"Expected both tensors to be on the same device. Found: {x.device} and {codebook.device}"
    )
    if x.is_cuda:
        return _cuda_nearest_neighbor_quantize(x, codebook)
    return torch.cdist(x, codebook).min(-1)[1]
