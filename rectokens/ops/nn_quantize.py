import torch

IS_GPU_AVAILABLE = torch.cuda.is_available()

if IS_GPU_AVAILABLE:
    from rectokens.kernels.nn_quantize import quantize_fwd
    from rectokens.kernels.nn_quantize import quantize_fwd_mm

torch.set_float32_matmul_precision("high")


def _cuda_nearest_neighbor_quantize(x: torch.Tensor, codebook: torch.Tensor):
    B, D = x.shape
    N, _ = codebook.shape
    if B <= 4096 and ((N <= 64 and D <= 128) or (N <= 128 and D <= 64)):
        return quantize_fwd(x, codebook)
    return quantize_fwd_mm(x, codebook)[0]


def nearest_neighbor_quantize(x: torch.Tensor, codebook: torch.Tensor):
    assert x.device == codebook.device, (
        f"Expected both tensors to be on the same device. Found: {x.device} and {codebook.device}"
    )
    if x.is_cuda:
        return _cuda_nearest_neighbor_quantize(x, codebook)
    return torch.cdist(x, codebook).min(-1)[1]
