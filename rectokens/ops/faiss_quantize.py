import torch
import faiss
import faiss.contrib.torch_utils  # patches faiss to accept CUDA tensors directly


def make_gpu_index(codebook: torch.Tensor) -> faiss.GpuIndex:
    """Build a FAISS flat L2 GPU index from a codebook tensor."""
    assert codebook.is_cuda and codebook.dtype == torch.float32
    _, D = codebook.shape
    res = faiss.StandardGpuResources()
    cpu_index = faiss.IndexFlatL2(D)
    gpu_index = faiss.index_cpu_to_gpu(res, codebook.device.index, cpu_index)
    gpu_index.add(codebook)
    return gpu_index


def faiss_quantize(x: torch.Tensor, codebook: torch.Tensor) -> torch.Tensor:
    """Nearest-neighbor quantization via FAISS-GPU flat L2 search.

    Builds the index from codebook then searches. Equivalent to the Triton
    kernels in terms of inputs/outputs but uses FAISS-GPU internally.
    """
    assert x.is_cuda and codebook.is_cuda
    x = x.contiguous().float()
    codebook = codebook.contiguous().float()
    gpu_index = make_gpu_index(codebook)
    _, indices = gpu_index.search(x, 1)
    return indices.squeeze(1)
