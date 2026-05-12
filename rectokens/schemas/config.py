from typing import Literal, NamedTuple


class GenerationConfig(NamedTuple):
    steps: int
    k: int = 1
    beam_size: int = 1
    temperature: float = 1.0
    csr_kernel: Literal["default", "sample", "topk"] = "default"
