from typing import NamedTuple


class GenerationConfig(NamedTuple):
    steps: int
    k: int = 1
    beam_size: int = 1
    temperature: float = 1.0
