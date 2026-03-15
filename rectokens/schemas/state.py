import torch
from rectokens.schemas.compact_csr_trie import CompactCSRTrie
from rectokens.schemas.config import GenerationConfig
from typing import NamedTuple
from typing import Optional


class GenerationState(NamedTuple):
    generated_ids: torch.Tensor
    kv_cache: dict[int, torch.Tensor]
    log_probas: Optional[torch.Tensor]


class ConstraintState(NamedTuple):
    step: int
    trie: CompactCSRTrie
    cur_node: Optional[torch.Tensor]


class ConstrainedGenerationState(NamedTuple):
    generation_config: GenerationConfig
    constraint_state: ConstraintState
    generation_state: Optional[GenerationState] = None
