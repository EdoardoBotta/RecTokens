import torch
from rectokens.schemas.compact_csr_trie import CompactCSRTrie
from typing import NamedTuple
from typing import Optional


class GenerationState(NamedTuple):
    generated_ids: torch.Tensor
    kv_cache: dict[int, torch.Tensor]
    log_probas: Optional[torch.Tensor]


class ConstraintState(NamedTuple):
    trie: CompactCSRTrie
    cur_node: Optional[torch.Tensor]


class ConstrainedGenerationState(NamedTuple):
    generation_state: Optional[GenerationState]
    constraint_state: Optional[ConstraintState]
