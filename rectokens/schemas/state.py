import torch
from rectokens.schemas.compact_csr_trie import CompactCSRTrie
from rectokens.schemas.config import GenerationConfig
from typing import NamedTuple
from typing import Optional
from torch import Tensor


class GenerationState(NamedTuple):
    generated_ids: torch.Tensor
    past_key_values: Optional[tuple[tuple[Tensor, Tensor], ...]]
    log_probas: Optional[torch.Tensor]
    attention_mask: Optional[Tensor] = None


class ConstraintState(NamedTuple):
    step: int
    trie: CompactCSRTrie
    cur_node: Optional[torch.Tensor]


class ConstrainedGenerationState(NamedTuple):
    generation_config: GenerationConfig
    constraint_state: ConstraintState
    generation_state: Optional[GenerationState] = None
