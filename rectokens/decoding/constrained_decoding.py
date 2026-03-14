import torch
import torch.nn.functional as F
from rectokens.decoding.schemas.compact_csr_trie import CompactCSRTrie
from rectokens.decoding.kernels.vtnk import constrained_node_transition
from typing import NamedTuple
from typing import Optional
from torch import nn


class GenerationState(NamedTuple):
    generated_ids: torch.Tensor
    kv_cache: dict[int, torch.Tensor]


class ConstraintState(NamedTuple):
    trie: CompactCSRTrie
    cur_node: Optional[torch.Tensor]


class ConstrainedGenerationState(NamedTuple):
    generation_state: Optional[GenerationState]
    constraint_state: Optional[ConstraintState]


class ModelInferenceOutput(NamedTuple):
    logits: torch.Tensor
    kv_cache: Optional[dict[int, torch.Tensor]]


class RandomModel(nn.Module):
    def __init__(self, vocab_size, hidden_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.linear = nn.Linear(hidden_size, vocab_size)

    def forward(self, input_ids, kv_cache=None):
        x = self.embedding(input_ids)
        logits = self.linear(x)
        return ModelInferenceOutput(logits=logits[:, -1], kv_cache=None)


@torch.inference_mode()
def decode_one_step(
    constrained_generation_state,
    model_fwd,
    input_ids,
    step,
    vocab_size,
    temperature=1.0,
):
    assert input_ids.ndim == 2  # (B, seq_len)

    trie = constrained_generation_state.constraint_state.trie
    kv_cache = (
        constrained_generation_state.generation_state.kv_cache
        if constrained_generation_state.generation_state is not None
        else None
    )
    model_output = model_fwd(input_ids, kv_cache)
    logits = model_output.logits

    assert logits.ndim == 2  # (B, vocab_size)

    next_node = None
    if step < len(trie.dense_mask_by_layer):
        layer_mask = trie.dense_mask_by_layer[step]
        if step > 0:
            generated_ids = constrained_generation_state.generation_state.generated_ids
            layer_mask = layer_mask[generated_ids.unbind(-1)]
        else:
            layer_mask = layer_mask.unsqueeze(0).expand(logits.shape[0], -1)
        logits[~layer_mask] = float("-inf")
    else:
        next_nodes, valid_idxs, logits = constrained_node_transition(
            logits,
            cur_node=constrained_generation_state.constraint_state.cur_node,
            constraint_transitions=trie,
            step=step,
            vocab_size=vocab_size,
        )

    probas_batched = F.softmax(logits / temperature, dim=-1)
    samples_batched = torch.multinomial(probas_batched, num_samples=1)

    generated_ids = (
        samples_batched
        if constrained_generation_state.generation_state is None
        else torch.cat(
            [
                constrained_generation_state.generation_state.generated_ids,
                samples_batched,
            ],
            dim=-1,
        )
    )

    if step == len(trie.dense_mask_by_layer) - 1:
        next_node = trie.dense_states[generated_ids.unbind(-1)]
    elif step >= len(trie.dense_mask_by_layer):
        next_node = next_nodes[samples_batched == valid_idxs]

    return ConstrainedGenerationState(
        generation_state=GenerationState(
            generated_ids=generated_ids, kv_cache=model_output.kv_cache
        ),
        constraint_state=ConstraintState(
            trie=trie,
            cur_node=next_node,
        ),
    )
