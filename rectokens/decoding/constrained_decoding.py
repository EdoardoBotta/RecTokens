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
    log_probas: Optional[torch.Tensor]


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
    k=1,
    beam_size=1,
):
    assert input_ids.ndim == 2  # (B, seq_len) on step 0, (B*k, 1) on subsequent steps

    trie = constrained_generation_state.constraint_state.trie
    generation_state = constrained_generation_state.generation_state
    is_first_step = generation_state is None

    kv_cache = generation_state.kv_cache if not is_first_step else None
    log_probas = generation_state.log_probas if not is_first_step else None

    model_output = model_fwd(input_ids, kv_cache)
    logits = model_output.logits

    assert logits.ndim == 2  # (B or B*k, vocab_size)

    current_batch_size = logits.shape[0]
    B = current_batch_size if is_first_step else current_batch_size // k

    # Flatten (B, k, t) -> (B*k, t) for internal indexing; first step has no prior ids
    prev_generated_ids_flat = (
        None if is_first_step
        else generation_state.generated_ids.reshape(B * k, -1)
    )

    next_node = None
    if step < len(trie.dense_mask_by_layer):
        layer_mask = trie.dense_mask_by_layer[step]
        if step > 0:
            assert prev_generated_ids_flat is not None
            layer_mask = layer_mask[prev_generated_ids_flat.unbind(-1)]
        else:
            layer_mask = layer_mask.unsqueeze(0).expand(current_batch_size, -1)
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
    # Sample beam_size candidates per current beam: (current_batch, beam_size)
    samples_batched = torch.multinomial(probas_batched, num_samples=beam_size)
    sampled_log_probas = torch.log(
        torch.gather(probas_batched, 1, samples_batched)
    )  # (current_batch, beam_size)

    if is_first_step:
        # Pick top-k candidates per batch item from beam_size samples
        top_k_log_probas, top_k_indices = sampled_log_probas.topk(k, dim=1)  # (B, k)
        top_k_samples = torch.gather(samples_batched, 1, top_k_indices)  # (B, k)

        generated_ids = top_k_samples.reshape(B * k, 1)
        new_log_probas = top_k_log_probas.reshape(B * k)

        # Each of the B originals produces k children: parent[b*k + i] = b
        flat_parent_ids = torch.arange(B, device=input_ids.device).repeat_interleave(k)

        # Expand kv_cache to match the new B*k batch size
        new_kv_cache = model_output.kv_cache
        if new_kv_cache is not None:
            new_kv_cache = {
                layer: tensor.repeat_interleave(k, dim=0)
                for layer, tensor in new_kv_cache.items()
            }
    else:
        assert log_probas is not None
        # Accumulate log-probas across beams: (B, k*beam_size)
        prev_log_probas_expanded = log_probas.reshape(B, k).repeat_interleave(
            beam_size, dim=1
        )
        total_log_probas = prev_log_probas_expanded + sampled_log_probas.reshape(
            B, k * beam_size
        )
        all_samples = samples_batched.reshape(B, k * beam_size)

        # Pick top-k per batch item
        top_k_log_probas, top_k_indices = total_log_probas.topk(k, dim=1)  # (B, k)

        # Identify parent beams and reorder generated_ids / kv_cache accordingly
        parent_beam_ids = top_k_indices // beam_size  # (B, k) — which beam [0,k) within each B
        flat_parent_ids = (
            torch.arange(B, device=input_ids.device).unsqueeze(1) * k + parent_beam_ids
        ).reshape(B * k)  # indices into [0, B*k)

        assert prev_generated_ids_flat is not None
        parent_generated_ids = prev_generated_ids_flat[flat_parent_ids]  # (B*k, step)
        top_k_samples = torch.gather(all_samples, 1, top_k_indices).reshape(B * k, 1)
        generated_ids = torch.cat([parent_generated_ids, top_k_samples], dim=-1)

        new_log_probas = top_k_log_probas.reshape(B * k)

        # Reorder kv_cache to match selected parent beams
        new_kv_cache = model_output.kv_cache
        if new_kv_cache is not None:
            new_kv_cache = {
                layer: tensor[flat_parent_ids]
                for layer, tensor in new_kv_cache.items()
            }

    if step == len(trie.dense_mask_by_layer) - 1:
        # dense_states is indexed by the full token path; works per-beam since each
        # row of generated_ids is an independent beam path.
        next_node = trie.dense_states[generated_ids.unbind(-1)]
    elif step >= len(trie.dense_mask_by_layer):
        # next_nodes / valid_idxs: (current_batch, max_branches) — computed for the
        # pre-selection beams.  Reorder by flat_parent_ids so row i aligns with the
        # parent of new beam i, then find which branch column matches the token chosen
        # for each new beam.
        next_nodes_reordered = next_nodes[flat_parent_ids]   # (B*k, max_branches)
        valid_idxs_reordered = valid_idxs[flat_parent_ids]   # (B*k, max_branches)
        selected_tokens = generated_ids[:, -1]               # (B*k,)
        branch_match = valid_idxs_reordered == selected_tokens.unsqueeze(1)  # (B*k, max_branches)
        next_node = next_nodes_reordered[branch_match]        # (B*k,)

    return ConstrainedGenerationState(
        generation_state=GenerationState(
            generated_ids=generated_ids.reshape(B, k, -1),
            kv_cache=new_kv_cache,
            log_probas=new_log_probas,
        ),
        constraint_state=ConstraintState(
            trie=trie,
            cur_node=next_node,
        ),
    )
