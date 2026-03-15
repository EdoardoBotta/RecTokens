import torch
import torch.nn.functional as F
from rectokens.schemas.compact_csr_trie import CompactCSRTrie
from rectokens.schemas.state import ConstrainedGenerationState
from rectokens.schemas.state import ConstraintState
from rectokens.schemas.state import GenerationState
from rectokens.ops.constrained_node_transition import constrained_node_transition
from rectokens.ops.constrained_node_transition import (
    fused_linear_constrained_node_transition,
)
from rectokens.modules.constrained_linear import ConstrainedLinear
from rectokens.modules.constraint_enforcer import ConstraintEnforcer
from typing import NamedTuple
from typing import Optional
from torch import nn


class ModelInferenceOutput(NamedTuple):
    logits: torch.Tensor
    kv_cache: Optional[dict[int, torch.Tensor]]


class RandomModel(nn.Module):
    def __init__(self, vocab_size, hidden_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.linear = nn.Linear(hidden_size, vocab_size, bias=False)

    def forward(self, input_ids, kv_cache=None):
        x = self.embedding(input_ids)
        hidden = x[:, -1]
        logits = self.linear(hidden)
        return ModelInferenceOutput(logits=logits, kv_cache=None)


@torch.inference_mode()
def autoregressive_generate(
    model: nn.Module,
    trie: CompactCSRTrie,
    input_ids: torch.Tensor,
    seq_len: int,
    vocab_size: int,
    attr_path: Optional[str] = None,
    temperature: float = 1.0,
    k: int = 1,
    beam_size: int = 1,
) -> torch.Tensor:
    """
    Run constrained autoregressive generation for `seq_len` steps.

    When `attr_path` is given, a ConstraintEnforcer is created and applied to
    the model before generation, replacing the output projection at that path
    with a ConstrainedLinear in-place.

    Returns:
        generated_ids: (B, k, seq_len) tensor of generated token ids.
    """
    if attr_path is not None:
        ConstraintEnforcer(attr_path).convert(model)

    state = ConstrainedGenerationState(
        generation_state=None,
        constraint_state=ConstraintState(trie=trie, cur_node=None),
    )

    for step in range(seq_len):
        state = decode_one_step(
            constrained_generation_state=state,
            model_fwd=model,
            input_ids=input_ids,
            step=step,
            vocab_size=vocab_size,
            temperature=temperature,
            k=k,
            beam_size=beam_size,
        )
        gen = state.generation_state.generated_ids  # (B, k, step+1)
        input_ids = gen.reshape(-1, gen.shape[-1])[:, -1:]  # (B*k, 1)

    return state.generation_state.generated_ids


def _reindex_kv_cache(
    kv_cache: Optional[dict], indices: torch.Tensor
) -> Optional[dict]:
    if kv_cache is None:
        return None
    return {layer: tensor[indices] for layer, tensor in kv_cache.items()}


@torch.inference_mode()
def decode_one_step(
    constrained_generation_state: ConstrainedGenerationState,
    model_fwd: nn.Module,
    input_ids: torch.Tensor,
    step: int,
    vocab_size: int,
    temperature: float = 1.0,
    k: int = 1,
    beam_size: int = 1,
) -> ConstrainedGenerationState:
    assert input_ids.ndim == 2  # (B, seq_len) on step 0, (B*k, 1) on subsequent steps

    trie = constrained_generation_state.constraint_state.trie
    generation_state = constrained_generation_state.generation_state
    is_first_step = generation_state is None

    if is_first_step:
        kv_cache, log_probas = None, None
    else:
        kv_cache = generation_state.kv_cache
        log_probas = generation_state.log_probas

    constrained_linear = (
        next((m for m in model_fwd.modules() if isinstance(m, ConstrainedLinear)), None)
        if isinstance(model_fwd, nn.Module)
        else None
    )

    if constrained_linear is not None and step >= len(trie.dense_mask_by_layer):
        cur_node = constrained_generation_state.constraint_state.cur_node
        constrained_linear.set_context(cur_node, trie, step)

    model_output = model_fwd(input_ids, kv_cache)
    logits = model_output.logits
    if constrained_linear is not None:
        constrained_linear.clear_context()

    assert logits.ndim == 2  # (B or B*k, vocab_size)

    current_batch_size = logits.shape[0]
    B = current_batch_size if is_first_step else current_batch_size // k

    # Flatten (B, k, t) -> (B*k, t) for internal indexing; first step has no prior ids
    prev_generated_ids_flat = (
        None if is_first_step else generation_state.generated_ids.reshape(B * k, -1)
    )

    next_node = None
    next_nodes = None
    valid_idxs = None
    if step < len(trie.dense_mask_by_layer):
        layer_mask = trie.dense_mask_by_layer[step]
        if is_first_step:
            layer_mask = layer_mask.unsqueeze(0).expand(current_batch_size, -1)
        else:
            assert prev_generated_ids_flat is not None
            layer_mask = layer_mask[prev_generated_ids_flat.unbind(-1)]
        logits[~layer_mask] = float("-inf")
    else:
        cur_node = constrained_generation_state.constraint_state.cur_node
        if constrained_linear is not None:
            next_nodes, valid_idxs = (
                constrained_linear.next_nodes,
                constrained_linear.valid_idxs,
            )
        else:
            next_nodes, valid_idxs, logits = constrained_node_transition(
                logits,
                cur_node=cur_node,
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
        expand_ids = torch.arange(B, device=input_ids.device).repeat_interleave(k)
        new_kv_cache = _reindex_kv_cache(model_output.kv_cache, expand_ids)
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
        parent_beam_ids = (
            top_k_indices // beam_size
        )  # (B, k) — which beam [0,k) within each B
        flat_parent_ids = (
            torch.arange(B, device=input_ids.device).unsqueeze(1) * k + parent_beam_ids
        ).reshape(B * k)  # indices into [0, B*k)

        assert prev_generated_ids_flat is not None
        parent_generated_ids = prev_generated_ids_flat[flat_parent_ids]  # (B*k, step)
        top_k_samples = torch.gather(all_samples, 1, top_k_indices).reshape(B * k, 1)
        generated_ids = torch.cat([parent_generated_ids, top_k_samples], dim=-1)

        new_log_probas = top_k_log_probas.reshape(B * k)

        # Reorder kv_cache to match selected parent beams
        new_kv_cache = _reindex_kv_cache(model_output.kv_cache, flat_parent_ids)

    if step == len(trie.dense_mask_by_layer) - 1:
        # dense_states is indexed by the full token path; works per-beam since each
        # row of generated_ids is an independent beam path.
        next_node = trie.dense_states[generated_ids.unbind(-1)]
    elif step >= len(trie.dense_mask_by_layer):
        # next_nodes / valid_idxs: (current_batch, max_branches) — computed for the
        # pre-selection beams.  Reorder by flat_parent_ids so row i aligns with the
        # parent of new beam i, then find which branch column matches the token chosen
        # for each new beam.
        next_nodes_reordered = next_nodes[flat_parent_ids]  # (B*k, max_branches)
        valid_idxs_reordered = valid_idxs[flat_parent_ids]  # (B*k, max_branches)
        selected_tokens = generated_ids[:, -1]  # (B*k,)
        branch_match = valid_idxs_reordered == selected_tokens.unsqueeze(
            1
        )  # (B*k, max_branches)
        next_node = next_nodes_reordered[branch_match]  # (B*k,)

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
