import torch
import torch.nn.functional as F
from rectokens.schemas.compact_csr_trie import CompactCSRTrie
from rectokens.schemas.config import GenerationConfig
from rectokens.schemas.state import ConstrainedGenerationState
from rectokens.schemas.state import ConstraintState
from rectokens.schemas.state import GenerationState
from rectokens.ops.constrained_node_transition import constrained_node_transition
from rectokens.modules.sparse_linear import SparseLinear
from rectokens.modules.constraint_enforcer import ConstraintEnforcer
from typing import Optional
from torch import nn
from transformers.modeling_outputs import CausalLMOutputWithPast


class RandomModel(nn.Module):
    def __init__(self, vocab_size, hidden_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.linear = nn.Linear(hidden_size, vocab_size, bias=False)

    def forward(
        self, input_ids, past_key_values=None, use_cache=False, attention_mask=None
    ):
        x = self.embedding(input_ids)
        logits = self.linear(x)  # (B, seq_len, vocab_size)
        return CausalLMOutputWithPast(logits=logits, past_key_values=None)


@torch.inference_mode()
def autoregressive_generate(
    model: nn.Module,
    trie: CompactCSRTrie,
    input_ids: torch.Tensor,
    generation_config: GenerationConfig,
    attr_path: Optional[str] = None,
    attention_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Run constrained autoregressive generation for `seq_len` steps.

    When `attr_path` is given, a ConstraintEnforcer is created and applied to
    the model before generation, replacing the output projection at that path
    with a SparseLinear in-place.

    Returns:
        generated_ids: (B, k, seq_len) tensor of generated token ids.
    """
    if attr_path is not None:
        ConstraintEnforcer(attr_path).prepare(model)

    state = ConstrainedGenerationState(
        generation_config=generation_config,
        generation_state=None,
        constraint_state=ConstraintState(step=0, trie=trie, cur_node=None),
    )

    for _ in range(generation_config.steps):
        state = decode_one_step(
            constrained_generation_state=state,
            model_fwd=model,
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        gen = state.generation_state.generated_ids  # (B, k, step+1)
        input_ids = gen.reshape(-1, gen.shape[-1])[:, -1:]  # (B*k, 1)
        attention_mask = None  # subsequent steps use accumulated mask from state

    return state.generation_state.generated_ids


def _reindex_past_key_values(
    past_key_values: Optional[tuple], indices: torch.Tensor
) -> Optional[tuple]:
    if past_key_values is None:
        return None
    return tuple((k[indices], v[indices]) for k, v in past_key_values)


@torch.inference_mode()
def decode_one_step(
    constrained_generation_state: ConstrainedGenerationState,
    model_fwd: nn.Module,
    input_ids: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
) -> ConstrainedGenerationState:
    assert input_ids.ndim == 2  # (B, seq_len) on step 0, (B*k, 1) on subsequent steps

    trie = constrained_generation_state.constraint_state.trie
    generation_state = constrained_generation_state.generation_state
    is_first_step = generation_state is None
    step = constrained_generation_state.constraint_state.step
    config = constrained_generation_state.generation_config
    k, beam_size = config.k, config.beam_size

    if is_first_step:
        past_key_values, log_probas = None, None
        # attention_mask comes from the function argument (prompt padding mask)
    else:
        past_key_values = generation_state.past_key_values
        log_probas = generation_state.log_probas
        # Accumulated mask from prior steps: (B*k, prompt_len + steps_done)
        attention_mask = generation_state.attention_mask

    constrained_linear = (
        next((m for m in model_fwd.modules() if isinstance(m, SparseLinear)), None)
        if isinstance(model_fwd, nn.Module)
        else None
    )

    if constrained_linear is not None and step >= len(trie.dense_mask_by_layer):
        constrained_linear.set_context(constrained_generation_state.constraint_state)

    model_output = model_fwd(
        input_ids,
        past_key_values=past_key_values,
        attention_mask=attention_mask,
        use_cache=True,
    )
    logits = model_output.logits[:, -1, :]  # (B, vocab_size)
    new_past_kv = model_output.past_key_values

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
        if constrained_linear is not None:
            next_nodes, valid_idxs = (
                constrained_linear.next_nodes,
                constrained_linear.valid_idxs,
            )
        else:
            next_nodes, valid_idxs, logits = constrained_node_transition(
                logits, constraint_state=constrained_generation_state.constraint_state
            )

    probas_batched = F.softmax(logits / config.temperature, dim=-1)
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

        # Expand past_key_values to match the new B*k batch size
        expand_ids = torch.arange(B, device=input_ids.device).repeat_interleave(k)
        new_past_kv = _reindex_past_key_values(new_past_kv, expand_ids)
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

        # Identify parent beams and reorder generated_ids / past_key_values accordingly
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

        # Reorder past_key_values to match selected parent beams
        new_past_kv = _reindex_past_key_values(new_past_kv, flat_parent_ids)

    # Build the accumulated attention mask for the next step.
    # flat_parent_ids (shape B*k) reindexes either the prompt (B rows → B*k rows on
    # the first step) or the current beams (B*k → B*k after parent selection).
    # Appending a column of 1s accounts for the token just generated.
    if attention_mask is not None:
        new_attention_mask = torch.cat(
            [
                attention_mask[flat_parent_ids],
                torch.ones(
                    B * k, 1, dtype=attention_mask.dtype, device=attention_mask.device
                ),
            ],
            dim=1,
        )
    else:
        new_attention_mask = None

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
        generation_config=config,
        generation_state=GenerationState(
            generated_ids=generated_ids.reshape(B, k, -1),
            past_key_values=new_past_kv,
            log_probas=new_log_probas,
            attention_mask=new_attention_mask,
        ),
        constraint_state=ConstraintState(
            step=step + 1,
            trie=trie,
            cur_node=next_node,
        ),
    )
