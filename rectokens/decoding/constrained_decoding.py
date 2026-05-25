import contextlib
import torch
import torch.nn.functional as F
from rectokens.schemas.compact_csr_trie import CompactCSRTrie
from rectokens.schemas.config import GenerationConfig
from rectokens.schemas.state import ConstrainedGenerationState
from rectokens.schemas.state import ConstraintState
from rectokens.schemas.state import GenerationState
from rectokens.ops.constrained_node_transition import constrained_node_transition
from rectokens.modules.sparse_linear import SparseLinear
from rectokens.modules.constraint_enforcer import SparseTrieConstraintEnforcer
from typing import Optional
from torch import nn


@torch.inference_mode()
def generate_with_item_constraints(
    model: nn.Module,
    input_ids: torch.Tensor,
    trie: CompactCSRTrie,
    item_sep_token_id: int,
    num_levels: int,
    max_new_tokens: int = 128,
    do_sample: bool = False,
    temperature: float = 1.0,
    eos_token_id: Optional[int] = None,
) -> torch.Tensor:
    """Token-by-token generation that constrains item codes to the valid catalog.

    Runs unconstrained autoregressive decoding until ``<|item_start|>`` is produced,
    then switches to trie-constrained decoding for exactly ``num_levels`` steps (one
    per RQ level), then returns to unconstrained decoding.  The switch repeats every
    time the model generates another ``<|item_start|>``, so multi-item responses are
    handled automatically.

    Only batch size 1 is supported; beam search is not used — use
    :func:`autoregressive_generate` when top-k retrieval is needed.

    Args:
        model: Causal LM with a KV-cache-compatible forward pass (``use_cache=True``).
        input_ids: Prompt token IDs, shape ``(1, seq_len)``.
        trie: :class:`~rectokens.schemas.compact_csr_trie.CompactCSRTrie` built via
            ``ItemAwareTokenizer.build_item_trie``.
        item_sep_token_id: HF vocab ID of ``<|item_start|>``.
        num_levels: Number of RQ levels; exactly this many tokens are constrained after
            each ``<|item_start|>``.
        max_new_tokens: Maximum total tokens to generate (includes constrained ones).
        do_sample: Sample from the distribution; greedy (argmax) when ``False``.
        temperature: Softmax temperature applied during sampling (ignored when
            ``do_sample=False``).
        eos_token_id: Stop generation when this token is produced.

    Returns:
        1-D long tensor of generated token IDs, **excluding** the input prompt.
    """
    assert input_ids.shape[0] == 1, (
        "generate_with_item_constraints supports batch_size=1 only"
    )
    device = input_ids.device

    generated: list[int] = []
    past_kv = None
    cur_input = input_ids

    # Constrained-phase bookkeeping
    constrained_steps_left: int = 0
    item_level: int = 0  # which RQ level we're constraining (0-indexed)
    cur_node: Optional[torch.Tensor] = None  # shape (1,) BFS node in trie
    code_tokens_so_far: list[int] = []  # HF token IDs chosen so far within this item

    for _ in range(max_new_tokens):
        output = model(cur_input, past_key_values=past_kv, use_cache=True)
        logits: torch.Tensor = output.logits[:, -1, :]  # (1, vocab_size)
        past_kv = output.past_key_values

        # Track CSR outputs so we can update cur_node after the token is chosen
        next_nodes: Optional[torch.Tensor] = None
        valid_idxs: Optional[torch.Tensor] = None

        # ── Constrained phase: mask logits to valid item tokens ──────────────
        if constrained_steps_left > 0:
            if item_level < len(trie.dense_mask_by_layer):
                # Dense lookup: trie.dense_mask_by_layer[l] is a bool tensor indexed
                # by all previously chosen item tokens at levels 0..l-1.
                layer_mask: torch.Tensor = trie.dense_mask_by_layer[item_level]
                if item_level > 0:
                    layer_mask = layer_mask[tuple(code_tokens_so_far)]
                logits[0, ~layer_mask] = float("-inf")
            else:
                # CSR trie: constrained_node_transition masks logits in-place and
                # returns the valid branch info needed to advance cur_node.
                constraint_state = ConstraintState(
                    step=item_level, trie=trie, cur_node=cur_node
                )
                next_nodes, valid_idxs, logits = constrained_node_transition(
                    logits, constraint_state
                )

        # ── Sample or pick greedily ───────────────────────────────────────────
        if do_sample and temperature > 0.0:
            probs = F.softmax(logits[0] / temperature, dim=-1)
            next_token = int(torch.multinomial(probs, num_samples=1).item())
        else:
            next_token = int(logits[0].argmax().item())

        # ── Advance constrained state ─────────────────────────────────────────
        if constrained_steps_left > 0:
            code_tokens_so_far.append(next_token)

            if item_level == len(trie.dense_mask_by_layer) - 1:
                # End of dense phase → compute the CSR root node from the full dense path.
                # dense_states is indexed by the tuple of dense-phase token IDs.
                cur_node = trie.dense_states[tuple(code_tokens_so_far)].reshape(1)
            elif item_level >= len(trie.dense_mask_by_layer):
                # CSR phase → find which branch corresponds to the chosen token.
                assert next_nodes is not None and valid_idxs is not None
                branch_match = valid_idxs[0] == next_token  # (max_branches,)
                cur_node = next_nodes[0, branch_match]  # (1,)

            item_level += 1
            constrained_steps_left -= 1

        else:
            # ── Unconstrained phase: watch for <|item_start|> ────────────────
            if next_token == item_sep_token_id:
                constrained_steps_left = num_levels
                item_level = 0
                cur_node = None
                code_tokens_so_far = []

        generated.append(next_token)
        cur_input = torch.tensor([[next_token]], dtype=torch.long, device=device)

        if eos_token_id is not None and next_token == eos_token_id:
            break

    return torch.tensor(generated, dtype=torch.long, device=device)


@torch.inference_mode()
def autoregressive_generate(
    model: nn.Module,
    trie: CompactCSRTrie,
    input_ids: torch.Tensor,
    generation_config: GenerationConfig,
    use_sparse_linear: bool = True,
    attention_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Run constrained autoregressive generation for `seq_len` steps.

    When ``use_sparse_linear=True`` (the default) and ``model.get_output_embeddings()``
    returns a plain ``nn.Linear``, the output projection is temporarily replaced with a
    ``SparseLinear`` for the duration of generation and restored afterwards.
    Pass ``use_sparse_linear=False`` to fall back to standard logit masking without the
    fused kernel.

    When ``generation_config.csr_kernel`` is ``"default"`` the appropriate fused kernel
    is selected automatically: ``"topk"`` for beam search (``beam_size > 1``), ``"sample"``
    for stochastic single-beam decoding (``temperature > 0``), and ``"default"`` otherwise.
    An explicit kernel choice in the config is always respected.

    Returns:
        generated_ids: (B, k, seq_len) tensor of generated token ids.
    """
    enforcer: Optional[SparseTrieConstraintEnforcer] = None
    if use_sparse_linear and hasattr(model, "get_output_embeddings"):
        output_emb = model.get_output_embeddings()
        if isinstance(output_emb, nn.Linear) and not isinstance(
            output_emb, SparseLinear
        ):
            enforcer = SparseTrieConstraintEnforcer()

    # Auto-select the fused CSR kernel when the caller left it at the default.
    # Prefer "topk" for beam search (beam_size > 1); fall back to "sample" for
    # stochastic single-beam decoding; keep "default" only for greedy single-beam.
    if enforcer is not None and generation_config.csr_kernel == "default":
        if generation_config.beam_size > 1:
            kernel = "topk"
        elif generation_config.temperature > 0:
            kernel = "sample"
        else:
            kernel = "default"
        generation_config = generation_config._replace(csr_kernel=kernel)

    ctx = enforcer.apply(model) if enforcer is not None else contextlib.nullcontext()
    with ctx:
        constrained_linear = (
            next((m for m in model.modules() if isinstance(m, SparseLinear)), None)
            if isinstance(model, nn.Module)
            else None
        )
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
                constrained_linear=constrained_linear,
                attention_mask=attention_mask,
            )
            gen = state.generation_state.generated_ids  # (B, k, step+1)
            input_ids = gen.reshape(-1, gen.shape[-1])[:, -1:]  # (B*k, 1)
            attention_mask = None  # subsequent steps use accumulated mask from state

        return state.generation_state.generated_ids


def _draw_beam_candidates(
    logits: torch.Tensor,
    beam_size: int,
    temperature: float,
    constrained_linear: Optional[SparseLinear],
    use_constrained: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample beam_size candidates per current beam.

    Returns (samples_batched, sampled_log_probas), both shape (current_batch, beam_size).
    """
    if use_constrained and constrained_linear.sample is not None:
        samples_batched = constrained_linear.sample.long().unsqueeze(1)  # (B, 1)
        sampled_log_probas = torch.zeros(
            logits.shape[0], 1, dtype=torch.float, device=logits.device
        )
    elif use_constrained and constrained_linear.topk_idxs is not None:
        samples_batched = constrained_linear.topk_idxs  # (B, beam_size) actual token IDs
        sampled_log_probas = F.log_softmax(logits, dim=-1)  # (B, beam_size)
    elif temperature == 0.0:
        _, samples_batched = logits.topk(beam_size, dim=-1)
        sampled_log_probas = F.log_softmax(logits, dim=-1).gather(1, samples_batched)
    else:
        probas_batched = F.softmax(logits / temperature, dim=-1)
        samples_batched = torch.multinomial(probas_batched, num_samples=beam_size)
        sampled_log_probas = probas_batched.log().gather(1, samples_batched)
    return samples_batched, sampled_log_probas


def _reindex_past_key_values(
    past_key_values: Optional[tuple], indices: torch.Tensor
) -> Optional[tuple]:
    if past_key_values is None:
        return None
    # HF DynamicCache (transformers >= 4.38): reorder in-place, no copy needed.
    if hasattr(past_key_values, "reorder_cache"):
        past_key_values.reorder_cache(indices)
        return past_key_values
    return tuple((k[indices], v[indices]) for k, v in past_key_values)


@torch.inference_mode()
def decode_one_step(
    constrained_generation_state: ConstrainedGenerationState,
    model_fwd: nn.Module,
    input_ids: torch.Tensor,
    constrained_linear: Optional[SparseLinear] = None,
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

    use_constrained = constrained_linear is not None and step >= len(
        trie.dense_mask_by_layer
    )
    ctx = (
        constrained_linear.constrained(
            constrained_generation_state.constraint_state,
            strategy=config.csr_kernel,
            temperature=config.temperature,
            k=config.beam_size,
        )
        if use_constrained
        else contextlib.nullcontext()
    )
    with ctx:
        model_output = model_fwd(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            use_cache=True,
        )
    logits = model_output.logits[:, -1, :]  # (B, vocab_size)
    new_past_kv = model_output.past_key_values

    assert logits.ndim == 2  # (B or B*k, vocab_size)

    current_batch_size = logits.shape[0]
    B = current_batch_size if is_first_step else current_batch_size // k

    # Flatten (B, k, t) -> (B*k, t) for internal indexing.
    # First step is treated as k_eff=1 with zero log-probas and empty history,
    # so the beam selection logic below is identical for both cases.
    k_eff = 1 if is_first_step else k
    prev_generated_ids_flat = (
        torch.empty(B * k_eff, 0, dtype=torch.long, device=input_ids.device)
        if is_first_step
        else generation_state.generated_ids.reshape(B * k, -1)
    )
    prev_log_probas = (
        torch.zeros(B * k_eff, device=input_ids.device)
        if is_first_step
        else log_probas
    )

    next_node = next_nodes = valid_idxs = None
    if step < len(trie.dense_mask_by_layer):
        layer_mask = trie.dense_mask_by_layer[step]
        if is_first_step:
            layer_mask = layer_mask.unsqueeze(0).expand(current_batch_size, -1)
        else:
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

    # Sample beam_size candidates per current beam: (current_batch, beam_size)
    samples_batched, sampled_log_probas = _draw_beam_candidates(
        logits, beam_size, config.temperature, constrained_linear, use_constrained
    )

    # Accumulate log-probas and pick top-k across all k_eff*n_cands candidates.
    # n_cands may be < beam_size when the trie has fewer valid branches than beam_size.
    # On the first step k_eff=1, so this reduces to selecting top-k from n_cands samples.
    n_cands = sampled_log_probas.shape[1]
    total_log_probas = prev_log_probas.reshape(B, k_eff).repeat_interleave(
        n_cands, dim=1
    ) + sampled_log_probas.reshape(B, k_eff * n_cands)
    all_samples = samples_batched.reshape(B, k_eff * n_cands)

    top_k_log_probas, top_k_indices = total_log_probas.topk(k, dim=1)  # (B, k)
    parent_beam_ids = top_k_indices // n_cands  # (B, k), values in [0, k_eff)
    flat_parent_ids = (
        torch.arange(B, device=input_ids.device).unsqueeze(1) * k_eff + parent_beam_ids
    ).reshape(B * k)

    top_k_samples = torch.gather(all_samples, 1, top_k_indices).reshape(B * k, 1)
    generated_ids = torch.cat([prev_generated_ids_flat[flat_parent_ids], top_k_samples], dim=-1)
    new_log_probas = top_k_log_probas.reshape(B * k)
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
