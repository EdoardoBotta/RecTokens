from __future__ import annotations

import unittest

import torch

if not torch.cuda.is_available():
    raise unittest.SkipTest("CUDA required")

from torch import nn

from rectokens.schemas.state import (
    ConstrainedGenerationState,
    ConstraintState,
    GenerationState,
)
from rectokens.schemas.config import GenerationConfig
from rectokens.decoding.constrained_decoding import (
    ModelInferenceOutput,
    RandomModel,
    decode_one_step,
)
from rectokens.schemas.compact_csr_trie import CompactCSRTrie

device = torch.device("cuda")

SEQS_5 = [
    [1, 2, 3, 4, 5],
    [1, 2, 7, 3, 6],
    [3, 1, 2, 5, 4],
    [2, 5, 1, 3, 7],
    [4, 6, 2, 1, 3],
]

SEQS_3 = [
    [1, 2, 3],
    [1, 2, 7],
    [3, 1, 2],
    [2, 5, 1],
    [4, 6, 2],
]


def make_trie(seqs, vocab_size=8, dense_lookup_layers=2):
    t = torch.tensor(sorted(seqs), dtype=torch.long, device=device)
    return CompactCSRTrie.from_sorted_batch(
        t, vocab_size=vocab_size, dense_lookup_layers=dense_lookup_layers
    )


class _ModelWithKVCache(nn.Module):
    """Like RandomModel but returns a non-None kv_cache so expansion can be tested."""

    def __init__(self, vocab_size, hidden_size, cache_dim=4):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.cache_dim = cache_dim

    def forward(self, input_ids, kv_cache=None):
        x = self.embedding(input_ids)
        logits = self.linear(x)
        new_cache = {
            0: torch.zeros(input_ids.shape[0], self.cache_dim, device=input_ids.device)
        }
        return ModelInferenceOutput(logits=logits[:, -1], kv_cache=new_cache)


def _run_beam_loop(trie, model, input_ids, vocab_size, seq_len, k, beam_size):
    state = ConstrainedGenerationState(
        generation_config=GenerationConfig(steps=seq_len, k=k, beam_size=beam_size),
        constraint_state=ConstraintState(step=0, trie=trie, cur_node=None),
    )
    for _ in range(seq_len):
        state = decode_one_step(
            constrained_generation_state=state,
            model_fwd=model,
            input_ids=input_ids,
        )
        gen = state.generation_state.generated_ids
        input_ids = gen.reshape(-1, gen.shape[-1])[:, -1:]
    return state


class TestConstrainedDecoding(unittest.TestCase):
    def test_decode_one_step_dense_path_xfail(self) -> None:
        """Exercise the dense-lookup branch (step < dense_lookup_layers)."""
        vocab_size = 8
        trie = make_trie(SEQS_5, vocab_size=vocab_size, dense_lookup_layers=2)
        model = RandomModel(vocab_size=vocab_size, hidden_size=16).to(device)

        B, seq_len = 2, 4
        input_ids = torch.randint(0, vocab_size, (B, seq_len), device=device)

        init_state = ConstrainedGenerationState(
            generation_config=GenerationConfig(steps=1, k=1, beam_size=1),
            constraint_state=ConstraintState(step=0, trie=trie, cur_node=None),
        )
        result = decode_one_step(
            constrained_generation_state=init_state,
            model_fwd=model,
            input_ids=input_ids,
        )
        assert isinstance(result, ConstrainedGenerationState)

    def test_autoregressive_generation_loop(self) -> None:
        """Run a full autoregressive generation loop for vocab_size steps."""
        vocab_size = 8
        sem_ids_length = 5
        trie = make_trie(SEQS_5, vocab_size=vocab_size, dense_lookup_layers=2)
        model = RandomModel(vocab_size=vocab_size, hidden_size=16).to(device)

        B, seq_len = 2, 4
        input_ids = torch.randint(0, vocab_size, (B, seq_len), device=device)

        state = ConstrainedGenerationState(
            generation_config=GenerationConfig(steps=sem_ids_length, k=1, beam_size=1),
            constraint_state=ConstraintState(step=0, trie=trie, cur_node=None),
        )
        for step in range(sem_ids_length):
            state = decode_one_step(
                constrained_generation_state=state,
                model_fwd=model,
                input_ids=input_ids,
            )
            assert isinstance(state, ConstrainedGenerationState)
            assert state.generation_state is not None
            assert state.generation_state.generated_ids.shape == (B, 1, step + 1)
            input_ids = state.generation_state.generated_ids[:, 0, -1:]

        generated = state.generation_state.generated_ids[:, 0, :]
        index = torch.tensor(SEQS_5, device=device)
        assert (index.unsqueeze(0) == generated.unsqueeze(1)).all(-1).any(-1).all(-1)

    def test_beam_search_shapes(self) -> None:
        """generated_ids is (B, k, step+1) and log_probas is (B*k,) at every step."""
        vocab_size = 8
        trie = make_trie(SEQS_5, vocab_size=vocab_size, dense_lookup_layers=2)
        model = RandomModel(vocab_size=vocab_size, hidden_size=16).to(device)

        B, seq_len, k, beam_size = 2, 4, 3, 6
        input_ids = torch.randint(0, vocab_size, (B, seq_len), device=device)

        state = ConstrainedGenerationState(
            generation_config=GenerationConfig(
                steps=len(SEQS_5[0]), k=k, beam_size=beam_size
            ),
            constraint_state=ConstraintState(step=0, trie=trie, cur_node=None),
        )
        for step in range(len(SEQS_5[0])):
            state = decode_one_step(
                constrained_generation_state=state,
                model_fwd=model,
                input_ids=input_ids,
            )
            gen = state.generation_state
            assert gen.generated_ids.shape == (B, k, step + 1)
            assert gen.log_probas.shape == (B * k,)
            input_ids = gen.generated_ids.reshape(-1, gen.generated_ids.shape[-1])[
                :, -1:
            ]

    def test_beam_search_all_outputs_valid_mixed_dense_nondense(self) -> None:
        """Every beam produces a valid trie sequence (dense_lookup_layers=2, seq_len=5)."""
        vocab_size = 8
        trie = make_trie(SEQS_5, vocab_size=vocab_size, dense_lookup_layers=2)
        model = RandomModel(vocab_size=vocab_size, hidden_size=16).to(device)

        B, seq_len, k, beam_size = 2, 4, 4, 8
        input_ids = torch.randint(0, vocab_size, (B, seq_len), device=device)

        state = _run_beam_loop(
            trie, model, input_ids, vocab_size, len(SEQS_5[0]), k, beam_size
        )

        generated = state.generation_state.generated_ids.reshape(-1, len(SEQS_5[0]))
        index = torch.tensor(SEQS_5, device=device)
        matches = (generated.unsqueeze(1) == index.unsqueeze(0)).all(-1).any(-1)
        assert matches.all()

    def test_beam_search_all_outputs_valid_dense_only(self) -> None:
        """Every beam is valid when dense_lookup_layers covers all steps."""
        vocab_size = 8
        sem_ids_length = len(SEQS_3[0])
        trie = make_trie(
            SEQS_3, vocab_size=vocab_size, dense_lookup_layers=sem_ids_length
        )
        model = RandomModel(vocab_size=vocab_size, hidden_size=16).to(device)

        B, seq_len, k, beam_size = 2, 4, 4, 8
        input_ids = torch.randint(0, vocab_size, (B, seq_len), device=device)

        state = _run_beam_loop(
            trie, model, input_ids, vocab_size, sem_ids_length, k, beam_size
        )

        generated = state.generation_state.generated_ids.reshape(-1, len(SEQS_3[0]))
        index = torch.tensor(SEQS_3, device=device)
        matches = (generated.unsqueeze(1) == index.unsqueeze(0)).all(-1).any(-1)
        assert matches.all()

    def test_beam_search_all_outputs_valid_nondense_dominated(self) -> None:
        """Every beam is valid when most steps use the non-dense (CSR kernel) path."""
        vocab_size = 8
        sem_ids_length = len(SEQS_3[0])
        trie = make_trie(SEQS_3, vocab_size=vocab_size, dense_lookup_layers=1)
        model = RandomModel(vocab_size=vocab_size, hidden_size=16).to(device)

        B, seq_len, k, beam_size = 2, 4, 4, 8
        input_ids = torch.randint(0, vocab_size, (B, seq_len), device=device)

        state = _run_beam_loop(
            trie, model, input_ids, vocab_size, sem_ids_length, k, beam_size
        )

        generated = state.generation_state.generated_ids.reshape(-1, len(SEQS_3[0]))
        index = torch.tensor(SEQS_3, device=device)
        matches = (generated.unsqueeze(1) == index.unsqueeze(0)).all(-1).any(-1)
        assert matches.all()

    def test_beam_search_cumulative_log_probas(self) -> None:
        """log_probas are <= 0 at every step and the running maximum never increases."""
        vocab_size = 8
        trie = make_trie(SEQS_5, vocab_size=vocab_size, dense_lookup_layers=2)
        model = RandomModel(vocab_size=vocab_size, hidden_size=16).to(device)

        B, seq_len, k, beam_size = 2, 4, 3, 6
        input_ids = torch.randint(0, vocab_size, (B, seq_len), device=device)

        state = ConstrainedGenerationState(
            generation_config=GenerationConfig(
                steps=len(SEQS_5[0]), k=k, beam_size=beam_size
            ),
            constraint_state=ConstraintState(step=0, trie=trie, cur_node=None),
        )
        prev_max = 0.0
        for step in range(len(SEQS_5[0])):
            state = decode_one_step(
                constrained_generation_state=state,
                model_fwd=model,
                input_ids=input_ids,
            )
            lp = state.generation_state.log_probas
            assert (lp <= 0).all()
            cur_max = lp.max().item()
            assert cur_max <= prev_max + 1e-6
            prev_max = cur_max
            gen = state.generation_state.generated_ids
            input_ids = gen.reshape(-1, gen.shape[-1])[:, -1:]

    def test_beam_search_kv_cache_expansion(self) -> None:
        """KV cache batch dim is B*k after step 0 and stays B*k on subsequent steps."""
        vocab_size = 8
        trie = make_trie(SEQS_5, vocab_size=vocab_size, dense_lookup_layers=2)
        model = _ModelWithKVCache(vocab_size=vocab_size, hidden_size=16).to(device)

        B, seq_len, k, beam_size = 2, 4, 3, 6
        input_ids = torch.randint(0, vocab_size, (B, seq_len), device=device)

        state = ConstrainedGenerationState(
            generation_config=GenerationConfig(
                steps=len(SEQS_5[0]), k=k, beam_size=beam_size
            ),
            constraint_state=ConstraintState(step=0, trie=trie, cur_node=None),
        )
        for step in range(len(SEQS_5[0])):
            state = decode_one_step(
                constrained_generation_state=state,
                model_fwd=model,
                input_ids=input_ids,
            )
            kv = state.generation_state.kv_cache
            assert kv is not None
            assert kv[0].shape[0] == B * k
            gen = state.generation_state.generated_ids
            input_ids = gen.reshape(-1, gen.shape[-1])[:, -1:]

    def test_beam_search_k1_beam1_is_valid(self) -> None:
        """k=1, beam_size=1 still produces valid sequences (should match greedy behaviour)."""
        vocab_size = 8
        trie = make_trie(SEQS_5, vocab_size=vocab_size, dense_lookup_layers=2)
        model = RandomModel(vocab_size=vocab_size, hidden_size=16).to(device)

        B, seq_len = 2, 4
        input_ids = torch.randint(0, vocab_size, (B, seq_len), device=device)

        state = _run_beam_loop(
            trie, model, input_ids, vocab_size, len(SEQS_5[0]), k=1, beam_size=1
        )

        generated = state.generation_state.generated_ids
        assert generated.shape == (B, 1, len(SEQS_5[0]))
        generated_flat = generated.squeeze(1)
        index = torch.tensor(SEQS_5, device=device)
        matches = (generated_flat.unsqueeze(1) == index.unsqueeze(0)).all(-1).any(-1)
        assert matches.all()
