import torch

from rectokens.decoding.constrained_decoding import (
    ConstrainedGenerationState,
    ConstraintState,
    GenerationState,
    ModelInferenceOutput,
    RandomModel,
    decode_one_step,
)
from rectokens.decoding.csr import csr_from_sorted_batch


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

device = torch.device("cuda")


def make_trie(seqs, vocab_size=8, dense_lookup_layers=2):
    t = torch.tensor(sorted(seqs), dtype=torch.long, device=device)
    return csr_from_sorted_batch(
        t, vocab_size=vocab_size, dense_lookup_layers=dense_lookup_layers
    )


def test_decode_one_step_dense_path_xfail():
    """
    Exercise the dense-lookup branch (step < dense_lookup_layers).
    Fails because of the ndim() bug.
    """
    vocab_size = 8
    seqs = [
        [1, 2, 3, 4, 5],
        [1, 2, 7, 3, 6],
        [3, 1, 2, 5, 4],
        [2, 5, 1, 3, 7],
        [4, 6, 2, 1, 3],
    ]
    trie = make_trie(seqs, vocab_size=vocab_size, dense_lookup_layers=2)

    model = RandomModel(vocab_size=vocab_size, hidden_size=16).to(device)

    B, seq_len = 2, 4
    input_ids = torch.randint(0, vocab_size, (B, seq_len), device=device)

    init_state = ConstrainedGenerationState(
        generation_state=None,
        constraint_state=ConstraintState(trie=trie, cur_node=None),
    )

    result = decode_one_step(
        constrained_generation_state=init_state,
        model_fwd=model,
        input_ids=input_ids,
        step=0,
        vocab_size=vocab_size,
    )
    assert isinstance(result, ConstrainedGenerationState)


def test_autoregressive_generation_loop():
    """
    Run a full autoregressive generation loop for vocab_size steps.
    At each step the newly sampled token is fed back as input_ids.
    """
    vocab_size = 8
    sem_ids_length = 5
    seqs = [
        [1, 2, 3, 4, 5],
        [1, 2, 7, 3, 6],
        [3, 1, 2, 5, 4],
        [2, 5, 1, 3, 7],
        [4, 6, 2, 1, 3],
    ]
    trie = make_trie(seqs, vocab_size=vocab_size, dense_lookup_layers=2)

    model = RandomModel(vocab_size=vocab_size, hidden_size=16).to(device)

    B, seq_len = 2, 4
    input_ids = torch.randint(0, vocab_size, (B, seq_len), device=device)

    state = ConstrainedGenerationState(
        generation_state=None,
        constraint_state=ConstraintState(trie=trie, cur_node=None),
    )

    for step in range(sem_ids_length):
        state = decode_one_step(
            constrained_generation_state=state,
            model_fwd=model,
            input_ids=input_ids,
            step=step,
            vocab_size=vocab_size,
        )
        assert isinstance(state, ConstrainedGenerationState)
        assert state.generation_state is not None
        # generated_ids grows by one token each step
        assert state.generation_state.generated_ids.shape == (B, step + 1)
        # feed the last sampled token as input for the next step
        input_ids = state.generation_state.generated_ids[:, -1:]

    generated = state.generation_state.generated_ids
    index = torch.tensor(seqs, device=device)

    # Check that all generated ids are valid.
    assert (index.unsqueeze(0) == generated.unsqueeze(1)).all(-1).any(-1).all(-1)


if __name__ == "__main__":
    # test_decode_one_step_dense_path_xfail()
    test_autoregressive_generation_loop()
