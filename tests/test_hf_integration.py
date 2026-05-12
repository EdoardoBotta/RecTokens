"""Tests for ItemAwareTokenizer, InterleavedSequenceCollator, ItemAwareCausalLM.from_causal_lm.

Verifies that:
- Token IDs for item semantic codes and the <|item_start|> separator are computed
  with the correct formula and fall within the extended vocabulary range.
- encode_sequence inserts <|item_start|> exactly when transitioning text→item and
  never between consecutive items.
- Semantic IDs in the encoded sequence match the raw codes produced by the item
  tokenizer mapped to HF vocab space.
- decode_sequence recovers text spans and item codes (round-trip).
- InterleavedSequenceCollator produces correct shapes, attention masks, labels,
  and loss masks for loss_on in {"all", "items", "text"}.
- ItemAwareCausalLM.from_causal_lm correctly expands embed_tokens/lm_head.
"""

from __future__ import annotations

import unittest

import torch
import torch.nn as nn
from tokenizers import Tokenizer as _BackendTokenizer
from tokenizers.models import WordLevel as _WordLevel
from tokenizers.pre_tokenizers import FixedLength as _FixedLength
from transformers import PreTrainedTokenizerFast

from rectokens.core.tokenizer import TokenSequence
from rectokens.integrations.hf.collator import InterleavedSequenceCollator
from rectokens.integrations.hf.model import ItemAwareCausalLM
from rectokens.integrations.hf.tokenizer import ItemAwareTokenizer
from rectokens.tokenizers.rq_kmeans import RQKMeansTokenizer
from rectokens.tokenizers.rqvae import RQVAETokenizer

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

NUM_LEVELS = 2
CODEBOOK_SIZE = 8
DIM = 4
HIDDEN_SIZE = 16
N_ITEMS = 200


# ---------------------------------------------------------------------------
# Minimal mocks
# ---------------------------------------------------------------------------


def _make_char_fast_tok() -> PreTrainedTokenizerFast:
    """Character-level ``PreTrainedTokenizerFast`` for testing.

    Vocab layout: 0=<pad>, 1=<unk>, 2..N = printable ASCII characters.
    Each input character is encoded as its own token (Split pre-tokenizer).
    """
    chars = "abcdefghijklmnopqrstuvwxyz ABCDEFGHIJKLMNOPQRSTUVWXYZ.,!?"
    vocab: dict[str, int] = {"<pad>": 0, "<unk>": 1}
    for ch in chars:
        if ch not in vocab:
            vocab[ch] = len(vocab)

    backend = _BackendTokenizer(_WordLevel(vocab=vocab, unk_token="<unk>"))
    # FixedLength(1) splits into single-character chunks — true character-level encoding.
    backend.pre_tokenizer = _FixedLength(length=1)

    return PreTrainedTokenizerFast(
        tokenizer_object=backend,
        pad_token="<pad>",
        unk_token="<unk>",
    )


class _MockHFModel(nn.Module):
    """Minimal mock of a HuggingFace PreTrainedModel with tied embeddings."""

    def __init__(self, vocab_size: int, hidden_size: int) -> None:
        super().__init__()
        self._emb = nn.Embedding(vocab_size, hidden_size)
        self._lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
        self._lm_head.weight = self._emb.weight  # tie weights

    def get_input_embeddings(self) -> nn.Embedding:
        return self._emb

    def get_output_embeddings(self) -> nn.Linear:
        return self._lm_head

    def resize_token_embeddings(self, new_size: int) -> nn.Embedding:
        old_emb = self._emb
        hidden = old_emb.embedding_dim
        n_copy = min(old_emb.num_embeddings, new_size)

        new_emb = nn.Embedding(new_size, hidden)
        with torch.no_grad():
            new_emb.weight[:n_copy] = old_emb.weight[:n_copy]

        new_lm = nn.Linear(hidden, new_size, bias=False)
        new_lm.weight = new_emb.weight  # tie

        self._emb = new_emb
        self._lm_head = new_lm
        return new_emb


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


def _make_aware(hf_tok=None, item_tok=None) -> tuple[ItemAwareTokenizer, torch.Tensor]:
    """Return (aware_tokenizer, item_data)."""
    torch.manual_seed(0)
    data = torch.randn(N_ITEMS, DIM)
    if item_tok is None:
        item_tok = RQKMeansTokenizer(
            num_levels=NUM_LEVELS, codebook_size=CODEBOOK_SIZE, dim=DIM
        )
        item_tok.fit_step(data)
    if hf_tok is None:
        hf_tok = _make_char_fast_tok()
    aware = ItemAwareTokenizer(
        hf_tok, item_tok, num_levels=NUM_LEVELS, codebook_size=CODEBOOK_SIZE
    )
    return aware, data


# ---------------------------------------------------------------------------
# ItemAwareTokenizer tests
# ---------------------------------------------------------------------------


class TestItemAwareTokenizerIds(unittest.TestCase):
    """Token-ID formula tests — no item embeddings needed."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.aware, _ = _make_aware()
        cls.orig = cls.aware.original_vocab_size

    def test_original_vocab_size_preserved(self) -> None:
        assert self.orig == self.aware.original_vocab_size

    def test_vocab_size_after_registration(self) -> None:
        expected = (
            self.orig + NUM_LEVELS * CODEBOOK_SIZE + 2
        )  # +2 for <|item_start|> and <|item_end|>
        assert self.aware.vocab_size == expected

    def test_item_token_id_level0_code0(self) -> None:
        assert self.aware.item_token_id(0, 0) == self.orig

    def test_item_token_id_formula(self) -> None:
        for l in range(NUM_LEVELS):
            for c in range(CODEBOOK_SIZE):
                assert (
                    self.aware.item_token_id(l, c) == self.orig + l * CODEBOOK_SIZE + c
                )

    def test_item_sep_token_id(self) -> None:
        assert self.aware.item_sep_token_id == self.orig + NUM_LEVELS * CODEBOOK_SIZE

    def test_item_token_ids_all_in_vocab(self) -> None:
        for l in range(NUM_LEVELS):
            for c in range(CODEBOOK_SIZE):
                tid = self.aware.item_token_id(l, c)
                assert self.orig <= tid < self.aware.vocab_size

    def test_sep_token_id_in_vocab(self) -> None:
        assert self.aware.item_sep_token_id < self.aware.vocab_size

    def test_sep_token_id_is_last_extended(self) -> None:
        # <|item_end|> should be the very last registered token
        assert self.aware.item_end_token_id == self.aware.vocab_size - 1

    def test_item_tokens_are_contiguous_per_level(self) -> None:
        # Level l tokens occupy [item_token_id(l,0), item_token_id(l, CODEBOOK_SIZE-1)]
        for l in range(NUM_LEVELS):
            start = self.aware.item_token_id(l, 0)
            end = self.aware.item_token_id(l, CODEBOOK_SIZE - 1)
            assert end - start == CODEBOOK_SIZE - 1

    def test_levels_are_non_overlapping(self) -> None:
        for l in range(NUM_LEVELS - 1):
            end_l = self.aware.item_token_id(l, CODEBOOK_SIZE - 1)
            start_next = self.aware.item_token_id(l + 1, 0)
            assert end_l < start_next


class TestItemAwareTokenizerNoItemTokenizer(unittest.TestCase):
    """ItemAwareTokenizer with item_tokenizer=None: metadata works, encode raises."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.aware = ItemAwareTokenizer(
            _make_char_fast_tok(),
            num_levels=NUM_LEVELS,
            codebook_size=CODEBOOK_SIZE,
        )

    def test_vocab_size(self) -> None:
        expected = self.aware.original_vocab_size + NUM_LEVELS * CODEBOOK_SIZE + 2
        assert self.aware.vocab_size == expected

    def test_item_token_id(self) -> None:
        assert self.aware.item_token_id(0, 0) == self.aware.original_vocab_size

    def test_encode_sequence_text_only_works(self) -> None:
        ids = self.aware.encode_sequence(["hello"])
        assert len(ids) > 0
        assert all(i < self.aware.original_vocab_size for i in ids)

    def test_encode_sequence_tensor_raises(self) -> None:
        with self.assertRaises(RuntimeError):
            self.aware.encode_sequence([torch.randn(DIM)])


class TestEncodeSequence(unittest.TestCase):
    """encode_sequence: separator placement and semantic ID correctness."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.aware, data = _make_aware()
        cls.orig = cls.aware.original_vocab_size
        cls.item_emb_a = data[0]  # (DIM,)
        cls.item_emb_b = data[1]
        cls.codes_a = cls.aware.item_tokenizer.encode(
            cls.item_emb_a.unsqueeze(0)
        ).codes[0]
        cls.codes_b = cls.aware.item_tokenizer.encode(
            cls.item_emb_b.unsqueeze(0)
        ).codes[0]

    def _item_ids(self, codes: torch.Tensor) -> list[int]:
        return [
            self.aware.item_token_id(l, int(codes[l])) for l in range(NUM_LEVELS)
        ] + [self.aware.item_end_token_id]

    def _text_ids(self, text: str) -> list[int]:
        return self.aware.text_tokenizer.encode(text, add_special_tokens=False)

    # --- text-only ---

    def test_text_only_no_item_tokens(self) -> None:
        ids = self.aware.encode_sequence(["ab"])
        assert ids == self._text_ids("ab")
        assert all(i < self.orig for i in ids)

    # --- item-only ---

    def test_item_first_no_separator(self) -> None:
        """Item as first element must NOT be preceded by a separator."""
        ids = self.aware.encode_sequence([self.item_emb_a])
        expected = self._item_ids(self.codes_a)
        assert ids == expected
        assert self.aware.item_sep_token_id not in ids

    def test_item_only_length(self) -> None:
        ids = self.aware.encode_sequence([self.item_emb_a])
        assert len(ids) == NUM_LEVELS + 1  # NUM_LEVELS codes + <|item_end|>

    def test_item_semantic_ids_correct_hf_ids(self) -> None:
        """Semantic ID at level l must equal item_token_id(l, code_l)."""
        ids = self.aware.encode_sequence([self.item_emb_a])
        for l in range(NUM_LEVELS):
            expected_id = self.aware.item_token_id(l, int(self.codes_a[l]))
            assert ids[l] == expected_id

    def test_item_ids_all_in_extended_vocab(self) -> None:
        ids = self.aware.encode_sequence([self.item_emb_a])
        assert all(self.orig <= i < self.aware.vocab_size for i in ids)

    # --- separator placement ---

    def test_text_then_item_inserts_separator(self) -> None:
        ids = self.aware.encode_sequence(["ab", self.item_emb_a])
        expected = (
            self._text_ids("ab")
            + [self.aware.item_sep_token_id]
            + self._item_ids(self.codes_a)
        )
        assert ids == expected

    def test_consecutive_items_no_separator(self) -> None:
        """item → item must NOT produce a separator."""
        ids = self.aware.encode_sequence([self.item_emb_a, self.item_emb_b])
        expected = self._item_ids(self.codes_a) + self._item_ids(self.codes_b)
        assert ids == expected
        assert self.aware.item_sep_token_id not in ids

    def test_item_then_text_no_separator(self) -> None:
        """item → text must NOT produce a separator."""
        ids = self.aware.encode_sequence([self.item_emb_a, "ab"])
        expected = self._item_ids(self.codes_a) + self._text_ids("ab")
        assert ids == expected
        assert self.aware.item_sep_token_id not in ids

    def test_text_item_text_item_separators(self) -> None:
        """Full interleaved: separator before each item that follows text."""
        ids = self.aware.encode_sequence(["a", self.item_emb_a, "b", self.item_emb_b])
        sep = self.aware.item_sep_token_id
        expected = (
            self._text_ids("a")
            + [sep]
            + self._item_ids(self.codes_a)
            + self._text_ids("b")
            + [sep]
            + self._item_ids(self.codes_b)
        )
        assert ids == expected

    def test_item_text_item_only_second_has_separator(self) -> None:
        """item → text → item: only the second item (after text) gets a separator."""
        ids = self.aware.encode_sequence([self.item_emb_a, "a", self.item_emb_b])
        sep = self.aware.item_sep_token_id
        expected = (
            self._item_ids(self.codes_a)
            + self._text_ids("a")
            + [sep]
            + self._item_ids(self.codes_b)
        )
        assert ids == expected

    def test_separator_count_equals_text_to_item_transitions(self) -> None:
        """Exactly two text→item transitions → exactly two separators."""
        ids = self.aware.encode_sequence(["x", self.item_emb_a, "y", self.item_emb_b])
        sep = self.aware.item_sep_token_id
        assert ids.count(sep) == 2

    def test_no_separator_for_item_item_item(self) -> None:
        ids = self.aware.encode_sequence(
            [self.item_emb_a, self.item_emb_b, self.item_emb_a]
        )
        assert self.aware.item_sep_token_id not in ids


class TestDecodeSequence(unittest.TestCase):
    """decode_sequence: correct text/item splitting and code recovery."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.aware, data = _make_aware()
        cls.orig = cls.aware.original_vocab_size
        cls.item_emb_a = data[0]
        cls.item_emb_b = data[1]
        cls.codes_a = cls.aware.item_tokenizer.encode(
            cls.item_emb_a.unsqueeze(0)
        ).codes[0]
        cls.codes_b = cls.aware.item_tokenizer.encode(
            cls.item_emb_b.unsqueeze(0)
        ).codes[0]

    def _item_ids(self, codes: torch.Tensor) -> list[int]:
        return [self.aware.item_token_id(l, int(codes[l])) for l in range(NUM_LEVELS)]

    def _text_ids(self, text: str) -> list[int]:
        return self.aware.text_tokenizer.encode(text, add_special_tokens=False)

    def test_decode_text_only_returns_str(self) -> None:
        ids = self._text_ids("abc")
        result = self.aware.decode_sequence(ids)
        assert len(result) == 1
        assert isinstance(result[0], str)

    def test_decode_item_only_returns_token_sequence(self) -> None:
        ids = self._item_ids(self.codes_a)
        result = self.aware.decode_sequence(ids)
        assert len(result) == 1
        assert isinstance(result[0], TokenSequence)

    def test_decode_item_codes_exact(self) -> None:
        ids = self._item_ids(self.codes_a)
        result = self.aware.decode_sequence(ids)
        assert isinstance(result[0], TokenSequence)
        assert (result[0].codes == self.codes_a).all()

    def test_decode_separator_is_stripped(self) -> None:
        """<|item_start|> token is consumed and not returned as a part."""
        ids = [self.aware.item_sep_token_id] + self._item_ids(self.codes_a)
        result = self.aware.decode_sequence(ids)
        # Only the TokenSequence part; no None / separator element
        assert len(result) == 1
        assert isinstance(result[0], TokenSequence)

    def test_encode_decode_roundtrip_text(self) -> None:
        ids = self.aware.encode_sequence(["hello"])
        decoded = self.aware.decode_sequence(ids)
        assert len(decoded) == 1
        assert isinstance(decoded[0], str)

    def test_encode_decode_roundtrip_item_codes(self) -> None:
        ids = self.aware.encode_sequence([self.item_emb_a])
        decoded = self.aware.decode_sequence(ids)
        assert len(decoded) == 1
        assert isinstance(decoded[0], TokenSequence)
        assert (decoded[0].codes == self.codes_a).all()

    def test_encode_decode_roundtrip_mixed_structure(self) -> None:
        """Text, item, text roundtrip: structure and item codes preserved."""
        ids = self.aware.encode_sequence(["hi", self.item_emb_a, "bye"])
        decoded = self.aware.decode_sequence(ids)
        assert len(decoded) == 3
        assert isinstance(decoded[0], str)
        assert isinstance(decoded[1], TokenSequence)
        assert isinstance(decoded[2], str)
        assert (decoded[1].codes == self.codes_a).all()

    def test_encode_decode_two_items_codes_correct(self) -> None:
        ids = self.aware.encode_sequence([self.item_emb_a, self.item_emb_b])
        decoded = self.aware.decode_sequence(ids)
        assert len(decoded) == 2
        assert isinstance(decoded[0], TokenSequence)
        assert isinstance(decoded[1], TokenSequence)
        assert (decoded[0].codes == self.codes_a).all()
        assert (decoded[1].codes == self.codes_b).all()


# ---------------------------------------------------------------------------
# InterleavedSequenceCollator tests
# ---------------------------------------------------------------------------


class TestInterleavedSequenceCollator(unittest.TestCase):
    """Shapes, padding, attention masks, labels, and loss masking."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.aware, cls.data = _make_aware()
        cls.orig = cls.aware.original_vocab_size
        cls.item_emb = cls.data[0]

    def _collator(self, loss_on="all", padding_side="right", pad_id=0, max_length=None):
        return InterleavedSequenceCollator(
            self.aware,
            padding_side=padding_side,
            pad_token_id=pad_id,
            loss_on=loss_on,
            max_length=max_length,
        )

    # --- output shapes ---

    def test_output_keys(self) -> None:
        batch = self._collator()([["ab"]])
        assert set(batch.keys()) == {"input_ids", "attention_mask", "labels"}

    def test_output_shapes_single(self) -> None:
        batch = self._collator()([["abc"]])
        L = batch["input_ids"].shape[1]
        assert batch["attention_mask"].shape == (1, L)
        assert batch["labels"].shape == (1, L)

    def test_output_shapes_batch(self) -> None:
        batch = self._collator()([["ab"], ["abcd"]])
        assert batch["input_ids"].shape[0] == 2
        assert batch["attention_mask"].shape == batch["input_ids"].shape
        assert batch["labels"].shape == batch["input_ids"].shape

    def test_max_length_truncation(self) -> None:
        batch = self._collator(max_length=2)([["abcdef"]])
        assert batch["input_ids"].shape[1] == 2

    # --- attention mask and padding ---

    def test_attention_mask_short_right_pad(self) -> None:
        # 1-char vs 3-char: shorter seq should be padded on the right
        batch = self._collator(padding_side="right")([["a"], ["abc"]])
        assert int(batch["attention_mask"][0].sum()) == 1
        assert int(batch["attention_mask"][1].sum()) == 3

    def test_padding_tokens_right(self) -> None:
        batch = self._collator(padding_side="right", pad_id=0)([["a"], ["abc"]])
        # Positions after the short sequence should be pad_id=0
        assert batch["input_ids"][0, 1:].eq(0).all()

    def test_attention_mask_left_pad(self) -> None:
        batch = self._collator(padding_side="left")([["a"], ["abc"]])
        assert int(batch["attention_mask"][0].sum()) == 1
        assert int(batch["attention_mask"][1].sum()) == 3

    # --- label construction ---

    def test_first_label_always_minus100(self) -> None:
        batch = self._collator(loss_on="all")([["abc"]])
        assert batch["labels"][0, 0].item() == -100

    def test_labels_match_input_ids_at_non_first_positions(self) -> None:
        """labels[j] == input_ids[j] for j >= 1 (HF will shift internally)."""
        batch = self._collator(loss_on="all")([["abc"]])
        ids = batch["input_ids"][0]
        labels = batch["labels"][0]
        seq_len = int(batch["attention_mask"][0].sum())
        for j in range(1, seq_len):
            assert labels[j].item() == ids[j].item(), f"mismatch at position {j}"

    def test_padding_positions_are_minus100(self) -> None:
        batch = self._collator(padding_side="right")([["a"], ["abc"]])
        # Short sequence: positions [1, 3) are padding → labels = -100
        assert batch["labels"][0, 1].item() == -100
        assert batch["labels"][0, 2].item() == -100

    # --- loss masking ---

    def test_loss_on_all_no_minus100_except_first_and_pad(self) -> None:
        batch = self._collator(loss_on="all")([["abc"]])
        labels = batch["labels"][0]
        seq_len = int(batch["attention_mask"][0].sum())
        for j in range(1, seq_len):
            assert labels[j].item() != -100

    def test_loss_on_items_only_item_tokens_labeled(self) -> None:
        """loss_on='items': text tokens in labels must be -100."""
        batch = self._collator(loss_on="items")([["a", self.item_emb]])
        labels = batch["labels"][0]
        for j in range(1, batch["input_ids"].shape[1]):
            lbl = labels[j].item()
            if lbl == -100:
                continue
            assert lbl >= self.orig, (
                f"label {lbl} at pos {j} is a text token under loss_on='items'"
            )

    def test_loss_on_text_only_text_tokens_labeled(self) -> None:
        """loss_on='text': item tokens in labels must be -100."""
        batch = self._collator(loss_on="text")([["a", self.item_emb]])
        labels = batch["labels"][0]
        for j in range(1, batch["input_ids"].shape[1]):
            lbl = labels[j].item()
            if lbl == -100:
                continue
            assert lbl < self.orig, (
                f"label {lbl} at pos {j} is an item token under loss_on='text'"
            )

    def test_loss_on_items_vs_all_differ_when_text_present(self) -> None:
        """Switching loss_on='items' vs 'all' must change label values when text is present.

        Use a sequence where a text token appears at a position > 0 so that
        loss_on='items' produces -100 there while loss_on='all' does not.
        e.g. [item_emb, "a"] → token ids [item0, item1, t_a]: label at index 2
        is t_a (text, < orig_vocab) under 'all' but -100 under 'items'.
        """
        example = [[self.item_emb, "a"]]
        batch_all = self._collator(loss_on="all")(example)
        batch_items = self._collator(loss_on="items")(example)
        assert not (batch_all["labels"] == batch_items["labels"]).all()

    def test_loss_on_sep_token_masked_as_item(self) -> None:
        """<|item_start|> separator has id >= orig_vocab; it is treated as item token."""
        batch = self._collator(loss_on="items")([["a", self.item_emb]])
        labels = batch["labels"][0]
        ids = batch["input_ids"][0]
        sep_id = self.aware.item_sep_token_id
        # Find the separator position in input_ids and check label is not -100
        sep_positions = (ids == sep_id).nonzero(as_tuple=True)[0].tolist()
        for pos in sep_positions:
            if pos == 0:
                continue  # first position is always -100
            assert labels[pos].item() == sep_id

    # --- input_ids correctness ---

    def test_input_ids_match_encode_sequence(self) -> None:
        parts = ["a", self.item_emb]
        expected = self.aware.encode_sequence(parts)
        batch = self._collator()([parts])
        n = len(expected)
        assert (
            batch["input_ids"][0, :n] == torch.tensor(expected, dtype=torch.long)
        ).all()


# ---------------------------------------------------------------------------
# ItemAwareCausalLM.from_causal_lm (resize and initialize) tests
# ---------------------------------------------------------------------------


class TestResizeAndInitialize(unittest.TestCase):
    """Model embedding resize and optional codebook-based initialization."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.aware, _ = _make_aware()
        cls.orig = cls.aware.original_vocab_size

    def test_embed_tokens_resized(self) -> None:
        model = _MockHFModel(self.orig, HIDDEN_SIZE)
        item_model = ItemAwareCausalLM.from_causal_lm(model, self.aware)
        assert (
            item_model.get_input_embeddings().weight.shape[0] == self.aware.vocab_size
        )

    def test_lm_head_resized(self) -> None:
        model = _MockHFModel(self.orig, HIDDEN_SIZE)
        item_model = ItemAwareCausalLM.from_causal_lm(model, self.aware)
        assert (
            item_model.get_output_embeddings().weight.shape[0] == self.aware.vocab_size
        )

    def test_existing_embeddings_preserved(self) -> None:
        """Original token embeddings must not be altered."""
        model = _MockHFModel(self.orig, HIDDEN_SIZE)
        orig_weight = model.get_input_embeddings().weight.data.clone()
        item_model = ItemAwareCausalLM.from_causal_lm(model, self.aware)
        new_weight = item_model.get_input_embeddings().weight.data
        assert torch.allclose(new_weight[: self.orig], orig_weight)


# ---------------------------------------------------------------------------
# ItemAwareCausalLM.from_causal_lm tests
# ---------------------------------------------------------------------------


class TestFromCausalLM(unittest.TestCase):
    """Verify ItemAwareCausalLM.from_causal_lm factory method."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.aware, _ = _make_aware()

    def _make_inner(self) -> _MockHFModel:
        return _MockHFModel(self.aware.original_vocab_size, HIDDEN_SIZE)

    def test_returns_item_aware_causal_lm(self) -> None:
        model = ItemAwareCausalLM.from_causal_lm(self._make_inner(), self.aware)
        assert isinstance(model, ItemAwareCausalLM)

    def test_inner_vocab_size_matches_extended_vocab(self) -> None:
        model = ItemAwareCausalLM.from_causal_lm(self._make_inner(), self.aware)
        actual = model.model.get_input_embeddings().weight.shape[0]
        assert actual == self.aware.vocab_size

    def test_lm_head_resized(self) -> None:
        model = ItemAwareCausalLM.from_causal_lm(self._make_inner(), self.aware)
        assert (
            model.model.get_output_embeddings().weight.shape[0] == self.aware.vocab_size
        )

    def test_config_num_levels(self) -> None:
        model = ItemAwareCausalLM.from_causal_lm(self._make_inner(), self.aware)
        assert model.config.num_levels == self.aware.num_levels

    def test_config_codebook_size(self) -> None:
        model = ItemAwareCausalLM.from_causal_lm(self._make_inner(), self.aware)
        assert model.config.codebook_size == self.aware.codebook_size

    def test_preloaded_module_used_directly(self) -> None:
        inner = self._make_inner()
        model = ItemAwareCausalLM.from_causal_lm(inner, self.aware)
        assert model.model is inner

    def test_embedding_delegation(self) -> None:
        """get_input/output_embeddings on the wrapper delegate to inner model."""
        model = ItemAwareCausalLM.from_causal_lm(self._make_inner(), self.aware)
        assert model.get_input_embeddings() is model.model.get_input_embeddings()
        assert model.get_output_embeddings() is model.model.get_output_embeddings()


if __name__ == "__main__":
    unittest.main()
