from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from rectokens.integrations.hf.tokenizer import ItemAwareTokenizer


def resize_and_initialize(
    hf_model: nn.Module,
    item_tokenizer: ItemAwareTokenizer,
    projection: Optional[nn.Module] = None,
) -> None:
    """Resize ``hf_model`` embeddings and lm_head to match the extended vocabulary.

    For multimodal models (e.g. Qwen2.5-VL) where the language model is nested
    (e.g. ``hf_model.model``), pass the language model sub-component as ``hf_model``.
    ``resize_token_embeddings`` handles both ``embed_tokens`` and ``lm_head``
    (including the tied-weight case).

    Args:
        hf_model: A ``PreTrainedModel`` (or language-model sub-component) whose
            embeddings will be resized.
        item_tokenizer: The ``ItemAwareTokenizer`` after token registration.
        projection: Optional ``nn.Module`` mapping latent codebook vectors
            ``(K, latent_dim)`` → ``(K, hidden_size)``.  When ``None`` (default),
            new item embeddings are initialized by HF's mean/covariance strategy.

    Note:
        ``ConstraintEnforcer`` requires the lm_head to be bias-free.  Most HF
        decoder models satisfy this; verify before calling
        ``autoregressive_generate`` with ``attr_path``.
    """
    orig_model_vocab = hf_model.get_input_embeddings().weight.shape[0]
    target_vocab = item_tokenizer.vocab_size
    n_new = target_vocab - item_tokenizer.original_vocab_size  # item tokens added

    hf_model.resize_token_embeddings(target_vocab)

    # --- Alignment verification ---
    emb = hf_model.get_input_embeddings()
    lm_head = hf_model.get_output_embeddings()

    assert emb.weight.shape[0] == target_vocab, (
        f"embed_tokens size {emb.weight.shape[0]} != expected {target_vocab}"
    )
    if lm_head is not None:
        assert lm_head.weight.shape[0] == target_vocab, (
            f"lm_head size {lm_head.weight.shape[0]} != expected {target_vocab} — "
            "if embeddings are tied this should never happen"
        )
    first_item = item_tokenizer.item_token_id(0, 0)
    last_item = item_tokenizer.item_token_id(
        item_tokenizer.num_levels - 1, item_tokenizer.codebook_size - 1
    )
    assert last_item < target_vocab, (
        f"last item token id {last_item} is out of range (vocab={target_vocab})"
    )

    tied = lm_head is not None and emb.weight is lm_head.weight
    print(
        f"[resize_and_initialize] vocab {orig_model_vocab} → {target_vocab} "
        f"(+{n_new} item tokens, ids {first_item}–{last_item})  "
        f"lm_head tied={tied}"
    )

    if projection is not None:
        emb = hf_model.get_input_embeddings()
        for l in range(item_tokenizer.num_levels):
            codebook = item_tokenizer.item_tokenizer.rq.levels[l].codebook
            all_codes = torch.arange(item_tokenizer.codebook_size)
            vecs = codebook.lookup(all_codes)       # (K, latent_dim)
            projected = projection(vecs)             # (K, hidden_size)
            start = item_tokenizer.item_token_id(l, 0)
            emb.weight.data[start : start + item_tokenizer.codebook_size] = (
                projected.detach()
            )
