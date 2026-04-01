from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModelForCausalLM, PreTrainedModel

from rectokens.decoding.constrained_decoding import autoregressive_generate
from rectokens.integrations.hf.configuration import ItemAwareCausalLMConfig
from rectokens.integrations.hf.tokenizer import ItemAwareTokenizer
from rectokens.schemas.compact_csr_trie import CompactCSRTrie
from rectokens.schemas.config import GenerationConfig


def _resize_and_initialize(
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

    # Keep config in sync so save_pretrained / from_pretrained round-trips work.
    if hasattr(hf_model, "config"):
        hf_model.config.vocab_size = target_vocab

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
        f"[_resize_and_initialize] vocab {orig_model_vocab} → {target_vocab} "
        f"(+{n_new} item tokens, ids {first_item}–{last_item})  "
        f"lm_head tied={tied}"
    )

    if projection is not None:
        if item_tokenizer.item_tokenizer is None:
            raise RuntimeError(
                "_resize_and_initialize: projection-based embedding initialization "
                "requires codebook access via item_tokenizer.item_tokenizer.rq, "
                "but item_tokenizer.item_tokenizer is None. "
                "Provide a real item tokenizer when using projection=."
            )
        emb = hf_model.get_input_embeddings()
        for l in range(item_tokenizer.num_levels):
            codebook = item_tokenizer.item_tokenizer.rq.levels[l].codebook
            all_codes = torch.arange(item_tokenizer.codebook_size)
            vecs = codebook.lookup(all_codes)  # (K, latent_dim)
            projected = projection(vecs)  # (K, hidden_size)
            start = item_tokenizer.item_token_id(l, 0)
            emb.weight.data[start : start + item_tokenizer.codebook_size] = (
                projected.detach()
            )


class ItemAwareCausalLM(PreTrainedModel):
    """HuggingFace ``PreTrainedModel`` subclass with item-aware generation.

    Wraps an inner causal-LM (``self.model``) while adding trie-based constrained
    generation for item retrieval.  All standard HF model methods — ``save_pretrained``,
    ``parameters``, ``state_dict``, ``to``, ``eval``, ``train``,
    ``get_input_embeddings``, etc. — are available through normal inheritance.

    ``generate`` is overridden to support both modes:

    * **Constrained** (``trie`` provided): runs constrained beam search via
      ``autoregressive_generate`` and returns ``(B, k, num_levels)`` item token IDs.
    * **Unconstrained** (``trie=None``): delegates to ``self.model.generate`` with all
      extra keyword arguments forwarded.

    Usage::

        hf_model = AutoModelForCausalLM.from_pretrained(...)
        model = ItemAwareCausalLM.from_causal_lm(hf_model, aware_tok)

        # constrained
        generated = model.generate(input_ids, trie=trie, generation_config=cfg)
        # unconstrained
        generated = model.generate(input_ids, max_new_tokens=50, do_sample=True)
        # HF methods work natively
        model.save_pretrained("checkpoint/")
        model.eval()
    """

    config_class = ItemAwareCausalLMConfig

    def __init__(
        self,
        config: ItemAwareCausalLMConfig,
        hf_model: Optional[nn.Module] = None,
    ) -> None:
        super().__init__(config)
        if hf_model is not None:
            self.model = hf_model
        elif config.base_model_config is not None:
            # Reconstruct inner model from the stored config (used by from_pretrained).
            base_cfg = AutoConfig.for_model(**config.base_model_config)
            self.model = AutoModelForCausalLM.from_config(base_cfg)
        else:
            raise ValueError(
                "Either hf_model or config.base_model_config must be provided."
            )

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def get_input_embeddings(self):
        return self.model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.model.set_input_embeddings(value)

    def get_output_embeddings(self):
        return self.model.get_output_embeddings()

    def set_output_embeddings(self, new_embeddings):
        self.model.set_output_embeddings(new_embeddings)

    def generate(
        self,
        input_ids: torch.Tensor,
        trie: Optional[CompactCSRTrie] = None,
        generation_config: Optional[GenerationConfig] = None,
        attr_path: Optional[str] = None,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """Generate tokens with optional trie-based item constraint.

        Args:
            input_ids: ``(B, seq_len)`` prompt token IDs.
            trie: ``CompactCSRTrie`` built via
                ``ItemAwareTokenizer.build_item_trie``, or ``None`` for
                unconstrained generation.
            generation_config: :class:`~rectokens.schemas.config.GenerationConfig`
                with ``steps``, ``k``, ``beam_size``, ``temperature``.
                Required when ``trie`` is provided; ignored otherwise.
            attr_path: Optional attribute path to the lm_head for
                ``SparseLinear`` constraint enforcement (constrained mode only).
            attention_mask: Optional ``(B, seq_len)`` padding mask.
            **kwargs: Forwarded verbatim to ``self.model.generate`` in
                unconstrained mode (e.g. ``max_new_tokens``, ``do_sample``).

        Returns:
            * Constrained: ``(B, k, num_levels)`` item token IDs in HF vocab space.
            * Unconstrained: whatever ``self.model.generate`` returns (typically
              ``(B, seq_len)``).
        """
        if trie is None:
            return self.model.generate(
                input_ids=input_ids, attention_mask=attention_mask, **kwargs
            )
        if generation_config is None:
            raise ValueError(
                "generation_config is required for constrained generation (trie is not None)"
            )
        with torch.inference_mode():
            return autoregressive_generate(
                model=self.model,
                trie=trie,
                input_ids=input_ids,
                generation_config=generation_config,
                attr_path=attr_path,
                attention_mask=attention_mask,
            )

    @classmethod
    def from_causal_lm(
        cls,
        model_name_or_path: "str | nn.Module",
        item_tokenizer: ItemAwareTokenizer,
        projection: Optional[nn.Module] = None,
        **kwargs,
    ) -> "ItemAwareCausalLM":
        """Factory: build an ``ItemAwareCausalLM`` from a name/path or pre-loaded model.

        Combines ``AutoModelForCausalLM.from_pretrained``, ``_resize_and_initialize``,
        and ``ItemAwareCausalLMConfig`` construction into a single call.

        Args:
            model_name_or_path: HuggingFace model identifier / local path (``str``),
                or an already-instantiated ``nn.Module``.  When a string is given the
                model is loaded via ``AutoModelForCausalLM.from_pretrained`` and all
                extra ``**kwargs`` are forwarded (e.g. ``torch_dtype``, ``device_map``).
                When a module is given, ``**kwargs`` are ignored.
            item_tokenizer: Fully initialised ``ItemAwareTokenizer`` (item tokens
                already registered).  Its ``num_levels`` and ``codebook_size``
                attributes are used to build the ``ItemAwareCausalLMConfig``.
            projection: Optional ``nn.Module`` mapping latent codebook vectors to the
                model's hidden size.  Forwarded verbatim to ``_resize_and_initialize``.
            **kwargs: Passed to ``AutoModelForCausalLM.from_pretrained`` when
                ``model_name_or_path`` is a string.  Silently ignored otherwise.

        Returns:
            Ready-to-use ``ItemAwareCausalLM`` with an extended vocabulary.

        Example::

            # From a HuggingFace hub name:
            model = ItemAwareCausalLM.from_causal_lm(
                "Qwen/Qwen2-1.5B", aware_tok, torch_dtype=torch.bfloat16
            )

            # From a pre-loaded module:
            inner = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2-1.5B")
            model = ItemAwareCausalLM.from_causal_lm(inner, aware_tok)
        """
        if isinstance(model_name_or_path, str):
            hf_model = AutoModelForCausalLM.from_pretrained(
                model_name_or_path, **kwargs
            )
        else:
            hf_model = model_name_or_path

        _resize_and_initialize(hf_model, item_tokenizer, projection=projection)

        config = ItemAwareCausalLMConfig(
            num_levels=item_tokenizer.num_levels,
            codebook_size=item_tokenizer.codebook_size,
            base_model_config=(
                hf_model.config.to_dict() if hasattr(hf_model, "config") else None
            ),
        )
        return cls(config, hf_model)


AutoConfig.register("item_aware_causal_lm", ItemAwareCausalLMConfig)
AutoModelForCausalLM.register(ItemAwareCausalLMConfig, ItemAwareCausalLM)
