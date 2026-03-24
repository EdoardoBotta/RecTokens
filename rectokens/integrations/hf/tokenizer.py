from __future__ import annotations

from typing import Any

import torch
from transformers import AutoTokenizer, PreTrainedTokenizerFast

from rectokens.core.tokenizer import Tokenizer, TokenSequence
from rectokens.schemas.compact_csr_trie import CompactCSRTrie


class ItemAwareTokenizer(PreTrainedTokenizerFast):
    """Unified tokenizer that merges a HuggingFace text tokenizer with a RecTokens
    item tokenizer.

    Extends :class:`~transformers.PreTrainedTokenizerFast` so that all standard HF
    tokenizer methods (``save_pretrained``, ``push_to_hub``, ``encode``, ``decode``,
    ``pad``, ``batch_decode``, etc.) are available natively through inheritance.

    Accepts either a plain ``PreTrainedTokenizerFast`` or any multimodal processor that
    exposes a ``.tokenizer`` attribute (e.g. ``Qwen2_5_VLProcessor``, ``LlavaProcessor``).
    Token registration and vocab operations always act on the underlying fast tokenizer
    backend; the image/video processor component is unaffected.

    Special tokens registered (in order, starting at ``original_vocab_size``):
      - ``<item_L{l}_C{c}>``  for l in [0, num_levels), c in [0, codebook_size)
      - ``<item_start>``       separator inserted between end-of-text and start of
                               the next item's semantic IDs.
    """

    def __init__(
        self,
        hf_tokenizer: Any,  # PreTrainedTokenizerFast or processor with .tokenizer
        item_tokenizer: Tokenizer,
        num_levels: int,
        codebook_size: int,
    ) -> None:
        # Duck-typing: any object with a .tokenizer attribute is treated as a processor.
        if hasattr(hf_tokenizer, "tokenizer"):
            self.processor = hf_tokenizer
            text_tokenizer = hf_tokenizer.tokenizer
        else:
            self.processor = None
            text_tokenizer = hf_tokenizer

        # Extract the tokenizers.Tokenizer backend and initialise the HF base class.
        # Special tokens are forwarded explicitly so they are preserved after re-init.
        # Use getattr with defaults for compatibility across transformers versions.
        super().__init__(
            tokenizer_object=text_tokenizer._tokenizer,
            bos_token=getattr(text_tokenizer, "bos_token", None),
            eos_token=getattr(text_tokenizer, "eos_token", None),
            unk_token=getattr(text_tokenizer, "unk_token", None),
            pad_token=getattr(text_tokenizer, "pad_token", None),
            additional_special_tokens=getattr(
                text_tokenizer, "additional_special_tokens", []
            )
            or [],
        )

        self.item_tokenizer = item_tokenizer
        self.num_levels = num_levels
        self.codebook_size = codebook_size

        # Save BEFORE registration so item_token_id offsets are stable.
        self._original_vocab_size = len(self)

        self._register_item_tokens()

    def __call__(self, *args, **kwargs):
        # Delegate to the processor for multimodal models, or the base class otherwise.
        if self.processor is not None:
            return self.processor(*args, **kwargs)
        return super().__call__(*args, **kwargs)

    def _register_item_tokens(self) -> None:
        # Skip if already registered (e.g. checkpoint reload).
        probe_id = self.convert_tokens_to_ids("<item_L0_C0>")
        if probe_id != self.unk_token_id:
            return
        new_tokens = [
            f"<item_L{l}_C{c}>"
            for l in range(self.num_levels)
            for c in range(self.codebook_size)
        ] + ["<item_start>"]  # separator: between end-of-text and item semantic IDs
        self.add_tokens(new_tokens, special_tokens=True)
        assert self.convert_tokens_to_ids("<item_L0_C0>") == self._original_vocab_size

    def item_token_id(self, level: int, code: int) -> int:
        """Return the HF vocab id for item level ``level``, code ``code``."""
        return self._original_vocab_size + level * self.codebook_size + code

    @property
    def item_sep_token_id(self) -> int:
        """Token id for ``<item_start>``: the separator inserted between the end of
        a text span and the start of the next item's semantic IDs."""
        return self._original_vocab_size + self.num_levels * self.codebook_size

    def encode_sequence(self, parts: list[str | torch.Tensor]) -> list[int]:
        """Encode a mixed text/item sequence to a flat list of HF token ids.

        Each element of ``parts`` is either:
        - ``str``: encoded by the text tokenizer (``add_special_tokens=False``).
        - ``torch.Tensor`` of shape ``(D,)``: a single item embedding, encoded by
          ``self.item_tokenizer`` and mapped to item token ids.

        An ``<item_start>`` separator is inserted before an item's semantic IDs
        whenever the immediately preceding element was a text span, marking the
        boundary between consecutive items' text and semantic IDs.
        """
        ids: list[int] = []
        for i, part in enumerate(parts):
            if isinstance(part, str):
                ids.extend(self.encode(part, add_special_tokens=False))
            else:
                # Insert <item_start> separator when transitioning from text → item.
                if i > 0 and isinstance(parts[i - 1], str):
                    ids.append(self.item_sep_token_id)
                # part: (D,) item embedding tensor — move to tokenizer's device/dtype.
                # Fall back to CPU float32 for non-Module tokenizers (e.g. RQKMeansTokenizer).
                try:
                    param = next(self.item_tokenizer.parameters())
                    device, dtype = param.device, param.dtype
                except (AttributeError, StopIteration):
                    device, dtype = torch.device("cpu"), torch.float32
                token_seq: TokenSequence = self.item_tokenizer.encode(
                    part.unsqueeze(0).to(device=device, dtype=dtype)
                )
                codes = token_seq.codes[0]  # (num_levels,)
                for l in range(self.num_levels):
                    ids.append(self.item_token_id(l, int(codes[l].item())))
        return ids

    def decode_sequence(self, ids: list[int]) -> list[str | TokenSequence]:
        """Decode a flat list of HF token ids back to text spans and item TokenSequences.

        Consecutive item tokens are grouped into a single ``TokenSequence`` per item.
        ``<item_start>`` separator tokens are treated as boundaries and skipped.
        """
        result: list[str | TokenSequence] = []
        text_run: list[int] = []
        item_run: list[int] = []

        def flush_text():
            if text_run:
                result.append(self.decode(text_run))
                text_run.clear()

        def flush_item():
            if item_run:
                # item_run contains ids for one full item (num_levels tokens)
                codes = torch.tensor(
                    [
                        (tid - self._original_vocab_size) % self.codebook_size
                        for tid in item_run
                    ],
                    dtype=torch.long,
                )
                result.append(TokenSequence(codes=codes))
                item_run.clear()

        sep_id = self.item_sep_token_id
        for tid in ids:
            if tid == sep_id:
                # Boundary marker — flush whatever is in progress and move on.
                flush_text()
                flush_item()
            elif tid < self._original_vocab_size:
                if item_run:
                    flush_item()
                text_run.append(tid)
            else:
                if text_run:
                    flush_text()
                item_run.append(tid)
                if len(item_run) == self.num_levels:
                    flush_item()

        flush_text()
        flush_item()
        return result

    def build_item_trie(
        self,
        catalog_codes: torch.Tensor,  # (N, num_levels) raw codes in [0, codebook_size)
        dense_lookup_layers: int = 1,
    ) -> CompactCSRTrie:
        """Build a ``CompactCSRTrie`` over the catalog for constrained generation.

        Args:
            catalog_codes: ``(N, num_levels)`` integer tensor of raw RQ codes.
            dense_lookup_layers: Passed to ``CompactCSRTrie.from_sorted_batch``.
                Default 1 keeps the dense mask shape ``(vocab_size,)`` — safe for
                GPT-scale vocabularies. Setting 2 allocates ``(vocab_size²,)`` bools.
        """
        N, L = catalog_codes.shape
        hf_ids = torch.stack(
            [catalog_codes[:, l] + self.item_token_id(l, 0) for l in range(L)],
            dim=1,
        )  # (N, L)

        # Lexicographic sort
        multipliers = torch.tensor(
            [self.codebook_size ** (L - 1 - l) for l in range(L)],
            dtype=torch.long,
            device=catalog_codes.device,
        )
        sort_key = (hf_ids * multipliers).sum(dim=1)
        hf_ids_sorted = hf_ids[sort_key.argsort()]

        return CompactCSRTrie.from_sorted_batch(
            hf_ids_sorted,
            vocab_size=self.vocab_size,
            dense_lookup_layers=dense_lookup_layers,
        )

    @property
    def vocab_size(self) -> int:
        """Vocabulary size after item token registration (includes ``<item_start>``)."""
        return len(self)

    @property
    def original_vocab_size(self) -> int:
        """Original vocabulary size before item token registration."""
        return self._original_vocab_size

    @property
    def text_tokenizer(self):
        """Return self — ``ItemAwareTokenizer`` is itself a ``PreTrainedTokenizerFast``."""
        return self


# ---------------------------------------------------------------------------
# AutoTokenizer registration
# ---------------------------------------------------------------------------
# Deferred import to avoid circular dependency (configuration imports nothing here).
from rectokens.integrations.hf.configuration import ItemAwareCausalLMConfig  # noqa: E402

AutoTokenizer.register(ItemAwareCausalLMConfig, fast_tokenizer_class=ItemAwareTokenizer)
