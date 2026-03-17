from __future__ import annotations

from typing import Literal, Optional

import torch

from rectokens.integrations.hf.tokenizer import ItemAwareTokenizer


class InterleavedSequenceCollator:
    """Collates mixed text/item sequences into padded batches for HF training.

    Each example is ``list[str | torch.Tensor]``.  The collator encodes every
    example via :meth:`ItemAwareTokenizer.encode_sequence`, pads to a uniform
    length, builds an attention mask, and applies loss masking according to
    ``loss_on``.

    Labels are NOT shifted — HF causal LM models perform the shift internally.
    The first token label is always ``-100`` (ignored).

    Args:
        item_tokenizer: Unified tokenizer used to encode each sequence.
        padding_side: ``"right"`` (default) or ``"left"``.
        pad_token_id: Token id used for padding positions.
        loss_on: Which positions contribute to the loss.

            - ``"all"``: all non-padding tokens.
            - ``"items"``: only item tokens (``id >= original_vocab_size``).
            - ``"text"``: only text tokens (``id < original_vocab_size``).
        max_length: If set, sequences are truncated to this length.
    """

    def __init__(
        self,
        item_tokenizer: ItemAwareTokenizer,
        padding_side: Literal["right", "left"] = "right",
        pad_token_id: int = 0,
        loss_on: Literal["all", "items", "text"] = "all",
        max_length: Optional[int] = None,
    ) -> None:
        self.item_tokenizer = item_tokenizer
        self.padding_side = padding_side
        self.pad_token_id = pad_token_id
        self.loss_on = loss_on
        self.max_length = max_length

    def __call__(
        self, examples: list[list[str | torch.Tensor]]
    ) -> dict[str, torch.Tensor]:
        encoded = [self.item_tokenizer.encode_sequence(ex) for ex in examples]

        if self.max_length is not None:
            encoded = [ids[: self.max_length] for ids in encoded]

        max_len = max(len(ids) for ids in encoded)
        B = len(encoded)

        input_ids = torch.full((B, max_len), self.pad_token_id, dtype=torch.long)
        attention_mask = torch.zeros(B, max_len, dtype=torch.long)
        labels = torch.full((B, max_len), -100, dtype=torch.long)

        orig_vocab = self.item_tokenizer.original_vocab_size

        for i, ids in enumerate(encoded):
            seq_len = len(ids)
            t = torch.tensor(ids, dtype=torch.long)

            if self.padding_side == "right":
                input_ids[i, :seq_len] = t
                attention_mask[i, :seq_len] = 1
                label_positions = slice(1, seq_len)  # first token always -100
                label_ids = t[1:]
            else:
                pad_len = max_len - seq_len
                input_ids[i, pad_len:] = t
                attention_mask[i, pad_len:] = 1
                label_start = pad_len + 1
                label_positions = slice(label_start, max_len)
                label_ids = t[1:]

            # Apply loss masking
            if self.loss_on == "all":
                masked = label_ids
            elif self.loss_on == "items":
                masked = torch.where(label_ids >= orig_vocab, label_ids, torch.tensor(-100, dtype=torch.long))
            else:  # "text"
                masked = torch.where(label_ids < orig_vocab, label_ids, torch.tensor(-100, dtype=torch.long))

            labels[i, label_positions] = masked

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }
