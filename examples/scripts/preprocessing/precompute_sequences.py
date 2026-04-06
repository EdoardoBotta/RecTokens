"""Precompute chat-formatted training samples mixing all 7 chat templates.

Creates a mixed dataset of:
  - description_to_sid:                    item description → SID
  - sid_to_description:                    SID → item description
  - sid_sequence_to_sid:                   SID history → next SID
  - description_sequence_to_sid:           description history → next SID
  - mixed_sequence_to_sid:                 interleaved (desc, SID) history → next SID
  - description_sequence_to_sid_and_description: description history → next SID + description
  - sid_sequence_to_sid_and_description:   SID history → next SID + description

All samples are formatted using Qwen's chat format (<|im_start|> / <|im_end|>) and
saved as {"input_ids": tensor, "labels": tensor} dicts for SFT with
PrecomputedSequenceCollator.  Labels are -100 on system and user turns; loss is
computed only on the assistant response.

Usage:
    python -m examples.scripts.preprocessing.precompute_sequences \\
        examples/configs/preprocessing/precompute_sequences_beauty.gin
"""

from __future__ import annotations

import os
import random

import gin
import torch
from transformers import AutoTokenizer

from examples.data.amazon import AmazonReviews
from examples.data.chat_templates import (
    description_sequence_to_sid,
    description_sequence_to_sid_and_description,
    description_to_sid,
    mixed_sequence_to_sid,
    sid_sequence_to_sid,
    sid_sequence_to_sid_and_description,
    sid_to_description,
)
from examples.utils import parse_config
from rectokens.integrations.hf.tokenizer import ItemAwareTokenizer
from rectokens.tokenizers.rqvae import RQVAETokenizer


SYSTEM_PROMPT = "You are a helpful recommendation assistant."


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_item_tokenizer(path: str, tok_type: str, device: torch.device):
    if tok_type == "rqvae":
        tok = RQVAETokenizer.load(path).to(device)
    else:
        from rectokens.tokenizers.rq_kmeans import RQKMeansTokenizer

        tok = RQKMeansTokenizer.load(path).to(device)
    tok.eval()
    for param in tok.parameters():
        param.requires_grad_(False)
    return tok


@torch.no_grad()
def encode_all_items(
    item_embs: torch.Tensor,
    item_tok,
    aware_tok: ItemAwareTokenizer,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    """Encode every item embedding → codes tensor (num_items, num_levels)."""
    num_items = item_embs.shape[0]
    all_codes = torch.zeros(num_items, aware_tok.num_levels, dtype=torch.long)

    try:
        param = next(item_tok.parameters())
        enc_dtype = param.dtype
    except StopIteration:
        enc_dtype = torch.float32

    for start in range(0, num_items, batch_size):
        end = min(start + batch_size, num_items)
        batch = item_embs[start:end].to(device=device, dtype=enc_dtype)
        token_seq = item_tok.encode(batch)
        all_codes[start:end] = token_seq.codes.cpu()
        if start % max(batch_size * 10, 1) == 0:
            print(f"  Encoded items {start}/{num_items}...", flush=True)

    print(f"  Encoded all {num_items} items.")
    return all_codes


# ---------------------------------------------------------------------------
# Chat-format helpers
# ---------------------------------------------------------------------------


def _get_special_token_id(aware_tok: ItemAwareTokenizer, token: str) -> int:
    tid = aware_tok.convert_tokens_to_ids(token)
    if tid is None or tid == aware_tok.unk_token_id:
        raise ValueError(f"Special token {token!r} not found in tokenizer vocabulary")
    return tid


def _render_content(content: list[dict], aware_tok: ItemAwareTokenizer) -> list[int]:
    """Convert a list of content blocks to a flat list of token IDs."""
    ids: list[int] = []
    for block in content:
        if block["type"] == "text":
            ids.extend(aware_tok.encode(block["text"], add_special_tokens=False))
        else:  # "item"
            ids.append(aware_tok.item_sep_token_id)
            for l, c in enumerate(block["codes"]):
                ids.append(aware_tok.item_token_id(l, c))
            ids.append(aware_tok.item_end_token_id)
    return ids


class ChatTokenizer:
    """Caches per-role token IDs and renders chat samples to {input_ids, labels}."""

    def __init__(self, aware_tok: ItemAwareTokenizer) -> None:
        self.aware_tok = aware_tok
        self.im_start = _get_special_token_id(aware_tok, "<|im_start|>")
        self.im_end = _get_special_token_id(aware_tok, "<|im_end|>")
        self.newline = aware_tok.encode("\n", add_special_tokens=False)
        self.system_role = aware_tok.encode("system", add_special_tokens=False)
        self.user_role = aware_tok.encode("user", add_special_tokens=False)
        self.assistant_role = aware_tok.encode("assistant", add_special_tokens=False)
        self.system_content = aware_tok.encode(SYSTEM_PROMPT, add_special_tokens=False)

        # Pre-build the system turn (always masked)
        self._system_turn: list[int] = (
            [self.im_start]
            + self.system_role
            + self.newline
            + self.system_content
            + [self.im_end]
            + self.newline
        )

    def encode(self, sample: dict) -> dict[str, torch.Tensor]:
        """Convert a chat_templates sample dict to {input_ids, labels} tensors.

        The system turn and every user turn are masked (-100 in labels).
        Loss is computed on the assistant's response tokens only.
        """
        input_ids: list[int] = list(self._system_turn)
        labels: list[int] = [-100] * len(self._system_turn)

        for message in sample["messages"]:
            role = message["role"]
            content_ids = _render_content(message["content"], self.aware_tok)
            role_ids = self.user_role if role == "user" else self.assistant_role

            prefix = [self.im_start] + role_ids + self.newline
            suffix = content_ids + [self.im_end] + self.newline

            input_ids.extend(prefix)
            input_ids.extend(suffix)

            if role == "assistant":
                labels.extend([-100] * len(prefix))
                labels.extend(suffix)  # loss on assistant response
            else:
                labels.extend([-100] * (len(prefix) + len(suffix)))

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


# ---------------------------------------------------------------------------
# Per-item samples (description_to_sid, sid_to_description)
# ---------------------------------------------------------------------------


def build_item_samples(
    codes_table: torch.Tensor,
    item_texts: list[str],
    chat_tok: ChatTokenizer,
) -> list[dict]:
    """Build two samples per item: description→SID and SID→description."""
    samples: list[dict] = []
    num_items = codes_table.shape[0]

    for iid in range(num_items):
        codes = codes_table[iid].tolist()
        text = str(item_texts[iid])

        samples.append(chat_tok.encode(description_to_sid(text, codes)))
        samples.append(chat_tok.encode(sid_to_description(codes, text)))

        if iid % 10000 == 0:
            print(f"  Built item samples {iid}/{num_items}...", flush=True)

    print(f"  Built {len(samples)} item samples for {num_items} items.")
    return samples


# ---------------------------------------------------------------------------
# Per-sequence samples (5 sequence-level templates)
# ---------------------------------------------------------------------------


def build_sequence_samples(
    seq_split: str,
    raw_data,
    codes_table: torch.Tensor,
    chat_tok: ChatTokenizer,
    max_seq_len: int,
) -> list[dict]:
    """Build 5 samples per user sequence using all sequence-level chat templates."""
    history = raw_data.data["user", "rated", "item"].history[seq_split]
    sequences_raw = history["itemId"]
    future_items = history["itemId_fut"]
    item_texts = raw_data.data["item"]["text"]

    samples: list[dict] = []
    n = len(sequences_raw)

    for idx in range(n):
        seq = sequences_raw[idx]
        item_ids = seq if isinstance(seq, list) else seq.tolist()
        item_ids = [i for i in item_ids if i >= 0]
        item_ids = item_ids[-max_seq_len:]

        fut = int(future_items[idx].item())
        if fut < 0 or len(item_ids) == 0:
            continue

        next_codes = codes_table[fut].tolist()
        next_text = str(item_texts[fut])
        history_codes = [codes_table[i].tolist() for i in item_ids]
        history_texts = [str(item_texts[i]) for i in item_ids]
        history_mixed = list(zip(history_texts, history_codes))

        for chat_sample in (
            sid_sequence_to_sid(history_codes, next_codes),
            description_sequence_to_sid(history_texts, next_codes),
            mixed_sequence_to_sid(history_mixed, next_codes),
            description_sequence_to_sid_and_description(
                history_texts, next_codes, next_text
            ),
            sid_sequence_to_sid_and_description(history_codes, next_codes, next_text),
        ):
            samples.append(chat_tok.encode(chat_sample))

        if idx % 1000 == 0:
            print(f"  Processed user {idx}/{n}...", flush=True)

    print(
        f"  Built {len(samples)} sequence samples from {n} users for split '{seq_split}'."
    )
    return samples


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


@gin.configurable
def main(
    root: str = "data/amazon",
    split: str = "beauty",
    seq_splits: tuple = ("train", "eval"),
    item_tok_path: str = gin.REQUIRED,
    item_tok_type: str = "rqvae",
    num_levels: int = 3,
    codebook_size: int = 256,
    model_name: str = "Qwen/Qwen3.5-2B",
    max_seq_len: int = 20,
    include_item_samples: bool = True,
    output_dir: str = gin.REQUIRED,
    batch_size: int = 512,
    seed: int = 42,
) -> None:
    random.seed(seed)
    device = get_device()
    print(f"Using device: {device}")
    os.makedirs(output_dir, exist_ok=True)

    # 1. Load dataset
    print(f"Loading AmazonReviews: root={root}, split={split}")
    raw_data = AmazonReviews(root=root, split=split)
    item_embs = raw_data.data["item"]["x"]
    print(f"  {item_embs.shape[0]} items, embedding dim={item_embs.shape[1]}")

    # 2. Load item tokenizer (frozen)
    print(f"Loading item tokenizer from {item_tok_path} (type={item_tok_type})")
    item_tok = load_item_tokenizer(item_tok_path, item_tok_type, device)

    # 3. Build ItemAwareTokenizer
    print(f"Loading HF tokenizer: {model_name}")
    hf_tokenizer = AutoTokenizer.from_pretrained(model_name)
    aware_tok = ItemAwareTokenizer(
        hf_tokenizer, item_tok, num_levels=num_levels, codebook_size=codebook_size
    )
    print(f"  Vocab size: {aware_tok.original_vocab_size} → {aware_tok.vocab_size}")

    # 4. Encode all item embeddings once
    print("Encoding all item embeddings...")
    codes_table = encode_all_items(item_embs, item_tok, aware_tok, batch_size, device)

    # 5. Build ChatTokenizer (caches role token IDs and special tokens)
    chat_tok = ChatTokenizer(aware_tok)

    item_texts = raw_data.data["item"]["text"]

    # 6. Build per-item samples (description↔SID) — shared across all splits
    item_samples: list[dict] = []
    if include_item_samples:
        print("\nBuilding per-item samples (description_to_sid, sid_to_description)...")
        item_samples = build_item_samples(codes_table, item_texts, chat_tok)

    # 7. For each seq_split, build sequence samples, shuffle, save
    for seq_split in seq_splits:
        print(f"\nBuilding sequence samples for seq_split='{seq_split}'...")
        seq_samples = build_sequence_samples(
            seq_split=seq_split,
            raw_data=raw_data,
            codes_table=codes_table,
            chat_tok=chat_tok,
            max_seq_len=max_seq_len,
        )

        random.shuffle(seq_samples)

        out_path = os.path.join(output_dir, f"{split}_{seq_split}.pt")
        torch.save(
            {
                "samples": seq_samples,
                "original_vocab_size": aware_tok.original_vocab_size,
                "num_levels": num_levels,
                "codebook_size": codebook_size,
                "meta": {
                    "split": split,
                    "seq_split": seq_split,
                    "model_name": model_name,
                    "item_tok_path": item_tok_path,
                    "max_seq_len": max_seq_len,
                    "templates": [
                        "sid_sequence_to_sid",
                        "description_sequence_to_sid",
                        "mixed_sequence_to_sid",
                        "description_sequence_to_sid_and_description",
                        "sid_sequence_to_sid_and_description",
                    ],
                },
            },
            out_path,
        )
        print(f"  Saved {len(seq_samples)} samples to {out_path}")

        # Save item samples (description↔SID) to a separate file
        if include_item_samples:
            item_out_path = os.path.join(output_dir, f"{split}_{seq_split}_item.pt")
            torch.save(
                {
                    "samples": item_samples,
                    "original_vocab_size": aware_tok.original_vocab_size,
                    "num_levels": num_levels,
                    "codebook_size": codebook_size,
                    "meta": {
                        "split": split,
                        "seq_split": seq_split,
                        "model_name": model_name,
                        "item_tok_path": item_tok_path,
                        "templates": [
                            "description_to_sid",
                            "sid_to_description",
                        ],
                    },
                },
                item_out_path,
            )
            print(f"  Saved {len(item_samples)} item samples to {item_out_path}")

    print("\nPrecomputation complete.")


if __name__ == "__main__":
    parse_config()
    main()
