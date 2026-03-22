"""Precompute interleaved token ID sequences for the full Amazon corpus.

Runs once before training.  Encodes all item embeddings into semantic IDs via
the item tokenizer, then assembles the full interleaved token-ID sequences for
every user (text tokens + item separator + semantic ID tokens) and saves them
to disk.  Training can then load pre-encoded integer tensors directly, avoiding
any neural-network inference at training time.

Usage:
    python precompute_sequences.py \
        --root data/amazon --split beauty \
        --seq_splits train,test \
        --item_tok_path checkpoints/rqvae/final.pt \
        --num_levels 3 --codebook_size 256 \
        --model_name Qwen/Qwen3.5-2B \
        --output_dir data/precomputed/beauty
"""

from __future__ import annotations

import argparse
import os

import torch
from transformers import AutoTokenizer

from examples.data.amazon import AmazonReviews
from rectokens.integrations.hf.tokenizer import ItemAwareTokenizer
from rectokens.tokenizers.rqvae import RQVAETokenizer


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Precompute interleaved token-ID sequences")
    p.add_argument("--root", type=str, default="data/amazon")
    p.add_argument(
        "--split",
        type=str,
        default="beauty",
        choices=["beauty", "sports", "toys", "yelp"],
    )
    p.add_argument(
        "--seq_splits",
        type=str,
        default="train,eval",
        help="Comma-separated sequence splits to precompute, e.g. train,eval,test",
    )
    p.add_argument(
        "--item_tok_path",
        type=str,
        required=True,
        help="Path to fitted .pt item tokenizer (RQVAETokenizer)",
    )
    p.add_argument(
        "--item_tok_type", type=str, default="rqvae", choices=["rqvae", "rqkmeans"]
    )
    p.add_argument("--num_levels", type=int, default=3)
    p.add_argument("--codebook_size", type=int, default=256)
    p.add_argument(
        "--model_name",
        type=str,
        default="Qwen/Qwen3.5-2B",
        help="HuggingFace model name for the text tokenizer",
    )
    p.add_argument("--max_seq_len", type=int, default=20)
    p.add_argument(
        "--include_future",
        action="store_true",
        help="Append the future item to each sequence",
    )
    p.add_argument(
        "--include_item_text",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include item text tokens in sequences (default: True). "
        "When --no-include_item_text is set, sequences contain only "
        "concatenated semantic IDs: <sep><semid_0>...<semid_L-1> per item.",
    )
    p.add_argument(
        "--build_item_dataset",
        action="store_true",
        help="Also build a per-item dataset of <semid> <text> sequences (one entry per item).",
    )
    p.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory where output .pt files will be written",
    )
    p.add_argument(
        "--batch_size",
        type=int,
        default=512,
        help="Batch size for item embedding encoding",
    )
    return p.parse_args()


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
    item_embs: torch.Tensor,  # (num_items, D)
    item_tok,
    aware_tok: ItemAwareTokenizer,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    """Encode every item embedding once → codes tensor (num_items, num_levels)."""
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
    return all_codes  # (num_items, num_levels)


def build_sequence_tokens(
    item_ids: list[int],
    codes_table: torch.Tensor,  # (num_items, num_levels)
    item_texts: list[str],
    aware_tok: ItemAwareTokenizer,
    include_item_text: bool = True,
) -> torch.Tensor:
    """Build the flat interleaved token-ID tensor for one user sequence.

    Mirrors ItemAwareTokenizer.encode_sequence() but uses the precomputed
    codes table instead of calling item_tokenizer.encode() per embedding.

    Args:
        include_item_text: If ``True`` (default), each item's text tokens are
            prepended before its semantic ID tokens, producing interleaved
            text+semid sequences.  If ``False``, text is omitted and the
            sequence contains only concatenated semantic ID tokens:
            ``<sep><semid_0>...<semid_L-1>  <sep><semid_0>...<semid_L-1>  ...``
    """
    ids: list[int] = []

    for iid in item_ids:
        if include_item_text:
            # Text tokens for this item
            text = str(item_texts[iid])
            text_ids = aware_tok._text_tokenizer.encode(text, add_special_tokens=False)
            ids.extend(text_ids)

        # <item_start> separator + semantic ID tokens
        ids.append(aware_tok.item_sep_token_id)
        codes = codes_table[iid]  # (num_levels,)
        for l in range(aware_tok.num_levels):
            ids.append(aware_tok.item_token_id(l, int(codes[l].item())))

    return torch.tensor(ids, dtype=torch.long)


def build_item_dataset(
    codes_table: torch.Tensor,  # (num_items, num_levels)
    item_texts: list[str],
    aware_tok: ItemAwareTokenizer,
) -> list[torch.Tensor]:
    """Build a per-item dataset of <semid> <text> sequences.

    Each entry corresponds to one item and contains:
        <item_sep> <semid_0> ... <semid_L-1> <text tokens>
    """
    results: list[torch.Tensor] = []
    num_items = codes_table.shape[0]

    for iid in range(num_items):
        ids: list[int] = []

        # Semantic ID tokens
        ids.append(aware_tok.item_sep_token_id)
        codes = codes_table[iid]
        for l in range(aware_tok.num_levels):
            ids.append(aware_tok.item_token_id(l, int(codes[l].item())))

        # Text tokens
        text = str(item_texts[iid])
        text_ids = aware_tok._text_tokenizer.encode(text, add_special_tokens=False)
        ids.extend(text_ids)

        results.append(torch.tensor(ids, dtype=torch.long))

        if iid % 10000 == 0:
            print(f"  Built item sequence {iid}/{num_items}...", flush=True)

    print(f"  Built all {num_items} item sequences.")
    return results


def precompute_split(
    seq_split: str,
    raw_data,
    codes_table: torch.Tensor,
    aware_tok: ItemAwareTokenizer,
    max_seq_len: int,
    include_future: bool,
    include_item_text: bool = True,
) -> list[torch.Tensor]:
    history = raw_data.data["user", "rated", "item"].history[seq_split]
    sequences_raw = history["itemId"]
    future_items = history["itemId_fut"]
    item_texts = raw_data.data["item"]["text"]

    results: list[torch.Tensor] = []
    n = len(sequences_raw)

    for idx in range(n):
        seq = sequences_raw[idx]
        item_ids = seq if isinstance(seq, list) else seq.tolist()
        item_ids = [i for i in item_ids if i >= 0]
        item_ids = item_ids[-max_seq_len:]

        if include_future:
            fut = int(future_items[idx].item())
            if fut >= 0:
                item_ids = item_ids + [fut]

        if len(item_ids) == 0:
            results.append(torch.zeros(0, dtype=torch.long))
            continue

        tokens = build_sequence_tokens(
            item_ids, codes_table, item_texts, aware_tok, include_item_text
        )
        results.append(tokens)

        if idx % 1000 == 0:
            print(f"  Processed user {idx}/{n}...", flush=True)

    print(f"  Processed all {n} users for split '{seq_split}'.")
    return results


def main() -> None:
    args = parse_args()
    device = get_device()
    print(f"Using device: {device}")

    os.makedirs(args.output_dir, exist_ok=True)

    # 1. Load dataset
    print(f"Loading AmazonReviews: root={args.root}, split={args.split}")
    raw_data = AmazonReviews(root=args.root, split=args.split)
    item_embs = raw_data.data["item"]["x"]  # (num_items, D)
    print(f"  {item_embs.shape[0]} items, embedding dim={item_embs.shape[1]}")

    # 2. Load item tokenizer (frozen)
    print(
        f"Loading item tokenizer from {args.item_tok_path} (type={args.item_tok_type})"
    )
    item_tok = load_item_tokenizer(args.item_tok_path, args.item_tok_type, device)

    # 3. Build ItemAwareTokenizer (registers item tokens in HF vocab)
    print(f"Loading HF tokenizer: {args.model_name}")
    hf_tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    aware_tok = ItemAwareTokenizer(
        hf_tokenizer, item_tok, args.num_levels, args.codebook_size
    )
    print(f"  Vocab size: {aware_tok.original_vocab_size} → {aware_tok.vocab_size}")

    # 4. Encode ALL item embeddings once
    print("Encoding all item embeddings...")
    codes_table = encode_all_items(
        item_embs, item_tok, aware_tok, args.batch_size, device
    )  # (num_items, num_levels)

    # 5. Optionally build per-item <semid> <text> dataset
    if args.build_item_dataset:
        print("\nBuilding per-item <semid> <text> dataset...")
        item_texts = raw_data.data["item"]["text"]
        item_sequences = build_item_dataset(codes_table, item_texts, aware_tok)
        out_path = os.path.join(args.output_dir, f"{args.split}_items.pt")
        torch.save(
            {
                "sequences": item_sequences,
                "original_vocab_size": aware_tok.original_vocab_size,
                "num_levels": args.num_levels,
                "codebook_size": args.codebook_size,
                "meta": {
                    "split": args.split,
                    "model_name": args.model_name,
                    "item_tok_path": args.item_tok_path,
                },
            },
            out_path,
        )
        print(f"  Saved {len(item_sequences)} item sequences to {out_path}")

    # 6. For each user seq_split, assemble sequences and save
    seq_splits = [s.strip() for s in args.seq_splits.split(",") if s.strip()]
    for seq_split in seq_splits:
        print(f"\nPrecomputing sequences for seq_split='{seq_split}'...")
        sequences = precompute_split(
            seq_split=seq_split,
            raw_data=raw_data,
            codes_table=codes_table,
            aware_tok=aware_tok,
            max_seq_len=args.max_seq_len,
            include_future=args.include_future,
            include_item_text=args.include_item_text,
        )

        out_path = os.path.join(args.output_dir, f"{args.split}_{seq_split}.pt")
        payload = {
            "sequences": sequences,
            "original_vocab_size": aware_tok.original_vocab_size,
            "num_levels": args.num_levels,
            "codebook_size": args.codebook_size,
            "meta": {
                "split": args.split,
                "seq_split": seq_split,
                "model_name": args.model_name,
                "item_tok_path": args.item_tok_path,
                "max_seq_len": args.max_seq_len,
                "include_future": args.include_future,
                "include_item_text": args.include_item_text,
            },
        }
        torch.save(payload, out_path)
        print(f"  Saved {len(sequences)} sequences to {out_path}")

    print("\nPrecomputation complete.")


if __name__ == "__main__":
    main()
