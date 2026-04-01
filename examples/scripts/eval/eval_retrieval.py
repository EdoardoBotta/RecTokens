"""Evaluate next-item retrieval using constrained beam search.

Uses precomputed token sequences as prompts (no re-encoding needed).
Appends <item_start> to each prompt, runs constrained beam search,
and reports Recall@1, @5, @10.

Usage:
    python examples/scripts/eval/eval_retrieval.py examples/configs/eval_retrieval_beauty.gin
"""

from __future__ import annotations

import random

import gin
import torch
from transformers import AutoTokenizer

from examples.data.amazon import AmazonReviews
from examples.utils import parse_config
from rectokens.tokenizers.rqvae import RQVAETokenizer
from rectokens.tokenizers.rq_kmeans import RQKMeansTokenizer
from rectokens.decoding.constrained_decoding import autoregressive_generate
from rectokens.integrations.hf.model import ItemAwareCausalLM
from rectokens.integrations.hf.tokenizer import ItemAwareTokenizer
from rectokens.schemas.config import GenerationConfig


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
        tok = RQKMeansTokenizer.load(path).to(device)
    tok.eval()
    for p in tok.parameters():
        p.requires_grad_(False)
    return tok


@torch.no_grad()
def encode_all_items(
    item_embs: torch.Tensor,
    item_tok,
    aware_tok: ItemAwareTokenizer,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    """Encode all item embeddings → codes (num_items, num_levels)."""
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
    return all_codes


def build_codes_to_item_id(codes_table: torch.Tensor) -> dict[tuple, int]:
    """Build a dict mapping codes tuple → item_id (0-indexed)."""
    mapping: dict[tuple, int] = {}
    for item_id, codes in enumerate(codes_table.tolist()):
        key = tuple(codes)
        if key not in mapping:
            mapping[key] = item_id
    return mapping


def generated_ids_to_codes(
    generated: torch.Tensor,  # (k, num_levels) in HF vocab space
    aware_tok: ItemAwareTokenizer,
) -> list[tuple]:
    """Convert generated HF token IDs to raw code tuples."""
    results = []
    for beam_idx in range(generated.shape[0]):
        codes = []
        for l in range(aware_tok.num_levels):
            token_id = int(generated[beam_idx, l].item())
            code = token_id - aware_tok.item_token_id(l, 0)
            codes.append(code)
        results.append(tuple(codes))
    return results


@gin.configurable
def main(
    model_dir: str = gin.REQUIRED,
    hf_model_name: str | None = None,
    item_tok_path: str = gin.REQUIRED,
    item_tok_type: str = "rqvae",
    precomputed_test_path: str = gin.REQUIRED,
    root: str = "data/amazon",
    split: str = "beauty",
    num_examples: int = 10,
    top_k: int = 10,
    beam_size: int = 20,
    encode_batch_size: int = 512,
    bf16: bool = False,
    seed: int = 42,
    attr_path: str | None = None,
) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    device = get_device()
    print(f"Device: {device}")

    # 1. Load precomputed test sequences
    print(f"Loading precomputed test sequences from {precomputed_test_path}")
    test_data = torch.load(precomputed_test_path, weights_only=False)
    sequences = test_data["sequences"]  # list of 1D long tensors
    num_levels = test_data["num_levels"]
    codebook_size = test_data["codebook_size"]
    original_vocab_size = test_data["original_vocab_size"]
    meta = test_data.get("meta", {})
    print(
        f"  {len(sequences)} sequences, num_levels={num_levels}, "
        f"codebook_size={codebook_size}"
    )
    assert not meta.get("include_future", True), (
        "Expected include_future=False in test split"
    )

    # 2. Load raw dataset for future item IDs (aligned 1:1 with sequences)
    print(f"Loading AmazonReviews for future item IDs: root={root}, split={split}")
    raw_data = AmazonReviews(root=root, split=split)
    item_embs = raw_data.data["item"]["x"]  # (num_items, D)
    test_history = raw_data.data["user", "rated", "item"].history["test"]
    future_items = test_history["itemId_fut"]  # tensor (num_users,)
    assert len(future_items) == len(sequences), (
        f"Sequence count mismatch: {len(sequences)} precomputed vs {len(future_items)} raw"
    )
    print(f"  {item_embs.shape[0]} items, {len(future_items)} test users")

    # 3. Load item tokenizer
    print(f"Loading item tokenizer ({item_tok_type}) from {item_tok_path}")
    item_tok = load_item_tokenizer(item_tok_path, item_tok_type, device)

    # 4. Build ItemAwareTokenizer
    resolved_model_name = hf_model_name or meta.get("model_name", "Qwen/Qwen3.5-2B")
    print(f"Loading HF tokenizer: {resolved_model_name}")
    hf_tokenizer = AutoTokenizer.from_pretrained(resolved_model_name)
    aware_tok = ItemAwareTokenizer(
        hf_tokenizer, item_tok, num_levels=num_levels, codebook_size=codebook_size
    )
    assert aware_tok.original_vocab_size == original_vocab_size, (
        f"Vocab size mismatch: tokenizer has {aware_tok.original_vocab_size}, "
        f"precomputed file expects {original_vocab_size}"
    )
    print(f"  Vocab: {aware_tok.original_vocab_size} → {aware_tok.vocab_size}")

    # 5. Load model
    dtype = torch.bfloat16 if bf16 else torch.float32
    print(f"Loading model from {model_dir} (dtype={dtype})")
    model = ItemAwareCausalLM.from_causal_lm(model_dir, aware_tok, torch_dtype=dtype)
    model = model.to(device)
    model.eval()

    # 6. Encode all items → codes table and lookup dict
    print("Encoding all item embeddings...")
    codes_table = encode_all_items(
        item_embs, item_tok, aware_tok, encode_batch_size, device
    )  # (num_items, num_levels)
    codes_to_item_id = build_codes_to_item_id(codes_table)
    print(
        f"  {len(codes_to_item_id)} unique code tuples for {item_embs.shape[0]} items"
    )

    # 7. Build item trie
    print("Building item trie...")
    trie = aware_tok.build_item_trie(codes_table.to(device))
    print("  Trie built.")

    # 8. Sample valid examples (non-empty sequence, valid future item)
    valid_indices = [
        i
        for i in range(len(sequences))
        if len(sequences[i]) > 0 and int(future_items[i].item()) >= 0
    ]
    sampled_indices = random.sample(
        valid_indices, min(num_examples, len(valid_indices))
    )
    print(f"\nEvaluating {len(sampled_indices)} examples")

    # 9. Run evaluation
    gen_config = GenerationConfig(
        steps=num_levels,
        k=top_k,
        beam_size=beam_size,
        temperature=1.0,
    )
    cutoffs = [c for c in [1, 5, 10] if c <= top_k]
    recall_at = {c: 0 for c in cutoffs}

    item_sep_id = aware_tok.item_sep_token_id

    for eval_idx, user_idx in enumerate(sampled_indices):
        # Prompt = precomputed context tokens + <item_start>
        context = sequences[user_idx]  # 1D long tensor
        prompt = torch.cat([context, torch.tensor([item_sep_id], dtype=torch.long)])
        input_ids = prompt.unsqueeze(0).to(device)  # (1, seq_len)

        fut_id = int(future_items[user_idx].item())

        print(
            f"\n[{eval_idx + 1}/{len(sampled_indices)}] user={user_idx}, "
            f"prompt_len={input_ids.shape[1]}, future_item={fut_id}"
        )

        with torch.inference_mode():
            generated = autoregressive_generate(
                model=model,
                trie=trie,
                input_ids=input_ids,
                generation_config=gen_config,
                attr_path=attr_path,
            )
        # generated: (1, k, num_levels)
        generated_beams = generated[0]  # (k, num_levels)

        code_tuples = generated_ids_to_codes(generated_beams, aware_tok)
        predicted_ids = [codes_to_item_id.get(ct, -1) for ct in code_tuples]

        print(f"  Top-5 predicted items: {predicted_ids[:5]}")
        print(f"  True future item:      {fut_id}")

        for cutoff in cutoffs:
            if fut_id in predicted_ids[:cutoff]:
                recall_at[cutoff] += 1

    # 10. Report
    n = len(sampled_indices)
    print("\n" + "=" * 40)
    print(f"Results on {n} examples  (split={split}, seq_split=test)")
    print(f"Model: {model_dir}")
    print("=" * 40)
    for cutoff in cutoffs:
        r = recall_at[cutoff] / n
        print(f"  Recall@{cutoff:<3}: {recall_at[cutoff]}/{n} = {r:.4f}")
    print("=" * 40)


if __name__ == "__main__":
    parse_config()
    main()
