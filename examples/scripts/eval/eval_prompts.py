"""Run arbitrary text prompts against a finetuned ItemAwareCausalLM.

Applies the same Qwen chat template used during finetuning via
``apply_chat_template``, then generates with hybrid constrained/unconstrained
decoding: unconstrained until ``<|item_start|>`` is produced, then trie-constrained
for ``num_levels`` steps to guarantee valid item codes, then unconstrained again.

Usage:
    python -m examples.scripts.eval.eval_prompts \\
        examples/configs/finetuning/eval_prompts_beauty.gin
"""

from __future__ import annotations

import gin
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from examples.data.amazon import AmazonReviews
from examples.utils import parse_config
from rectokens.tokenizers.rqvae import RQVAETokenizer
from rectokens.tokenizers.rq_kmeans import RQKMeansTokenizer
from rectokens.integrations.hf.model import ItemAwareCausalLM
from rectokens.integrations.hf.tokenizer import ItemAwareTokenizer
from rectokens.decoding.constrained_decoding import generate_with_item_constraints


SYSTEM_PROMPT = "You are a helpful recommendation assistant."


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_prompt_ids(
    prompt: str,
    hf_tokenizer: PreTrainedTokenizerBase,
    aware_tok: ItemAwareTokenizer,
    device: torch.device,
) -> torch.Tensor:
    """Format a plain-text user prompt with the Qwen chat template and encode it.

    Uses the same system prompt and turn structure as finetuning:
        <|im_start|>system\\n{SYSTEM_PROMPT}<|im_end|>\\n
        <|im_start|>user\\n{prompt}<|im_end|>\\n
        <|im_start|>assistant\\n

    Returns:
        ``(1, seq_len)`` long tensor ready for generation.
    """
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]
    text: str = hf_tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    ids = aware_tok.encode(text, add_special_tokens=False)
    return torch.tensor(ids, dtype=torch.long).unsqueeze(0).to(device)


@torch.no_grad()
def encode_all_items(
    item_embs: torch.Tensor,
    item_tok: RQVAETokenizer,
    aware_tok: ItemAwareTokenizer,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    """Encode all item embeddings to code tensors of shape ``(num_items, num_levels)``."""
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


@gin.configurable
def main(
    model_dir: str = gin.REQUIRED,  # type: ignore[assignment]
    hf_model_name: str = gin.REQUIRED,  # type: ignore[assignment]
    item_tok_path: str = gin.REQUIRED,  # type: ignore[assignment]
    item_tok_type: str = "rqvae",
    num_levels: int = gin.REQUIRED,  # type: ignore[assignment]
    codebook_size: int = gin.REQUIRED,  # type: ignore[assignment]
    root: str = "data/amazon",
    split: str = "beauty",
    prompts: list = gin.REQUIRED,  # type: ignore[assignment]
    max_new_tokens: int = 128,
    do_sample: bool = False,
    temperature: float = 1.0,
    encode_batch_size: int = 512,
    bf16: bool = False,
    seed: int = 42,
) -> None:
    """
    Args:
        model_dir: Path to the finetuned model directory (output of finetune_qwen.py).
        hf_model_name: Base HF model name used during finetuning (e.g. Qwen/Qwen3.5-2B).
        item_tok_path: Path to the item tokenizer checkpoint (.pt).
        item_tok_type: "rqvae" or "rq_kmeans".
        num_levels: Number of RQ levels (must match finetuning config).
        codebook_size: Codebook size per level (must match finetuning config).
        root: Root directory for the AmazonReviews dataset (used to build the item trie).
        split: Dataset split name (e.g. "beauty").
        prompts: List of plain-text user prompt strings.
        max_new_tokens: Maximum total tokens to generate per prompt.
        do_sample: Use sampling; if False greedy decoding is used.
        temperature: Sampling temperature (only used when do_sample=True).
        encode_batch_size: Batch size for encoding item embeddings into codes.
        bf16: Load the model in bfloat16.
        seed: Random seed.
    """
    torch.manual_seed(seed)
    device = get_device()
    print(f"Device: {device}")

    # 1. Load item tokenizer
    print(f"Loading item tokenizer ({item_tok_type}) from {item_tok_path}")
    if item_tok_type == "rqvae":
        item_tok = RQVAETokenizer.load(item_tok_path).to(device)
        item_tok.eval()
        for p in item_tok.parameters():
            p.requires_grad_(False)
    else:
        item_tok = RQKMeansTokenizer.load(item_tok_path)

    # 2. Build ItemAwareTokenizer
    print(f"Loading HF tokenizer: {hf_model_name}")
    hf_tokenizer = AutoTokenizer.from_pretrained(hf_model_name)
    aware_tok = ItemAwareTokenizer(
        hf_tokenizer,
        item_tok,
        num_levels=num_levels,
        codebook_size=codebook_size,
    )
    print(f"  Vocab: {aware_tok.original_vocab_size} → {aware_tok.vocab_size}")

    # 3. Load model
    dtype = torch.bfloat16 if bf16 else torch.float32
    print(f"Loading model from {model_dir} (dtype={dtype})")
    model: ItemAwareCausalLM = ItemAwareCausalLM.from_causal_lm(
        model_dir, aware_tok, torch_dtype=dtype
    )
    model.to(device)
    model.eval()

    # 4. Build item trie from the full item catalog
    print(f"Loading item catalog: root={root}, split={split}")
    raw_data = AmazonReviews(root=root, split=split)
    item_embs = raw_data.data["item"]["x"]
    print(f"  {item_embs.shape[0]} items — encoding to codes...")
    codes_table = encode_all_items(
        item_embs, item_tok, aware_tok, encode_batch_size, device
    )
    print("  Building item trie...")
    trie = aware_tok.build_item_trie(codes_table.to(device))
    print("  Trie built.")

    # 5. Run prompts
    print("\n" + "=" * 60)
    for i, prompt in enumerate(prompts):
        print(f"\n[{i + 1}/{len(prompts)}] Prompt: {prompt!r}")

        input_ids = build_prompt_ids(prompt, hf_tokenizer, aware_tok, device)

        with torch.inference_mode():
            new_token_ids = generate_with_item_constraints(
                model=model,
                input_ids=input_ids,
                trie=trie,
                item_sep_token_id=aware_tok.item_sep_token_id,
                num_levels=num_levels,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature if do_sample else 1.0,
                eos_token_id=hf_tokenizer.eos_token_id,
            )

        # Decode using decode_sequence so item tokens display as code tuples
        parts = aware_tok.decode_sequence(new_token_ids.tolist())
        response_parts: list[str] = []
        for part in parts:
            if isinstance(part, str):
                response_parts.append(part)
            else:
                response_parts.append(str(tuple(part.codes.tolist())))
        print(f"Response: {''.join(response_parts).strip()}")
        print("-" * 60)


if __name__ == "__main__":
    parse_config()
    main()
