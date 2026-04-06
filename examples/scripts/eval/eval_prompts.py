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
from torch import nn
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from examples.data.amazon import AmazonReviews
from examples.utils import parse_config
from rectokens.core.tokenizer import Tokenizer
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
    text = hf_tokenizer.apply_chat_template(  # type: ignore[assignment]
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    assert isinstance(text, str)
    ids = aware_tok.encode(text, add_special_tokens=False)
    return torch.tensor(ids, dtype=torch.long).unsqueeze(0).to(device)


@torch.no_grad()
def encode_all_items(
    item_embs: torch.Tensor,
    item_tok: Tokenizer,
    aware_tok: ItemAwareTokenizer,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    """Encode all item embeddings to code tensors of shape ``(num_items, num_levels)``."""
    num_items = item_embs.shape[0]
    all_codes = torch.zeros(num_items, aware_tok.num_levels, dtype=torch.long)
    enc_dtype = torch.float32
    if isinstance(item_tok, nn.Module):
        params = list(item_tok.parameters())
        if params:
            enc_dtype = params[0].dtype

    for start in range(0, num_items, batch_size):
        end = min(start + batch_size, num_items)
        batch = item_embs[start:end].to(device=device, dtype=enc_dtype)
        token_seq = item_tok.encode(batch)
        all_codes[start:end] = token_seq.codes.cpu()
    return all_codes


def find_expected_item(
    prompt: str,
    item_texts: list[str],
    codes_table: torch.Tensor,
) -> tuple[int, tuple[int, ...]] | None:
    """Search item_texts for one whose text is a substring of the prompt.

    Returns ``(item_id, codes)`` for the first match, or ``None``.
    """
    for item_id, text in enumerate(item_texts):
        if text.strip() in prompt:
            return item_id, tuple(codes_table[item_id].tolist())
    return None


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
    model.to(device)  # type: ignore[call-arg]
    model.eval()

    # 4. Build item trie and lookup structures from the full item catalog
    print(f"Loading item catalog: root={root}, split={split}")
    raw_data = AmazonReviews(root=root, split=split)
    item_embs = raw_data.data["item"]["x"]
    item_texts: list[str] = [str(t) for t in raw_data.data["item"]["text"]]
    print(f"  {item_embs.shape[0]} items — encoding to codes...")
    codes_table = encode_all_items(
        item_embs, item_tok, aware_tok, encode_batch_size, device
    )

    # Reverse lookup: code tuple → item_id
    codes_to_item_id: dict[tuple[int, ...], int] = {}
    for item_id, codes in enumerate(codes_table.tolist()):
        key = tuple(codes)
        if key not in codes_to_item_id:
            codes_to_item_id[key] = item_id

    print("  Building item trie...")
    trie = aware_tok.build_item_trie(codes_table.to(device))
    print("  Trie built.")

    # 5. Run prompts
    print("\n" + "=" * 60)
    for i, prompt in enumerate(prompts):
        print(f"\n[{i + 1}/{len(prompts)}] Prompt: {prompt!r}")

        # Find expected item by matching its text against the prompt
        expected = find_expected_item(prompt, item_texts, codes_table)

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

        # Decode: item tokens → code tuples, text tokens → plain string
        parts = aware_tok.decode_sequence(new_token_ids.tolist())
        response_parts: list[str] = []
        generated_item_ids: list[int] = []
        for part in parts:
            if isinstance(part, str):
                response_parts.append(part)
            else:
                codes = tuple(part.codes.tolist())
                response_parts.append(str(codes))
                item_id = codes_to_item_id.get(codes)
                if item_id is not None:
                    generated_item_ids.append(item_id)

        print(f"Response:      {''.join(response_parts).strip()}")

        # Show generated item text(s)
        for rank, item_id in enumerate(generated_item_ids):
            label = (
                "Generated item"
                if len(generated_item_ids) == 1
                else f"Generated item {rank + 1}"
            )
            print(f"{label} [{item_id}]: {item_texts[item_id]}")

        # Show expected item text if found in the prompt
        if expected is not None:
            exp_id, exp_codes = expected
            print(f"Expected SID:  {exp_codes}")
            print(f"Expected item [{exp_id}]: {item_texts[exp_id]}")
        print("-" * 60)


if __name__ == "__main__":
    parse_config()
    main()
