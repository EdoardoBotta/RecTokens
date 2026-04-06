"""Finetune Qwen on precomputed interleaved semantic ID + text sequences.

Loss is computed on the full sequence (text + item tokens) by default.
Sequences must be precomputed with precompute_sequences.py before running.

Usage:
    python examples/scripts/training/finetune_qwen.py examples/configs/finetuning/finetune_qwen_beauty.gin
"""

from __future__ import annotations

import os

import gin
import wandb
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

from examples.data.amazon import PrecomputedSequenceDataset
from examples.utils import parse_config
from rectokens.integrations.hf.collator import PrecomputedSequenceCollator
from rectokens.integrations.hf.model import ItemAwareCausalLM
from rectokens.integrations.hf.tokenizer import ItemAwareTokenizer


def freeze_pretrained_parameters(model: nn.Module, original_vocab_size: int) -> None:
    """Freeze everything except new-token rows in embed_tokens and lm_head.

    All transformer weights and the original vocabulary rows in the embedding
    and lm_head matrices are frozen.  Only the rows at indices
    ``[original_vocab_size, vocab_size)`` — i.e. the newly added item tokens —
    receive gradient updates, which is enforced via a backward hook that zeroes
    gradients for the frozen rows.

    Args:
        model: The model whose parameters should be (partially) frozen.
        original_vocab_size: Number of tokens in the base model's vocabulary
            before item tokens were added.  Rows at indices
            ``[:original_vocab_size]`` in embed_tokens / lm_head are frozen.
    """
    for param in model.parameters():
        param.requires_grad_(False)

    def _mask_pretrained_grad(grad: torch.Tensor) -> torch.Tensor:
        masked = grad.clone()
        masked[:original_vocab_size] = 0.0
        return masked

    embed_weight = model.get_input_embeddings().weight
    embed_weight.requires_grad_(True)
    embed_weight.register_hook(_mask_pretrained_grad)

    lm_head_weight = model.get_output_embeddings().weight
    if lm_head_weight is not embed_weight:
        lm_head_weight.requires_grad_(True)
        lm_head_weight.register_hook(_mask_pretrained_grad)

    n_total = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    new_tokens = embed_weight.shape[0] - original_vocab_size
    tied = lm_head_weight is embed_weight
    print(
        f"[freeze_pretrained] Frozen all params except new-token rows "
        f"(new_tokens={new_tokens}, lm_head_tied={tied}). "
        f"Trainable params: {n_trainable:,} / {n_total:,}"
    )


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@gin.configurable
def train(
    precomputed_path: str = gin.REQUIRED,
    precomputed_eval_path: str | None = None,
    model_name: str = "Qwen/Qwen3.5-2B",
    num_levels: int = 3,
    codebook_size: int = 256,
    batch_size: int = 8,
    grad_accum: int = 4,
    num_epochs: int = 3,
    lr: float = 2e-4,
    weight_decay: float = 1e-2,
    max_length: int = 256,
    no_expand_vocab: bool = False,
    align_sid_finetuning: bool = False,
    output_dir: str = "checkpoints",
    log_every: int = 10,
    save_every: int = 5000,
    eval_every: int = 0,
    bf16: bool = False,
    wandb_project: str | None = None,
    wandb_run_name: str | None = None,
) -> None:
    device = get_device()

    # 1. Load precomputed datasets
    dataset = PrecomputedSequenceDataset(precomputed_path)
    eval_dataset = None
    if precomputed_eval_path and eval_every > 0:
        eval_dataset = PrecomputedSequenceDataset(precomputed_eval_path)

    # 2. HF text tokenizer (needed for pad_token_id)
    hf_tokenizer = AutoTokenizer.from_pretrained(model_name)
    pad_token_id = hf_tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = hf_tokenizer.eos_token_id

    # 3. Build ItemAwareTokenizer to register item tokens and compute the extended
    #    vocab size. No item_tokenizer needed for precomputed-sequence training.
    aware_tokenizer = ItemAwareTokenizer(
        hf_tokenizer,
        num_levels=num_levels,
        codebook_size=codebook_size,
    )
    # Sanity-check: dataset metadata must match gin config
    if dataset.original_vocab_size != aware_tokenizer.original_vocab_size:
        raise ValueError(
            f"original_vocab_size mismatch: precomputed file has "
            f"{dataset.original_vocab_size}, but tokenizer has "
            f"{aware_tokenizer.original_vocab_size}. "
            "Make sure --model_name matches the one used during precomputation."
        )
    if dataset.num_levels != num_levels:
        raise ValueError(
            f"num_levels mismatch: precomputed file has {dataset.num_levels}, "
            f"but gin config has {num_levels}."
        )
    if dataset.codebook_size != codebook_size:
        raise ValueError(
            f"codebook_size mismatch: precomputed file has {dataset.codebook_size}, "
            f"but gin config has {codebook_size}."
        )

    # 4. Load model and (optionally) resize embeddings for item tokens
    dtype = torch.bfloat16 if bf16 else torch.float32
    if not no_expand_vocab:
        model = ItemAwareCausalLM.from_causal_lm(
            model_name, aware_tokenizer, torch_dtype=dtype
        ).to(device)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype).to(
            device
        )
        print(
            f"[finetune_qwen] Skipping vocabulary expansion "
            f"(expected vocab size: {aware_tokenizer.vocab_size})"
        )

    # 5. (Optional) freeze all pretrained parameters; train only new-token rows
    if align_sid_finetuning:
        if no_expand_vocab:
            raise ValueError(
                "freeze_pretrained=True requires vocabulary expansion "
                "(set no_expand_vocab=False)"
            )
        freeze_pretrained_parameters(model, aware_tokenizer.original_vocab_size)

    # 6. Collator — no GPU calls, safe to use multiple workers
    collator = PrecomputedSequenceCollator(
        pad_token_id=pad_token_id,
        max_length=max_length,
    )

    # 7. W&B setup
    if wandb_project:
        wandb.login()
        wandb.init(project=wandb_project, name=wandb_run_name)
        report_to = "wandb"
    else:
        report_to = "none"
    report_to = "wandb"

    # 8. TrainingArguments
    save_steps = save_every if save_every > 0 else log_every * 10
    eval_strategy = "steps" if (eval_dataset is not None and eval_every > 0) else "no"
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=lr,
        weight_decay=weight_decay,
        bf16=bf16,
        logging_steps=log_every,
        logging_strategy="steps",
        eval_strategy=eval_strategy,
        eval_steps=eval_every if eval_strategy == "steps" else None,
        save_strategy="steps",
        save_steps=save_steps,
        save_total_limit=3,
        save_only_model=True,
        report_to=report_to,
        dataloader_num_workers=4,  # collator is CPU-only — workers are safe
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
    )

    trainer.train()
    trainer.save_model(os.path.join(output_dir, "final"))
    hf_tokenizer.save_pretrained(os.path.join(output_dir, "final"))
    print(f"Training complete. Final model saved to {output_dir}/final")


if __name__ == "__main__":
    parse_config()
    train()
