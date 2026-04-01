"""Finetune Qwen on precomputed interleaved semantic ID + text sequences.

Loss is computed on the full sequence (text + item tokens) by default.
Sequences must be precomputed with precompute_sequences.py before running.

Usage:
    python examples/scripts/training/finetune_qwen.py examples/configs/finetune_qwen_beauty.gin
"""

from __future__ import annotations

import os

import gin
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

from examples.data.amazon import PrecomputedSequenceDataset
from examples.utils import parse_config
from rectokens.integrations.hf.collator import PrecomputedSequenceCollator
from rectokens.integrations.hf.model import ItemAwareCausalLM
from rectokens.integrations.hf.tokenizer import ItemAwareTokenizer


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
    max_length: int = 512,
    no_expand_vocab: bool = False,
    loss_on: str = "all",
    output_dir: str = "checkpoints",
    log_every: int = 10,
    save_every: int = 5000,
    eval_every: int = 0,
    bf16: bool = False,
    gradient_checkpointing: bool = False,
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
    hf_tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-2B")
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
    # Sanity-check: original_vocab_size from dataset must match
    if dataset.original_vocab_size != aware_tokenizer.original_vocab_size:
        raise ValueError(
            f"original_vocab_size mismatch: precomputed file has "
            f"{dataset.original_vocab_size}, but tokenizer has "
            f"{aware_tokenizer.original_vocab_size}. "
            "Make sure --model_name matches the one used during precomputation."
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

    # 5. Collator — no GPU calls, safe to use multiple workers
    collator = PrecomputedSequenceCollator(
        original_vocab_size=dataset.original_vocab_size,
        pad_token_id=pad_token_id,
        loss_on=loss_on,
        max_length=max_length,
    )

    # 6. W&B setup
    if wandb_project:
        os.environ["WANDB_PROJECT"] = wandb_project
        if wandb_run_name:
            os.environ["WANDB_NAME"] = wandb_run_name
        report_to = "wandb"
    else:
        report_to = "none"

    # 7. TrainingArguments
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
        gradient_checkpointing=gradient_checkpointing,
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
