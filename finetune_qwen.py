"""Finetune Qwen on precomputed interleaved semantic ID + text sequences.

Loss is computed on the full sequence (text + item tokens) by default.
Sequences must be precomputed with scripts/preprocessing/precompute_sequences.py
before running this script.

Usage:
    python scripts/training/finetune_qwen.py \
        --model_name Qwen/Qwen3.5-2B \
        --precomputed_path data/precomputed/beauty/beauty_train.pt \
        --precomputed_eval_path data/precomputed/beauty/beauty_eval.pt \
        --num_levels 3 --codebook_size 256 \
        --batch_size 8 --max_length 512 --loss_on all \
        --bf16 --gradient_checkpointing \
        --wandb_project my-project --wandb_run_name beauty-run
"""

from __future__ import annotations

import argparse
import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

from examples.data.amazon import PrecomputedSequenceDataset
from rectokens.integrations.hf.collator import PrecomputedSequenceCollator
from rectokens.integrations.hf.model import resize_and_initialize
from rectokens.integrations.hf.tokenizer import ItemAwareTokenizer


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Finetune Qwen on precomputed Amazon sequences"
    )
    # Precomputed data paths
    p.add_argument(
        "--precomputed_path",
        type=str,
        required=True,
        help="Path to precomputed training sequences (.pt)",
    )
    p.add_argument(
        "--precomputed_eval_path",
        type=str,
        default=None,
        help="Path to precomputed eval sequences (.pt). "
        "If omitted, no evaluation is run.",
    )
    # Model
    p.add_argument("--model_name", type=str, default="Qwen/Qwen3.5-2B")
    p.add_argument(
        "--num_levels",
        type=int,
        default=3,
        help="Number of RQ levels (needed for vocab sizing)",
    )
    p.add_argument(
        "--codebook_size",
        type=int,
        default=256,
        help="Codebook size per level (needed for vocab sizing)",
    )
    # Training hyperparameters
    p.add_argument(
        "--batch_size", type=int, default=8, help="Per-device training batch size"
    )
    p.add_argument(
        "--grad_accum", type=int, default=4, help="Gradient accumulation steps"
    )
    p.add_argument("--num_epochs", type=int, default=3)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--weight_decay", type=float, default=1e-2)
    p.add_argument(
        "--max_length", type=int, default=512, help="Truncate sequences to this length"
    )
    p.add_argument(
        "--no_expand_vocab",
        action="store_true",
        help="Skip vocabulary expansion and embedding resize. Use when finetuning a "
        "model whose vocabulary has already been expanded.",
    )
    p.add_argument(
        "--loss_on",
        type=str,
        default="all",
        choices=["all", "items", "text"],
        help="Which token positions contribute to the loss",
    )
    # Output / checkpointing
    p.add_argument("--output_dir", type=str, default="checkpoints")
    p.add_argument("--log_every", type=int, default=10)
    p.add_argument(
        "--save_every",
        type=int,
        default=5000,
        help="Save checkpoint every N steps (0 = epoch-end only)",
    )
    p.add_argument(
        "--eval_every",
        type=int,
        default=0,
        help="Evaluate every N steps (0 = no mid-training eval)",
    )
    # Precision / memory
    p.add_argument("--bf16", action="store_true")
    p.add_argument("--gradient_checkpointing", action="store_true")
    # Logging
    p.add_argument("--wandb_project", type=str, default=None)
    p.add_argument("--wandb_run_name", type=str, default=None)
    return p.parse_args()


def train(args: argparse.Namespace) -> None:
    device = get_device()

    # 1. Load precomputed datasets
    dataset = PrecomputedSequenceDataset(args.precomputed_path)
    eval_dataset = None
    if args.precomputed_eval_path and args.eval_every > 0:
        eval_dataset = PrecomputedSequenceDataset(args.precomputed_eval_path)

    # 2. HF text tokenizer (needed for pad_token_id and resize_and_initialize)
    hf_tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-2B")
    pad_token_id = hf_tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = hf_tokenizer.eos_token_id

    # 3. Build a dummy ItemAwareTokenizer solely to register item tokens and
    #    compute the extended vocab size used to resize the model embeddings.
    #    No item_tokenizer neural net is needed at training time.
    class _DummyItemTokenizer:
        """Minimal stand-in — encode() is never called during training."""

        def parameters(self):
            return iter([])

    aware_tokenizer = ItemAwareTokenizer(
        hf_tokenizer,
        _DummyItemTokenizer(),
        args.num_levels,
        args.codebook_size,
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
    dtype = torch.bfloat16 if args.bf16 else torch.float32
    model = AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype=dtype).to(
        device
    )
    if not args.no_expand_vocab:
        resize_and_initialize(model, aware_tokenizer)
    else:
        print(
            f"[finetune_qwen] Skipping vocabulary expansion "
            f"(expected vocab size: {aware_tokenizer.vocab_size})"
        )

    # 5. Collator — no GPU calls, safe to use multiple workers
    collator = PrecomputedSequenceCollator(
        original_vocab_size=dataset.original_vocab_size,
        pad_token_id=pad_token_id,
        loss_on=args.loss_on,
        max_length=args.max_length,
    )

    # 6. W&B setup
    if args.wandb_project:
        os.environ["WANDB_PROJECT"] = args.wandb_project
        if args.wandb_run_name:
            os.environ["WANDB_NAME"] = args.wandb_run_name
        report_to = "wandb"
    else:
        report_to = "none"

    # 7. TrainingArguments
    save_steps = args.save_every if args.save_every > 0 else args.log_every * 10
    eval_strategy = (
        "steps" if (eval_dataset is not None and args.eval_every > 0) else "no"
    )
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        bf16=args.bf16,
        gradient_checkpointing=args.gradient_checkpointing,
        logging_steps=args.log_every,
        logging_strategy="steps",
        eval_strategy=eval_strategy,
        eval_steps=args.eval_every if eval_strategy == "steps" else None,
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
    trainer.save_model(os.path.join(args.output_dir, "final"))
    hf_tokenizer.save_pretrained(os.path.join(args.output_dir, "final"))
    print(f"Training complete. Final model saved to {args.output_dir}/final")


if __name__ == "__main__":
    train(parse_args())
