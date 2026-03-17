"""Finetune Qwen on interleaved semantic ID + text sequences using HF Trainer.

Loss is computed on the full sequence (text + item tokens) by default.

Usage:
    python -m examples.training.finetune_qwen \
        --root data/amazon --split beauty --seq_split train \
        --item_tok_path item_tok.pt --num_levels 3 --codebook_size 256 \
        --model_name Qwen/Qwen3.5-2B --batch_size 2 --num_epochs 1 \
        --max_seq_len 10 --bf16 --gradient_checkpointing --log_every 5 \
        --wandb_project my-project --wandb_run_name beauty-run
"""

from __future__ import annotations

import argparse
import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

from examples.data.amazon import UserSequenceDataset
from rectokens.integrations.hf.collator import InterleavedSequenceCollator
from rectokens.integrations.hf.model import resize_and_initialize
from rectokens.integrations.hf.tokenizer import ItemAwareTokenizer
from rectokens.tokenizers.rqvae import RQVAETokenizer


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Finetune Qwen on Amazon sequences")
    p.add_argument("--root", type=str, default="data/amazon")
    p.add_argument("--split", type=str, default="beauty",
                   choices=["beauty", "sports", "toys"])
    p.add_argument("--seq_split", type=str, default="train",
                   choices=["train", "eval", "test"])
    p.add_argument("--max_seq_len", type=int, default=20)
    p.add_argument("--item_tok_path", type=str, required=True,
                   help="Path to fitted .pt item tokenizer")
    p.add_argument("--num_levels", type=int, default=3)
    p.add_argument("--codebook_size", type=int, default=256)
    p.add_argument("--model_name", type=str, default="Qwen/Qwen3.5-2B")
    p.add_argument("--batch_size", type=int, default=2,
                   help="Per-device training batch size")
    p.add_argument("--grad_accum", type=int, default=8,
                   help="Gradient accumulation steps")
    p.add_argument("--num_epochs", type=int, default=3)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--weight_decay", type=float, default=1e-2)
    p.add_argument("--max_length", type=int, default=512)
    p.add_argument("--loss_on", type=str, default="all",
                   choices=["all", "items", "text"],
                   help="Which token positions contribute to the loss")
    p.add_argument("--output_dir", type=str, default="checkpoints")
    p.add_argument("--bf16", action="store_true")
    p.add_argument("--gradient_checkpointing", action="store_true",
                   help="Enable gradient checkpointing to reduce GPU memory usage")
    p.add_argument("--log_every", type=int, default=10,
                   help="Log metrics every N steps")
    p.add_argument("--save_every", type=int, default=500,
                   help="Save checkpoint every N steps (0 = only at epoch end)")
    p.add_argument("--wandb_project", type=str, default=None,
                   help="W&B project name. If set, enables wandb logging.")
    p.add_argument("--wandb_run_name", type=str, default=None,
                   help="W&B run name (optional)")
    return p.parse_args()


def train(args: argparse.Namespace) -> None:
    device = get_device()

    # 1. Load pre-fitted item tokenizer (frozen)
    item_tok = RQVAETokenizer.load(args.item_tok_path).to(device)
    item_tok.eval()
    for p in item_tok.parameters():
        p.requires_grad_(False)

    # 2. HF text tokenizer + ItemAwareTokenizer
    hf_tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    aware_tokenizer = ItemAwareTokenizer(
        hf_tokenizer, item_tok, args.num_levels, args.codebook_size
    )

    # Fall back to eos_token_id if there is no dedicated pad token
    pad_token_id = hf_tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = hf_tokenizer.eos_token_id

    # 3. Load model, resize embeddings for item tokens
    dtype = torch.bfloat16 if args.bf16 else torch.float32
    model = AutoModelForCausalLM.from_pretrained(args.model_name, dtype=dtype).to(device)
    resize_and_initialize(model, aware_tokenizer)

    # 4. Dataset + collator
    dataset = UserSequenceDataset(
        args.root, args.split, args.seq_split, max_seq_len=args.max_seq_len
    )
    collator = InterleavedSequenceCollator(
        aware_tokenizer,
        loss_on=args.loss_on,
        pad_token_id=pad_token_id,
        max_length=args.max_length,
    )

    # 5. W&B setup
    if args.wandb_project:
        os.environ["WANDB_PROJECT"] = args.wandb_project
        if args.wandb_run_name:
            os.environ["WANDB_NAME"] = args.wandb_run_name
        report_to = "wandb"
    else:
        report_to = "none"

    # 6. TrainingArguments — Trainer handles the loop, logging, and checkpointing
    save_steps = args.save_every if args.save_every > 0 else args.log_every * 10
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
        save_strategy="steps",
        save_steps=save_steps,
        save_total_limit=3,
        report_to=report_to,
        dataloader_num_workers=0,  # collator uses GPU (item_tok) — must stay in main process
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=collator,
    )

    trainer.train()
    trainer.save_model(os.path.join(args.output_dir, "final"))
    hf_tokenizer.save_pretrained(os.path.join(args.output_dir, "final"))
    print(f"Training complete. Final model saved to {args.output_dir}/final")


if __name__ == "__main__":
    train(parse_args())
