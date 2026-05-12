<p align="center">
  <img src="assets/rectokens_logo_v2.svg" alt="RecTokens" width="100%"/>
</p>

# RecTokens

A tokenizer library for sequential recommendation systems. RecTokens converts continuous item embeddings into discrete multi-level token sequences using **Residual Quantization (RQ)**, enabling large item catalogs to be represented as compact token IDs suitable for Language model finetuning and/or classic Transformer-based recommendation models.

## Overview

Modern sequential recommendation models treat item retrieval as a language modeling problem: given a user's interaction history, generate the next item's token sequence autoregressively. RecTokens provides three components for this pipeline:

1. **Tokenizers** — Convert item feature vectors into discrete token sequences (item IDs).
2. **Constrained Decoding** — At inference time, efficiently restrict the model's generation to only valid item token sequences.
3. **HuggingFace Integration** — Plug item tokens directly into any pretrained LLM from HuggingFace. `ItemAwareTokenizer` extends an HF text tokenizer with item-level tokens, `ItemAwareCausalLM.from_causal_lm` adapts the model's embedding table, and `InterleavedSequenceCollator` / `PrecomputedSequenceCollator` prepare mixed text+item batches for standard HF `Trainer` fine-tuning. This lets you fine-tune models like Qwen on recommendation sequences with minimal boilerplate.

### Prior Work

RecTokens builds on several lines of research in generative recommendation. The idea of treating item retrieval as a language modeling problem over discrete semantic IDs originates from Rajput et al., [**"Recommender Systems with Generative Retrieval"**](https://arxiv.org/abs/2305.05065) (NeurIPS 2023), which introduced the TIGER system and demonstrated that items can be represented as compact hierarchical token sequences produced by Residual Quantization. He et al., [**"PLUM: Adapting Pre-trained Language Models for Industrial-scale Generative Recommendations"**](https://arxiv.org/abs/2510.07784) (Google, 2025) and Zhou et al., [**"OneRec Technical Report"**](https://arxiv.org/abs/2506.13695) (2025) extend this paradigm to production-scale LLM fine-tuning, validating the approach at the scale of real recommendation systems.

The constrained decoding components — the CSR trie structure and the fused Triton kernels for masked linear projection — are directly inspired by Su et al., [**"Vectorizing the Trie: Efficient Constrained Decoding for LLM-based Generative Retrieval on Accelerators"**](https://arxiv.org/abs/2602.22647) (2026), which introduces the VTNK kernel and shows how to reformulate trie-constrained beam search as a GPU-friendly sparse matrix operation, eliminating the bottleneck of sequential CPU trie traversal at large batch sizes. The fused sampling kernel (`fused_linear_constrained_node_transition_sampling`) additionally incorporates the Gumbel-max trick as described in Ruiz et al., [**"FlashSampling: Fast and Memory-Efficient Exact Sampling"**](https://arxiv.org/abs/2603.15854) (2026), fusing categorical sampling directly into the linear projection pass to avoid materializing the full logit tensor.

### Residual Quantization

RQ encodes a `D`-dimensional embedding as `L` discrete codes, one per level:

```
r_0 = x
code_l, q_l = quantizer_l.encode(r_{l-1})
r_l = r_{l-1} - q_l     (residual)
```

With `K` codes per level and `L` levels, the scheme supports `K^L` unique item IDs. For example, `K=256, L=3` yields 16.7 million possible item IDs.

## Installation

```bash
pip install rectokens
```

For development or editable installs:

```bash
pip install -e .

# With development dependencies
pip install -e ".[dev]"
```

**Requirements:** Python ≥ 3.10, PyTorch ≥ 2.0, NumPy ≥ 1.24. CUDA is required for constrained decoding with GPU kernels.

## End-to-End Workflow

This section walks through the full pipeline for training a generative recommendation model on the Amazon Reviews dataset.

### Step 0 — Data

`AmazonReviews` (in `examples/data/amazon.py`) auto-downloads and processes the dataset on first use. Point `--root` to an empty directory and the data will be downloaded there.

Supported `--split` values: `beauty`, `sports`, `toys`.

### Step 1 — Train an item tokenizer

Choose **RQKMeans** (fast, no GPU required) or **RQVAE** (better reconstruction, requires GPU).

**RQKMeans:**
```bash
python -m examples.scripts.training.train_rqkmeans examples/configs/pretraining/train_rqkmeans_beauty.gin
```

**RQVAE:**
```bash
python -m examples.scripts.training.train_rqvae examples/configs/pretraining/train_rqvae_beauty.gin
```

### Step 2 — Precompute interleaved sequences

Encode all item embeddings once with the fitted tokenizer and assemble the flat token-ID sequences for every user (text tokens + semantic ID tokens). This avoids repeated neural-network inference during training.

```bash
python -m examples.scripts.preprocessing.precompute_sequences examples/configs/preprocessing/precompute_sequences_beauty.gin
```

Key config parameters (`examples/configs/preprocessing/precompute_sequences_beauty.gin`):
- `main.item_tok_type` — `"rqvae"` (default) or `"rqkmeans"`
- `main.seq_splits` — tuple of splits: `("train", "eval", "test")`
- `main.include_future` — append the held-out future item to each sequence
- `main.batch_size` — batch size for item embedding encoding (default 512)

### Step 3 — Finetune Qwen on precomputed sequences

```bash
python -m examples.scripts.training.finetune_qwen examples/configs/finetuning/finetune_qwen_beauty.gin
```

Key config parameters (`examples/configs/finetuning/finetune_qwen_beauty.gin`):
- `train.loss_on` — `"all"` (default), `"items"`, or `"text"`
- `train.bf16` / `train.gradient_checkpointing` — recommended for GPU training
- `train.wandb_project` / `train.wandb_run_name` — optional W&B logging (`None` for no logging)
- `train.precomputed_eval_path` — if set, enables mid-training evaluation

## Tokenizers

### RQKMeansTokenizer

Fits codebooks via mini-batch K-means. No training loop required — call `fit_step` on batches of embeddings, then encode.

```python
from rectokens.tokenizers.rq_kmeans import RQKMeansTokenizer
import numpy as np
import torch

tokenizer = RQKMeansTokenizer(
    num_levels=3,       # number of RQ levels (token sequence length)
    codebook_size=256,  # codes per level
    dim=128,            # embedding dimension
)

data = torch.from_numpy(np.random.randn(10_000, 128).astype("float32"))
for start in range(0, len(data), 256):
    tokenizer.fit_step(data[start : start + 256])

features = torch.randn(8, 128)
tokens = tokenizer.encode(features)   # TokenSequence, codes shape: (8, 3)
reconstructed = tokenizer.decode(tokens)  # Tensor shape: (8, 128)

tokenizer.save("tokenizer.pt")
tokenizer = RQKMeansTokenizer.load("tokenizer.pt")
```

### RQVAETokenizer

Learns tokenization end-to-end via an encoder–quantizer–decoder architecture. Uses a Vector Quantization (VQ) objective with EMA codebook updates and a dead-code restart mechanism.

```python
from rectokens.tokenizers.rqvae import RQVAETokenizer
import torch
import torch.nn.functional as F

tokenizer = RQVAETokenizer(
    input_dim=128,
    latent_dim=64,
    hidden_dim=256,
    num_levels=3,
    codebook_size=256,
    commitment_weight=0.25,
    ema_decay=0.99,
)

optimizer = torch.optim.Adam(tokenizer.parameters(), lr=1e-3)

for batch in data_loader:
    out = tokenizer(batch)
    loss = F.mse_loss(out.recon, batch) + out.commitment_loss
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

tokenizer._fitted = True
tokenizer.eval()

tokens = tokenizer.encode(features)
reconstructed = tokenizer.decode(tokens)

tokenizer.save("tokenizer.pt")
tokenizer = RQVAETokenizer.load("tokenizer.pt")
```

The `forward` pass returns an `RQVAEOutput` NamedTuple with fields `recon` (reconstruction), `commitment_loss`, `codes`, and `p_unique_ids` (fraction of distinct token tuples in the batch).

## Constrained Decoding

At inference time, a recommendation model must generate token sequences that correspond to actual items in the catalog. RecTokens provides a GPU-accelerated trie for this constraint.

### CompactCSRTrie

A trie over all valid item token sequences, stored in Compressed Sparse Row (CSR) format for efficient GPU tensor operations. The first few layers use dense lookup tables for O(1) indexing; deeper layers use sparse CSR traversal.

```python
from rectokens.schemas.compact_csr_trie import CompactCSRTrie

# Build from the codes tensor of all items
sem_ids = tokenizer.encode(all_item_features)
trie = CompactCSRTrie.from_sorted_batch(
    sem_ids.codes,
    vocab_size=256,
    dense_lookup_layers=2,
)
```

### Autoregressive Generation

```python
from rectokens.decoding.constrained_decoding import autoregressive_generate
from rectokens.schemas.config import GenerationConfig

config = GenerationConfig(
    steps=3,       # token sequence length
    k=10,          # number of items to retrieve
    beam_size=50,  # beam width
    temperature=1.0,
)

# model: any nn.Module whose forward returns logits over the vocab
generated = autoregressive_generate(
    model=model,
    trie=trie,
    input_ids=user_history_ids,
    generation_config=config,
)
# generated shape: (B, k, steps)
```

### Constrained Node Transition

The core primitive for constrained decoding is a masked linear projection. Two implementations are provided:

- **`vtnk_pytorch`** — Pure PyTorch; applies a validity mask to logits before sampling.
- **`fused_linear_constrained_node_transition`** — Custom Triton kernel that fuses the matrix multiply and constraint masking into a single GPU kernel for maximum throughput.
- **`fused_linear_constrained_node_transition_sampling`** — Extends the fused kernel to additionally fuse multinomial sampling into the same GPU pass, eliminating a separate `torch.multinomial` call.

## HuggingFace Integration

`rectokens.integrations.hf` bridges RecTokens item tokenizers with HuggingFace models for end-to-end training of generative recommendation models. The integration has three components:

- **`ItemAwareTokenizer`** — extends a HF text tokenizer with item tokens (`<item_L{l}_C{c}>`), one per level/code pair. Encodes mixed text+item sequences to flat token id lists, decodes them back, and builds a `CompactCSRTrie` over the catalog for constrained generation.
- **`InterleavedSequenceCollator`** — collates mixed text/item example lists into padded `input_ids`, `attention_mask`, and `labels` tensors ready for `model(**batch)`. Supports `loss_on="all"|"items"|"text"` to mask loss to specific token types.
- **`PrecomputedSequenceCollator`** — lightweight collator for training on pre-encoded integer tensors produced by `precompute_sequences.py`. No neural-network calls at collation time.
- **`ItemAwareCausalLM`** (`rectokens.integrations.hf.model`) — wraps any HF causal-LM with item-aware constrained generation. Use `ItemAwareCausalLM.from_causal_lm` to load a model, resize its embedding table and `lm_head` to include item tokens, and optionally initialize item embeddings from projected RQ codebook vectors.

### Finetuning Qwen on item sequences (online encoding)

Use this path when you want to encode sequences on-the-fly during training, or for quick prototyping. For large-scale training, prefer the precomputed workflow above.

```python
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from rectokens.integrations.hf.tokenizer import ItemAwareTokenizer
from rectokens.integrations.hf.collator import PrecomputedSequenceCollator
from rectokens.integrations.hf.model import ItemAwareCausalLM
from rectokens.tokenizers.rqvae import RQVAETokenizer

# 1. Load a pre-fitted item tokenizer (frozen)
item_tok = RQVAETokenizer.load("item_tok.pt").eval()

# 2. Wrap HF tokenizer to register item tokens in the vocabulary
hf_tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-2B")
aware_tokenizer = ItemAwareTokenizer(
    hf_tokenizer, item_tokenizer=item_tok, num_levels=3, codebook_size=256
)
# vocab now includes 3×256 + 2 = 770 new tokens: <item_L0_C0> … <item_L2_C255> + <|item_start|> + <|item_end|>

# 3. Load model and resize embeddings to the extended vocab
model = ItemAwareCausalLM.from_causal_lm(
    "Qwen/Qwen3.5-2B", aware_tokenizer, torch_dtype=torch.bfloat16
).cuda()

# 4. Encode item embeddings on-the-fly to integer semantic IDs (codes)
item_embs = torch.randn(3, item_tok.input_dim)  # (N, D) — replace with real embeddings
with torch.no_grad():
    codes = item_tok.encode(item_embs).codes.tolist()  # list of [c0, c1, c2] per item

# 5. Build flat token-id sequences — items are semantic ID special tokens:
#    <|item_start|> <item_L0_Cx> <item_L1_Cy> <item_L2_Cz> <|item_end|>
def item_ids(c): return [aware_tokenizer.item_sep_token_id, *[aware_tokenizer.item_token_id(l, v) for l, v in enumerate(c)], aware_tokenizer.item_end_token_id]
def text_ids(s): return aware_tokenizer.encode(s, add_special_tokens=False)

sequences = [
    torch.tensor(text_ids("User watched ") + item_ids(codes[0]) + text_ids(" then ") + item_ids(codes[1]) + text_ids(". Next: ")),
    torch.tensor(text_ids("User bought ") + item_ids(codes[2]) + text_ids(". Recommend: ")),
]

# 6. Build examples: labels are -100 at text positions, item token ids otherwise
orig_vocab = aware_tokenizer.original_vocab_size
examples = [
    {"input_ids": s, "labels": torch.where(s >= orig_vocab, s, torch.full_like(s, -100))}
    for s in sequences
]

# 7. Collate and train
collator = PrecomputedSequenceCollator(pad_token_id=hf_tokenizer.eos_token_id, max_length=512)
loader = DataLoader(examples, batch_size=2, collate_fn=collator)

optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4)
model.train()
for batch in loader:
    optimizer.zero_grad()
    out = model(**{k: v.cuda() for k, v in batch.items()})
    out.loss.backward()
    optimizer.step()
```

### Training on precomputed sequences

Use `PrecomputedSequenceCollator` with `PrecomputedSequenceDataset` (from `examples.data.amazon`) when training on sequences produced by `precompute_sequences.py`. No item tokenizer neural net is needed at training time — omit `item_tokenizer` (defaults to `None`).

```python
import torch
from transformers import AutoTokenizer
from rectokens.integrations.hf.tokenizer import ItemAwareTokenizer
from rectokens.integrations.hf.collator import PrecomputedSequenceCollator
from rectokens.integrations.hf.model import ItemAwareCausalLM
from examples.data.amazon import PrecomputedSequenceDataset
from torch.utils.data import DataLoader

# 1. Load precomputed sequences
dataset = PrecomputedSequenceDataset("data/precomputed/beauty/beauty_train.pt")

# 2. HF tokenizer + ItemAwareTokenizer (no item_tokenizer neural net needed)
hf_tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-2B")

aware_tokenizer = ItemAwareTokenizer(
    hf_tokenizer,
    num_levels=dataset.num_levels,
    codebook_size=dataset.codebook_size,
)

# 3. Load model and resize embeddings
model = ItemAwareCausalLM.from_causal_lm(
    "Qwen/Qwen3.5-2B", aware_tokenizer, torch_dtype=torch.bfloat16
).cuda()

# 4. Collate precomputed integer sequences
# Labels are already masked in the precomputed data (loss on assistant turn only)
collator = PrecomputedSequenceCollator(
    pad_token_id=hf_tokenizer.eos_token_id,
    max_length=512,
)

loader = DataLoader(dataset, batch_size=8, collate_fn=collator)

optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4)
model.train()
for batch in loader:
    optimizer.zero_grad()
    out = model(**{k: v.cuda() for k, v in batch.items()})
    out.loss.backward()
    optimizer.step()
```

### Constrained generation at inference time

```python
# Build a trie over the item catalog (catalog_codes: (N, num_levels) int tensor)
trie = aware_tokenizer.build_item_trie(catalog_codes, dense_lookup_layers=1)

# Use autoregressive_generate with the extended vocab trie
from rectokens.decoding.constrained_decoding import autoregressive_generate
from rectokens.schemas.config import GenerationConfig

generated = autoregressive_generate(
    model=model,
    trie=trie,
    input_ids=user_history_ids,   # (B, T) in the extended HF vocab
    generation_config=GenerationConfig(steps=3, k=10, beam_size=50),
)
# generated: (B, k, 3) item token sequences in the extended HF vocab space
```

## Scripts Reference

| Script | Purpose |
|--------|---------|
| `examples/scripts/training/train_rqkmeans.py` | Train `RQKMeansTokenizer` on Amazon item embeddings |
| `examples/scripts/training/train_rqvae.py` | Train `RQVAETokenizer` on Amazon item embeddings |
| `examples/scripts/training/finetune_qwen.py` | Finetune Qwen on precomputed sequences via HF `Trainer` |
| `examples/scripts/preprocessing/precompute_sequences.py` | Encode items + assemble interleaved token-ID sequences for all users |
| `examples/scripts/eval/eval_retrieval.py` | Evaluate next-item retrieval with constrained beam search |
| `examples/scripts/benchmark/benchmark_vtnk.py` | Benchmark constrained decoding implementations |
| `examples/scripts/benchmark/benchmark_fused_sample.py` | Benchmark fused sampling kernel vs sparse PyTorch + multinomial |
| `examples/scripts/benchmark/benchmark_nn_quantize.py` | Benchmark nearest-neighbor quantization kernels |

All scripts accept a single positional gin config file. Ready-made configs live in `examples/configs/` organized into `pretraining/`, `finetuning/`, and `preprocessing/` subdirectories.

## Module Structure

```
rectokens/
├── core/               # Abstract base classes (Tokenizer, Quantizer, Codebook)
├── tokenizers/         # RQKMeansTokenizer, RQVAETokenizer
├── quantizers/         # KMeansQuantizer, ResidualQuantizer
├── codebooks/          # EuclideanCodebook (vectorized L2 nearest-neighbor)
├── decoding/           # vtnk_pytorch, autoregressive_generate, Trie (CPU)
├── schemas/            # CompactCSRTrie, GenerationConfig, GenerationState
├── ops/                # Python wrappers for kernels
├── kernels/            # Triton GPU kernels
├── modules/            # SparseLinear, ConstraintEnforcer (abstract), SparseTrieConstraintEnforcer
└── integrations/
    └── hf/             # ItemAwareTokenizer, InterleavedSequenceCollator,
                        # PrecomputedSequenceCollator, ItemAwareCausalLM

examples/
├── configs/
│   ├── pretraining/
│   │   ├── train_rqvae_beauty.gin        # RQ-VAE tokenizer — Amazon Beauty
│   │   ├── train_rqvae_sports.gin        # RQ-VAE tokenizer — Amazon Sports
│   │   ├── train_rqkmeans_beauty.gin     # RQ-KMeans tokenizer — Amazon Beauty
│   │   └── train_rqkmeans_sports.gin     # RQ-KMeans tokenizer — Amazon Sports
│   ├── finetuning/
│   │   ├── finetune_qwen_beauty.gin
│   │   └── eval_retrieval_beauty.gin
│   └── preprocessing/
│       └── precompute_sequences_beauty.gin
├── data/
│   └── amazon.py       # AmazonReviews, ItemData, UserSequenceDataset,
│                       # PrecomputedSequenceDataset
├── scripts/
│   ├── training/
│   │   ├── train_rqkmeans.py   # Train RQKMeansTokenizer on Amazon item embeddings
│   │   ├── train_rqvae.py      # Train RQVAETokenizer on Amazon item embeddings
│   │   └── finetune_qwen.py    # Finetune Qwen via HF Trainer on precomputed data
│   ├── eval/
│   │   └── eval_retrieval.py   # Evaluate next-item retrieval with beam search
│   ├── preprocessing/
│   │   └── precompute_sequences.py    # Precompute interleaved token-ID sequences
│   └── benchmark/
│       ├── benchmark_vtnk.py          # Benchmark constrained decoding
│       ├── benchmark_fused_sample.py  # Benchmark fused sampling kernel
│       └── benchmark_nn_quantize.py   # Benchmark nearest-neighbor quantization
└── utils.py            # parse_config() gin helper
```

## Key Types

| Type | Description |
|------|-------------|
| `TokenSequence` | Output of `encode()`; holds `.codes` tensor of shape `(N, num_levels)` |
| `QuantizerOutput` | Single-level quantizer output: `codes`, `quantized`, `residuals`, `commitment_loss` |
| `ResidualQuantizerOutput` | Multi-level output: `codes` `(B, L)`, `quantized` `(B, D)`, `level_outputs` |
| `GenerationConfig` | Beam search config: `steps`, `k`, `beam_size`, `temperature` |
| `CompactCSRTrie` | GPU-resident CSR trie encoding valid item token sequences |

## Testing

Run the full test suite with:

```bash
python -m pytest tests/
```

To run a specific test file:

```bash
python -m pytest tests/test_constrained_decoding.py
python -m pytest tests/test_trie.py
python -m pytest tests/test_kernel.py
python -m pytest tests/test_tokenizers.py
python -m pytest tests/test_hf_integration.py
```

## Performance

The `examples/scripts/benchmark/benchmark_vtnk.py` script benchmarks constrained decoding implementations across batch sizes (`B ∈ {256, 1024, 4096}`) and vocabulary sizes (`N ∈ {512, 1024, 8192, 150000}`). All results below were obtained with **1% sparsity** (i.e. the trie has at most `0.01 × N` valid next tokens at each node), with hidden dim K=512 fixed.

```bash
python -m examples.scripts.benchmark.benchmark_vtnk
```

### Fused Kernel Speedup Heatmaps

This section benchmarks `fused_linear_constrained_node_transition` — the kernel that fuses the linear projection and CSR trie constraint into a single GPU pass — against four baselines. Each heatmap reports the speedup ratio (values > 1 mean the fused kernel is faster) across batch sizes (B ∈ {256, 1024, 4096}) and vocabulary sizes (N ∈ {512, 1024, 8192, 150000}), with hidden dim K=512 and 1% sparsity.

**Summary of findings.** The fused kernel consistently outperforms all GPU baselines. Against the dense PyTorch baseline (`compiled_linear+vtnk_pytorch`) it is **4–9× faster** across all grid points, with peak advantage at B ∈ {256, 1024} (8.2–8.8×) and a reduced but still clear margin at B=4096 (4.3–6.6×). The margin narrows at N=150k (5.2× for B ∈ {256, 1024}, 4.5× for B=4096) but remains substantial throughout. Against sparse PyTorch the advantage is **10.9–12.4×** at N ≤ 1024 for B ∈ {256, 1024}, dropping to **4.9–7.4×** at B=4096; at N=8192 it narrows to **1.7–8.5×**, and at N=150k the fused kernel is **1.4–2.2×** ahead since sparse PyTorch already skips most of the matmul work at high sparsity.

Against the two-kernel approach (separate matmul + constraint pass) the advantage is strongly N-driven: at N=150k the fused kernel is **81–95× faster**, at N=8192 **7.4–8.3×**, and at N=1024 **1.5–3.5×**. At N=512 the fused kernel leads at all batch sizes: **1.1×** at B=256, rising to **1.9–3.1×** at larger batch sizes.

Against CPU trie traversal (benchmarked for B ∈ {256, 1024} and N ∈ {512, 1024, 8192} — the CPU baseline was too slow at larger settings) the speedups are massive: **175–578×** at N=512, **289–974×** at N=1024, and **1582–2267×** at N=8192. The dominant axis is batch size: the GPU processes B items in parallel while the CPU traverses them serially, driving the speedup proportionally with B.

**vs PyTorch (dense)** — `torch.compile(nn.Linear)` followed by `vtnk_pytorch`, which applies a validity mask to the logits in a separate GPU pass after the matmul.
![fused vs pytorch](out/heatmap_fused_vs_pytorch.jpg)

**vs Sparse PyTorch** — `torch.compile(sparse_linear_pytorch)`, which skips columns corresponding to invalid tokens during the matmul using a sparse weight representation, but remains within the PyTorch runtime.
![fused vs sparse pytorch](out/heatmap_fused_vs_sparse_pytorch.jpg)

**vs Custom Kernel** — `torch.compile(nn.Linear)` followed by `constrained_node_transition`, a standalone Triton kernel that applies the CSR trie mask to precomputed logits. The matmul and masking are still two separate kernel launches.
![fused vs kernel](out/heatmap_fused_vs_kernel.jpg)

**vs CPU Trie** — Pure Python traversal of an in-memory `Trie` on CPU, iterating over each batch item to collect valid next tokens. Included as the reference baseline for the constrained decoding problem.
![fused vs cpu trie](out/heatmap_fused_vs_trie_cpu.jpg)

### Fused Sample Kernel Speedup

The `examples/scripts/benchmark/benchmark_fused_sample.py` script benchmarks `fused_linear_constrained_node_transition_sampling` — a Triton kernel that fuses the linear projection, CSR trie masking, **and** multinomial sampling into a single GPU pass — against `torch.compile(sparse_linear_pytorch)` followed by a separate `torch.softmax` + `torch.multinomial` call. Grid and setup are the same as above (K=512, 1% sparsity).

```bash
python -m examples.scripts.benchmark.benchmark_fused_sample
```

**Summary of findings.** The fused sample kernel is faster than sparse PyTorch + multinomial at every point in the benchmark grid (B ∈ {256, 1024, 4096}, N ∈ {512, 1024, 8192, 150000}).

At N ≤ 1024 the fused kernel is **9.7–11.1× faster** at B ∈ {256, 1024} and **6.3–6.4× faster** at B=4096: fusing the sampling step eliminates a `softmax` pass and a separate `torch.multinomial` kernel launch whose overhead rivals the matmul cost at small N. At N=8192 the advantage ranges from **5.0–9.3×** across batch sizes. At N=150k the fused kernel remains consistently faster: **8.6–11.7×** at B ∈ {256, 1024} and **5.7×** at B=4096, because sparse PyTorch still requires the separate multinomial pass over ~1500 valid candidates regardless of batch size. `fused_linear_constrained_node_transition_sampling` is the recommended choice across all measured settings.

**vs Sparse PyTorch + multinomial** — `torch.compile(sparse_linear_pytorch)` (skips invalid-token columns during the matmul) followed by `torch.softmax` and `torch.multinomial`. This is the two-step baseline that separates projection from sampling.
![fused sample vs sparse pytorch](out/heatmap_fused_sample_vs_sparse_pytorch.jpg)

## Nearest-Neighbor Quantization Kernels

The core operation in residual quantization is finding the nearest codebook entry for each embedding vector — a `(B, D) × (N, D)` nearest-neighbor search returning `B` integer indices. RecTokens provides four implementations, benchmarked across batch sizes `B ∈ {32, …, 65536}`, embedding dims `D ∈ {64, 128, 256}`, and codebook sizes `N ∈ {64, 128, 256, 512}`.

### Summary of takeaways

- **`quantize_fwd_mm` is the default choice for N ≥ 128.** Its matrix-multiply tile structure maps well onto GPU SIMD for any batch size, and its lead widens as both N and D grow. It beats `cdist_compiled` at every point in the benchmark grid for N ≥ 128, and beats FAISS-GPU at every point regardless of N.
- **`quantize_fwd` wins only at N ≤ 64.** Its sequential-scan approach is efficient when the inner loop over N is short enough that tiling overhead is not yet amortized. At N=64 and small batch it is up to 1.6× faster than `quantize_fwd_mm`.
- **The crossover between `fwd` and `mm` is governed by N × D.** At small N the sequential scan wins; as N grows the scan's serial inner loop becomes the bottleneck and the MM tile kernel pulls ahead. Higher D accelerates this crossover.
- **`cdist_compiled` is a reasonable fallback** — simpler to deploy than Triton kernels, and its cost grows predictably with B×N. It is however consistently slower than `quantize_fwd_mm` and loses to `quantize_fwd` at small D.
- **FAISS-GPU (search only, pre-built index) is the slowest implementation across virtually all settings.** Its per-call dispatch overhead dominates at small N and large B. The only region where it is marginally competitive is large N (≥ 256) combined with large D (256) and very small batch (B ≤ 256) — a regime uncommon in production rec systems. The Triton kernels are otherwise 2–21× faster.
- **All implementations benefit from large B, but the Triton kernels benefit most.** Their speedup over FAISS and cdist grows monotonically with batch size because they achieve near-linear GPU utilization scaling while FAISS's fixed dispatch cost remains constant.

### Kernel descriptions

#### `quantize_fwd` — Triton sequential scan

Each Triton block handles one or more rows of `x` and iterates over the full codebook of size N in sequence, accumulating the minimum L2 distance in registers.

**Strengths:** Very low launch overhead; minimal shared memory pressure; optimal at N ≤ 64 where the inner loop is short; memory access pattern is sequential and cache-friendly.

**Weaknesses:** The inner loop over N is serial within each thread block, so kernel time scales linearly with N. At large N or large D the kernel stalls waiting for memory, and the MM-style kernel's parallelism over N tiles wins decisively. At N=512, D=256, `fwd` is over 3× slower than `mm`.

#### `quantize_fwd_mm` — Triton MM-style tiled kernel

Reformulates the L2 nearest-neighbor problem as a matrix multiplication over B×N tiles. Multiple thread blocks cooperate over the N dimension in parallel.

**Strengths:** Strong GPU utilization at any (B, N, D) combination; tiling amortizes launch overhead effectively; consistently fastest at N ≥ 128; no failure modes — it always beats FAISS.

**Weaknesses:** Higher launch overhead and shared memory pressure than the sequential kernel; slight disadvantage at N ≤ 64 + small batch where the tile setup cost is not yet amortized.

#### `cdist_compiled` — `torch.compile(torch.cdist + argmin)`

Computes the full pairwise distance matrix `(B, N)` using `torch.cdist`, then takes `argmin` along N. Compiled with `torch.compile` for fused kernel dispatch.

**Strengths:** Zero external dependencies beyond PyTorch; leverages cuBLAS for the pairwise distance core; D-insensitive performance because the matmul is fully blocked.

**Weaknesses:** Allocates an intermediate `(B, N)` distance buffer that grows with both B and N; two-pass execution (matmul then argmin) prevents full fusion. Consistently slower than both Triton kernels at large B, and loses to `quantize_fwd` at small D.

#### `faiss_search` — FAISS-GPU flat L2 (pre-built index)

Uses a FAISS `IndexFlatL2` GPU index built once from the codebook (`make_gpu_index`) and queried at inference time with `index.search(x, 1)`.

**Strengths:** Battle-tested implementation; plugs into the broader FAISS ecosystem for approximate search extensions; cuBLAS-backed distance computation is highly optimized for large D.

**Weaknesses:** Large per-call dispatch overhead even on GPU, not amortized at small N or large B. The Triton kernels are 2–21× faster across the benchmark grid. Only marginally competitive at N ≥ 256, D=256, and B ≤ 256 — a narrow corner of the operating space.

### Benchmark setup

```bash
python -m examples.scripts.benchmark.benchmark_nn_quantize
```

Grid: `B ∈ {32, 256, 1024, 4096, 16384, 32768, 65536}`, `D ∈ {64, 128, 256}`, `N ∈ {64, 128, 256, 512}`. Heatmap axes are batch size (B, rows) vs embedding dim (D, columns). Speedup values > 1 mean the left-hand kernel is faster. All FAISS timings exclude index build time (static codebook).

### Heatmaps

#### `quantize_fwd` vs `quantize_fwd_mm`

**N=64 — `fwd` leads at small batch and low D, `mm` takes over at large D or large B**

![fwd vs mm N=64](out/heatmap_fwd_vs_mm_N64.jpg)

At N=64 the sequential scan is fast enough to outrun the MM tile kernel across most of the grid. `quantize_fwd` is up to 1.62× faster at (B=32, D=64), where the tile setup overhead of `mm` is not yet amortized and the codebook is small enough that the serial inner loop completes quickly. The advantage decays along two axes: increasing D multiplies the per-row scan cost, and increasing B eventually saturates the GPU such that the parallel tile structure of `mm` becomes the more efficient fit. At D=256 the advantage collapses entirely: `fwd` falls to 0.54–0.84× of `mm` for large B. The crossover diagonal runs from (small B, high D) to (large B, low D). For N=64 with D ≤ 128 and B ≤ 4096, `fwd` is the faster choice.

**N=256 — `mm` dominates; `fwd` never recovers**

![fwd vs mm N=256](out/heatmap_fwd_vs_mm_N256.jpg)

With N=256 the sequential inner loop in `quantize_fwd` is 4× longer than at N=64, and the MM tile kernel's parallel decomposition over N fully materializes. `fwd` is slower at nearly every cell, ranging from 0.45× (B=32, D=256 — more than 2× slower) to a single near-parity cell at (B=65536, D=128: 1.03×). The D=256 column is universally dark: 0.45–0.61× regardless of batch size. The gradient along the D axis is the dominant signal — wider embeddings amplify the per-row scan cost linearly while the MM kernel's tile width absorbs the extra work with no additional penalty. At N=256 and above, `quantize_fwd_mm` is unambiguously the correct choice.

#### `quantize_fwd` vs `cdist_compiled`

**N=256 — `fwd` wins at small D with growing margin, but loses at large D + small batch**

![fwd vs cdist N=256](out/heatmap_fwd_vs_cdist_N256.jpg)

At D=64, `quantize_fwd` beats `cdist_compiled` at every batch size (0.98–3.08×), and the speedup grows monotonically with B — as the batch size grows the Triton kernel's per-row parallelism scales efficiently while `cdist`'s two-pass (matmul + argmin) dispatch cost amortizes more slowly. At D=128 the advantage shrinks to 0.72–1.94× and is only reliable for B ≥ 1024. At D=256, `fwd` loses at small batch (0.47–0.75× for B ≤ 256) and barely reaches parity at the largest batch sizes (B=65536, 0.97×). This failure at high D reflects the sequential kernel's sensitivity to embedding width: `cdist` delegates to a cuBLAS batched matmul that remains equally efficient regardless of D, while `fwd`'s inner loop cost grows with D. For N=256 and D=256, `cdist_compiled` is actually preferable to `quantize_fwd` at small batch.

#### `quantize_fwd_mm` vs `cdist_compiled`

**N=256 — `mm` beats cdist at every point; speedup driven by B and D**

![mm vs cdist N=256](out/heatmap_mm_vs_cdist_N256.jpg)

Unlike `quantize_fwd`, the MM kernel beats `cdist_compiled` at every cell in this grid without exception. The minimum speedup is 1.03× (B=32, D=256) and the maximum is 3.72× (B=65536, D=64). The speedup grows primarily along the B axis — as batch size increases the tile structure of `mm` better saturates the GPU while `cdist`'s two-kernel dispatch overhead becomes the dominant cost. The secondary gradient runs along D inversely: higher D slightly reduces `mm`'s relative advantage because `cdist`'s cuBLAS core is particularly well-optimized for wide embeddings. Even so, the D=256 column reaches 1.61× at B=65536, a meaningful and reliable margin. This heatmap confirms `quantize_fwd_mm` as the dominant general-purpose kernel for N ≥ 128.

#### `quantize_fwd` vs `faiss_search`

**N=64 — large and growing advantage for `fwd` across all batch sizes**

![fwd vs faiss N=64](out/heatmap_fwd_vs_faiss_N64.jpg)

At N=64, FAISS-GPU's fixed per-call dispatch overhead dwarfs the actual distance computation, making the Triton kernel overwhelmingly faster. `quantize_fwd` is 2.34–20.85× faster across the grid. The speedup pattern is strikingly regular: it is nearly uniform across D at small batch (B=32: 2.34–4.05×, increasing with D), and then fans out rapidly as B grows — at B=65536 the margin reaches 4.51–20.85×. This proportional growth with B is the direct signature of FAISS's fixed overhead: the Triton kernel's useful work scales with B while FAISS's dispatch cost stays constant, driving the ratio linearly. Even at the smallest batch size tested (B=32), `fwd` is 2–4× faster, meaning there is no operating point at N=64 where FAISS is competitive.

**N=256 — `fwd` loses at large D + small batch; FAISS's cuBLAS core becomes visible**

![fwd vs faiss N=256](out/heatmap_fwd_vs_faiss_N256.jpg)

At N=256 the sequential scan's increasing cost at large D begins to rival FAISS's fixed overhead at small batch. The D=256 column dips below 1× for B ≤ 256 (0.73× at B=32, 0.80× at B=256): the serial inner loop over 256 codebook entries × 256 dimensions is slow enough that FAISS's cuBLAS-backed distance kernel, despite its overhead, matches or slightly beats it. At D=64 the advantage remains strong and B-driven (1.76–9.17×) — FAISS's overhead is not amortized here. The practical takeaway: if you are using N=256, D=256, and very small batch sizes (B ≤ 256), FAISS is a viable alternative to `quantize_fwd` specifically; for all other settings `fwd` wins.

#### `quantize_fwd_mm` vs `faiss_search`

**N=64 — `mm` wins everywhere, uniform strength across D**

![mm vs faiss N=64](out/heatmap_mm_vs_faiss_N64.jpg)

At N=64, `quantize_fwd_mm` beats FAISS-GPU at every cell: 2.24–20.28×. The overall pattern mirrors the `fwd` vs FAISS heatmap — B-driven growth, large margins at large batch — but with one key difference: the D=256 column is competitive (2.24–8.37×) rather than being the weakest column. Where `fwd` has elevated cost at large D (sequential scan), `mm`'s tiling keeps cost controlled, so the D=256 minimum (B=32, 2.24×) is only marginally lower than D=64 (B=32, 2.50×). `mm` provides a more uniform advantage profile than `fwd` against FAISS because it does not have the sequential scan's D-scaling weakness.

**N=256 — `mm` wins at every cell; no failure mode at large D + small batch**

![mm vs faiss N=256](out/heatmap_mm_vs_faiss_N256.jpg)

This heatmap is the clearest illustration of `quantize_fwd_mm`'s robustness. While `quantize_fwd` fell below FAISS at N=256, D=256, small B, `mm` maintains a positive margin everywhere: 1.60–11.06×. The D=256 column ranges from 1.60× (B=32) to 3.19× (B=65536) — a clear win even in the exact regime where `fwd` lost. The D=64 column reaches 11.06× at B=65536 driven by B-scaling. The bottom-right region (large B, any D) is brightest, consistent with FAISS's fixed overhead being overwhelmed by the volume of useful work. This heatmap, paired with the N=256 fwd vs faiss heatmap, is the strongest argument for preferring `mm` over `fwd` as the default kernel for standard rec-system codebook sizes.

## License

Apache 2.0
