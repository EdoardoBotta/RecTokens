# ---------------------------------------------------------------------------
# RQVAE training loop
# ---------------------------------------------------------------------------

import torch
import torch.nn.functional as F
from torch.utils.data import BatchSampler
from torch.utils.data import DataLoader
from torch.utils.data import RandomSampler
from torch.optim import AdamW
from examples.data.amazon import ItemData

from rectokens.tokenizers.rqvae import RQVAETokenizer


def get_device() -> torch.device:
    """Return the best available device: MPS (Apple Silicon) > CUDA > CPU."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def train_rqvae(
    dataset: ItemData,
    *,
    latent_dim: int = 64,
    hidden_dim: int = 256,
    num_levels: int = 3,
    codebook_size: int = 256,
    num_epochs: int = 20,
    batch_size: int = 32,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    log_every: int = 1,
) -> RQVAETokenizer:
    """Train an RQVAETokenizer on an ItemData dataset.

    Each batch is drawn from ``dataset`` via a PyTorch DataLoader.
    ``ItemData.__getitem__`` returns a ``SeqBatch`` namedtuple; the default
    collate stacks each field, so ``batch.x`` has shape ``(B, D)``.

    Loss = MSE reconstruction + VQ commitment loss (summed across levels).
    Codebook entries are updated via EMA inside the forward pass, so AdamW
    only optimises the encoder and decoder MLP weights.

    Args:
        dataset: An :class:`~examples.data.amazon.ItemData` instance.
        latent_dim: Encoder output / codebook embedding dimension.
        hidden_dim: MLP hidden layer width.
        num_levels: Number of RQ levels.
        codebook_size: Codes per level.
        num_epochs: Training epochs.
        batch_size: Mini-batch size for the DataLoader.
        lr: AdamW learning rate.
        weight_decay: AdamW weight decay.
        log_every: Print a log line every this many epochs.

    Returns:
        Fitted :class:`~rectokens.RQVAETokenizer`.
    """
    device = get_device()
    print(f"Training on: {device}")

    input_dim = dataset[0].x.shape[0]
    model = RQVAETokenizer(
        input_dim=input_dim,
        latent_dim=latent_dim,
        hidden_dim=hidden_dim,
        num_levels=num_levels,
        codebook_size=codebook_size,
        learnable_codebook=False,
    ).to(device)

    # AdamW only sees encoder + decoder weights — codebook embeddings are
    # registered as non-trainable buffers and updated via EMA in forward().
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    train_sampler = BatchSampler(RandomSampler(dataset), batch_size, False)
    loader = DataLoader(dataset, sampler=train_sampler, batch_size=None, collate_fn=lambda batch: batch)
    

    model.train()
    for epoch in range(1, num_epochs + 1):
        total_recon = 0.0
        total_commit = 0.0
        total_unique = 0.0

        for batch in loader:
            x = batch.x.float().to(device)
            optimizer.zero_grad()

            out = model(x)
            recon_loss = F.mse_loss(out["recon"], x)
            commit_loss = out["commitment_loss"]
            (recon_loss + commit_loss).backward()
            optimizer.step()

            total_recon += recon_loss.item()
            total_commit += commit_loss.item()
            total_unique += out["p_unique_ids"].item()

        if epoch % log_every == 0:
            n = len(loader)
            print(
                f"epoch {epoch:3d}/{num_epochs}"
                f"  recon={total_recon / n:.4f}"
                f"  commit={total_commit / n:.4f}"
                f"  total={(total_recon + total_commit) / n:.4f}"
                f"  p_unique={total_unique / n:.3f}"
            )

    model._fitted = True
    return model


if __name__ == "__main__":
    torch.manual_seed(0)

    dataset = ItemData(root="data/amazon", split="beauty", train_test_split="train")
    print(f"Dataset: {len(dataset)} items, dim={dataset[0].x.shape[0]}")

    model = train_rqvae(
        dataset,
        latent_dim=64,
        hidden_dim=512,
        num_levels=3,
        codebook_size=256,
        num_epochs=100,
        batch_size=640,
        lr=1e-3,
    )

    # Sanity check
    model.eval()
    device = next(model.parameters()).device
    tokens = model.encode(dataset.item_data[:4].to(device))
    recon = model.decode(tokens)
    print(f"\nEncoded shape : {tokens.codes.shape}")
    print(f"Decoded shape : {recon.shape}")
    print(f"Token tuples  : {tokens.to_tuple_ids()}")
