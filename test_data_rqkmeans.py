# ---------------------------------------------------------------------------
# RQKMeans training loop
# ---------------------------------------------------------------------------

import torch
import torch.nn.functional as F
from torch.utils.data import BatchSampler
from torch.utils.data import DataLoader
from torch.utils.data import RandomSampler
from examples.amazon import ItemData

from rectokens.tokenizers.rq_kmeans import RQKMeansTokenizer


def train_rqkmeans(
    dataset: ItemData,
    *,
    num_levels: int = 3,
    codebook_size: int = 256,
    num_epochs: int = 20,
    batch_size: int = 640,
    log_every: int = 1,
) -> RQKMeansTokenizer:
    """Train an RQKMeansTokenizer on an ItemData dataset.

    Each batch is drawn from ``dataset`` via a PyTorch DataLoader.
    ``fit_step`` is called on each batch to update the K-means centroids
    via a mini-batch running-average update.

    After each epoch, reconstruction MSE and p_unique_ids are logged.
    There is no optimizer — K-means updates are closed-form.

    Args:
        dataset: An :class:`~examples.data.amazon.ItemData` instance.
        num_levels: Number of RQ levels.
        codebook_size: Codes per level.
        num_epochs: Number of training epochs.
        batch_size: Mini-batch size for the DataLoader.
        log_every: Print a log line every this many epochs.

    Returns:
        Fitted :class:`~rectokens.RQKMeansTokenizer`.
    """
    input_dim = dataset[0].x.shape[0]
    model = RQKMeansTokenizer(
        num_levels=num_levels,
        codebook_size=codebook_size,
        dim=input_dim,
    )

    train_sampler = BatchSampler(RandomSampler(dataset), batch_size, False)
    loader = DataLoader(
        dataset, sampler=train_sampler, batch_size=None, collate_fn=lambda batch: batch
    )

    for epoch in range(1, num_epochs + 1):
        total_recon = 0.0
        total_unique = 0.0

        for batch in loader:
            x = batch.x.float()

            # Mini-batch K-means update (no gradients)
            model.fit_step(x)

            # Evaluate reconstruction and uniqueness on this batch
            with torch.no_grad():
                tokens = model.encode(x)
                recon = model.decode(tokens)
                total_recon += F.mse_loss(recon, x).item()

                codes = tokens.codes  # (B, num_levels)
                eq = (codes.unsqueeze(0) == codes.unsqueeze(1)).all(dim=-1)
                p_unique = (~torch.triu(eq, diagonal=1)).all(dim=1).float().mean()
                total_unique += p_unique.item()

        if epoch % log_every == 0:
            n = len(loader)
            print(
                f"epoch {epoch:3d}/{num_epochs}"
                f"  recon={total_recon / n:.4f}"
                f"  p_unique={total_unique / n:.3f}"
            )

    return model


if __name__ == "__main__":
    torch.manual_seed(0)

    dataset = ItemData(root="data/amazon", split="beauty", train_test_split="train")
    print(f"Dataset: {len(dataset)} items, dim={dataset[0].x.shape[0]}")

    model = train_rqkmeans(
        dataset,
        num_levels=3,
        codebook_size=256,
        num_epochs=20,
        batch_size=640,
        log_every=1,
    )

    # Sanity check
    tokens = model.encode(dataset.item_data[:4].float())
    recon = model.decode(tokens)
    print(f"\nEncoded shape : {tokens.codes.shape}")
    print(f"Decoded shape : {recon.shape}")
    print(f"Token tuples  : {tokens.to_tuple_ids()}")
