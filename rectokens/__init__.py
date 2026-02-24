"""RecTokens — a tokenizer library for sequential recommendation systems.

Quickstart (RQ-Kmeans)::

    import numpy as np
    import torch
    from rectokens import RQKMeansTokenizer, NumpyDataset

    data = np.random.randn(10_000, 128).astype("float32")
    tok = RQKMeansTokenizer(num_levels=3, codebook_size=256, dim=128)
    tok.fit(NumpyDataset(data))

    features = torch.randn(8, 128)
    tokens = tok.encode(features)   # TokenSequence  codes: (8, 3)
    recon  = tok.decode(tokens)     # Tensor (8, 128)
    print(tokens.to_tuple_ids())    # [(c0, c1, c2), ...]
"""

# Core abstractions
from rectokens.core.codebook import Codebook, SearchResult
from rectokens.core.dataset import ItemDataset
from rectokens.core.quantizer import Quantizer, QuantizerOutput, ResidualQuantizerOutput
from rectokens.core.tokenizer import TokenSequence, Tokenizer

# Concrete codebooks
from rectokens.codebooks.euclidean import EuclideanCodebook

# Concrete quantizers
from rectokens.quantizers.kmeans import KMeansQuantizer
from rectokens.quantizers.residual import ResidualQuantizer

# Concrete tokenizers
from rectokens.tokenizers.rq_kmeans import RQKMeansTokenizer
from rectokens.tokenizers.rqvae import RQVAETokenizer, VQQuantizer

# Dataset adapters
from rectokens.datasets import NumpyDataset, TensorDataset

# Registry
from rectokens.registry import TokenizerRegistry

# Register built-in tokenizers so they're available by name out-of-the-box
TokenizerRegistry.register("rq_kmeans")(RQKMeansTokenizer)
TokenizerRegistry.register("rqvae")(RQVAETokenizer)

__all__ = [
    # Core
    "Codebook",
    "SearchResult",
    "ItemDataset",
    "Quantizer",
    "QuantizerOutput",
    "ResidualQuantizerOutput",
    "TokenSequence",
    "Tokenizer",
    # Codebooks
    "EuclideanCodebook",
    # Quantizers
    "KMeansQuantizer",
    "ResidualQuantizer",
    "VQQuantizer",
    # Tokenizers
    "RQKMeansTokenizer",
    "RQVAETokenizer",
    # Datasets
    "NumpyDataset",
    "TensorDataset",
    # Registry
    "TokenizerRegistry",
]
