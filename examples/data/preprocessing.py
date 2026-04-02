import numpy as np
from tqdm import tqdm
import pandas as pd
import polars as pl
import torch
import torch.nn.functional as F
from einops import rearrange
from sentence_transformers import SentenceTransformer
from transformers import AutoModel, AutoTokenizer
from typing import List

FUT_SUFFIX = "_fut"

QWEN3_EMBEDDING_MODEL_ID = "Qwen/Qwen3-Embedding-0.6B"
SENTENCE_T5_MODEL_ID = "sentence-transformers/sentence-t5-large"


def _last_token_pool(last_hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    left_padding = attention_mask[:, -1].sum() == attention_mask.shape[0]
    if left_padding:
        return last_hidden_states[:, -1]
    sequence_lengths = attention_mask.sum(dim=1) - 1
    batch_size = last_hidden_states.shape[0]
    return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]


def _encode_with_qwen3(text_feat, model_id=QWEN3_EMBEDDING_MODEL_ID, batch_size=2, max_length=8192):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_id, padding_side="left")
    model = AutoModel.from_pretrained(model_id).to(device).eval()

    all_embeddings = []
    sentences = list(text_feat)
    for start in tqdm(range(0, len(sentences), batch_size), desc="Encoding with Qwen3"):
        batch = sentences[start : start + batch_size]
        encoded = tokenizer(
            batch, padding=True, truncation=True, max_length=max_length, return_tensors="pt"
        ).to(device)
        with torch.no_grad():
            outputs = model(**encoded)
        pooled = _last_token_pool(outputs.last_hidden_state, encoded["attention_mask"])
        pooled = F.normalize(pooled, p=2, dim=-1)
        all_embeddings.append(pooled.cpu())

    return torch.cat(all_embeddings, dim=0)


def _encode_with_sentence_t5(text_feat, model_id=SENTENCE_T5_MODEL_ID, batch_size=2):
    model = SentenceTransformer(model_id)
    return model.encode(
        batch_size=batch_size,
        sentences=text_feat,
        show_progress_bar=True,
        convert_to_tensor=True,
    ).cpu()


class PreprocessingMixin:
    @staticmethod
    def _encode_text_feature(text_feat, encoder: str = "qwen3", batch_size: int = 16):
        """Encode text features into dense embeddings.

        Args:
            text_feat: Iterable of strings to encode.
            encoder: ``"qwen3"`` (default) uses Qwen/Qwen3-0.6B with mean pooling;
                     ``"sentence-t5"`` uses sentence-transformers/sentence-t5-large.
            batch_size: Batch size for encoding (only used by the qwen3 encoder).
        """
        if encoder == "sentence-t5":
            return _encode_with_sentence_t5(text_feat)
        return _encode_with_qwen3(text_feat, batch_size=batch_size)

    @staticmethod
    def _ordered_train_test_split(df, on, train_split=0.8):
        threshold = df.select(pl.quantile(on, train_split)).item()
        return df.with_columns(is_train=pl.col(on) <= threshold)

    @staticmethod
    def _df_to_tensor_dict(df, features):
        out = {
            feat: torch.from_numpy(
                rearrange(df.select(feat).to_numpy().squeeze().tolist(), "b d -> b d")
            )
            if df.select(
                pl.col(feat).list.len().max() == pl.col(feat).list.len().min()
            ).item()
            else df.get_column("itemId").to_list()
            for feat in features
        }
        fut_out = {
            feat + FUT_SUFFIX: torch.from_numpy(df.select(feat + FUT_SUFFIX).to_numpy())
            for feat in features
        }
        out.update(fut_out)
        out["userId"] = torch.from_numpy(df.select("userId").to_numpy())
        return out
