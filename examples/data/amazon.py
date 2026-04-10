import gzip
import json
import numpy as np
import os
import os.path as osp
import pandas as pd
import polars as pl
import torch

from collections import defaultdict
from examples.data.preprocessing import PreprocessingMixin
from torch.utils.data import Dataset
from torch_geometric.data import download_google_url
from torch_geometric.data import extract_zip
from torch_geometric.data import HeteroData
from torch_geometric.data import InMemoryDataset
from torch_geometric.io import fs
from typing import Callable
from typing import List
from typing import NamedTuple
from typing import Optional


class SeqBatch(NamedTuple):
    user_ids: torch.Tensor
    ids: torch.Tensor
    ids_fut: torch.Tensor
    x: torch.Tensor
    x_fut: torch.Tensor
    seq_mask: torch.Tensor


def parse(path):
    g = gzip.open(path, "r")
    for l in g:
        yield eval(l)


class AmazonReviews(InMemoryDataset, PreprocessingMixin):
    gdrive_id = "1qGxgmx7G_WB7JE4Cn_bEcZ_o_NAJLE3G"
    gdrive_filename = "P5_data.zip"

    def __init__(
        self,
        root: str,
        split: str,  # 'beauty', 'sports', 'toys'
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        force_reload: bool = False,
    ) -> None:
        self.split = split
        super(AmazonReviews, self).__init__(
            root, transform, pre_transform, force_reload
        )
        self.load(self.processed_paths[0], data_cls=HeteroData)

    @property
    def raw_file_names(self) -> List[str]:
        return [self.split]

    @property
    def processed_file_names(self) -> str:
        return f"data_{self.split}.pt"

    def download(self) -> None:
        path = download_google_url(self.gdrive_id, self.root, self.gdrive_filename)
        extract_zip(path, self.root)
        os.remove(path)
        folder = osp.join(self.root, "data")
        fs.rm(self.raw_dir)
        os.rename(folder, self.raw_dir)

    def _remap_ids(self, x):
        return x - 1

    def train_test_split(self, max_seq_len=20):
        splits = ["train", "eval", "test"]
        sequences = {sp: defaultdict(list) for sp in splits}
        user_ids = []
        with open(
            os.path.join(self.raw_dir, self.split, "sequential_data.txt"), "r"
        ) as f:
            for line in f:
                parsed_line = list(map(int, line.strip().split()))
                user_ids.append(parsed_line[0])
                items = [self._remap_ids(id) for id in parsed_line[1:]]

                # We keep the whole sequence without padding. Allows flexible training-time subsampling.
                train_items = items[:-2]
                sequences["train"]["itemId"].append(train_items)
                sequences["train"]["itemId_fut"].append(items[-2])

                eval_items = items[-(max_seq_len + 1) : -1]
                sequences["eval"]["itemId"].append(
                    eval_items + [-1] * (max_seq_len - len(eval_items))
                )
                sequences["eval"]["itemId_fut"].append(items[-1])

                test_items = items[-(max_seq_len + 1) : -1]
                sequences["test"]["itemId"].append(
                    test_items + [-1] * (max_seq_len - len(test_items))
                )
                sequences["test"]["itemId_fut"].append(items[-1])

        for sp in splits:
            sequences[sp]["userId"] = user_ids
            sequences[sp] = pl.from_dict(sequences[sp])
        return sequences

    def process(self, max_seq_len=20) -> None:
        data = HeteroData()

        with open(os.path.join(self.raw_dir, self.split, "datamaps.json"), "r") as f:
            data_maps = json.load(f)

        # Construct user sequences
        sequences = self.train_test_split(max_seq_len=max_seq_len)
        data["user", "rated", "item"].history = {
            k: self._df_to_tensor_dict(v, ["itemId"]) for k, v in sequences.items()
        }

        # Compute item features
        asin2id = pd.DataFrame(
            [
                {"asin": k, "id": self._remap_ids(int(v))}
                for k, v in data_maps["item2id"].items()
            ]
        )
        item_data = (
            pd.DataFrame(
                [
                    meta
                    for meta in parse(
                        path=os.path.join(self.raw_dir, self.split, "meta.json.gz")
                    )
                ]
            )
            .merge(asin2id, on="asin")
            .sort_values(by="id")
            .fillna({"brand": "Unknown", "description": ""})
        )

        sentences = item_data.apply(
            lambda row: (
                str(row["title"])
                + " by "
                + str(row["brand"])
                + " is categorized under "
                + ", ".join(row["categories"][0])
                + " and is priced at "
                + str(row["price"])
                + "."
            ),
            axis=1,
        )

        item_emb = self._encode_text_feature(sentences)
        data["item"].x = item_emb
        data["item"].text = np.array(sentences)

        gen = torch.Generator()
        gen.manual_seed(42)
        data["item"].is_train = torch.rand(item_emb.shape[0], generator=gen) > 0.05

        self.save([data], self.processed_paths[0])


class PrecomputedSequenceDataset(Dataset):
    """Dataset that loads precomputed chat-formatted training samples from disk.

    Each item is a dict ``{"input_ids": tensor, "labels": tensor}`` produced by
    ``scripts/preprocessing/precompute_sequences.py``.  Labels are already
    masked to ``-100`` on the user turn; loss is computed only on the assistant
    turn.  Loading these avoids any neural-network inference during training.

    Args:
        path: Path to a ``.pt`` file saved by the preprocessing script.
    """

    def __init__(self, path: str) -> None:
        data = torch.load(path, weights_only=False)
        self.samples: list[dict] = data["samples"]
        self.original_vocab_size: int = data["original_vocab_size"]
        self.num_levels: int = data["num_levels"]
        self.codebook_size: int = data["codebook_size"]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        return self.samples[idx]  # {"input_ids": tensor, "labels": tensor}


class ItemData(Dataset):
    def __init__(
        self,
        root: str,
        *args,
        force_process: bool = False,
        train_test_split: str = "all",
        **kwargs,
    ) -> None:

        raw_data = AmazonReviews(root=root, *args, **kwargs)

        processed_data_path = raw_data.processed_paths[0]
        if not os.path.exists(processed_data_path) or force_process:
            raw_data.process(max_seq_len=20)

        if train_test_split == "train":
            filt = raw_data.data["item"]["is_train"]
        elif train_test_split == "eval":
            filt = ~raw_data.data["item"]["is_train"]
        elif train_test_split == "all":
            filt = torch.ones_like(raw_data.data["item"]["x"][:, 0], dtype=bool)

        self.item_data, self.item_text = (
            raw_data.data["item"]["x"][filt],
            raw_data.data["item"]["text"][filt],
        )

    def __len__(self):
        return self.item_data.shape[0]

    def __getitem__(self, idx):
        item_ids = (
            torch.tensor(idx).unsqueeze(0) if not isinstance(idx, torch.Tensor) else idx
        )
        x = self.item_data[idx]
        return SeqBatch(
            user_ids=-1 * torch.ones_like(item_ids.squeeze(0)),
            ids=item_ids,
            ids_fut=-1 * torch.ones_like(item_ids.squeeze(0)),
            x=x,
            x_fut=-1 * torch.ones_like(item_ids.squeeze(0)),
            seq_mask=torch.ones_like(item_ids, dtype=bool),
        )
