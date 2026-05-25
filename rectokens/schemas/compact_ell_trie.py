from __future__ import annotations

from typing import NamedTuple
from typing import TYPE_CHECKING

import torch
from torch import Tensor

if TYPE_CHECKING:
    from rectokens.schemas.compact_csr_trie import CompactCSRTrie


class CompactELLTrie(NamedTuple):
    """A trie encoded in ELLPACK (ELL) format for GPU-accelerated constrained decoding.

    Compared to CompactCSRTrie, ELL eliminates the dependent row-pointer load by storing
    adjacency data in a padded (num_nodes, 2, max_branches) matrix. Branch k of node i
    sits at ell_cols_vals[i, :, k], so its address is pure arithmetic from cur_node — no
    prior row_ptrs load is required. This shortens the dependent load chain from depth 3
    to depth 2 in Triton kernels.

    Layout (num_nodes, 2, max_branches):
      ell_cols_vals[node, 0, branch] = col (token index), -1 = padding
      ell_cols_vals[node, 1, branch] = val (child node id), -1 = padding

    Cols and vals for the same node are max_branches apart in memory (stride(1) = max_branches),
    keeping both in the same cache region rather than being separated by node_count*max_branches.

    max_branches = max(layer_max_branches). The dense-phase fields are identical to CompactCSRTrie.
    """

    ell_cols_vals: Tensor       # (num_nodes, 2, max_branches): [cols, vals], -1 = padding
    n_children: Tensor          # (num_nodes,): child count per node — used as a block-level gate
    layer_max_branches: list[int]
    dense_mask_by_layer: list[Tensor]
    dense_states: Tensor
    vocab_size: int

    @classmethod
    def from_csr(cls, csr: CompactCSRTrie) -> CompactELLTrie:
        row_ptrs = csr.row_ptrs
        device = row_ptrs.device
        num_nodes = len(row_ptrs)
        nnz = csr.stacked_cols_vals.shape[1] - 1  # exclude sentinel
        max_branches = max(csr.layer_max_branches) if csr.layer_max_branches else 0

        row_ptrs_ext = torch.cat(
            [row_ptrs, torch.tensor([nnz], dtype=row_ptrs.dtype, device=device)]
        )
        n_children = row_ptrs_ext.diff()  # (num_nodes,)

        ell = torch.full(
            (num_nodes, 2, max_branches), -1, dtype=torch.long, device=device
        )

        if nnz > 0:
            edge_node_id = torch.repeat_interleave(
                torch.arange(num_nodes, device=device), n_children
            )  # (nnz,)
            edge_local_idx = (
                torch.arange(nnz, device=device) - row_ptrs[edge_node_id]
            )  # (nnz,)

            ell[edge_node_id, 0, edge_local_idx] = csr.stacked_cols_vals[0, :nnz]
            ell[edge_node_id, 1, edge_local_idx] = csr.stacked_cols_vals[1, :nnz]

        return cls(
            ell_cols_vals=ell,
            n_children=n_children,
            layer_max_branches=csr.layer_max_branches,
            dense_mask_by_layer=csr.dense_mask_by_layer,
            dense_states=csr.dense_states,
            vocab_size=csr.vocab_size,
        )
