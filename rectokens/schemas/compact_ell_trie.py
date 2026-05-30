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

    def update(self, new_seqs: list[list[int]]) -> "CompactELLTrie":
        """Add new sequences incrementally without full reconstruction.

        Traverses the existing trie for each new sequence, inserting only the
        nodes and edges that do not already exist.  Cost is O(M * L * B) where
        M = len(new_seqs), L = sequence length, B = avg children per node —
        versus O(N * L) for a CSR full rebuild over N total sequences.

        Note: layer_max_branches, dense_mask_by_layer, and dense_states are
        preserved as-is and will be stale for nodes added by this call.
        """
        if not new_seqs:
            return self

        ell_cv = self.ell_cols_vals.clone()  # (num_nodes, 2, max_branches)
        n_ch = self.n_children.clone()
        max_branches = ell_cv.shape[2]
        num_nodes = ell_cv.shape[0]

        # Pre-allocate capacity for worst-case new nodes (one per token in new_seqs)
        capacity = sum(len(s) for s in new_seqs)
        ell_cv = torch.cat([
            ell_cv,
            torch.full((capacity, 2, max_branches), -1, dtype=torch.long),
        ])
        n_ch = torch.cat([n_ch, torch.zeros(capacity, dtype=torch.long)])

        for seq in new_seqs:
            node = 0
            for token in seq:
                nc = int(n_ch[node])
                child = -1
                if nc > 0:
                    hit = (ell_cv[node, 0, :nc] == token).nonzero(as_tuple=False)
                    if len(hit):
                        child = int(ell_cv[node, 1, hit[0, 0]])

                if child == -1:
                    if nc == max_branches:
                        new_mb = max_branches * 2
                        expanded = torch.full(
                            (ell_cv.shape[0], 2, new_mb), -1,
                            dtype=torch.long, device=ell_cv.device,
                        )
                        expanded[:, :, :max_branches] = ell_cv
                        ell_cv = expanded
                        max_branches = new_mb
                    child = num_nodes
                    num_nodes += 1
                    ell_cv[node, 0, nc] = token
                    ell_cv[node, 1, nc] = child
                    n_ch[node] += 1

                node = child

        return CompactELLTrie(
            ell_cols_vals=ell_cv[:num_nodes].contiguous(),
            n_children=n_ch[:num_nodes].contiguous(),
            layer_max_branches=self.layer_max_branches,
            dense_mask_by_layer=self.dense_mask_by_layer,
            dense_states=self.dense_states,
            vocab_size=self.vocab_size,
        )


class MutableELLTrie:
    """ELL trie with pre-allocated capacity for zero-copy incremental updates.

    Unlike CompactELLTrie.update(), which clones the full tensor on every call,
    MutableELLTrie pre-allocates a fixed capacity buffer and writes new nodes
    directly into it.  Each update() call is purely O(M * L * B) — proportional
    to the new sequences, not the total trie size.

    Use this when you need repeated incremental updates without rebuilding.
    """

    def __init__(
        self,
        ell: CompactELLTrie,
        extra_capacity: int,
    ) -> None:
        num_nodes = ell.ell_cols_vals.shape[0]
        max_branches = ell.ell_cols_vals.shape[2]
        device = ell.ell_cols_vals.device

        # Allocate buffer large enough for existing nodes + future inserts.
        total = num_nodes + extra_capacity
        self.ell_cv = torch.full((total, 2, max_branches), -1, dtype=torch.long, device=device)
        self.ell_cv[:num_nodes] = ell.ell_cols_vals

        self.n_ch = torch.zeros(total, dtype=torch.long, device=device)
        self.n_ch[:num_nodes] = ell.n_children

        self.num_nodes = num_nodes
        self.max_branches = max_branches
        self.vocab_size = ell.vocab_size

    def update(self, new_seqs: list[list[int]]) -> None:
        """Insert new sequences in-place.  No allocation is performed unless
        a node needs more children than max_branches (rare; triggers realloc)."""
        for seq in new_seqs:
            node = 0
            for token in seq:
                nc = int(self.n_ch[node])
                child = -1
                if nc > 0:
                    hit = (self.ell_cv[node, 0, :nc] == token).nonzero(as_tuple=False)
                    if len(hit):
                        child = int(self.ell_cv[node, 1, hit[0, 0]])

                if child == -1:
                    if nc == self.max_branches:
                        new_mb = self.max_branches * 2
                        expanded = torch.full(
                            (self.ell_cv.shape[0], 2, new_mb), -1,
                            dtype=torch.long, device=self.ell_cv.device,
                        )
                        expanded[:, :, :self.max_branches] = self.ell_cv
                        self.ell_cv = expanded
                        self.max_branches = new_mb
                    child = self.num_nodes
                    self.num_nodes += 1
                    self.ell_cv[node, 0, nc] = token
                    self.ell_cv[node, 1, nc] = child
                    self.n_ch[node] += 1

                node = child
