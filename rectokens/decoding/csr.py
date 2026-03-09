import torch
from collections import deque
from rectokens.decoding.trie import Trie
from rectokens.decoding.schemas.compact_csr_trie import CompactCSRTrie
from typing import NamedTuple


def csr_from_trie(trie: Trie, vocab_size: int, dense_lookup_layers: int = 2) -> CompactCSRTrie:
    row_ptrs = []
    col_idxs = []
    max_children_at_depth: dict[int, int] = {}
    dense_lookup_mask = torch.zeros([vocab_size] * dense_lookup_layers, dtype=torch.bool)
    dense_states = torch.zeros([vocab_size] * dense_lookup_layers, dtype=torch.long)

    next_bfs_idx = 1  # root is BFS row 0
    frontier: deque[tuple] = deque([(trie.root, 0, ())])
    while frontier:
        node, depth, path = frontier.popleft()
        row_ptrs.append(len(col_idxs))

        n_children = len(node.children)
        max_children_at_depth[depth] = max(max_children_at_depth.get(depth, 0), n_children)

        for token, child in node.children.items():
            col_idxs.append(token)
            child_path = path + (token,)

            if len(child_path) == dense_lookup_layers:
                dense_lookup_mask[child_path] = True
                dense_states[child_path] = next_bfs_idx

            frontier.append((child, depth + 1, child_path))
            next_bfs_idx += 1

    col_idxs.append(-1)
    values = list(range(1, len(col_idxs))) + [-1]

    layer_max_branches = [
        max_children_at_depth.get(d, 0)
        for d in range(max(max_children_at_depth, default=-1) + 1)
    ]

    return CompactCSRTrie(
        row_ptrs=torch.tensor(row_ptrs),
        stacked_cols_vals=torch.stack([torch.tensor(col_idxs), torch.tensor(values)]),
        layer_max_branches=layer_max_branches,
        dense_lookup_mask=dense_lookup_mask,
        dense_states=dense_states,
    )

def csr_from_sorted_batch(sem_ids: torch.Tensor, vocab_size: int, dense_lookup_layers: int = 2):
    """
    Expects sem_ids to be a 2D tensor with rows lexicographically sorted.
    """
    N, L = sem_ids.shape
    device = sem_ids.device

    sem_ids_aug = torch.cat([torch.full_like(sem_ids[:1], -1), sem_ids], dim=0)
    is_new_node = (sem_ids_aug[1:] != sem_ids_aug[:-1]).cumsum(dim=-1) > 0  # (N, L)

    is_new_node_T = is_new_node.T.contiguous()  # (L, N)
    node_ids = is_new_node_T.flatten().cumsum(0).view(L, N)

    cols = torch.cat([
        sem_ids.T[is_new_node_T],
        torch.tensor([-1], dtype=sem_ids.dtype, device=device)
    ])

    # Parent IDs for each edge — sorted by construction (node_ids is a cumsum)
    parts = [torch.zeros(int(is_new_node[:, 0].sum()), dtype=torch.long, device=device)]
    layer_max_branches = [len(parts[0])]
    for d in range(1, L):
        t = node_ids[d - 1][is_new_node[:, d]]
        parts.append(t)
        layer_max_branches.append(int(t.bincount().max()) if len(t) > 0 else 0)
    parents = torch.cat(parts)  # (nnz,) sorted

    changes = torch.where(parents.diff() != 0)[0] + 1 if len(parents) > 1 else torch.zeros(0, dtype=torch.long, device=device)

    n_edges = len(cols) - 1
    rows = torch.cat([
        torch.zeros(1, dtype=torch.long, device=device),
        changes,
        torch.full((n_edges - len(changes),), n_edges, dtype=torch.long, device=device),
    ])

    vals = torch.arange(1, len(cols) + 1, device=device)
    vals[-1] = -1

    mask = torch.zeros([vocab_size]*dense_lookup_layers, dtype=torch.bool, device=sem_ids.device)
    mask[sem_ids[:, :dense_lookup_layers].unbind(-1)] = True

    dense_states = torch.zeros([vocab_size] * dense_lookup_layers, dtype=torch.long, device=device)
    dense_states[sem_ids[:, :dense_lookup_layers].unbind(-1)] = node_ids[dense_lookup_layers - 1]

    return CompactCSRTrie(
        row_ptrs=rows,
        stacked_cols_vals=torch.stack([cols, vals]),
        layer_max_branches=layer_max_branches,
        dense_lookup_mask=mask,
        dense_states=dense_states,
    )





    

