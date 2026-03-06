import torch
from rectokens.decoding.trie import Trie
from torch import sparse_csr_tensor

def csr_from_trie(trie: Trie):
    row_ptrs = []
    col_idxs = []
    
    frontier = deque([trie.root])
    while frontier:
        node = frontier.popleft()
        row_ptrs.append(len(col_idxs))
        for idx, child in node.children.items():
            col_idxs.append(idx)
            frontier.append(child)
    
    col_idxs.append(-1)
    values = [i for i in range(1, len(col_idxs))] + [-1]
    return sparse_csr_tensor(row_ptrs, col_idxs, values)

def csr_from_sorted_batch(sem_ids: torch.Tensor):
    """
    Expects sem_ids to be a 2D tensor with rows lexicographically sorted.
    """
    N, L = sem_ids.shape
    device = sem_ids.device

    sem_ids_aug = torch.cat([torch.full_like(sem_ids[0:1], -1), sem_ids], dim=0)
    is_diff = (sem_ids_aug[1:] != sem_ids_aug[:-1])
    is_new_node = is_diff.cumsum(dim=-1) > 0          # (N, L)
    is_new_node_T = is_new_node.T.contiguous()         # (L, N) cached
    node_ids = is_new_node_T.flatten().cumsum(0).reshape(L, N)  # (L, N)

    cols = torch.cat([sem_ids.T[is_new_node_T],
                      -torch.ones(1, dtype=sem_ids.dtype, device=device)])

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

    return sparse_csr_tensor(rows, cols, vals), layer_max_branches





    

