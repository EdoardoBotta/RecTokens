import torch

def sparse_linear_pytorch(a, weight, cur_node, trie, step):
    """
    PyTorch impl that only computes logits for valid (constrained) tokens.
    weight shape: (N, K)  — standard nn.Linear weight layout.
    """
    device = trie.row_ptrs.device
    B, K = a.shape
    N = weight.shape[0]

    idx_start = trie.row_ptrs[cur_node]
    n_children = trie.row_ptrs[cur_node + 1] - idx_start

    slice_len = trie.layer_max_branches[step]
    slice_idxs = idx_start.unsqueeze(-1) + torch.arange(slice_len, device=device)

    cols, vals = trie.stacked_cols_vals[:, slice_idxs].unbind()

    valid_range = torch.arange(slice_len, device=device) < n_children.unsqueeze(-1)
    valid_idxs = torch.where(valid_range, cols, -1)
    next_node = torch.where(valid_range, vals, -1)

    # Gather weight rows for valid tokens only, compute dot products
    clamped_idxs = valid_idxs.clamp(min=0)          # (B, max_branches)
    valid_weights = weight[clamped_idxs]              # (B, max_branches, K)
    logits_valid = (a.unsqueeze(1) * valid_weights).sum(dim=-1)  # (B, max_branches)

    # Scatter results into full logits tensor (rest stays -inf)
    corrected_logits = torch.full((B, N), float('-inf'), dtype=torch.float32, device=device)
    b_idx = torch.arange(B, device=device).unsqueeze(-1).expand_as(valid_idxs)
    valid = valid_idxs >= 0
    corrected_logits[b_idx[valid], valid_idxs[valid]] = logits_valid[valid]

    return next_node, valid_idxs, corrected_logits


def vtnk_pytorch(logits, cur_node, trie, step):
    device = trie.row_ptrs.device
    B, vocab_size = logits.shape
    assert cur_node.dim() > 0

    idx_start = trie.row_ptrs[cur_node]           # (B,)
    n_children = trie.row_ptrs[cur_node + 1] - idx_start  # (B,)

    slice_len = trie.layer_max_branches[step]
    slice_idxs = idx_start.unsqueeze(-1) + torch.arange(slice_len, device=device)  # (B, slice_len)

    cols, vals = trie.stacked_cols_vals[:, slice_idxs].unbind()

    valid_range = torch.arange(slice_len, device=device) < n_children.unsqueeze(-1)  # (B, slice_len)
    valid_idxs = torch.where(valid_range, cols, -1)
    next_node = torch.where(valid_range, vals, -1)

    mask = torch.zeros(B, vocab_size, dtype=torch.bool, device=device)
    b_idx = torch.arange(B, device=device).unsqueeze(-1).expand_as(valid_idxs)
    valid = valid_idxs >= 0
    mask[b_idx[valid], valid_idxs[valid]] = True

    corrected_logits = torch.where(mask, logits, float('-inf'))

    return next_node, valid_idxs, corrected_logits
