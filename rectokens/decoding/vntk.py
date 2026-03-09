import torch

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
