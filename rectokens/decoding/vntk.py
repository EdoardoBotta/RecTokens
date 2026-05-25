import torch


def _sparse_ell_branch_logits(a, weight, cur_node, ell_trie, step):
    """Shared inner: ELL trie traversal + dot products for valid branches only.

    Returns (next_node, valid_idxs, branch_logits) where branch_logits has
    shape (B, max_branches) with -inf padding — no (B, N) scatter.
    weight shape: (N, K) — standard nn.Linear weight layout.
    """
    max_branches = ell_trie.layer_max_branches[step]

    # Direct index by node ID — no row_ptr indirection (ELL advantage over CSR)
    cols = ell_trie.ell_cols_vals[cur_node, 0, :max_branches]  # (B, max_branches)
    vals = ell_trie.ell_cols_vals[cur_node, 1, :max_branches]  # (B, max_branches)

    valid_range = cols >= 0  # -1 sentinel marks padding
    valid_idxs = torch.where(valid_range, cols, -1)
    next_node = torch.where(valid_range, vals, -1)

    clamped_idxs = valid_idxs.clamp(min=0)   # (B, max_branches)
    valid_weights = weight[clamped_idxs]       # (B, max_branches, K)
    logits = (a.unsqueeze(1) * valid_weights).to(torch.float32).sum(dim=-1)  # (B, max_branches)
    branch_logits = torch.where(valid_range, logits, float("-inf"))

    return next_node, valid_idxs, branch_logits


def sparse_linear_ell_pytorch(a, weight, cur_node, ell_trie, step):
    """
    PyTorch impl for ELL-format trie that only computes logits for valid tokens.
    Equivalent to sparse_linear_pytorch but uses ELL format (no row_ptr lookup).
    weight shape: (N, K) — standard nn.Linear weight layout.
    """
    device = ell_trie.ell_cols_vals.device
    B = a.shape[0]
    N = weight.shape[0]

    next_node, valid_idxs, branch_logits = _sparse_ell_branch_logits(
        a, weight, cur_node, ell_trie, step
    )

    corrected_logits = torch.full(
        (B, N), float("-inf"), dtype=torch.float32, device=device
    )
    b_idx = torch.arange(B, device=device).unsqueeze(-1).expand_as(valid_idxs)
    valid = valid_idxs >= 0
    corrected_logits[b_idx[valid], valid_idxs[valid]] = branch_logits[valid]

    return next_node, valid_idxs, corrected_logits


def sparse_linear_compact_ell_pytorch(a, weight, cur_node, ell_trie, step):
    """Like sparse_linear_ell_pytorch but skips the (B, N) scatter.

    Returns (next_node, valid_idxs, branch_logits) where branch_logits has
    shape (B, max_branches). Avoids allocating the full vocab-sized logit
    matrix — top-k can be applied directly on the compact buffer.
    weight shape: (N, K) — standard nn.Linear weight layout.
    """
    return _sparse_ell_branch_logits(a, weight, cur_node, ell_trie, step)


def _sparse_branch_logits(a, weight, cur_node, trie, step):
    """Shared inner: trie traversal + dot products for valid branches only.

    Returns (next_node, valid_idxs, branch_logits) where branch_logits has
    shape (B, max_branches) with -inf padding — no (B, N) scatter.
    weight shape: (N, K) — standard nn.Linear weight layout.
    """
    device = trie.row_ptrs.device
    B, K = a.shape

    idx_start = trie.row_ptrs[cur_node]
    n_children = trie.row_ptrs[cur_node + 1] - idx_start

    slice_len = trie.layer_max_branches[step]
    slice_idxs = idx_start.unsqueeze(-1) + torch.arange(slice_len, device=device)

    cols, vals = trie.stacked_cols_vals[:, slice_idxs].unbind()

    valid_range = torch.arange(slice_len, device=device) < n_children.unsqueeze(-1)
    valid_idxs = torch.where(valid_range, cols, -1)
    next_node = torch.where(valid_range, vals, -1)

    clamped_idxs = valid_idxs.clamp(min=0)  # (B, max_branches)
    valid_weights = weight[clamped_idxs]  # (B, max_branches, K)
    # Multiply in bf16, accumulate in fp32 — matches the Triton kernel's compute pattern.
    logits = (a.unsqueeze(1) * valid_weights).to(torch.float32).sum(dim=-1)  # (B, max_branches)
    branch_logits = torch.where(valid_range, logits, float("-inf"))

    return next_node, valid_idxs, branch_logits


def sparse_linear_pytorch(a, weight, cur_node, trie, step):
    """
    PyTorch impl that only computes logits for valid (constrained) tokens.
    weight shape: (N, K)  — standard nn.Linear weight layout.
    """
    device = trie.row_ptrs.device
    B = a.shape[0]
    N = weight.shape[0]

    next_node, valid_idxs, branch_logits = _sparse_branch_logits(
        a, weight, cur_node, trie, step
    )

    # Scatter compact logits into full (B, N) tensor (rest stays -inf)
    corrected_logits = torch.full(
        (B, N), float("-inf"), dtype=torch.float32, device=device
    )
    b_idx = torch.arange(B, device=device).unsqueeze(-1).expand_as(valid_idxs)
    valid = valid_idxs >= 0
    corrected_logits[b_idx[valid], valid_idxs[valid]] = branch_logits[valid]

    return next_node, valid_idxs, corrected_logits


def sparse_linear_compact_pytorch(a, weight, cur_node, trie, step):
    """Like sparse_linear_pytorch but skips the (B, N) scatter.

    Returns (next_node, valid_idxs, branch_logits) where branch_logits has
    shape (B, max_branches). Avoids allocating the full vocab-sized logit
    matrix — top-k can be applied directly on the compact buffer.
    weight shape: (N, K)  — standard nn.Linear weight layout.
    """
    return _sparse_branch_logits(a, weight, cur_node, trie, step)


def vtnk_pytorch(logits, cur_node, trie, step):
    device = trie.row_ptrs.device
    B, vocab_size = logits.shape
    assert cur_node.dim() > 0

    idx_start = trie.row_ptrs[cur_node]  # (B,)
    n_children = trie.row_ptrs[cur_node + 1] - idx_start  # (B,)

    slice_len = trie.layer_max_branches[step]
    slice_idxs = idx_start.unsqueeze(-1) + torch.arange(
        slice_len, device=device
    )  # (B, slice_len)

    cols, vals = trie.stacked_cols_vals[:, slice_idxs].unbind()

    valid_range = torch.arange(slice_len, device=device) < n_children.unsqueeze(
        -1
    )  # (B, slice_len)
    valid_idxs = torch.where(valid_range, cols, -1)
    next_node = torch.where(valid_range, vals, -1)

    mask = torch.zeros(B, vocab_size, dtype=torch.bool, device=device)
    b_idx = torch.arange(B, device=device).unsqueeze(-1).expand_as(valid_idxs)
    valid = valid_idxs >= 0
    mask[b_idx[valid], valid_idxs[valid]] = True

    corrected_logits = torch.where(mask, logits, float("-inf"))

    return next_node, valid_idxs, corrected_logits
