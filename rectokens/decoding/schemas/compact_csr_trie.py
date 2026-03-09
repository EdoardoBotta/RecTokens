from typing import NamedTuple
from torch import Tensor


class CompactCSRTrie(NamedTuple):
    row_ptrs: Tensor
    stacked_cols_vals: Tensor
    layer_max_branches: list[int]
    dense_lookup_mask: Tensor
    dense_states: Tensor 