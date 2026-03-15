import torch
from rectokens.ops.constrained_node_transition import (
    fused_linear_constrained_node_transition,
)
from rectokens.schemas.compact_csr_trie import CompactCSRTrie
from torch import nn
from typing import Optional


class ConstrainedLinear(nn.Module):
    def __init__(self, base_linear: nn.Linear):
        super().__init__()
        self.base_linear = base_linear
        self._ctx: Optional[tuple] = None
        self.next_nodes: Optional[torch.Tensor] = None
        self.valid_idxs: Optional[torch.Tensor] = None

    def set_context(
        self, cur_node: torch.Tensor, trie: CompactCSRTrie, step: int
    ) -> None:
        self._ctx = (cur_node, trie, step)

    def clear_context(self) -> None:
        self._ctx = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.ndim == 2, "Fused kernel does not support dim > 2"
        if self._ctx is None:
            return self.base_linear(x)
        cur_node, trie, step = self._ctx
        next_nodes, valid_idxs, logits = fused_linear_constrained_node_transition(
            x, self.base_linear.weight, cur_node, trie, step, bias=self.base_linear.bias
        )
        self.next_nodes = next_nodes
        self.valid_idxs = valid_idxs
        return logits
