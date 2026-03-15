import torch
from rectokens.ops.constrained_node_transition import (
    fused_linear_constrained_node_transition,
)
from rectokens.schemas.state import ConstraintState
from torch import nn
from typing import Optional


class SparseLinear(nn.Module):
    def __init__(self, base_linear: nn.Linear):
        super().__init__()
        self.base_linear = base_linear
        self._ctx: Optional[ConstraintState] = None
        self.next_nodes: Optional[torch.Tensor] = None
        self.valid_idxs: Optional[torch.Tensor] = None

    def set_context(self, constraint_state) -> None:
        self._ctx = constraint_state

    def clear_context(self) -> None:
        self._ctx = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.ndim == 2, "Fused kernel does not support dim > 2"
        if self._ctx is None:
            return self.base_linear(x)
        next_nodes, valid_idxs, logits = fused_linear_constrained_node_transition(
            x, self.base_linear.weight, self._ctx, bias=self.base_linear.bias
        )
        self.next_nodes = next_nodes
        self.valid_idxs = valid_idxs
        return logits
