import torch
from contextlib import contextmanager
from rectokens.ops.constrained_node_transition import (
    fused_linear_constrained_node_transition,
    fused_linear_constrained_node_transition_sampling,
    fused_linear_constrained_node_transition_topk,
)
from rectokens.schemas.state import ConstraintState
from torch import nn
from typing import Literal, Optional


class SparseLinear(nn.Module):
    def __init__(self, base_linear: nn.Linear):
        super().__init__()
        self.base_linear = base_linear
        self._ctx: Optional[ConstraintState] = None
        self._strategy: Literal["default", "sample", "topk"] = "default"
        self._temperature: Optional[float] = None
        self._k: int = 1
        self._rng_seed: Optional[int] = None
        self.next_nodes: Optional[torch.Tensor] = None
        self.valid_idxs: Optional[torch.Tensor] = None
        self.sample: Optional[torch.Tensor] = None
        self.topk_logits: Optional[torch.Tensor] = None
        self.topk_idxs: Optional[torch.Tensor] = None

    @contextmanager
    def constrained(
        self,
        constraint_state: ConstraintState,
        strategy: Literal["default", "sample", "topk"] = "default",
        temperature: Optional[float] = None,
        k: int = 1,
        rng_seed: Optional[int] = None,
    ):
        self._ctx = constraint_state
        self._strategy = strategy
        self._temperature = temperature
        self._k = k
        self._rng_seed = rng_seed
        try:
            yield
        finally:
            self._ctx = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.ndim == 2, "Fused kernel does not support dim > 2"
        # Clear outputs from the previous step.
        self.next_nodes = self.valid_idxs = None
        self.sample = self.topk_logits = self.topk_idxs = None

        if self._ctx is None:
            return self.base_linear(x)

        w, bias = self.base_linear.weight, self.base_linear.bias

        if self._strategy == "sample":
            next_nodes, valid_idxs, sample = fused_linear_constrained_node_transition_sampling(
                x, w, self._ctx, bias=bias,
                temperature=self._temperature,
                rng_seed=self._rng_seed,
            )
            self.next_nodes, self.valid_idxs, self.sample = next_nodes, valid_idxs, sample
            # Return zeros — decode_one_step uses self.sample directly.
            return x.new_zeros(x.shape[0], w.shape[0])

        if self._strategy == "topk":
            next_nodes, valid_idxs, topk_logits, topk_idxs = fused_linear_constrained_node_transition_topk(
                x, w, self._ctx, self._k, bias=bias,
            )
            self.next_nodes, self.valid_idxs = next_nodes, valid_idxs
            self.topk_logits, self.topk_idxs = topk_logits, topk_idxs
            return topk_logits  # (B, k)

        next_nodes, valid_idxs, logits = fused_linear_constrained_node_transition(
            x, w, self._ctx, bias=bias,
        )
        self.next_nodes, self.valid_idxs = next_nodes, valid_idxs
        return logits
