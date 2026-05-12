from abc import ABC, abstractmethod
from contextlib import AbstractContextManager, contextmanager
from rectokens.modules.sparse_linear import SparseLinear
from rectokens.schemas.state import ConstraintState
from typing import Literal, Optional
from torch import nn


class ConstraintEnforcer(ABC, nn.Module):
    @abstractmethod
    def prepare(self, model: nn.Module) -> nn.Module:
        """Modify `model` in-place to enable constraint enforcement."""
        ...

    @abstractmethod
    def constrained(
        self, constraint_state: ConstraintState, **kwargs
    ) -> AbstractContextManager:
        """Context manager that scopes constraint enforcement to one forward pass."""
        ...

    def forward(self, *args, **kwargs):
        raise NotImplementedError(
            "ConstraintEnforcer is not a forward model; call .prepare() to prepare a model "
            "then pass it to autoregressive_generate."
        )


class SparseTrieConstraintEnforcer(ConstraintEnforcer):
    """
    Wraps a model and replaces its output projection with a SparseLinear.

    Args:
        attr_path: Dotted path to the output projection layer, e.g. "lm_head".
    """

    def __init__(self, attr_path: str):
        super().__init__()
        self.attr_path = attr_path
        self.constrained_linear: Optional[SparseLinear] = None

    @classmethod
    def convert_to_sparse_linear(cls, model: nn.Module, attr_path: str) -> SparseLinear:
        """
        Replace the output projection layer at `attr_path` with a SparseLinear.

        Navigates the dotted attribute path to find the parent module and target
        attribute, validates the layer is a bias-free nn.Linear, then swaps it
        in-place.  Returns the SparseLinear that was inserted.

        Args:
            model:     The model to modify in-place.
            attr_path: Dotted path to the linear layer, e.g. "lm_head" or
                       "model.embed_out".

        Returns:
            The SparseLinear now installed at attr_path.

        Raises:
            AttributeError: If any component of attr_path does not exist.
            TypeError:      If the target attribute is not an nn.Linear.
            ValueError:     If the target nn.Linear has a bias.
        """
        parts = attr_path.split(".")
        parent = model
        for part in parts[:-1]:
            parent = getattr(parent, part)

        attr_name = parts[-1]
        linear = getattr(parent, attr_name)

        if not isinstance(linear, nn.Linear):
            raise TypeError(
                f"Expected nn.Linear at '{attr_path}', got {type(linear).__name__}"
            )

        constrained = SparseLinear(linear)
        setattr(parent, attr_name, constrained)
        return constrained

    def prepare(self, model: nn.Module) -> nn.Module:
        """Replace the output projection in `model` with a SparseLinear in-place."""
        self.constrained_linear = self.convert_to_sparse_linear(model, self.attr_path)
        return model

    @contextmanager
    def constrained(
        self,
        constraint_state: ConstraintState,
        strategy: Literal["default", "sample", "topk"] = "default",
        temperature: Optional[float] = None,
        k: int = 1,
        rng_seed: Optional[int] = None,
    ):
        """Context manager that scopes constraint enforcement to one forward pass."""
        if self.constrained_linear is None:
            raise RuntimeError("Call .prepare(model) before using .constrained()")
        with self.constrained_linear.constrained(
            constraint_state,
            strategy=strategy,
            temperature=temperature,
            k=k,
            rng_seed=rng_seed,
        ):
            yield
