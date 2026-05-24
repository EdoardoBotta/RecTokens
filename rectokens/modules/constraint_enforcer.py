from abc import ABC, abstractmethod
from contextlib import AbstractContextManager, contextmanager
from rectokens.modules.sparse_linear import SparseLinear
from rectokens.schemas.state import ConstraintState
from typing import Literal, Optional
from torch import nn


class ConstraintEnforcer(ABC):
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


class SparseTrieConstraintEnforcer(ConstraintEnforcer):
    """Replaces a model's output projection with a SparseLinear for fused-kernel constraint enforcement.

    Uses ``model.get_output_embeddings()`` / ``model.set_output_embeddings()`` to locate
    and swap the layer, so it works with any HF ``PreTrainedModel`` regardless of internal
    attribute naming, including PEFT-wrapped models and models from the ``adapters`` library.

    Call ``prepare(model)`` before generation and ``restore(model)`` after to leave the
    model in its original state.
    """

    def __init__(self) -> None:
        self.constrained_linear: Optional[SparseLinear] = None
        self._original_linear: Optional[nn.Linear] = None

    def prepare(self, model: nn.Module) -> nn.Module:
        """Replace the output projection in ``model`` with a SparseLinear in-place.

        Raises:
            RuntimeError: If called again before :meth:`restore`.
            RuntimeError: If ``model.get_output_embeddings()`` returns ``None``.
            TypeError:    If the output embedding is not an ``nn.Linear``.
        """
        if self._original_linear is not None:
            raise RuntimeError(
                "prepare() already called; call restore() before calling prepare() again"
            )
        linear = model.get_output_embeddings()
        if linear is None:
            raise RuntimeError("model.get_output_embeddings() returned None")
        if not isinstance(linear, nn.Linear):
            raise TypeError(
                f"Expected nn.Linear from get_output_embeddings(), got {type(linear).__name__}"
            )
        self._original_linear = linear
        sparse = SparseLinear(linear)
        model.set_output_embeddings(sparse)
        self.constrained_linear = sparse
        return model

    def restore(self, model: nn.Module) -> None:
        """Reinstate the original output projection replaced by :meth:`prepare`."""
        if self._original_linear is None:
            return
        model.set_output_embeddings(self._original_linear)
        self._original_linear = None
        self.constrained_linear = None

    @contextmanager
    def apply(self, model: nn.Module):
        """Context manager that calls :meth:`prepare` on enter and :meth:`restore` on exit."""
        self.prepare(model)
        try:
            yield
        finally:
            self.restore(model)

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
