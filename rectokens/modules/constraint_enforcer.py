from rectokens.modules.constrained_linear import ConstrainedLinear
from typing import Optional
from torch import nn


class ConstraintEnforcer(nn.Module):
    """
    Wraps a model and replaces its output projection with a ConstrainedLinear.

    Args:
        attr_path: Dotted path to the output projection layer, e.g. "lm_head".
    """

    def __init__(self, attr_path: str):
        super().__init__()
        self.attr_path = attr_path
        self.constrained_linear: Optional[ConstrainedLinear] = None

    @classmethod
    def convert_to_constrained_linear(cls, model: nn.Module, attr_path: str) -> ConstrainedLinear:
        """
        Replace the output projection layer at `attr_path` with a ConstrainedLinear.

        Navigates the dotted attribute path to find the parent module and target
        attribute, validates the layer is a bias-free nn.Linear, then swaps it
        in-place.  Returns the ConstrainedLinear that was inserted.

        Args:
            model:     The model to modify in-place.
            attr_path: Dotted path to the linear layer, e.g. "lm_head" or
                       "model.embed_out".

        Returns:
            The ConstrainedLinear now installed at attr_path.

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

        constrained = ConstrainedLinear(linear)
        setattr(parent, attr_name, constrained)
        return constrained

    def convert(self, model: nn.Module) -> nn.Module:
        """Replace the output projection in `model` with a ConstrainedLinear in-place."""
        self.constrained_linear = self.convert_to_constrained_linear(model, self.attr_path)
        return model

    def forward(self, *args, **kwargs):
        raise NotImplementedError(
            "ConstraintEnforcer is not a forward model; call .convert() to prepare a model "
            "then pass it to autoregressive_generate."
        )
