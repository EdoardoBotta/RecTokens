from __future__ import annotations

from transformers import PretrainedConfig


class ItemAwareCausalLMConfig(PretrainedConfig):
    """Configuration for :class:`ItemAwareCausalLM`.

    Stores the item-tokenizer hyperparameters (``num_levels``, ``codebook_size``)
    and the base causal-LM configuration so the model can be fully reconstructed
    from a checkpoint via ``AutoModelForCausalLM.from_pretrained``.

    Args:
        num_levels: Number of RQ levels (depth of the residual quantizer).
        codebook_size: Number of codes per RQ level.
        base_model_config: ``hf_model.config.to_dict()`` of the wrapped causal LM.
            Serialised into ``config.json`` so ``from_pretrained`` can recreate the
            inner model architecture without a separate config file.
        **kwargs: Forwarded to :class:`~transformers.PretrainedConfig`.
    """

    model_type = "item_aware_causal_lm"

    def __init__(
        self,
        num_levels: int = 4,
        codebook_size: int = 256,
        base_model_config: dict | None = None,
        **kwargs,
    ) -> None:
        self.num_levels = num_levels
        self.codebook_size = codebook_size
        self.base_model_config = base_model_config
        super().__init__(**kwargs)
