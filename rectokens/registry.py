from __future__ import annotations

from typing import Type

from rectokens.core.tokenizer import Tokenizer


class TokenizerRegistry:
    """A global registry mapping string names to :class:`~rectokens.core.tokenizer.Tokenizer` classes.

    Use the :meth:`register` decorator to add new tokenizers::

        from rectokens.registry import TokenizerRegistry
        from rectokens.core.tokenizer import Tokenizer

        @TokenizerRegistry.register("my_tokenizer")
        class MyTokenizer(Tokenizer):
            ...

    Then instantiate by name::

        tok = TokenizerRegistry.create("my_tokenizer", dim=64, ...)

    This decouples user code from concrete imports and makes it easy to
    swap tokenizer implementations via configuration strings.
    """

    _registry: dict[str, Type[Tokenizer]] = {}

    @classmethod
    def register(cls, name: str):
        """Class decorator that registers ``tokenizer_cls`` under ``name``.

        Args:
            name: Unique string key for the tokenizer.

        Returns:
            The decorator function.  The decorated class is returned unchanged.
        """
        def decorator(tokenizer_cls: Type[Tokenizer]) -> Type[Tokenizer]:
            if name in cls._registry:
                raise ValueError(
                    f"A tokenizer named '{name}' is already registered.  "
                    "Use a different name or unregister the existing entry first."
                )
            cls._registry[name] = tokenizer_cls
            return tokenizer_cls

        return decorator

    @classmethod
    def create(cls, name: str, **kwargs) -> Tokenizer:
        """Instantiate a registered tokenizer by name.

        Args:
            name: Key passed to :meth:`register`.
            **kwargs: Constructor arguments forwarded to the tokenizer class.

        Returns:
            A new tokenizer instance.

        Raises:
            KeyError: If ``name`` is not registered.
        """
        if name not in cls._registry:
            available = list(cls._registry.keys())
            raise KeyError(
                f"Unknown tokenizer '{name}'.  "
                f"Registered tokenizers: {available}"
            )
        return cls._registry[name](**kwargs)

    @classmethod
    def list_tokenizers(cls) -> list[str]:
        """Return the names of all registered tokenizers."""
        return list(cls._registry.keys())

    @classmethod
    def unregister(cls, name: str) -> None:
        """Remove a tokenizer from the registry.

        Args:
            name: Key to remove.

        Raises:
            KeyError: If ``name`` is not registered.
        """
        if name not in cls._registry:
            raise KeyError(f"'{name}' is not registered.")
        del cls._registry[name]
