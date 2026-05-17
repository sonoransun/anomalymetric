"""Loader protocol + entry-point-based plugin registry.

Third-party packages can register loaders by declaring an
`anomalymetric.loaders` entry point pointing at a `SpectrumLoader`
implementation.
"""

from __future__ import annotations

from importlib.metadata import entry_points
from typing import Any, Protocol

from anomalymetric.spectrum import Spectrum


class SpectrumLoader(Protocol):
    """A loader knows how to turn a source identifier into a `Spectrum`."""

    name: str

    def load(self, source: str, **kwargs: Any) -> Spectrum: ...


_REGISTRY: dict[str, SpectrumLoader] = {}


def register(loader: SpectrumLoader) -> None:
    _REGISTRY[loader.name] = loader


def discover() -> dict[str, SpectrumLoader]:
    """Discover loaders via entry points; cache in-process."""
    if _REGISTRY:
        return dict(_REGISTRY)
    eps = entry_points(group="anomalymetric.loaders")
    for ep in eps:
        try:
            cls = ep.load()
            instance = cls() if isinstance(cls, type) else cls
            _REGISTRY[ep.name] = instance
        except Exception:
            # Loaders requiring optional extras may fail to import; that's OK.
            continue
    return dict(_REGISTRY)


def get(name: str) -> SpectrumLoader:
    reg = discover()
    if name not in reg:
        raise KeyError(
            f"No loader '{name}' registered. Available: {sorted(reg)}"
        )
    return reg[name]
