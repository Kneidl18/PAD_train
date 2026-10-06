"""Registry that makes feature extractors plug and play."""

from __future__ import annotations

from collections.abc import Callable
from typing import overload

import numpy as np
from numpy.typing import NDArray

from pad_train.data import Image

#: A feature extractor maps one grayscale image to a 1-D feature vector. The
#: vector length must not depend on the image size.
FeatureExtractor = Callable[[Image], NDArray[np.float64]]

_REGISTRY: dict[str, FeatureExtractor] = {}


@overload
def register_extractor(name: str) -> Callable[[FeatureExtractor], FeatureExtractor]: ...
@overload
def register_extractor(name: str, extractor: FeatureExtractor) -> FeatureExtractor: ...
def register_extractor(
    name: str, extractor: FeatureExtractor | None = None
) -> FeatureExtractor | Callable[[FeatureExtractor], FeatureExtractor]:
    """Register a feature extractor under ``name``.

    Usable as a plain call or as a decorator::

        register_extractor("fourier_low", make_fourier_extractor(bands=range(10)))

        @register_extractor("mean")
        def mean_intensity(image: Image) -> NDArray[np.float64]:
            return np.array([image.mean()])

    Args:
        name: Name the extractor is looked up by. Existing names are replaced.
        extractor: The extractor. Omit it to get a decorator.

    Returns:
        The extractor itself, or a decorator when ``extractor`` is omitted.
    """

    def _register(func: FeatureExtractor) -> FeatureExtractor:
        _REGISTRY[name] = func
        return func

    return _register if extractor is None else _register(extractor)


def get_extractor(name: str) -> FeatureExtractor:
    """Return the extractor registered under ``name``.

    Raises:
        KeyError: No extractor of that name is registered.
    """
    try:
        return _REGISTRY[name]
    except KeyError:
        raise KeyError(
            f"Unknown feature extractor '{name}'. Available: {available_extractors()}"
        ) from None


def available_extractors() -> list[str]:
    """Return the names of all registered extractors."""
    return sorted(_REGISTRY)
