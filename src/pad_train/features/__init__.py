"""Feature extraction schemes, selectable by name.

Importing this package registers the built-in extractors. To plug in a new
scheme, add a module here (or anywhere in your own code) that calls
:func:`register_extractor`, then pass its name to :func:`pad_train.evaluate`.
"""

from pad_train.features.fourier import fourier_features, make_fourier_extractor, thirds_to_bands
from pad_train.features.haralick import haralick_features
from pad_train.features.registry import (
    FeatureExtractor,
    available_extractors,
    get_extractor,
    register_dataset_extractor,
    register_extractor,
)

__all__ = [
    "FeatureExtractor",
    "available_extractors",
    "fourier_features",
    "get_extractor",
    "haralick_features",
    "make_fourier_extractor",
    "register_dataset_extractor",
    "register_extractor",
    "thirds_to_bands",
]
