"""Haralick texture feature extraction."""

from __future__ import annotations

import mahotas
import numpy as np
from numpy.typing import NDArray

from pad_train.data import Image
from pad_train.features.registry import register_extractor


@register_extractor("haralick")
def haralick_features(image: Image) -> NDArray[np.float64]:
    """Extract the Haralick texture features of an image.

    The 13 classic Haralick statistics (contrast, correlation, entropy, ...)
    are computed from the gray-level co-occurrence matrix of each of the four
    directions and averaged over the directions.

    Args:
        image: 2-D grayscale image.

    Returns:
        Feature vector of length 13. The features are not scaled and differ by
        orders of magnitude.
    """
    features: NDArray[np.float64] = mahotas.features.haralick(image, return_mean=True)
    return features.ravel()
