"""Haralick and GLCM texture feature extraction."""

from __future__ import annotations

from collections.abc import Sequence

import mahotas
import numpy as np
from numpy.typing import NDArray
from skimage.feature import graycomatrix, graycoprops

from pad_train.data import Image
from pad_train.features.registry import register_extractor

#: Standard four directions in radians (0°, 45°, 90°, 135°).
_ANGLES_4: tuple[float, ...] = (0.0, np.pi / 4, np.pi / 2, 3 * np.pi / 4)


@register_extractor("haralick_mahotas")
def haralick_mahotas_features(image: Image) -> NDArray[np.float64]:
    """Extract the Haralick texture features of an image using mahotas.

    The 13 classic Haralick statistics (contrast, correlation, entropy, ...)
    are computed from the gray-level co-occurrence matrix of each of the four
    directions and averaged over the directions using mahotas.

    Args:
        image: 2-D grayscale image.

    Returns:
        Feature vector of length 13.
    """
    features: NDArray[np.float64] = mahotas.features.haralick(image, return_mean=True)
    return features.ravel()


haralick_mahotas_features.requires_scaling = True  # type: ignore[attr-defined]


def _extract_glcm_properties(
    image: Image,
    properties: Sequence[str],
    distances: Sequence[int] = (1,),
    angles: Sequence[float] = _ANGLES_4,
) -> NDArray[np.float64]:
    """Compute GLCM and extract requested texture properties using scikit-image."""
    img = np.ascontiguousarray(image)
    if not img.flags.writeable:
        img = img.copy()
    glcm = graycomatrix(  # type: ignore[no-untyped-call]
        img,
        distances=distances,
        angles=angles,
        levels=256,
        symmetric=True,
        normed=True,
    )
    feature_list: list[NDArray[np.float64]] = []
    for prop in properties:
        prop_values = graycoprops(glcm, prop)  # type: ignore[no-untyped-call]
        feature_list.append(prop_values.ravel())

    return np.concatenate(feature_list).astype(np.float64)


@register_extractor("haralick_3_best_features")
def haralick_3_best_features(image: Image) -> NDArray[np.float64]:
    """Top 3 separating GLCM texture features across 4 directions (d=1).

    Computes contrast, correlation, and entropy along angles 0°, 45°, 90°, and 135°,
    yielding a 12-dimensional feature vector.
    """
    return _extract_glcm_properties(image, properties=("contrast", "correlation", "entropy"))


haralick_3_best_features.requires_scaling = True  # type: ignore[attr-defined]


@register_extractor("haralick_3_worst_features")
def haralick_3_worst_features(image: Image) -> NDArray[np.float64]:
    """Bottom 3 separating GLCM texture features across 4 directions (d=1).

    Computes homogeneity, ASM, and energy along angles 0°, 45°, 90°, and 135°,
    yielding a 12-dimensional feature vector.
    """
    return _extract_glcm_properties(image, properties=("homogeneity", "ASM", "energy"))


haralick_3_worst_features.requires_scaling = True  # type: ignore[attr-defined]
