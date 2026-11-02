"""Local binary pattern (LBP) texture feature extraction."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from skimage.feature import local_binary_pattern

from pad_train.data import Image
from pad_train.features.registry import register_extractor

#: The two LBP scales as ``(points, radius)``: 8 neighbours at radius 1 and 16
#: neighbours at radius 2, the setting used for finger vein PAD by Vorderleitner
#: and Uhl (SAC 2025).
LBP_SCALES: tuple[tuple[int, int], ...] = ((8, 1), (16, 2))


@register_extractor("lbp")
def lbp_features(
    image: Image, scales: tuple[tuple[int, int], ...] = LBP_SCALES
) -> NDArray[np.float64]:
    """Extract uniform LBP histograms of an image.

    For every ``(points, radius)`` in ``scales`` the uniform LBP code of each
    pixel is computed (``points + 2`` possible codes: the ``points + 1``
    uniform patterns plus one bin for all non-uniform patterns) and summarised
    as a histogram normalised to sum 1. The histograms are concatenated.

    Args:
        image: 2-D grayscale image.
        scales: LBP scales as ``(points, radius)`` pairs.

    Returns:
        Feature vector of length ``sum(points + 2)``, 28 for the default
        scales. The length does not depend on the image size.
    """
    histograms: list[NDArray[np.float64]] = []
    for points, radius in scales:
        # scikit-image ships no annotations for this function.
        codes = local_binary_pattern(image, points, radius, method="uniform")  # type: ignore[no-untyped-call]
        n_bins = points + 2
        histogram, _ = np.histogram(codes, bins=n_bins, range=(0, n_bins))
        histograms.append(histogram / max(histogram.sum(), 1))
    return np.concatenate(histograms).astype(np.float64)
