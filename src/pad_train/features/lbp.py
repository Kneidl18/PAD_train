"""Local binary pattern (LBP) feature extraction."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from pad_train.data import Image
from pad_train.features.registry import register_extractor

#: Offsets (row, column) of the eight neighbours, in clockwise order.
_NEIGHBOURS: tuple[tuple[int, int], ...] = (
    (-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1),
)  # fmt: skip


@register_extractor("lbp")
def lbp_features(image: Image) -> NDArray[np.float64]:
    """Extract the normalized histogram of basic 3x3 LBP codes.

    Every interior pixel is compared with its eight neighbours; a neighbour
    that is at least as bright contributes one bit to the pixel's 8-bit code.

    Args:
        image: 2-D grayscale image.

    Returns:
        Feature vector of length 256 that sums to one.
    """
    rows, cols = image.shape
    centre = image[1:-1, 1:-1]
    codes = np.zeros(centre.shape, dtype=np.uint8)
    for bit, (dy, dx) in enumerate(_NEIGHBOURS):
        neighbour = image[1 + dy : rows - 1 + dy, 1 + dx : cols - 1 + dx]
        codes |= (neighbour >= centre).astype(np.uint8) << bit

    histogram = np.bincount(codes.ravel(), minlength=256).astype(np.float64)
    normalized: NDArray[np.float64] = histogram / max(histogram.sum(), 1.0)
    return normalized
