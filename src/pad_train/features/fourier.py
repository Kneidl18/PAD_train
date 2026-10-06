"""Fourier feature extraction: energy of the power spectrum in radial bands.

Only the magnitude of the spectrum is used, the phase is discarded.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
from numpy.typing import NDArray

from pad_train.data import Image
from pad_train.features.registry import (
    FeatureExtractor,
    register_dataset_extractor,
    register_extractor,
)


def _band_index(shape: tuple[int, ...], n_bands: int) -> NDArray[np.intp]:
    """Assign every frequency bin of a centred spectrum to a radial band.

    The bands are concentric rings of equal width, reaching from the spectrum
    centre (band 0, low frequencies) to its corner. The corner bin itself gets
    the out-of-range index ``n_bands``.
    """
    rows, cols = shape
    y, x = np.ogrid[:rows, :cols]
    dist = np.hypot(x - cols // 2, y - rows // 2)
    edges = np.linspace(0.0, float(dist.max()), n_bands + 1)
    index: NDArray[np.intp] = np.digitize(dist, edges) - 1
    return index


def fourier_features(
    image: Image,
    n_bands: int = 30,
    bands: Iterable[int] | None = None,
    log_compression: bool = True,
    normalize: bool = True,
) -> NDArray[np.float64]:
    """Extract the spectral energy per radial frequency band.

    Args:
        image: 2-D grayscale image.
        n_bands: Number of concentric radial bands.
        bands: Indices of the bands to return (0 = lowest frequencies). ``None``
            returns all of them. The selection happens after normalization.
        log_compression: Apply ``log(1 + x)`` to the band energies.
        normalize: L2-normalize the feature vector.

    Returns:
        Feature vector with one entry per selected band.
    """
    # Centred 2-D FFT: the zero frequency sits in the middle of the spectrum.
    # Taking the absolute value keeps the magnitude and drops the phase.
    spectrum = np.fft.fftshift(np.fft.fft2(image))
    power = np.abs(spectrum) ** 2

    # Sum the power of all bins falling into the same band.
    index = _band_index(image.shape, n_bands)
    features = np.bincount(index.ravel(), weights=power.ravel(), minlength=n_bands + 1)[:n_bands]

    if log_compression:
        features = np.log1p(features)
    if normalize:
        features = features / (np.linalg.norm(features) + 1e-12)
    return features if bands is None else features[list(bands)]


def thirds_to_bands(thirds: Iterable[int], n_bands: int = 30) -> list[int]:
    """Translate thirds of the spectrum (1 = low, 2 = middle, 3 = high) into band indices.

    Raises:
        ValueError: A third is not 1, 2 or 3, or none is given.
    """
    selected = sorted(set(thirds))
    if not selected or not set(selected) <= {1, 2, 3}:
        raise ValueError(f"Thirds must be a non-empty selection of 1, 2, 3, got {selected}")
    size = n_bands // 3
    # The last third also takes the remaining bands if n_bands is not divisible by 3.
    return [
        band
        for third in selected
        for band in range((third - 1) * size, n_bands if third == 3 else third * size)
    ]


def make_fourier_extractor(
    n_bands: int = 30,
    bands: Iterable[int] | None = None,
    log_compression: bool = True,
    normalize: bool = True,
    thirds: Iterable[int] | None = None,
) -> FeatureExtractor:
    """Build a Fourier extractor with fixed parameters.

    Args:
        n_bands: See :func:`fourier_features`.
        bands: See :func:`fourier_features`.
        log_compression: See :func:`fourier_features`.
        normalize: See :func:`fourier_features`.
        thirds: Alternative to ``bands``: thirds of the spectrum to keep, see
            :func:`thirds_to_bands`.

    Raises:
        ValueError: Both ``bands`` and ``thirds`` are given.
    """
    if bands is not None and thirds is not None:
        raise ValueError("Pass either 'bands' or 'thirds', not both")
    selected = thirds_to_bands(thirds, n_bands) if thirds is not None else bands
    selected = None if selected is None else list(selected)

    def extractor(image: Image) -> NDArray[np.float64]:
        return fourier_features(image, n_bands, selected, log_compression, normalize)

    return extractor


# Without a dataset all bands are used; with one, its configured thirds.
register_extractor("fourier", make_fourier_extractor())
register_dataset_extractor("fourier", lambda ds: make_fourier_extractor(thirds=ds.fourier_thirds))
