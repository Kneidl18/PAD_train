"""PAD error rates according to ISO/IEC 30107-3."""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
from numpy.typing import NDArray

#: Class label of bona fide presentations.
BONA_FIDE: int = 0
#: Class label of attack presentations (real and synthetic alike).
ATTACK: int = 1


class PadMetrics(NamedTuple):
    """Error rates of a PAD system, each in ``[0, 1]``.

    Attributes:
        apcer: Attack Presentation Classification Error Rate, the share of
            attacks accepted as bona fide.
        bpcer: Bona fide Presentation Classification Error Rate, the share of
            bona fide presentations rejected as attacks.
        acer: Average Classification Error Rate, the mean of the two.
    """

    apcer: float
    bpcer: float
    acer: float


def pad_metrics(y_true: NDArray[np.integer], y_pred: NDArray[np.integer]) -> PadMetrics:
    """Compute APCER, BPCER and ACER from true and predicted class labels.

    Args:
        y_true: Ground truth labels (:data:`BONA_FIDE` or :data:`ATTACK`).
        y_pred: Predicted labels.

    Returns:
        The three error rates. A rate is ``nan`` if its class is absent.
    """
    attacks = y_true == ATTACK
    bona_fide = y_true == BONA_FIDE
    apcer = float(np.mean(y_pred[attacks] == BONA_FIDE)) if attacks.any() else float("nan")
    bpcer = float(np.mean(y_pred[bona_fide] == ATTACK)) if bona_fide.any() else float("nan")
    return PadMetrics(apcer, bpcer, (apcer + bpcer) / 2)
