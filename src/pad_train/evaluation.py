"""Five-fold cross-validated k-NN PAD evaluation.

Two training protocols are evaluated on identical, subject-disjoint folds. Both
are always *tested* on bona fide (type 1) vs. real attack (type 2) samples:

``baseline``
    Trained on bona fide vs. real attack samples of the training subjects.
``synthetic``
    Trained on bona fide samples of the training subjects vs. synthetic attack
    (type 3) samples. No real attack sample is ever seen during training.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.model_selection import GroupKFold
from sklearn.neighbors import KNeighborsClassifier

from pad_train.config import Settings, settings
from pad_train.data import Image, PadData, load_data
from pad_train.features.registry import FeatureExtractor, get_extractor
from pad_train.metrics import ATTACK, BONA_FIDE, pad_metrics

#: Names of the training protocols.
PROTOCOLS: tuple[str, ...] = ("baseline", "synthetic")

#: Columns identifying one result row, followed by the metric columns.
ID_COLUMNS: list[str] = ["dataset", "extractor", "protocol", "train_attack"]
METRIC_COLUMNS: list[str] = ["APCER", "BPCER", "ACER"]

Features = NDArray[np.float64]


def extract_features(images: Sequence[Image], extractor: FeatureExtractor) -> Features:
    """Stack the feature vectors of ``images`` into an ``(n, d)`` matrix."""
    return np.vstack([extractor(image) for image in images])


def _knn_predict(
    x_train: Features, y_train: NDArray[np.intp], x_test: Features, n_neighbors: int
) -> NDArray[np.intp]:
    """Fit a k-NN classifier and return its predictions for ``x_test``."""
    classifier = KNeighborsClassifier(n_neighbors=n_neighbors)
    classifier.fit(x_train, y_train)
    predictions: NDArray[np.intp] = classifier.predict(x_test)
    return predictions


def cross_validate(
    real: Features,
    real_subjects: NDArray[np.str_],
    spoof: Features,
    spoof_subjects: NDArray[np.str_],
    synthetic: Mapping[str, Features],
    n_neighbors: int,
    protocols: Sequence[str] = PROTOCOLS,
    balance_synthetic: bool = True,
    n_folds: int = 5,
    random_state: int = 42,
) -> pd.DataFrame:
    """Run both protocols on precomputed feature matrices.

    Args:
        real: Features of the bona fide samples.
        real_subjects: Subject id per bona fide sample.
        spoof: Features of the real attack samples.
        spoof_subjects: Subject id per real attack sample.
        synthetic: Features of the synthetic attack samples per source name.
        n_neighbors: ``k`` of the k-NN classifier.
        protocols: Protocols to run, a subset of :data:`PROTOCOLS`.
        balance_synthetic: Train on as many synthetic attack samples as there
            are bona fide training samples (randomly drawn per fold). If
            ``False`` all synthetic samples are used in every fold.
        n_folds: Number of subject-disjoint folds.
        random_state: Seed for the synthetic sub-sampling.

    Returns:
        One row per protocol, training attack source and fold with the columns
        ``protocol``, ``train_attack``, ``fold``, ``APCER``, ``BPCER``, ``ACER``.
    """
    unknown = set(protocols) - set(PROTOCOLS)
    if unknown:
        raise ValueError(f"Unknown protocols {sorted(unknown)}. Available: {PROTOCOLS}")

    # The folds partition the *subjects*, so no person is in training and test.
    x_all = np.vstack([real, spoof])
    y_all = np.concatenate([np.full(len(real), BONA_FIDE), np.full(len(spoof), ATTACK)])
    subjects = np.concatenate([real_subjects, spoof_subjects])
    folds = GroupKFold(n_splits=n_folds).split(x_all, y_all, groups=subjects)

    rng = np.random.default_rng(random_state)
    rows: list[dict[str, object]] = []
    for fold, (train_idx, test_idx) in enumerate(folds):
        x_test, y_test = x_all[test_idx], y_all[test_idx]
        # Training sets of this fold as (protocol, train_attack, features, labels).
        runs: list[tuple[str, str, Features, NDArray[np.intp]]] = []

        if "baseline" in protocols:
            runs.append(("baseline", "spoof", x_all[train_idx], y_all[train_idx]))

        if "synthetic" in protocols:
            # Bona fide samples of the training subjects only.
            x_real = x_all[train_idx][y_all[train_idx] == BONA_FIDE]
            for name, x_synthetic in synthetic.items():
                if balance_synthetic and len(x_synthetic) > len(x_real):
                    chosen = rng.choice(len(x_synthetic), size=len(x_real), replace=False)
                    x_synthetic = x_synthetic[chosen]
                x_train = np.vstack([x_real, x_synthetic])
                y_train = np.concatenate(
                    [np.full(len(x_real), BONA_FIDE), np.full(len(x_synthetic), ATTACK)]
                )
                runs.append(("synthetic", name, x_train, y_train))

        for protocol, train_attack, x_train, y_train in runs:
            metrics = pad_metrics(y_test, _knn_predict(x_train, y_train, x_test, n_neighbors))
            rows.append(
                {"protocol": protocol, "train_attack": train_attack, "fold": fold}
                | dict(zip(METRIC_COLUMNS, metrics, strict=True))
            )

    return pd.DataFrame(rows)


def evaluate_dataset(
    data: PadData,
    extractor: FeatureExtractor,
    n_neighbors: int,
    protocols: Sequence[str] = PROTOCOLS,
    balance_synthetic: bool = True,
    n_folds: int = 5,
    random_state: int = 42,
) -> pd.DataFrame:
    """Extract features from ``data`` and cross-validate, see :func:`cross_validate`."""
    return cross_validate(
        real=extract_features(data.real, extractor),
        real_subjects=data.real_subjects,
        spoof=extract_features(data.spoof, extractor),
        spoof_subjects=data.spoof_subjects,
        synthetic={k: extract_features(v, extractor) for k, v in data.synthetic.items()},
        n_neighbors=n_neighbors,
        protocols=protocols,
        balance_synthetic=balance_synthetic,
        n_folds=n_folds,
        random_state=random_state,
    )


def evaluate(
    datasets: Sequence[str] | None = None,
    extractors: Sequence[str | FeatureExtractor] = ("fourier",),
    synthetic: Sequence[str] | None = None,
    protocols: Sequence[str] = PROTOCOLS,
    n_neighbors: int | None = None,
    balance_synthetic: bool = True,
    per_fold: bool = False,
    download: bool = True,
    config: Settings | None = None,
) -> pd.DataFrame:
    """Evaluate the PAD system and return APCER, BPCER and ACER as a table.

    Args:
        datasets: Dataset names. ``None`` evaluates all configured datasets.
        extractors: Feature extractors, given by registered name or as a
            callable (which is then listed under its ``__name__``).
        synthetic: Names of the synthetic sources to train on. ``None`` uses
            all configured ones.
        protocols: Protocols to run, a subset of :data:`PROTOCOLS`.
        n_neighbors: ``k`` of the k-NN classifier. ``None`` uses the
            per-dataset value from the configuration.
        balance_synthetic: See :func:`cross_validate`.
        per_fold: Return one row per fold (with a ``fold`` column) instead of
            the mean over the folds.
        download: Fetch missing data sources.
        config: Settings to use instead of the package-wide ones.

    Returns:
        A :class:`pandas.DataFrame` with the columns ``dataset``, ``extractor``,
        ``protocol``, ``train_attack`` (``"spoof"`` for the baseline, otherwise
        the synthetic source name), ``APCER``, ``BPCER`` and ``ACER``. The rates
        are fractions in ``[0, 1]``.
    """
    config = config or settings
    needs_synthetic = "synthetic" in protocols
    named = [
        (e, get_extractor(e)) if isinstance(e, str) else (getattr(e, "__name__", repr(e)), e)
        for e in extractors
    ]

    frames: list[pd.DataFrame] = []
    for dataset in datasets if datasets is not None else list(config.datasets):
        data = load_data(dataset, synthetic if needs_synthetic else (), download, config)
        k = n_neighbors if n_neighbors is not None else config.datasets[dataset].n_neighbors
        for name, extractor in named:
            frame = evaluate_dataset(
                data,
                extractor,
                k,
                protocols,
                balance_synthetic,
                config.n_folds,
                config.random_state,
            )
            frames.append(frame.assign(dataset=dataset, extractor=name))

    columns = [*ID_COLUMNS, "fold", *METRIC_COLUMNS]
    if not frames:
        return pd.DataFrame(columns=columns if per_fold else [*ID_COLUMNS, *METRIC_COLUMNS])
    result: pd.DataFrame = pd.concat(frames, ignore_index=True)[columns]
    if not per_fold:
        result = result.groupby(ID_COLUMNS, sort=False)[METRIC_COLUMNS].mean().reset_index()
    return result
