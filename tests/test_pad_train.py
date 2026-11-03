"""Tests for metrics, feature extractors, data handling and the evaluation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import StandardScaler

import pad_train
from pad_train import Settings
from pad_train.evaluation import _knn_predict
from pad_train.features import (
    fourier_features,
    haralick_3_best_features,
    haralick_3_worst_features,
    haralick_mahotas_features,
    lbp_features,
    thirds_to_bands,
)

from .conftest import N_SUBJECTS, SAMPLES_PER_SUBJECT


def test_pad_metrics() -> None:
    y_true = np.array([0, 0, 0, 0, 1, 1])
    y_pred = np.array([0, 1, 0, 0, 0, 0])
    metrics = pad_train.pad_metrics(y_true, y_pred)
    assert metrics.apcer == 1.0  # both attacks accepted
    assert metrics.bpcer == 0.25  # one of four bona fides rejected
    assert metrics.acer == 0.625


@pytest.mark.parametrize("shape", [(32, 48), (65, 31)])
def test_extractors_are_size_independent(shape: tuple[int, int]) -> None:
    image = np.random.default_rng(0).integers(0, 256, size=shape).astype(np.uint8)
    assert fourier_features(image).shape == (30,)
    assert fourier_features(image, bands=range(10)).shape == (10,)
    assert np.isclose(np.linalg.norm(fourier_features(image)), 1.0)
    assert haralick_mahotas_features(image).shape == (13,)
    assert np.isfinite(haralick_mahotas_features(image)).all()
    assert haralick_3_best_features(image).shape == (12,)
    assert np.isfinite(haralick_3_best_features(image)).all()
    assert haralick_3_worst_features(image).shape == (12,)
    assert np.isfinite(haralick_3_worst_features(image)).all()
    lbp = lbp_features(image)
    assert lbp.shape == (28,)
    # One normalised histogram per scale: 10 bins for 8 neighbours, 18 bins for 16.
    assert np.isclose(lbp[:10].sum(), 1.0) and np.isclose(lbp[10:].sum(), 1.0)


def test_fourier_band_energies_sum_to_total_power() -> None:
    image = np.random.default_rng(1).integers(0, 256, size=(16, 16)).astype(np.uint8)
    features = fourier_features(image, n_bands=4, log_compression=False, normalize=False)
    spectrum = np.fft.fftshift(np.fft.fft2(image))
    # Everything but the single corner bin belongs to exactly one band.
    assert np.isclose(features.sum(), (np.abs(spectrum) ** 2).sum() - np.abs(spectrum[0, 0]) ** 2)


def test_fourier_thirds() -> None:
    assert thirds_to_bands([1]) == list(range(10))
    assert thirds_to_bands([2, 1]) == list(range(20))
    assert thirds_to_bands([1, 2, 3]) == list(range(30))
    with pytest.raises(ValueError, match="Thirds"):
        thirds_to_bands([4])

    image = np.random.default_rng(0).integers(0, 256, size=(32, 48)).astype(np.uint8)
    full = fourier_features(image)
    datasets = pad_train.Settings().datasets
    expected = {"PLUS": full, "IDIAP": full[:20], "SCUT": full[:10]}
    for name, features in expected.items():
        extractor = pad_train.get_extractor("fourier", datasets[name])
        assert np.array_equal(extractor(image), features)
    # A changed setting is picked up, and no dataset means all bands.
    datasets["SCUT"].fourier_thirds = (3,)
    assert np.array_equal(pad_train.get_extractor("fourier", datasets["SCUT"])(image), full[20:])
    assert np.array_equal(pad_train.get_extractor("fourier")(image), full)


def test_registry() -> None:
    assert {
        "fourier",
        "haralick_mahotas",
        "haralick_3_best_features",
        "haralick_3_worst_features",
        "lbp",
    } <= set(pad_train.available_extractors())
    with pytest.raises(KeyError, match="Unknown feature extractor"):
        pad_train.get_extractor("does-not-exist")


def test_load_data_downloads_and_balances(config: Settings) -> None:
    data = pad_train.load_data("TOY", config=config)
    assert len(data.real) == len(data.spoof) == N_SUBJECTS * SAMPLES_PER_SUBJECT
    assert len(set(data.real_subjects)) == N_SUBJECTS
    assert len(data.synthetic["synth"]) == 40
    assert data.real[0].dtype == np.uint8 and data.real[0].shape == (32, 48)
    # Archives are unwrapped into the configured paths and removed afterwards.
    assert (config.data_root / "real" / "TOY").is_dir()
    assert not list(config.data_root.glob("*.zip"))


def test_load_data_without_download_raises(config: Settings) -> None:
    with pytest.raises(FileNotFoundError, match="pad-train download"):
        pad_train.load_data("TOY", download=False, config=config)


def test_evaluate(config: Settings) -> None:
    results = pad_train.evaluate(extractors=["fourier", "haralick_mahotas"], config=config)
    assert list(results.columns) == [
        "dataset", "extractor", "protocol", "train_attack", "APCER", "BPCER", "ACER",
    ]  # fmt: skip
    assert list(results["train_attack"]) == ["spoof", "synth"] * 2
    assert results[["APCER", "BPCER", "ACER"]].to_numpy().max() <= 1.0
    # Noise vs. smoothed noise is trivially separable in the Fourier domain.
    assert (results.loc[results["extractor"] == "fourier", "ACER"] < 0.1).all()

    per_fold = pad_train.evaluate(per_fold=True, protocols=["baseline"], config=config)
    assert sorted(per_fold["fold"]) == list(range(config.n_folds))


def test_workers_do_not_change_results(config: Settings) -> None:
    serial = pad_train.evaluate(extractors=["fourier", "haralick_mahotas"], config=config)
    parallel = pad_train.evaluate(
        extractors=["fourier", "haralick_mahotas"], config=config, workers=4
    )
    pd.testing.assert_frame_equal(serial, parallel)


def test_equal_synthetic_sources_give_equal_results(config: Settings) -> None:
    config.synthetic_sources["copy"] = config.synthetic_sources["synth"]
    results = pad_train.evaluate(protocols=["synthetic"], per_fold=True, config=config)
    by_source = results.set_index(["train_attack", "fold"])[["APCER", "BPCER"]]
    assert by_source.loc["synth"].equals(by_source.loc["copy"])


def _record_scaler_fits(monkeypatch: pytest.MonkeyPatch) -> list[np.ndarray]:
    """Make ``StandardScaler.fit`` record every matrix it is fitted on."""
    seen: list[np.ndarray] = []
    original_fit = StandardScaler.fit

    def recording_fit(
        self: StandardScaler, x: np.ndarray, *args: object, **kwargs: object
    ) -> StandardScaler:
        seen.append(np.array(x))
        return original_fit(self, x, *args, **kwargs)

    monkeypatch.setattr(StandardScaler, "fit", recording_fit)
    return seen


def test_knn_scaling_changes_prediction() -> None:
    # Feature 0 separates the classes, feature 1 is uninformative and a thousand times larger.
    x_train = np.array([[0.0, 0.0], [0.0, 1000.0], [1.0, 300.0], [1.0, 700.0]])
    y_train = np.array([0, 0, 1, 1])
    x_test = np.array([[1.0, 10.0], [0.0, 690.0]])
    # Unscaled, the Euclidean distance is dominated by feature 1 and both answers are wrong.
    assert list(_knn_predict(x_train, y_train, x_test, 1)) == [0, 1]
    assert list(_knn_predict(x_train, y_train, x_test, 1, scale_features=True)) == [1, 0]


def test_knn_scaler_is_fitted_on_training_rows_only(monkeypatch: pytest.MonkeyPatch) -> None:
    seen = _record_scaler_fits(monkeypatch)
    x_train = np.array([[0.0, 0.0], [1.0, 10.0], [2.0, 20.0]])
    y_train = np.array([0, 1, 1])
    x_test = np.array([[9.0, 90.0], [8.0, 80.0]])
    _knn_predict(x_train, y_train, x_test, 1, scale_features=True)
    assert len(seen) == 1 and np.array_equal(seen[0], x_train)
    _knn_predict(x_train, y_train, x_test, 1)
    assert len(seen) == 1  # no scaler without scaling


def test_evaluate_scale_features(config: Settings, monkeypatch: pytest.MonkeyPatch) -> None:
    seen = _record_scaler_fits(monkeypatch)
    pad_train.evaluate(protocols=["baseline"], config=config)
    assert not seen  # off by default for fourier
    # Extractor with requires_scaling triggers scaling automatically
    pad_train.evaluate(
        extractors=["haralick_3_best_features"], protocols=["baseline"], config=config
    )
    assert len(seen) == config.n_folds
    # Explicit scale_features=False suppresses it
    pad_train.evaluate(
        extractors=["haralick_3_best_features"],
        protocols=["baseline"],
        config=config,
        scale_features=False,
    )
    assert len(seen) == config.n_folds
    pad_train.evaluate(protocols=["baseline"], config=config, scale_features=True)
    # One scaler per fold, fitted on the training share of the samples only.
    n_train = 2 * N_SUBJECTS * SAMPLES_PER_SUBJECT * (config.n_folds - 1) // config.n_folds
    assert [len(x) for x in seen] == [n_train] * (2 * config.n_folds)
    config.scale_features = True
    pad_train.evaluate(protocols=["baseline"], config=config)
    assert len(seen) == 3 * config.n_folds


def test_evaluate_accepts_custom_extractor(config: Settings) -> None:
    @pad_train.register_extractor("mean_std")
    def mean_std(image: pad_train.data.Image) -> np.ndarray:
        return np.array([image.mean(), image.std()])

    results = pad_train.evaluate(extractors=["mean_std", mean_std], config=config)
    assert set(results["extractor"]) == {"mean_std"}
