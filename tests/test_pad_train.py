"""Tests for metrics, feature extractors, data handling and the evaluation."""

from __future__ import annotations

import numpy as np
import pytest

import pad_train
from pad_train import Settings
from pad_train.features import fourier_features, lbp_features

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
    assert lbp_features(image).shape == (256,)
    assert np.isclose(lbp_features(image).sum(), 1.0)


def test_fourier_band_energies_sum_to_total_power() -> None:
    image = np.random.default_rng(1).integers(0, 256, size=(16, 16)).astype(np.uint8)
    features = fourier_features(image, n_bands=4, log_compression=False, normalize=False)
    spectrum = np.fft.fftshift(np.fft.fft2(image))
    # Everything but the single corner bin belongs to exactly one band.
    assert np.isclose(features.sum(), (np.abs(spectrum) ** 2).sum() - np.abs(spectrum[0, 0]) ** 2)


def test_registry() -> None:
    assert {"fourier", "lbp"} <= set(pad_train.available_extractors())
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
    results = pad_train.evaluate(extractors=["fourier", "lbp"], config=config)
    assert list(results.columns) == [
        "dataset", "extractor", "protocol", "train_attack", "APCER", "BPCER", "ACER",
    ]  # fmt: skip
    assert list(results["train_attack"]) == ["spoof", "synth"] * 2
    assert results[["APCER", "BPCER", "ACER"]].to_numpy().max() <= 1.0
    # Noise vs. smoothed noise is trivially separable in the Fourier domain.
    assert (results.loc[results["extractor"] == "fourier", "ACER"] < 0.1).all()

    per_fold = pad_train.evaluate(per_fold=True, protocols=["baseline"], config=config)
    assert sorted(per_fold["fold"]) == list(range(config.n_folds))


def test_evaluate_accepts_custom_extractor(config: Settings) -> None:
    @pad_train.register_extractor("mean_std")
    def mean_std(image: pad_train.data.Image) -> np.ndarray:
        return np.array([image.mean(), image.std()])

    results = pad_train.evaluate(extractors=["mean_std", mean_std], config=config)
    assert set(results["extractor"]) == {"mean_std"}
