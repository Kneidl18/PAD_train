"""Shared fixtures: a tiny on-disk dataset in the layout of the real archives."""

from __future__ import annotations

import zipfile
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from pad_train import DatasetConfig, DataSource, Settings

N_SUBJECTS = 10
SAMPLES_PER_SUBJECT = 3


def _save(path: Path, rng: np.random.Generator, smooth: bool) -> None:
    """Write a noise image; "attacks" are smoothed to make the classes separable."""
    pixels = rng.integers(0, 256, size=(32, 48)).astype(np.float64)
    if smooth:
        pixels = (pixels + np.roll(pixels, 1, 0) + np.roll(pixels, 1, 1)) / 3
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(pixels.astype(np.uint8)).save(path)


@pytest.fixture
def config(tmp_path: Path) -> Settings:
    """Settings pointing at a generated toy dataset; sources are zip archives."""
    rng = np.random.default_rng(0)
    build = tmp_path / "build"

    # Real source: wrapped in a top-level folder, like the original archive.
    for subject in range(N_SUBJECTS):
        for sample in range(SAMPLES_PER_SUBJECT):
            name = f"{subject:03d}/{subject:03d}_{sample}.png"
            _save(build / "real/Wrapped/TOY/real" / name, rng, smooth=False)
            _save(build / "real/Wrapped/TOY/spoof" / name, rng, smooth=True)
    # One bona fide sample without attack counterpart, must be dropped.
    _save(build / "real/Wrapped/TOY/real/000/000_unmatched.png", rng, smooth=False)

    # Synthetic source: dataset folders at the archive top level.
    for sample in range(40):
        _save(build / f"synth/TOY/spoof/samples/sample_{sample}.png", rng, smooth=True)

    archives = tmp_path / "archives"
    archives.mkdir()
    for name in ("real", "synth"):
        with zipfile.ZipFile(archives / f"{name}.zip", "w") as zip_file:
            for file in sorted((build / name).rglob("*.png")):
                zip_file.write(file, file.relative_to(build / name))

    return Settings(
        data_root=tmp_path / "data",
        real_source=DataSource(
            "real", Path("real"), (archives / "real.zip").as_uri(), archive_root="Wrapped"
        ),
        synthetic_sources={
            "synth": DataSource("synth", Path("synth"), (archives / "synth.zip").as_uri())
        },
        datasets={
            "TOY": DatasetConfig("TOY", real_dir="TOY", synthetic_dir="TOY", pair_by_filename=True)
        },
    )
