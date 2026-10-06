"""Central configuration: where the data lives and how the datasets are laid out.

Everything is plain Python. To change a value, mutate the module-level
:data:`settings` object before calling into the package::

    from pathlib import Path
    from pad_train import settings

    settings.data_root = Path("/mnt/datasets/finger_vein")

The data root can also be set through the ``PAD_TRAIN_DATA_ROOT`` environment
variable, which is read once at import time.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

#: Environment variable overriding the default data root.
DATA_ROOT_ENV_VAR: str = "PAD_TRAIN_DATA_ROOT"


@dataclass
class DataSource:
    """A directory of samples, optionally backed by a downloadable zip archive.

    Attributes:
        name: Short identifier, used in logs and in the result table.
        path: Directory holding the per-dataset folders. Relative paths are
            resolved against :attr:`Settings.data_root`.
        url: Zip archive to fetch when ``path`` does not exist. ``None`` marks a
            purely local source (e.g. samples produced by your own code).
        archive_root: Folder inside the archive that holds the per-dataset
            folders. ``None`` if they sit at the top level of the archive.
    """

    name: str
    path: Path
    url: str | None = None
    archive_root: str | None = None


@dataclass
class DatasetConfig:
    """Layout and classifier settings of one finger vein dataset.

    Attributes:
        name: Dataset identifier (``"PLUS"``, ``"IDIAP"``, ``"SCUT"``).
        real_dir: Dataset folder inside the real/spoof source.
        synthetic_dir: Dataset folder inside a synthetic source.
        subsets: Sub-folders of ``real_dir`` that each contain a ``real`` and a
            ``spoof`` folder. ``("",)`` means they sit directly in ``real_dir``.
        n_neighbors: ``k`` of the k-NN classifier.
        pair_by_filename: Real and spoof captures of the same finger share a
            file name, so bona fide samples can be matched to their attack
            sample one-to-one. If ``False`` the matching is done per subject.
    """

    name: str
    real_dir: str
    synthetic_dir: str
    subsets: tuple[str, ...] = ("",)
    n_neighbors: int = 1
    pair_by_filename: bool = False


def _default_data_root() -> Path:
    """Return the data root from the environment, falling back to ``./data``."""
    return Path(os.environ.get(DATA_ROOT_ENV_VAR, "data")).expanduser()


def _default_synthetic_sources() -> dict[str, DataSource]:
    """Return the two diffusion-based synthetic attack sources."""
    return {
        "AE_Res": DataSource(
            name="AE_Res",
            path=Path("AE_Res-diff"),
            url="https://cosy.sbg.ac.at/~uhl/AE_Res-diff.zip",
        ),
        "VAE": DataSource(
            name="VAE",
            path=Path("VAE-diff"),
            url="https://cosy.sbg.ac.at/~uhl/VAE-diff.zip",
        ),
    }


def _default_datasets() -> dict[str, DatasetConfig]:
    """Return the layout of the three supported datasets."""
    return {
        "PLUS": DatasetConfig(
            name="PLUS",
            real_dir="PLUS",
            synthetic_dir="PLUS_matched",
            n_neighbors=6,
            pair_by_filename=True,
        ),
        "IDIAP": DatasetConfig(
            name="IDIAP",
            real_dir="IDIAP/full",
            synthetic_dir="IDIAP",
            # ``test`` is unlabelled in the archive and therefore unusable.
            subsets=("train", "dev"),
            pair_by_filename=True,
        ),
        "SCUT": DatasetConfig(
            name="SCUT",
            real_dir="SCUT/full",
            synthetic_dir="SCUT",
            subsets=("train", "dev", "test"),
        ),
    }


@dataclass
class Settings:
    """All tunable values of the package.

    Attributes:
        data_root: Base directory for every relative :attr:`DataSource.path`.
        real_source: Source of the bona fide (type 1) and real attack (type 2)
            samples.
        synthetic_sources: Sources of synthetic attack (type 3) samples, keyed
            by name. Add an entry to evaluate another generator.
        synthetic_attack_subdir: Folder below a synthetic dataset folder that
            holds the attack samples.
        datasets: Supported datasets, keyed by name.
        n_folds: Number of subject-disjoint cross-validation folds.
        random_state: Seed for every random choice (synthetic sub-sampling).
    """

    data_root: Path = field(default_factory=_default_data_root)
    real_source: DataSource = field(
        default_factory=lambda: DataSource(
            name="real",
            path=Path("Data_FV_Spoofing_WS2025_26"),
            url="https://www.cosy.sbg.ac.at/~uhl/Data_FV_Spoofing_WS2025_26.zip",
            archive_root="Data_FV_Spoofing_WS2025_26",
        )
    )
    synthetic_sources: dict[str, DataSource] = field(default_factory=_default_synthetic_sources)
    synthetic_attack_subdir: str = "spoof/samples"
    datasets: dict[str, DatasetConfig] = field(default_factory=_default_datasets)
    n_folds: int = 5
    random_state: int = 42

    def resolve(self, source: DataSource) -> Path:
        """Return the absolute-or-root-relative directory of ``source``."""
        path = Path(source.path).expanduser()
        return path if path.is_absolute() else Path(self.data_root).expanduser() / path

    def synthetic_folder(self, source: str, dataset: str) -> Path:
        """Return the folder holding the attack samples of ``dataset`` in ``source``."""
        root = self.resolve(self.synthetic_sources[source])
        return root / self.datasets[dataset].synthetic_dir / self.synthetic_attack_subdir

    def sources(self) -> list[DataSource]:
        """Return every configured source, the real one first."""
        return [self.real_source, *self.synthetic_sources.values()]


#: Package-wide default settings. Mutate this object to reconfigure the package.
settings: Settings = Settings()
