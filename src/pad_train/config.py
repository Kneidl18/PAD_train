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
        members: Folders of the archive (paths relative to its top level) to
            extract. ``None`` extracts everything. If the server supports
            byte ranges, only these parts of the archive are downloaded.
        folders: Per dataset, the folder below ``path`` that holds the attack
            samples. Overrides the default ``<synthetic_dir>/spoof/samples``.
            Only the datasets listed here are covered by the source.
        datasets: Datasets the source covers if ``folders`` is not given.
            ``None`` means all of them.
        rotate: Folders of the archive whose images are stored turned, with
            the number of quarter turns (counter-clockwise) that bring them
            into the orientation of the real samples. Applied once, when the
            archive is extracted.
    """

    name: str
    path: Path
    url: str | None = None
    archive_root: str | None = None
    members: tuple[str, ...] | None = None
    folders: dict[str, str] | None = None
    datasets: tuple[str, ...] | None = None
    rotate: dict[str, int] | None = None

    def covers(self, dataset: str) -> bool:
        """Return whether the source holds attack samples of ``dataset``."""
        if self.folders is not None:
            return dataset in self.folders
        return self.datasets is None or dataset in self.datasets


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
        fourier_best_thirds: Thirds of the Fourier magnitude spectrum used by
            the ``fourier_best`` extractor for this dataset: 1 = low,
            2 = middle, 3 = high frequencies. Change them in
            :func:`_default_datasets`.
        fourier_worst_thirds: The same for the ``fourier_worst`` extractor.
        real_source: Source of the bona fide and real attack samples of this
            dataset. ``None`` uses :attr:`Settings.real_source`.
        kind_dirs: Names of the bona fide and the real attack folder.
    """

    name: str
    real_dir: str
    synthetic_dir: str
    subsets: tuple[str, ...] = ("",)
    n_neighbors: int = 1
    pair_by_filename: bool = False
    fourier_best_thirds: tuple[int, ...] = (1, 2, 3)
    fourier_worst_thirds: tuple[int, ...] = (1, 2, 3)
    real_source: DataSource | None = None
    kind_dirs: tuple[str, str] = ("real", "spoof")


def _default_data_root() -> Path:
    """Return the data root from the environment, falling back to ``./data``."""
    return Path(os.environ.get(DATA_ROOT_ENV_VAR, "data")).expanduser()


#: Archive with the samples of four GANs, from the Salzburg master thesis on
#: CycleGAN and its derivatives.
PREPARED_URL: str = "https://www.cosy.sbg.ac.at/~uhl/Data_prepared.zip"

#: GAN source name -> its folder name in the archive (spelled as there).
PREPARED_GANS: dict[str, str] = {
    "CycleGAN": "spoofed_synthethic_cyclegan",
    "DistanceGAN": "spoofed_synthethic_distancegan",
    "DRIT": "spoofed_synthethic_drit",
    "StarGANv2": "spoofed_synthethic_stargan-v2",
}

#: Variant of the thesis used per GAN and database, see ``Description.pdf`` in
#: the archive, as ``<variant>/<style>``. The style is ``reference`` or, for
#: StarGANv2 only, ``latent``.
PREPARED_VARIANTS: dict[str, dict[str, str]] = {
    "CycleGAN": {"PLUS": "003/reference", "IDIAP": "009/reference", "SCUT": "008/reference"},
    "DistanceGAN": {"PLUS": "001/reference", "IDIAP": "009/reference", "SCUT": "008/reference"},
    "DRIT": {"PLUS": "003/reference", "IDIAP": "009/reference", "SCUT": "007/reference"},
    "StarGANv2": {"PLUS": "001/reference", "IDIAP": "009/latent", "SCUT": "007/reference"},
}


def _prepared_folder(gan: str, database: str) -> str:
    """Return the archive folder with all five folds, resized to the original size."""
    variant, style = PREPARED_VARIANTS[gan][database].split("/")
    return f"{database}/{PREPARED_GANS[gan]}/{variant}/all_rs/{style}"


def _prepared_source(name: str = "prepared", folders: dict[str, str] | None = None) -> DataSource:
    """Return a source backed by :data:`PREPARED_URL`; all of them share one folder."""
    members = tuple(
        _prepared_folder(gan, db) for gan, dbs in PREPARED_VARIANTS.items() for db in dbs
    )
    # The SCUT samples of all four GANs are stored turned clockwise
    # (288 x 640 instead of the 640 x 288 of the real SCUT images).
    rotate = {_prepared_folder(gan, "SCUT"): 1 for gan in PREPARED_VARIANTS}
    return DataSource(
        name=name,
        path=Path("Data_prepared"),
        url=PREPARED_URL,
        members=members,
        folders=folders,
        rotate=rotate,
    )


def _default_synthetic_sources() -> dict[str, DataSource]:
    """Return the two diffusion-based and the four GAN-based synthetic attack sources."""
    diffusion = ("PLUS", "IDIAP", "SCUT")
    sources = {
        "AE_Res": DataSource(
            name="AE_Res",
            path=Path("AE_Res-diff"),
            url="https://cosy.sbg.ac.at/~uhl/AE_Res-diff.zip",
            datasets=diffusion,
        ),
        "VAE": DataSource(
            name="VAE",
            path=Path("VAE-diff"),
            url="https://cosy.sbg.ac.at/~uhl/VAE-diff.zip",
            datasets=diffusion,
        ),
    }
    for name, databases in PREPARED_VARIANTS.items():
        folders = {db: _prepared_folder(name, db) for db in databases}
        sources[name] = _prepared_source(name, folders)
    return sources


def _default_datasets() -> dict[str, DatasetConfig]:
    """Return the layout of the three supported datasets.

    The Fourier thirds were chosen from all seven combinations by the mean ACER
    of the ``synthetic`` protocol over ``AE_Res`` and ``VAE``, ties broken by the
    baseline ACER (IDIAP: low alone ties with low + middle).
    """
    return {
        "PLUS": DatasetConfig(
            name="PLUS",
            real_dir="PLUS",
            synthetic_dir="PLUS_matched",
            n_neighbors=6,
            pair_by_filename=True,
            fourier_best_thirds=(1, 2),
            fourier_worst_thirds=(2,),
        ),
        "IDIAP": DatasetConfig(
            name="IDIAP",
            real_dir="IDIAP/full",
            synthetic_dir="IDIAP",
            # ``test`` is unlabelled in the archive and therefore unusable.
            subsets=("train", "dev"),
            pair_by_filename=True,
            fourier_best_thirds=(1, 2),
            fourier_worst_thirds=(2, 3),
        ),
        "SCUT": DatasetConfig(
            name="SCUT",
            real_dir="SCUT/full",
            synthetic_dir="SCUT",
            subsets=("train", "dev", "test"),
            fourier_best_thirds=(1,),
            fourier_worst_thirds=(3,),
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
        scale_features: Standardise the features with a scaler fitted on the
            training samples of each fold before the k-NN. None defers to the
            extractor's ``requires_scaling`` flag (False if unset).
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
    scale_features: bool | None = None

    def resolve(self, source: DataSource) -> Path:
        """Return the absolute-or-root-relative directory of ``source``."""
        path = Path(source.path).expanduser()
        return path if path.is_absolute() else Path(self.data_root).expanduser() / path

    def synthetic_folder(self, source: str, dataset: str) -> Path:
        """Return the folder holding the attack samples of ``dataset`` in ``source``."""
        data_source = self.synthetic_sources[source]
        root = self.resolve(data_source)
        if data_source.folders is not None:
            return root / data_source.folders[dataset]
        return root / self.datasets[dataset].synthetic_dir / self.synthetic_attack_subdir

    def real_source_of(self, dataset: str) -> DataSource:
        """Return the source of the bona fide and real attack samples of ``dataset``."""
        return self.datasets[dataset].real_source or self.real_source

    def sources(self) -> list[DataSource]:
        """Return every configured source once, the real ones first.

        Sources that share a folder (several GANs from one archive) are listed
        once, since fetching one of them fetches all.
        """
        real = [self.real_source, *(d.real_source for d in self.datasets.values() if d.real_source)]
        unique: dict[Path, DataSource] = {}
        for source in [*real, *self.synthetic_sources.values()]:
            unique.setdefault(self.resolve(source), source)
        return list(unique.values())


#: Package-wide default settings. Mutate this object to reconfigure the package.
settings: Settings = Settings()
