"""Dataset download and loading."""

from __future__ import annotations

import logging
import shutil
import tempfile
import urllib.request
import zipfile
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from PIL import Image as PILImage

from pad_train.config import DataSource, Settings, settings

logger = logging.getLogger(__name__)

#: A grayscale image as a 2-D ``uint8`` array.
Image = NDArray[np.uint8]

#: File suffixes (lower case) that are read as images.
IMAGE_SUFFIXES: frozenset[str] = frozenset({".png", ".bmp", ".jpg", ".jpeg"})

#: Bytes read per chunk while downloading.
_CHUNK_SIZE: int = 1 << 20


@dataclass
class PadData:
    """All samples of one dataset, ready for evaluation.

    The bona fide and attack sets are balanced: only bona fide samples with a
    matching real attack sample are kept (and vice versa).

    Attributes:
        dataset: Dataset name.
        real: Bona fide (type 1) images.
        real_subjects: Subject id of every bona fide image.
        spoof: Real presentation attack (type 2) images.
        spoof_subjects: Subject id of every attack image.
        synthetic: Synthetic attack (type 3) images, keyed by source name. They
            carry no subject identity.
    """

    dataset: str
    real: list[Image]
    real_subjects: NDArray[np.str_]
    spoof: list[Image]
    spoof_subjects: NDArray[np.str_]
    synthetic: dict[str, list[Image]] = field(default_factory=dict)


def _download(url: str, target: Path) -> None:
    """Stream ``url`` to ``target``, going through a ``.part`` file."""
    partial = target.with_name(target.name + ".part")
    logger.info("Downloading %s", url)
    with urllib.request.urlopen(url) as response, partial.open("wb") as out:
        total = int(response.headers.get("Content-Length") or 0)
        done, next_report = 0, 0.1
        while chunk := response.read(_CHUNK_SIZE):
            out.write(chunk)
            done += len(chunk)
            if total and done / total >= next_report:
                logger.info("  %3.0f %% of %.0f MB", 100 * done / total, total / 1e6)
                next_report += 0.1
    partial.replace(target)


def _extract(archive: Path, target: Path, archive_root: str | None) -> None:
    """Extract ``archive`` so that ``archive_root`` (or its top level) becomes ``target``.

    Extraction happens in a temporary sibling directory which is moved into
    place at the end, so an existing ``target`` is always complete.
    """
    logger.info("Extracting %s", archive.name)
    with tempfile.TemporaryDirectory(dir=target.parent, prefix=f".{target.name}-") as tmp:
        staging = Path(tmp) / "content"
        with zipfile.ZipFile(archive) as zip_file:
            zip_file.extractall(staging)
        payload = staging / archive_root if archive_root else staging
        if not payload.is_dir():
            raise FileNotFoundError(f"'{archive_root}' not found in {archive.name}")
        shutil.move(payload, target)


def ensure_source(
    source: DataSource, download: bool = True, config: Settings | None = None
) -> Path:
    """Return the directory of ``source``, fetching it first if it is missing.

    Args:
        source: The data source to locate.
        download: Download and extract the archive if the directory is missing.
        config: Settings to use instead of the package-wide ones.

    Returns:
        The existing directory of the source.

    Raises:
        FileNotFoundError: The directory is missing and could not be fetched.
    """
    config = config or settings
    path = config.resolve(source)
    if path.is_dir():
        return path
    if not download or source.url is None:
        raise FileNotFoundError(
            f"Data source '{source.name}' not found at {path}. "
            "Run `pad-train download` or call load_data(download=True)."
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    archive = path.parent / Path(source.url).name
    if not archive.is_file():
        _download(source.url, archive)
    _extract(archive, path, source.archive_root)
    archive.unlink()
    return path


def download_data(config: Settings | None = None) -> None:
    """Fetch every configured data source that is not on disk yet."""
    config = config or settings
    for source in config.sources():
        if source.url is not None:
            logger.info("%s -> %s", source.name, ensure_source(source, config=config))


def image_files(folder: Path) -> list[Path]:
    """Return all image files below ``folder`` in a stable order."""
    return sorted(p for p in folder.rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES)


def parallel_map[T, R](function: Callable[[T], R], items: Sequence[T], workers: int = 1) -> list[R]:
    """Return ``[function(item) for item in items]``, computed by ``workers`` threads.

    Threads suit the work in this package: image decoding and the numerical
    libraries release the interpreter lock, and nothing has to be copied.
    """
    if workers <= 1 or len(items) <= 1:
        return [function(item) for item in items]
    with ThreadPoolExecutor(max_workers=workers) as executor:
        return list(executor.map(function, items))


def read_image(path: Path) -> Image:
    """Read an image file as a grayscale array."""
    with PILImage.open(path) as image:
        return np.asarray(image.convert("L"), dtype=np.uint8)


def load_data(
    dataset: str,
    synthetic: Sequence[str] | None = None,
    download: bool = True,
    config: Settings | None = None,
    workers: int = 1,
) -> PadData:
    """Load one dataset, downloading the underlying archives when necessary.

    Args:
        dataset: Dataset name, a key of :attr:`Settings.datasets`.
        synthetic: Names of the synthetic sources to load. ``None`` loads all
            configured ones, an empty sequence loads none.
        download: Fetch missing sources. If ``False`` a missing source raises.
        config: Settings to use instead of the package-wide ones.
        workers: Number of threads reading the images.

    Returns:
        The balanced bona fide / attack samples plus the synthetic samples.

    Raises:
        KeyError: Unknown dataset or synthetic source name.
        FileNotFoundError: A source is missing (and ``download`` is ``False``),
            or it contains no samples for this dataset.
    """
    config = config or settings
    layout = config.datasets[dataset]
    real_root = ensure_source(config.real_source, download, config) / layout.real_dir

    # Subject = name of the folder a capture lives in; key = what makes a bona
    # fide sample and an attack sample "belong together".
    files: dict[str, list[Path]] = {"real": [], "spoof": []}
    for subset in layout.subsets:
        for kind, found in files.items():
            found.extend(image_files(real_root / subset / kind))

    def key(path: Path) -> tuple[str, ...]:
        subject = path.parent.name
        return (subject, path.name) if layout.pair_by_filename else (subject,)

    # Balanced sets: drop everything without a counterpart in the other class.
    common = {key(p) for p in files["real"]} & {key(p) for p in files["spoof"]}
    real_files = [p for p in files["real"] if key(p) in common]
    spoof_files = [p for p in files["spoof"] if key(p) in common]
    if not real_files:
        raise FileNotFoundError(f"No matching real/spoof samples for {dataset} in {real_root}")

    names = list(config.synthetic_sources) if synthetic is None else list(synthetic)
    synthetic_images: dict[str, list[Image]] = {}
    for name in names:
        ensure_source(config.synthetic_sources[name], download, config)
        folder = config.synthetic_folder(name, dataset)
        synthetic_images[name] = parallel_map(read_image, image_files(folder), workers)
        if not synthetic_images[name]:
            raise FileNotFoundError(f"No synthetic samples for {dataset} in {folder}")

    return PadData(
        dataset=dataset,
        real=parallel_map(read_image, real_files, workers),
        real_subjects=np.array([p.parent.name for p in real_files]),
        spoof=parallel_map(read_image, spoof_files, workers),
        spoof_subjects=np.array([p.parent.name for p in spoof_files]),
        synthetic=synthetic_images,
    )
