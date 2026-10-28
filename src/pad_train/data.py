"""Dataset download and loading."""

from __future__ import annotations

import http.client
import io
import json
import logging
import shutil
import tempfile
import time
import urllib.error
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

#: Seconds without any data after which a connection counts as broken.
_TIMEOUT: float = 60.0

#: Attempts per download step before giving up; the wait doubles each time.
_ATTEMPTS: int = 8

#: Gaps between needed parts of an archive up to this size are downloaded
#: along with them, which saves a request per part.
_MAX_GAP: int = 4 << 20


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


def _request(url: str, start: int = 0, end: int | None = None) -> http.client.HTTPResponse:
    """Open ``url``, from byte ``start`` (to ``end``, inclusive) if either is given."""
    headers = {"User-Agent": "pad-train"}
    if start or end is not None:
        headers["Range"] = f"bytes={start}-{'' if end is None else end}"
    response: http.client.HTTPResponse = urllib.request.urlopen(
        urllib.request.Request(url, headers=headers), timeout=_TIMEOUT
    )
    return response


def _retrying[T](action: Callable[[], T], what: str) -> T:
    """Run ``action``; retry network errors, waiting 2, 4, 8, ... (at most 60) seconds."""
    for attempt in range(1, _ATTEMPTS + 1):
        try:
            return action()
        except (OSError, http.client.HTTPException) as error:
            permanent = isinstance(error, urllib.error.HTTPError) and error.code < 500
            if permanent or attempt == _ATTEMPTS:
                raise
            wait = min(2**attempt, 60)
            logger.warning(
                "%s: %s; retry %d of %d in %d s", what, error, attempt, _ATTEMPTS - 1, wait
            )
            time.sleep(wait)
    raise AssertionError("unreachable")


class _Progress:
    """Logs how much of a download is done, with speed and remaining time."""

    def __init__(self, what: str, total: int, done: int = 0) -> None:
        self.what, self.total, self.done = what, total, done
        self.started, self.start_done, self.last = time.monotonic(), done, 0.0

    def add(self, size: int) -> None:
        self.done += size
        now = time.monotonic()
        if now - self.last >= 30 or self.done >= self.total:
            self.last = now
            rate = (self.done - self.start_done) / max(now - self.started, 1e-9)
            left = (self.total - self.done) / rate if rate > 0 else float("inf")
            logger.info(
                "  %s: %.0f of %.0f MB (%.0f %%), %.1f MB/s, about %s left",
                self.what,
                self.done / 1e6,
                self.total / 1e6,
                100 * self.done / max(self.total, 1),
                rate / 1e6,
                f"{left / 60:.0f} min" if left != float("inf") else "?",
            )


def _download(url: str, target: Path) -> None:
    """Stream ``url`` to ``target`` through a ``.part`` file.

    An interrupted download resumes where it stopped, in this call (after a
    network error) as well as in a later one (after the program was stopped),
    if the server supports byte ranges.
    """
    partial = target.with_name(target.name + ".part")
    logger.info("Downloading %s", url)
    progress: list[_Progress] = []

    def attempt() -> None:
        done = partial.stat().st_size if partial.is_file() else 0
        with _request(url, start=done) as response:
            if done and response.status != 206:  # the server ignored the range
                done = 0
            size = int(response.headers.get("Content-Length") or 0)
            if not progress:
                progress.append(_Progress(target.name, done + size, done))
            progress[0].done = done
            received = 0
            with partial.open("ab" if done else "wb") as out:
                while chunk := response.read(_CHUNK_SIZE):
                    out.write(chunk)
                    received += len(chunk)
                    progress[0].add(len(chunk))
            # A dropped connection can end the stream without an error.
            if size and received < size:
                raise OSError(f"connection closed after {received} of {size} bytes")

    _retrying(attempt, f"Downloading {url}")
    partial.replace(target)


class _RemoteFile(io.RawIOBase):
    """A remote file read through byte ranges; every byte read is copied into ``mirror``."""

    def __init__(self, url: str, size: int, mirror: io.BufferedRandom) -> None:
        self.url, self.size, self.mirror, self.position = url, size, mirror, 0

    def seekable(self) -> bool:
        return True

    def readable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.position

    def seek(self, offset: int, whence: int = 0) -> int:
        self.position = {0: offset, 1: self.position + offset, 2: self.size + offset}[whence]
        return self.position

    def read(self, size: int = -1) -> bytes:
        end = self.size if size < 0 else min(self.size, self.position + size)
        if end <= self.position:
            return b""
        start = self.position

        def fetch() -> bytes:
            with _request(self.url, start, end - 1) as response:
                return bytes(response.read())

        data = _retrying(fetch, f"Reading {self.url}")
        self.mirror.seek(start)
        self.mirror.write(data)
        self.position += len(data)
        return data

    def readinto(self, buffer: bytearray | memoryview) -> int:  # type: ignore[override]
        data = self.read(len(buffer))
        buffer[: len(data)] = data
        return len(data)


def _matches(name: str, members: Sequence[str]) -> bool:
    """Return whether the archive entry ``name`` lies in one of the ``members`` folders."""
    return any(name == m or name.startswith(m.rstrip("/") + "/") for m in members)


def _remote_size(url: str) -> int | None:
    """Return the size of ``url`` if its server answers byte-range requests, else ``None``."""
    if not url.startswith(("http://", "https://")):
        return None
    try:
        with _request(url, 0, 0) as response:
            content_range = response.headers.get("Content-Range", "")
            if response.status != 206 or "/" not in content_range:
                return None
            return int(content_range.rsplit("/", 1)[1])
    except (OSError, http.client.HTTPException, ValueError):
        return None


def _download_members(url: str, target: Path, members: Sequence[str]) -> bool:
    """Download only the parts of the remote zip ``url`` that hold ``members``.

    The result at ``target`` is a sparse copy of the archive: the directory at
    its end and the needed entries are real, everything else is a hole that
    takes no disk space. Finished parts are noted in a ``.ranges.json`` file,
    so a stopped download resumes with the first missing part.

    Returns:
        ``False`` if the server does not support byte ranges; nothing is
        downloaded then.
    """
    size = _remote_size(url)
    if size is None:
        return False
    partial = target.with_name(target.name + ".ranges")
    state_file = target.with_name(target.name + ".ranges.json")
    state = json.loads(state_file.read_text()) if state_file.is_file() and partial.is_file() else {}
    if state.get("size") != size:
        state = {"size": size, "done": []}
        with partial.open("wb") as out:
            out.truncate(size)

    with partial.open("r+b") as mirror:
        logger.info("Reading the table of contents of %s", url)
        remote = _RemoteFile(url, size, mirror)
        with zipfile.ZipFile(remote) as zip_file:
            entries = sorted(zip_file.infolist(), key=lambda info: info.header_offset)
            directory_start = zip_file.start_dir
        ends = [e.header_offset for e in entries[1:]] + [directory_start]
        needed = [(e.header_offset, end) for e, end in zip(entries, ends, strict=True)]
        needed = [
            span for span, e in zip(needed, entries, strict=True) if _matches(e.filename, members)
        ]

        spans: list[list[int]] = []
        for start, end in needed:
            if spans and start - spans[-1][1] <= _MAX_GAP:
                spans[-1][1] = end
            else:
                spans.append([start, end])
        done = {tuple(span) for span in state["done"]}
        todo = [span for span in spans if tuple(span) not in done]
        total = sum(end - start for start, end in spans)
        logger.info(
            "Downloading %.0f MB of %.0f MB (%d parts) from %s",
            total / 1e6,
            size / 1e6,
            len(spans),
            url,
        )
        progress = _Progress(target.name, total, total - sum(e - s for s, e in todo))

        for start, end in todo:
            position = [start]

            def fetch(position: list[int] = position, start: int = start, end: int = end) -> None:
                with _request(url, position[0], end - 1) as response:
                    if response.status != 206:
                        raise OSError("the server stopped answering byte ranges")
                    mirror.seek(position[0])
                    while chunk := response.read(_CHUNK_SIZE):
                        mirror.write(chunk)
                        position[0] += len(chunk)
                        progress.add(len(chunk))
                if position[0] < end:
                    raise OSError(f"connection closed after {position[0] - start} bytes")

            _retrying(fetch, f"Downloading {url}")
            mirror.flush()
            state["done"].append([start, end])
            state_file.write_text(json.dumps(state))

    partial.replace(target)
    state_file.unlink()
    return True


def rotate_images(folder: Path, turns: int) -> int:
    """Turn every image below ``folder`` by ``turns`` quarter turns counter-clockwise.

    Returns:
        The number of images turned.
    """
    files = image_files(folder)
    for path in files:
        with PILImage.open(path) as image:
            turned = np.rot90(np.asarray(image), turns)
        PILImage.fromarray(np.ascontiguousarray(turned)).save(path)
    return len(files)


def _extract(
    archive: Path,
    target: Path,
    archive_root: str | None,
    members: Sequence[str] | None = None,
    rotate: dict[str, int] | None = None,
) -> None:
    """Extract ``archive`` so that ``archive_root`` (or its top level) becomes ``target``.

    Extraction happens in a temporary sibling directory which is moved into
    place at the end, so an existing ``target`` is always complete. With
    ``members`` only those folders (relative to the archive's top level) are
    extracted; ``rotate`` turns the images of some folders, see
    :attr:`DataSource.rotate`.
    """
    logger.info("Extracting %s", archive.name)
    with tempfile.TemporaryDirectory(dir=target.parent, prefix=f".{target.name}-") as tmp:
        staging = Path(tmp) / "content"
        with zipfile.ZipFile(archive) as zip_file:
            names = None
            if members is not None:
                names = [n for n in zip_file.namelist() if _matches(n, members)]
                if not names:
                    raise FileNotFoundError(f"None of {list(members)} found in {archive.name}")
            zip_file.extractall(staging, members=names)
        for folder, turns in (rotate or {}).items():
            if (staging / folder).is_dir():
                logger.info(
                    "Turning %d images in %s", rotate_images(staging / folder, turns), folder
                )
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
        partly = source.members is not None and _download_members(
            source.url, archive, source.members
        )
        if not partly:
            _download(source.url, archive)
    _extract(archive, path, source.archive_root, source.members, source.rotate)
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
            configured ones, an empty sequence loads none. Sources that do not
            cover the dataset (see :meth:`DataSource.covers`) are skipped.
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
    real_root = ensure_source(config.real_source_of(dataset), download, config) / layout.real_dir

    # Subject = name of the folder a capture lives in; key = what makes a bona
    # fide sample and an attack sample "belong together".
    files: dict[str, list[Path]] = {"real": [], "spoof": []}
    for subset in layout.subsets:
        for kind, kind_dir in zip(files, layout.kind_dirs, strict=True):
            files[kind].extend(image_files(real_root / subset / kind_dir))

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
    skipped = [n for n in names if not config.synthetic_sources[n].covers(dataset)]
    if skipped:
        logger.info("%s: no samples in %s, skipped", dataset, ", ".join(skipped))
    synthetic_images: dict[str, list[Image]] = {}
    for name in (n for n in names if n not in skipped):
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
