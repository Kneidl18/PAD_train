"""Downloads: resuming, fetching only parts of an archive, and datasets from such archives."""

from __future__ import annotations

import http.server
import threading
import zipfile
from collections.abc import Iterator
from pathlib import Path
from typing import ClassVar

import numpy as np
import pytest
from PIL import Image

import pad_train
from pad_train import DatasetConfig, DataSource, Settings
from pad_train import data as data_module


class _RangeHandler(http.server.SimpleHTTPRequestHandler):
    """Serves files with byte ranges; ``fail_after`` bytes into a response it drops once."""

    fail_after: ClassVar[int | None] = None
    requests: ClassVar[list[str]] = []

    def log_message(self, *args: object) -> None:
        pass

    def do_GET(self) -> None:
        path = Path(self.translate_path(self.path))
        content = path.read_bytes()
        header = self.headers.get("Range")
        _RangeHandler.requests.append(header or "")
        if header:
            first, _, last = header.removeprefix("bytes=").partition("-")
            start, end = int(first), int(last) if last else len(content) - 1
            self.send_response(206)
            self.send_header("Content-Range", f"bytes {start}-{end}/{len(content)}")
        else:
            start, end = 0, len(content) - 1
            self.send_response(200)
        body = content[start : end + 1]
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        if _RangeHandler.fail_after is not None and len(body) > _RangeHandler.fail_after:
            self.wfile.write(body[: _RangeHandler.fail_after])
            _RangeHandler.fail_after = None
            self.close_connection = True
            return
        self.wfile.write(body)


@pytest.fixture
def server(tmp_path: Path) -> Iterator[tuple[str, Path]]:
    """An HTTP server with byte ranges for the folder ``served``."""
    served = tmp_path / "served"
    served.mkdir()

    class Handler(_RangeHandler):
        def __init__(self, *args: object, **kwargs: object) -> None:
            super().__init__(*args, directory=str(served), **kwargs)  # type: ignore[arg-type]

    _RangeHandler.requests = []
    _RangeHandler.fail_after = None
    httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}", served
    httpd.shutdown()


@pytest.fixture(autouse=True)
def no_waiting(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(data_module.time, "sleep", lambda seconds: None)


def _image(rng: np.random.Generator, smooth: bool) -> Image.Image:
    pixels = rng.integers(0, 256, size=(32, 48)).astype(np.float64)
    if smooth:
        pixels = (pixels + np.roll(pixels, 1, 0) + np.roll(pixels, 1, 1)) / 3
    return Image.fromarray(pixels.astype(np.uint8))


def _archive(path: Path, entries: dict[str, Image.Image | bytes]) -> None:
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_STORED) as zip_file:
        for name, item in entries.items():
            if isinstance(item, bytes):
                zip_file.writestr(name, item)
            else:
                target = path.parent / "_tmp.png"
                item.save(target)
                zip_file.write(target, name)
                target.unlink()


def test_download_resumes_after_a_dropped_connection(
    server: tuple[str, Path], tmp_path: Path
) -> None:
    url, served = server
    payload = bytes(range(256)) * 4000
    (served / "big.bin").write_bytes(payload)
    _RangeHandler.fail_after = 300_000

    data_module._download(f"{url}/big.bin", tmp_path / "big.bin")

    assert (tmp_path / "big.bin").read_bytes() == payload
    assert _RangeHandler.requests == ["", "bytes=300000-"]


def test_only_the_needed_members_are_downloaded(server: tuple[str, Path], tmp_path: Path) -> None:
    url, served = server
    noise = np.random.default_rng(0).integers(0, 256, 6_000_000, dtype=np.uint8).tobytes()
    _archive(
        served / "data.zip",
        {"A/keep/1.bin": b"one", "B/skip/big.bin": noise, "A/keep/2.bin": b"two", "C/x.bin": b"x"},
    )
    target = tmp_path / "out" / "data"
    target.parent.mkdir()
    _RangeHandler.fail_after = 10  # the first part breaks off and is resumed

    data_module._download_members(f"{url}/data.zip", target.parent / "data.zip", ["A/keep"])
    data_module._extract(target.parent / "data.zip", target, None, ["A/keep"])

    assert sorted(p.relative_to(target).as_posix() for p in target.rglob("*.bin")) == [
        "A/keep/1.bin",
        "A/keep/2.bin",
    ]
    assert (target / "A/keep/2.bin").read_bytes() == b"two"
    # The 6 MB entry between the two needed ones was not fetched.
    fetched = sum(
        int(end) - int(start) + 1
        for start, end in (r.removeprefix("bytes=").split("-") for r in _RangeHandler.requests if r)
    )
    assert fetched < 1_000_000


def test_servers_without_byte_ranges_fall_back_to_the_whole_archive(tmp_path: Path) -> None:
    archive = tmp_path / "data.zip"
    _archive(archive, {"A/1.bin": b"one"})
    assert not data_module._download_members(archive.as_uri(), tmp_path / "copy.zip", ["A"])


def test_dataset_with_its_own_real_source_and_partial_coverage(
    server: tuple[str, Path], config: Settings
) -> None:
    url, served = server
    rng = np.random.default_rng(1)
    entries: dict[str, Image.Image | bytes] = {}
    for subject in range(10):
        for sample in range(3):
            entries[f"HAND/genuine/{subject:03d}/g{sample}.png"] = _image(rng, smooth=False)
            entries[f"HAND/spoofed/{subject:03d}/s{sample}.png"] = _image(rng, smooth=True)
    for sample in range(30):
        entries[f"HAND/gan/variant/all/s{sample}.png"] = _image(rng, smooth=True)
        entries[f"HAND/unused/s{sample}.png"] = _image(rng, smooth=True)
    _archive(served / "prepared.zip", entries)

    members = ("HAND/genuine", "HAND/spoofed", "HAND/gan/variant/all")
    prepared = {"path": Path("prepared"), "url": f"{url}/prepared.zip", "members": members}
    config.datasets["HAND"] = DatasetConfig(
        "HAND",
        real_dir="HAND",
        synthetic_dir="HAND",
        real_source=DataSource("prepared", **prepared),  # type: ignore[arg-type]
        kind_dirs=("genuine", "spoofed"),
    )
    config.synthetic_sources["synth"].datasets = ("TOY",)
    config.synthetic_sources["gan"] = DataSource(
        "gan",
        folders={"HAND": "HAND/gan/variant/all"},
        **prepared,  # type: ignore[arg-type]
    )

    results = pad_train.evaluate(extractors=["fourier"], config=config)

    rows = set(zip(results["dataset"], results["train_attack"], strict=True))
    assert rows == {("TOY", "spoof"), ("TOY", "synth"), ("HAND", "spoof"), ("HAND", "gan")}
    assert not (config.data_root / "prepared" / "HAND" / "unused").exists()
    assert len(config.sources()) == 3  # the real, the toy and the shared prepared source


def test_turned_folders_are_turned_back_when_extracted(tmp_path: Path) -> None:
    pixels = np.arange(6 * 4, dtype=np.uint8).reshape(6, 4)
    stored = Image.fromarray(np.rot90(pixels, -1).copy())  # stored turned clockwise
    _archive(tmp_path / "data.zip", {"A/turned/x.png": stored, "A/plain/x.png": stored})

    data_module._extract(tmp_path / "data.zip", tmp_path / "out", None, None, {"A/turned": 1})

    turned = np.asarray(Image.open(tmp_path / "out/A/turned/x.png"))
    assert (turned == pixels).all()
    assert np.asarray(Image.open(tmp_path / "out/A/plain/x.png")).shape == (4, 6)
