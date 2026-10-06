"""Command line interface: ``pad-train download`` and ``pad-train evaluate``."""

from __future__ import annotations

import argparse
import logging
from collections.abc import Sequence
from pathlib import Path

from pad_train.config import settings
from pad_train.data import download_data
from pad_train.evaluation import PROTOCOLS, evaluate
from pad_train.features import available_extractors


def _build_parser() -> argparse.ArgumentParser:
    """Create the argument parser."""
    parser = argparse.ArgumentParser(prog="pad-train", description=__doc__)
    parser.add_argument(
        "--data-root", type=Path, help=f"data directory (default: {settings.data_root})"
    )
    commands = parser.add_subparsers(dest="command", required=True)

    commands.add_parser("download", help="fetch all datasets that are not on disk yet")

    run = commands.add_parser("evaluate", help="run the evaluation and print the result table")
    run.add_argument("--datasets", nargs="+", choices=list(settings.datasets), help="default: all")
    run.add_argument("--extractors", nargs="+", default=["fourier"], choices=available_extractors())
    run.add_argument(
        "--synthetic", nargs="+", choices=list(settings.synthetic_sources), help="default: all"
    )
    run.add_argument("--protocols", nargs="+", default=list(PROTOCOLS), choices=PROTOCOLS)
    run.add_argument("--n-neighbors", type=int, help="k of the k-NN (default: per dataset)")
    run.add_argument("--per-fold", action="store_true", help="report every fold, not the mean")
    run.add_argument("--no-download", action="store_true", help="fail if data is missing")
    run.add_argument("--output", type=Path, help="also write the table to this CSV file")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    """Run the command line interface."""
    args = _build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    if args.data_root is not None:
        settings.data_root = args.data_root

    if args.command == "download":
        download_data()
        return

    results = evaluate(
        datasets=args.datasets,
        extractors=args.extractors,
        synthetic=args.synthetic,
        protocols=args.protocols,
        n_neighbors=args.n_neighbors,
        per_fold=args.per_fold,
        download=not args.no_download,
    )
    print(results.to_string(index=False, float_format="{:.4f}".format))
    if args.output is not None:
        results.to_csv(args.output, index=False)


if __name__ == "__main__":
    main()
