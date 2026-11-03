"""k-NN presentation attack detection trained on real or synthetic attack samples.

Typical use::

    import pad_train

    pad_train.settings.data_root = "/path/to/data"   # optional
    results = pad_train.evaluate(extractors=["fourier", "haralick"])   # pandas.DataFrame
"""

from importlib.metadata import PackageNotFoundError, version

from pad_train.config import DatasetConfig, DataSource, Settings, settings
from pad_train.data import PadData, download_data, load_data
from pad_train.evaluation import cross_validate, evaluate, evaluate_dataset
from pad_train.features import (
    FeatureExtractor,
    available_extractors,
    get_extractor,
    make_fourier_extractor,
    register_dataset_extractor,
    register_extractor,
)
from pad_train.metrics import PadMetrics, pad_metrics

try:
    __version__: str = version("pad_train")
except PackageNotFoundError:  # running from a source tree without installation
    __version__ = "0.0.0"

__all__ = [
    "DataSource",
    "DatasetConfig",
    "FeatureExtractor",
    "PadData",
    "PadMetrics",
    "Settings",
    "__version__",
    "available_extractors",
    "cross_validate",
    "download_data",
    "evaluate",
    "evaluate_dataset",
    "get_extractor",
    "load_data",
    "make_fourier_extractor",
    "pad_metrics",
    "register_dataset_extractor",
    "register_extractor",
    "settings",
]
