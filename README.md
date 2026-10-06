# pad_train

k-NN presentation attack detection (PAD) for finger vein images. The package
answers one question: **can a PAD system that is trained on synthetic attack
samples reliably tell real attacks from bona fide samples?**

It returns APCER, BPCER and ACER as a `pandas.DataFrame`.

## Method

Three sample types are involved:

| Type | Meaning                                   | Source                       |
|------|-------------------------------------------|------------------------------|
| (1)  | bona fide ("real")                        | `Data_FV_Spoofing_WS2025_26` |
| (2)  | real presentation attack ("spoof")        | `Data_FV_Spoofing_WS2025_26` |
| (3)  | synthetic presentation attack (diffusion) | `AE_Res-diff`, `VAE-diff`    |

Each dataset (PLUS, IDIAP, SCUT) is evaluated with a five-fold cross-validation.
The folds partition the **subjects**, so no person is in training and test at
the same time. Only type (1) samples that have a type (2) counterpart are used
(balanced sets). Two protocols run on identical folds:

| `protocol`  | Training (4/5 of the subjects) | Test (remaining 1/5) |
|-------------|--------------------------------|----------------------|
| `baseline`  | (1) vs. (2)                    | (1) vs. (2)          |
| `synthetic` | (1) vs. (3)                    | (1) vs. (2)          |

The classifier is a k-nearest-neighbour classifier on top of an exchangeable
feature extractor. The reported metrics follow ISO/IEC 30107-3:

- **APCER** – share of attacks classified as bona fide
- **BPCER** – share of bona fide samples classified as attack
- **ACER** – mean of the two

Details worth knowing:

- The synthetic samples carry no subject identity, so they cannot be split by
  subject. By default each fold draws as many synthetic samples as it has bona
  fide training samples (`balance_synthetic=True`), which keeps the training set
  the same size as in the baseline. Synthetic samples are never part of a test set.
- "Has a counterpart" is decided per file name for PLUS and IDIAP (real and
  spoof captures share their names) and per subject for SCUT.
- The metrics are computed per fold and then averaged.

## Installation

As a dependency (SSH access to the repository required):

```bash
pip install "git+ssh://git@github.com/Kneidl18/PAD_train.git"          # latest main
pip install "git+ssh://git@github.com/Kneidl18/PAD_train.git@0.0.1"    # a release
```

For local development:

```bash
git clone git@github.com:Kneidl18/PAD_train.git
cd PAD_train
conda env create -f environment.yml    # installs the package in editable mode
conda activate pad_train
```

## Data

The three archives (about 3 GB in total) are downloaded on demand into the
data root and removed after extraction:

```text
<data root>/
├── Data_FV_Spoofing_WS2025_26/   # types (1) and (2)
├── AE_Res-diff/                  # type (3), residual auto encoder
└── VAE-diff/                     # type (3), variational auto encoder
```

The data root defaults to `./data` (relative to the working directory). Change
it in Python, through the environment, or on the command line:

```python
from pathlib import Path
import pad_train

pad_train.settings.data_root = Path("/mnt/datasets/finger_vein")
```

```bash
export PAD_TRAIN_DATA_ROOT=/mnt/datasets/finger_vein
pad-train --data-root /mnt/datasets/finger_vein download
```

All paths, URLs and dataset layouts live in
[`src/pad_train/config.py`](src/pad_train/config.py).

Fetching happens in one of two ways:

- **Explicitly:** `pad-train download` (or `pad_train.download_data()`).
- **Implicitly:** `load_data()` and `evaluate()` download whatever is missing
  and simply use what is already there. Pass `download=False` to forbid it.

> `pip` cannot run code at install time for wheels, so there is no
> `pip install --download-datasets`. Run `pad-train download` after installing.

## Usage

```python
import pad_train

results = pad_train.evaluate(extractors=["fourier", "haralick"])
print(results)
```

```text
dataset extractor  protocol train_attack   APCER   BPCER    ACER
   PLUS   fourier  baseline        spoof  ...
   PLUS   fourier synthetic       AE_Res  ...
   PLUS   fourier synthetic          VAE  ...
   ...
```

`train_attack` names the attack samples used for training: `spoof` for the
baseline, otherwise the synthetic source. All rates are fractions in `[0, 1]`.

Useful arguments of `evaluate()`:

| Argument            | Default        | Meaning                                         |
|---------------------|----------------|-------------------------------------------------|
| `datasets`          | all            | e.g. `["PLUS"]`                                 |
| `extractors`        | `("fourier",)` | registered names or callables                   |
| `synthetic`         | all            | e.g. `["VAE"]`                                  |
| `protocols`         | both           | `"baseline"` and/or `"synthetic"`               |
| `n_neighbors`       | per dataset    | `k` of the k-NN (PLUS 6, IDIAP 1, SCUT 1)       |
| `balance_synthetic` | `True`         | sub-sample the synthetic training samples       |
| `per_fold`          | `False`        | one row per fold instead of the mean            |
| `download`          | `True`         | fetch missing data                              |

Lower-level building blocks: `load_data()` returns the images of one dataset,
`evaluate_dataset()` evaluates images, `cross_validate()` evaluates precomputed
feature matrices.

The same from the command line:

```bash
pad-train evaluate --datasets PLUS IDIAP --extractors fourier haralick --output results.csv
pad-train evaluate --help
```

## Feature extractors

A feature extractor is any callable that maps a grayscale image (2-D `uint8`
array) to a 1-D feature vector whose length does not depend on the image size.

| Name       | Description                                                              |
|------------|--------------------------------------------------------------------------|
| `fourier`  | log energy of the magnitude spectrum in radial bands, thirds per dataset |
| `haralick` | 13 Haralick texture statistics, averaged over the four directions        |

### Fourier settings

`fourier` uses only the magnitude of the spectrum; the phase is discarded. The
30 bands are grouped into thirds (1 = low, 2 = middle, 3 = high frequencies),
and each dataset uses its own selection:

| Dataset | `fourier_thirds` | Bands |
|---------|------------------|-------|
| PLUS    | `(1, 2, 3)`      | 0–29  |
| IDIAP   | `(1, 2)`         | 0–19  |
| SCUT    | `(1,)`           | 0–9   |

`evaluate()` and the command line apply these automatically. To change them,
edit `fourier_thirds` in `_default_datasets()` in
[`src/pad_train/config.py`](src/pad_train/config.py), or override them at run time:

```python
pad_train.settings.datasets["SCUT"].fourier_thirds = (1, 2)
```

### Adding an extractor

Adding one takes a single decorator, in this package
(`src/pad_train/features/`) or in your own code:

```python
import numpy as np
import pad_train


@pad_train.register_extractor("mean_std")
def mean_std(image):
    return np.array([image.mean(), image.std()])


# a parametrised variant of a built-in one: the 10 lowest frequency bands
pad_train.register_extractor("fourier_low", pad_train.make_fourier_extractor(bands=range(10)))

pad_train.evaluate(extractors=["fourier", "fourier_low", "mean_std"])
```

## Additional synthetic sources

Any folder with the layout `<path>/<dataset>/spoof/samples/*.png` can be used
as a source of type (3) samples, for example post-processed versions of the
provided ones:

```python
from pathlib import Path
from pad_train import DataSource, evaluate, settings

settings.synthetic_sources["VAE_cleaned"] = DataSource("VAE_cleaned", Path("/data/VAE_cleaned"))
evaluate(synthetic=["VAE", "VAE_cleaned"])
```

The dataset folder names are `PLUS_matched`, `IDIAP` and `SCUT`.

## Development

```bash
ruff format .     # formatting
ruff check .      # linting (annotations and docstrings included)
mypy              # static type checking
pytest            # tests; they generate their own toy data, no download needed
```

`main` is protected. Work happens on `dev` (or on branches merged into `dev`)
and reaches `main` through a pull request. Releases are tags on `main`.
