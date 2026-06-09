# HEGM: Hierarchical Exponential-Gaussian Mixtures for Watch-Time Distribution Prediction

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776ab?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.11.0-ee4c2c?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-Apache--2.0-9cf?style=flat-square)](https://opensource.org/licenses/Apache-2.0)

**Sofia Gulevskaia** ¹ ²  ·  **Mikhail Trapeznikov** ¹ ²  ·  **Aleksandr Poslavsky** ¹ ²  ·  **Alexander D'yakonov** ¹

¹ AI VK, VK, Moscow, Russia  ·  ² Lomonosov Moscow State University, Moscow, Russia

[Quick start](#quick-start)  ·  [Results](#results)  ·  [Training](#training)  ·  [Data](#data)  ·  [Citation](#citation)

---

HEGM is a distributional watch-time prediction head for short-video recommendation.
It models the full conditional watch-time distribution as an exponential component
for quick skips combined with a Gaussian mixture over `K` components for engaged
watching, separated by a learned skip gate. This repository releases the code,
preprocessing scripts, dataset splits, configurations, and pre-trained checkpoints to
reproduce all KuaiRec and VK-LSVD experiments reported in the paper.

## Quick start

Three commands reproduce the numbers in the [results table](#results) without touching
any raw data:

```bash
pip install -r requirements.txt
python scripts/download_artifacts.py --all
python evaluate.py --config configs/vklsvd_k3.yaml --checkpoint checkpoints/vklsvd_k3/last.ckpt
```

The second command downloads the processed dataset splits to `data/processed/` and the
pre-trained checkpoints to `checkpoints/`. The third command then evaluates the
`vklsvd_k3` checkpoint on the corresponding test split and prints `MSE 480.0518 / MAE
15.7046 / XAUC 0.6202`, which is the `vklsvd_k3` row in the results table. Replacing the
config and the checkpoint with any of the other seven `<dataset>_k<K>` pairs reproduces
the remaining rows.

Rebuilding the data from raw sources or training a model from scratch is also
supported; see [Data](#data) and [Training](#training).

## Results

Each row is the output of `evaluate.py` on the corresponding released checkpoint:

```bash
python evaluate.py --config configs/<name>.yaml --checkpoint checkpoints/<name>/last.ckpt
```

| Dataset  | K  | MSE (sec²) | MAE (sec) | XAUC   | Config / checkpoint name |
| -------- | -- | ---------- | --------- | ------ | ------------------------ |
| VK-LSVD  | 3  | 480.0518   | 15.7046   | 0.6202 | `vklsvd_k3`              |
| VK-LSVD  | 6  | 478.9167   | 15.7449   | 0.6206 | `vklsvd_k6`              |
| VK-LSVD  | 9  | 479.8697   | 15.7717   | 0.6197 | `vklsvd_k9`              |
| VK-LSVD  | 12 | 479.6837   | 15.7679   | 0.6202 | `vklsvd_k12`             |
| KuaiRec  | 3  | 59.2050    | 4.5223    | 0.5625 | `kuairec_k3`             |
| KuaiRec  | 6  | 59.3619    | 4.5056    | 0.5600 | `kuairec_k6`             |
| KuaiRec  | 9  | 59.2296    | 4.5099    | 0.5555 | `kuairec_k9`             |
| KuaiRec  | 12 | 59.2950    | 4.5094    | 0.5537 | `kuairec_k12`            |

## Repository layout

```
hegm/                         Model and core components
  ├── model.py                  Embedders, DCN encoder, HEGM decoder, ranker
  ├── hegm.py                   HEGM decoding and training objective
  ├── data.py                   Dataset and Lightning data module
  ├── features.py               Bucket edges and categorical cardinalities
  └── metrics.py                XAUC
scripts/
  ├── download_artifacts.py     Download processed splits and checkpoints
  ├── preprocess_kuairec.py     Build KuaiRec splits from raw data
  └── preprocess_vklsvd.py      Build VK-LSVD splits from raw data
configs/                      One config per dataset and K in {3, 6, 9, 12}
train.py
evaluate.py
requirements.txt
```

## Requirements

```bash
pip install -r requirements.txt
```

Tested on Python 3.10 and CUDA 12.3 with NVIDIA H100 GPUs; a single GPU is sufficient
for all public-dataset experiments. All package versions are pinned in
`requirements.txt`.

## Data

There are two ways to obtain the data. Use Option A to reproduce the released numbers
exactly; use Option B to rebuild the splits from the original sources.

### Option A — Download processed splits and checkpoints

`scripts/download_artifacts.py` fetches two archives from Google Drive with `gdown` and
unpacks them in place: `splits.zip` into `data/processed/` (the dataset splits used by
the model) and `checkpoints.zip` into `checkpoints/` (the pre-trained models behind
the [results table](#results)).

```bash
python scripts/download_artifacts.py --all          # both splits and checkpoints
python scripts/download_artifacts.py --splits       # only splits
python scripts/download_artifacts.py --checkpoints  # only checkpoints
```

After unpacking the layout is:

```
data/processed/
├── KuaiRec/                          train.parquet, val.parquet, test.parquet, meta.json
└── VK-LSVD/up0.01_ir0.01/            train.parquet, val.parquet, test.parquet, meta.json

checkpoints/
├── kuairec_k{3,6,9,12}/last.ckpt
└── vklsvd_k{3,6,9,12}/last.ckpt
```

This is the layout that `train.py` and `evaluate.py` expect; the configs point to the
same paths.

### Option B — Rebuild splits from raw data

**VK-LSVD** ([deepvk/VK-LSVD](https://huggingface.co/datasets/deepvk/VK-LSVD)). The raw
`up0.01_ir0.01` subsample is downloaded automatically from the Hugging Face Hub. The
script also fits the iALS features used by the model.

```bash
python scripts/preprocess_vklsvd.py
# writes data/processed/VK-LSVD/up0.01_ir0.01/{train,val,test}.parquet and meta.json
```

**KuaiRec** ([kuairec.com](https://kuairec.com)). Download the *big-matrix* release
(the small-matrix version does not contain the required user features) and place the
following files in `data/raw/KuaiRec/`:

```
data/raw/KuaiRec/
├── big_matrix.csv
├── user_features.csv
├── item_daily_features.csv
├── item_categories.csv
└── kuairec_caption_category.csv
```

Then run:

```bash
python scripts/preprocess_kuairec.py
# writes data/processed/KuaiRec/{train,val,test}.parquet and meta.json
```

Both datasets are split chronologically (KuaiRec 80/10/10 by interactions; VK-LSVD
5/1/1 by weeks) and targets are globally normalized by the 99th percentile of watch
time (60 s for KuaiRec, 157 s for VK-LSVD), stored in `meta.json`.

## Training

```bash
python train.py --config configs/<name>.yaml
```

| Argument     | Required | Description                                                     |
| ------------ | -------- | --------------------------------------------------------------- |
| `--config`   | yes      | Path to a YAML config in `configs/`.                            |

All hyperparameters are read from the config: dataset paths, embedding dimension,
number of Gaussian components, batch size, learning rate, number of epochs, dropout.
The eight configs in `configs/` cover every `(dataset, K)` cell of the results table:
`{kuairec,vklsvd}_k{3,6,9,12}.yaml`.

Training uses `seed = 42`, the Adam optimizer, and 30 epochs. Dataset-specific
settings follow the paper: learning rate 1e-5 for KuaiRec and 1e-4 for VK-LSVD,
dropout 0.3 and 0.1 respectively, batch size 4096, embedding dimension 8. The final
checkpoint is written to `checkpoints/<dataset>_k<K>/last.ckpt`.

## Evaluation

```bash
python evaluate.py --config configs/<name>.yaml --checkpoint checkpoints/<name>/last.ckpt
```

| Argument       | Required | Description                                                  |
| -------------- | -------- | ------------------------------------------------------------ |
| `--config`     | yes      | Path to the YAML config used for the corresponding checkpoint. |
| `--checkpoint` | yes      | Path to the `.ckpt` file.                                    |

The script prints MSE (sec²), MAE (sec), and XAUC.

## Notes on reproducibility

Evaluation of a released checkpoint on the released splits is deterministic and
reproduces the numbers in the [results table](#results). Minor differences from the
values reported in the paper are possible due to non-deterministic GPU operations
during training.

Rebuilding the data from raw sources is supported. The KuaiRec preprocessing pipeline
is fully deterministic. The VK-LSVD pipeline fits iALS, whose factors depend on BLAS
threading and the hardware, so the splits used for the released checkpoints are
distributed directly via `download_artifacts.py --splits`.

## Citation

```bibtex
@inproceedings{gulevskaia2026hegm,
  title     = {Hierarchical Exponential-Gaussian Mixtures for Watch-Time Distribution Prediction},
  author    = {Gulevskaia, Sofia and Trapeznikov, Mikhail and Poslavsky, Aleksandr and D'yakonov, Alexander},
  booktitle = {Proceedings of the IEEE International Conference on Data Mining (ICDM)},
  year      = {2026}
}
```

## License

This repository is released under the Apache License 2.0 (see `LICENSE`). The KuaiRec
and VK-LSVD datasets are distributed under their own licenses by the respective
authors; please refer to the original releases for terms of use. If you use this code
in your research, we kindly ask that you cite the paper as described in
[Citation](#citation).
