# KGDirectAU — ComplEx / ComplEx-AU

Training and evaluation for **ComplEx** (adversarial BCE + filtered 1-N negatives) and **ComplEx-AU** (alignment-uniformity / KGAU).

## Repository layout

```
KGDirectAU_root/
├── base/
│   ├── evaluator.py        # Link prediction and triple classification
│   └── model.py            # KGE scorer / embedder / model bases
├── configs/
│   ├── config.py           # Argument parser and config resolution
│   └── ComplEx*.json       # Per-experiment hyperparameters
├── data/
│   ├── WN18RR/             # Raw WN18RR splits (download separately; see below)
│   ├── FB15k237/           # Raw FB15k-237 splits (download separately)
│   ├── hetionet_subset/    # Hetionet subset splits
│   ├── dataset.py
│   ├── dict_hub.py
│   └── preprocess.py
├── metrics/
│   ├── classification.py
│   └── ranking.py
├── models/
│   ├── embedders/lookup_embedder.py
│   ├── complex.py          # ComplEx scorer
│   ├── losses/
│   │   ├── adversarial_bce_loss.py
│   │   └── au_loss.py
│   ├── samplers/filtered_1_to_n_sampler.py
│   ├── strategies/
│   │   ├── negsamp_strategy.py
│   │   └── kgau_strategy.py
│   └── builder.py
├── scripts/
│   ├── hpo/                # Optuna search for ComplEx-AU
│   └── run_ComplEx-AU_WN18RR_chunked_nohup.sh
├── utils/
├── main.py
└── requirements.txt
```

## Quickstart

0) Create and activate a virtual environment.

```bash
python -m venv .venv
# Linux/macOS
source .venv/bin/activate
# Windows (PowerShell)
.venv\Scripts\Activate.ps1
```

1) Install dependencies (recommended inside a virtualenv).

PyTorch is **not** pinned in `requirements.txt`; install it first from the CUDA index that matches your driver (see comments in `requirements.txt`), then install the rest:

```bash
# Example (CUDA 12.6): pick the index URL that matches your setup
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126

pip install -r requirements.txt
```

Core runtime packages: `torch`, `numpy`, `tqdm`, `scikit-learn`. HPO scripts additionally use `optuna`, `pandas`, and `matplotlib`.

2) Download raw dataset files.

The repo ships only a subset of raw splits. Before preprocessing, place the full dataset under the preset directory (case-sensitive on Linux):

**WN18RR** (`data/WN18RR/`):
- `train.txt`, `valid.txt`, `test.txt` — tab-separated triples (`head\trelation\ttail`)
- `valid_w_label.txt`, `test_w_label.txt` — labeled triples for triple classification (partially checked in)
- `wordnet-mlj12-definitions.txt` — entity names and descriptions (required for text enrichment)

**FB15k-237** (`data/FB15k237/`):
- `train.txt`, `valid.txt`, `test.txt`
- `FB15k_mid2name.txt` — entity name mapping (partially checked in)

Standard sources: [WN18RR](https://github.com/TimDettmers/ConvE), [FB15k-237](https://web.informatik.uni-mannheim.de/LinkedData/FB15K237/).

3) Preprocess the dataset (one-time setup).

```bash
# For WN18RR  →  reads data/WN18RR/, writes data/WN18RR/preprocessed/
python data/preprocess.py --dataset wn18rr

# For FB15k-237  →  reads data/FB15k237/, writes data/FB15k237/preprocessed/
python data/preprocess.py --dataset fb15k237
```

The script generates `train.txt.json`, `valid.txt.json`, `test.txt.json`, and (when raw label files exist) `valid_w_label.txt.json` / `test_w_label.txt.json`. Preprocessed output is gitignored and must be generated locally.

4) Train and evaluate with a configuration.

ComplEx uses adversarial BCE negative sampling. ComplEx-AU swaps in `au_loss.py` + `kgau_strategy.py` (no sampler).

```bash
# WN18RR
python main.py --config-path configs/ComplEx_WN18RR_adversarial_bce.json
python main.py --config-path configs/ComplEx-AU_WN18RR.json

# FB15k-237
python main.py --config-path configs/ComplEx_FB15k237_adversarial_bce.json
python main.py --config-path configs/ComplEx-AU_FB15k237.json

# Hetionet subset
python main.py --config-path configs/ComplEx_hetionet_subset_adversarial_bce.json
python main.py --config-path configs/ComplEx-AU_hetionet_subset.json

# Run only link prediction; override config values from CLI
python main.py --config-path configs/ComplEx-AU_WN18RR.json --task lp
python main.py --config-path configs/ComplEx-AU_WN18RR.json --batch-size 32 --epochs 100

# Shorthand: @configs/foo.json expands to --config-path configs/foo.json
python main.py @configs/ComplEx_WN18RR_adversarial_bce.json
```

Outputs are saved to `logs/<model-dataset>_<yyyy-mm-dd>_<hh-mm-ss>/` by default. The JSON config is the primary source of settings; CLI flags override on re-parse.

## Preprocessing (WN18RR)

- **Input files** (in `data/WN18RR/`):
  - `wordnet-mlj12-definitions.txt`: entity words and descriptions used to enrich examples.
  - `train.txt`: raw training triples (`head\trelation\ttail`).
  - `valid.txt`, `test.txt`: raw validation/test triples.
  - `valid_w_label.txt`, `test_w_label.txt`: labeled triples for triple classification (auto-discovered when label paths are omitted).

- **Output files** (written to `data/WN18RR/preprocessed/`):
  - `entities.json`, `relations.json`: per-entity/relation metadata.
  - `entity2id.json`, `relation2id.json`: string-to-index mappings.
  - Per-split JSON files named after the input basename with `.json` appended (e.g. `train.txt.json`, `valid_w_label.txt.json`).

- **Notes:**
  - Preprocessing converts raw tab-separated triples into JSON objects with fields `head`, `relation`, `tail`, `head_desc`, `tail_desc`, etc.
  - Each split JSON is written only when the corresponding raw `*.txt` file is present (or an explicit path is passed).
  - Training configs resolve data paths from the `dataset` field via `configs/config.py` (e.g. `data/WN18RR/preprocessed/train.txt.json`).

## Model source layout

Each model is assembled from five composable pieces (`models/builder.py`):

- **embedder** (`models/embedders/lookup_embedder.py`): real/imaginary lookup tables.
- **scorer** (`models/complex.py`): Hermitian ComplEx scoring.
- **loss** (`models/losses/`): `adversarial_bce_loss.py` (ComplEx) or `au_loss.py` (ComplEx-AU).
- **sampler** (`models/samplers/filtered_1_to_n_sampler.py`): used by ComplEx; omitted for ComplEx-AU.
- **strategy** (`models/strategies/`): `negsamp_strategy.py` (ComplEx) or `kgau_strategy.py` (ComplEx-AU).

Embedder and scorer paths are inferred from the `model` field when omitted from JSON; see `configs/config.py`.

## Shipped configs and components

| Model | Config (`configs/`) | Scorer | Loss | Sampler | Strategy |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **ComplEx** | `ComplEx_WN18RR_adversarial_bce.json` | `complex.py` | `adversarial_bce_loss.py` | `filtered_1_to_n_sampler.py` | `negsamp_strategy.py` |
| **ComplEx-AU** | `ComplEx-AU_WN18RR.json` | `complex.py` | `au_loss.py` | *(none)* | `kgau_strategy.py` |

FB15k-237 and hetionet-subset variants use the same wiring under `configs/ComplEx*_FB15k237*.json` and `configs/ComplEx*_hetionet_subset*.json`.

> KGAU (Alignment and Uniformity) strategies compute loss directly on positive batch embeddings and do not require negative sampling.
