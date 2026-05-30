# KGDirectAU

Repository layout
```
KGDirectAU_root/
├── base/
│   ├── evaluator.py        # Testing loop
│   ├── model.py            # Abstract Base
│   └── trainer.py          # Training loop
├── configs/
│   ├── config.py           # Argument parser
│   └── <model-dataset configs>.json # Hyperparameters
├── data/
│   ├── <dataset>/
│   ├── dataloader.py
│   ├── dataset.py
│   ├── dict_hub.py
│   └── preprocess.py
├── metrics/
│   ├── classification.py   # for Triple classification
│   └── ranking.py          # for Link prediction
├── models/
│   ├── encoders/      
│   │   └── <encoder models>.py
│   ├── losses/
│   │   ├── pointwise.py    # Sigmoid / Logistic
│   │   ├── pairwise.py     # Margin / Ranking
│   │   ├── listwise.py     # NCE / Cross-Entropy
│   │   └── <other loss formulas>.py
│   ├── samplers/
│   │   ├── uniform.py
│   │   └── <other sampling strategies>.py
│   └── strategies/         
│       ├── standard.py     # Simple encoder + uniform sampling + pairwise loss
│       ├── kgdirectau.py 
│       └── <other training strategies>.py 
├── utils/
│   ├── checkpoint.py   # Save/Load weights
│   ├── device.py       # GPU setup, parameter reporting, DDP unwrapping
│   └── logger.py       # ProgressMeter, AverageMeter, logging setup
├── logs/
│   └── <model-dataset logs>/
│       ├── train.log       # Training text output
│       ├── results.txt     # Final result metrics, best valid, time, configs
│       ├── best_model.mdl   # Best model's weights
│       └── last_model.mdl   # Last trained model's weights
├── main.py                 # THE START BUTTON
├── README.md
└── requirements.txt
```

Quickstart

0) Start from scratch by creating a virtual environment and activating it.

```bash
python -m venv .venv
# Linux/macOS
source .venv/bin/activate

# Windows (PowerShell)
.venv\Scripts\Activate.ps1
```

1) Install dependencies (recommended inside a virtualenv):

```bash
pip install -r requirements.txt
```

2) Preprocess the dataset (one-time setup).

```bash
# For WN18RR
python data/preprocess.py --dataset wn18rr

# For FB15k-237
python data/preprocess.py --dataset fb15k237
```

By default, the script reads from `data/<dataset>/` and writes the processed files into `data/<dataset>/preprocessed/`. It generates `train.txt.json`, `valid.txt.json`, `test.txt.json`, and optionally `valid_w_label.txt.json` / `test_w_label.txt.json` for triple classification tasks.

3) Train and evaluate with a configuration.

```bash
# Train with default WN18RR config
## SimKGC
python main.py --config-path configs/SimKGC_WN18RR.json

## DistMult
python main.py --config-path configs/DistMult_WN18RR.json

## ComplEx
python main.py --config-path configs/ComplEx_WN18RR.json

# Run only link prediction task
python main.py --config-path configs/SimKGC_WN18RR.json --task lp

# Override config values from command line
python main.py --config-path configs/SimKGC_WN18RR.json --batch-size 32 --num-epochs 100
```

Outputs are saved to `logs/<model-dataset>_<yyyy-mm-dd>_<hh-mm-ss>/` by default.

4) Train and evaluate from `main.py`.

```bash
python main.py --config-path configs/SimKGC_WN18RR.json
```

You can override any config value from the command line, but the JSON file remains the primary source of configuration. For example, `--task` can be used to run only link prediction, only triple classification, or both.

```bash
python main.py --config-path configs/SimKGC_WN18RR.json --task lp
```

Notes
- The repo is organized to separate encoders, loss formulations, samplers and training strategies so you can mix-and-match components.
- Ensure `torch`, `transformers`, `numpy`, and `tqdm` are installed in your environment before running training or evaluation.

Preprocessing (WN18RR)

- Input files:
	- [data/WN18RR/wordnet-mlj12-definitions.txt](data/WN18RR/wordnet-mlj12-definitions.txt): entity words and descriptions used to enrich examples.
	- [data/WN18RR/train.txt](data/WN18RR/train.txt): raw training triples (head\trelation\ttail).
	- [data/WN18RR/valid.txt](data/WN18RR/valid.txt), [data/WN18RR/test.txt](data/WN18RR/test.txt): raw validation/test triples.
	- [data/WN18RR/valid_w_label.txt](data/WN18RR/valid_w_label.txt), [data/WN18RR/test_w_label.txt](data/WN18RR/test_w_label.txt): labeled triples for triple-classification (auto-discovered when label paths are omitted).

- Output files (written to `data/WN18RR/preprocessed/` by `python data/preprocess.py --dataset wn18rr`):
	- [data/WN18RR/preprocessed/entities.json](data/WN18RR/preprocessed/entities.json): per-entity metadata (id, name, description).
	- [data/WN18RR/preprocessed/relations.json](data/WN18RR/preprocessed/relations.json): per-relation metadata.
	- [data/WN18RR/preprocessed/entity2id.json](data/WN18RR/preprocessed/entity2id.json): mapping from entity id string to integer index.
	- [data/WN18RR/preprocessed/relation2id.json](data/WN18RR/preprocessed/relation2id.json): mapping from relation id string to integer index.
	- Per-split JSON files named after the input basename with `.json` appended (e.g. `valid_w_label.txt.json`, `test_w_label.txt.json`, or `train.txt.json` if provided): enriched triples with text and description fields.

- Purpose / notes:
	- The preprocess script loads dataset-specific metadata (for WN18RR it uses `wordnet-mlj12-definitions.txt`) to provide human-readable entity names and descriptions used by text-based encoders.
	- It converts raw tab-separated triples into JSON `TripleExample` objects with fields `head`, `relation`, `tail`, `head_desc`, `tail_desc`, etc., and writes these into the `preprocessed/` folder.
	- The script auto-discovers labeled splits (`valid_w_label.txt`, `test_w_label.txt`) if label paths are not explicitly provided, but it does not automatically synthesize `train.txt.json`/`valid.txt.json`/`test.txt.json` unless the corresponding raw `*.txt` files are provided or paths are passed in. Running `python data/preprocess.py --dataset wn18rr` will pick up the common filenames in `data/WN18RR/`.
	- Output files are used by the training and evaluation pipeline (see `configs/*.json` `train_path` / `valid_path` / `test_path` settings).

Model source layout
-------------------
Each model implementation should be split into four composable pieces so the repo pipeline can mix-and-match components:

- **encoder**: model-specific encoder implementation (place under `models/encoders/`).
	- Example files: `models/encoders/transe_encoder.py`, `models/encoders/distmult_encoder.py`
- **loss**: loss function definitions (place under `models/losses/`).
	- Example files: `models/losses/pairwise.py`, `models/losses/kbgan_loss.py`
- **negative sampling**: negative-sampler implementations (place under `models/samplers/`).
	- Example files: `models/samplers/uniform.py`, `models/samplers/sans.py`
- **core logic / strategy**: training strategy tying encoder + loss + sampler (place under `models/strategies/`).
	- Example files: `models/strategies/transe.py`, `models/strategies/kbgan.py`

The `model_def` field in each JSON config should point to the strategy implementation (e.g. `models/strategies/transe.py`) and specify which encoder, loss, and sampler to use. This keeps the training loop (`main.py` / `base/trainer.py`) generic and lets you add new models by dropping well-formed components into these folders.

If you'd like, I can also update `requirements.txt` to pin versions or add a short `examples/` section with commands for FB15k-237 and Wikidata5M.