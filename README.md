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
.venv\Scripts\activate
```

1) Install dependencies (recommended inside a virtualenv):

```bash
pip install -r requirements.txt
```

2) Preprocess the dataset (one-time setup).

```bash
# For WN18RR
python data/preprocess.py --dataset wn18rr --data-dir data/WN18RR --output-dir data/WN18RR

# For FB15k-237
python data/preprocess.py --dataset fb15k237 --data-dir data/FB15k237 --output-dir data/FB15k237
```

This generates `train.txt.json`, `valid.txt.json`, `test.txt.json`, and optionally `valid_w_label.txt.json` / `test_w_label.txt.json` for triple classification tasks.

3) Train and evaluate with a configuration.

```bash
# Train with default WN18RR config
python main.py --config-path configs/SimKGC_WN18RR.json

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