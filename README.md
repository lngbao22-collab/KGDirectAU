# KGDirectAU

Lightweight, modular KG training and evaluation framework extracted from SimKGC.

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
│       ├── train.log       # Text output
│       ├── results.txt     # Final metrics
│       └── best_model.mdl   # Model weights
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

2) Prepare the dataset once.

`main.py` consumes preprocessed split files such as `train.txt.json`, `valid.txt.json`, `test.txt.json`, and, when available, `valid_w_label.txt.json` / `test_w_label.txt.json` for triple classification. Run `data/preprocess.py` once per dataset to generate those JSON files from the raw `.txt` splits under `data/<dataset>/`.

3) Pick a JSON config in `configs/` and use it with `main.py`.

The default WN18RR config is [configs/SimKGC_WN18RR.json](configs/SimKGC_WN18RR.json). Pass it with `--config-path`; this file is the main place to define the model protocol through `model_def`, which points to the encoder, loss, sampler, and strategy implementation in `models/`.

When `output_dir` is omitted or left at the default placeholder, `config.py` resolves it to a timestamped run directory such as `logs/SimKGC_WN18RR_<yyyy-mm-dd>_<hh-mm-ss>/`.

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

If you'd like, I can also update `requirements.txt` to pin versions or add a short `examples/` section with commands for FB15k-237 and Wikidata5M.