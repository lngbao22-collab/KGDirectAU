# KGDirectAU Training Pipeline

This document explains how a training run flows through the repository: which components exist, what each one does, and how they connect.

---

## 1. End-to-end lifecycle

```mermaid
flowchart TB
    subgraph prep["Data preparation (one-time)"]
        RAW["Raw splits<br/>train.txt, valid.txt, test.txt"]
        PRE["data/preprocess.py"]
        JSON["Preprocessed JSON<br/>*.txt.json, entities.json, relation2id.json"]
        RAW --> PRE --> JSON
    end

    subgraph config["Configuration"]
        CFG["configs/*.json<br/>experiment recipe"]
        PARSER["configs/config.py<br/>build_parser() + CLI overrides"]
        ARGS["args namespace"]
        CFG --> PARSER --> ARGS
    end

    subgraph run["Runtime (main.py)"]
        HW["utils/device.py<br/>init_hardware()"]
        BUILD["models/builder.py<br/>build_pipeline()"]
        TRAIN["Strategy.train_loop()"]
        CKPT["utils/checkpoint.py<br/>best_model.mdl / last_model.mdl"]
        EVAL["base/evaluator.py<br/>Evaluator"]
        OUT["logs/&lt;run&gt;/<br/>run.log, results.txt"]
    end

    JSON --> ARGS
    ARGS --> HW --> BUILD --> TRAIN --> CKPT
    CKPT --> EVAL --> OUT
    TRAIN --> OUT
```

**Entry point:** `main.py` loads `args` from JSON + CLI, initializes GPU/seed, calls `build_pipeline()`, runs `trainer.train_loop()`, then loads the best checkpoint into `Evaluator` for test metrics.

---

## 2. The five composable pillars

Every experiment is assembled in `build_pipeline()` from five swappable pieces. The JSON config points to each one via path fields.

```mermaid
flowchart LR
    subgraph pillars["Five pillars (models/builder.py)"]
        E["1. Embedder<br/>model_embedder_path"]
        S["2. Scorer<br/>model_scorer_path"]
        L["3. Loss<br/>model_loss_path"]
        N["4. Sampler<br/>model_sampler_path"]
        T["5. Strategy<br/>model_strategy_path"]
    end

    E --> M["KGEModel / TextKGEModel<br/>(base/model.py)"]
    S --> M
    M --> T
    L --> T
    N --> T

    T --> LOOP["train_loop()"]
```

| Pillar | Config key | Role | Typical location |
|--------|------------|------|------------------|
| **Embedder** | `model_embedder_path` | Maps entity/relation IDs (or text tokens) to vectors | `models/embedders/lookup_embedder.py`, `text_embedder.py` |
| **Scorer** | `model_scorer_path` | Pure scoring math on embeddings (DistMult, ComplEx, RotatE, …) | `models/scorers/*_scorer.py` |
| **Loss** | `model_loss_path` | Objective: BCE, CE, AU alignment/uniformity, InfoNCE, … | `models/losses/*_loss.py` |
| **Sampler** | `model_sampler_path` | Generates corrupted triples for negative sampling (optional) | `models/samplers/*_sampler.py` |
| **Strategy** | `model_strategy_path` | Owns the training loop: batches, forward, backward, validation | `models/strategies/*_strategy.py` |

The strategy is the **orchestrator**. It receives the bound model, loss, and (when needed) sampler, then implements `train_loop()`, `train_batch()`, and validation hooks.

---

## 3. Model assembly

Embedders and scorer are **bound** into a single `nn.Module` before the strategy runs.

```mermaid
flowchart TB
    ARGS["args (dim, dropout, reciprocal relations, …)"]

    subgraph embed["Embedders"]
        ENT["Entity embedder<br/>LookupEmbedder or TextEmbedder"]
        REL["Relation embedder<br/>LookupEmbedder (index KGE only)"]
        AUX["Aux embedders (DaBR dr_emb)"]
    end

    subgraph score["Scorer"]
        SC["KGEScorer subclass<br/>score_spo, score_sp_, au_representations, …"]
    end

    ARGS --> ENT & REL & AUX & SC

    ENT --> DECIDE{input_mode<br/>== tokens?}
    DECIDE -->|yes| TEXT["TextKGEModel<br/>+ TextQueryEmbedder"]
    DECIDE -->|no| KGE["KGEModel<br/>ent + rel + scorer"]
    REL --> KGE
    AUX --> KGE
    SC --> TEXT & KGE
```

**Responsibility split**

- **Embedder:** index lookup or text encoding → `[batch, dim]` vectors.
- **Scorer:** tensor math only (no embedding tables). Implements 1-to-1 scoring, 1-vs-all broadcasting, candidate scoring, and (for AU models) representation extraction for KGAU.
- **KGEModel:** wires embedders to scorer, handles reciprocal relations, optional L2 normalization for LP/AU, and delegates `score_sp_` / `predict_tail_sp_` to the scorer.

---

## 4. Data layer

```mermaid
flowchart TB
    PATHS["train_path / valid_path / test_path<br/>(JSON or .txt)"]

    subgraph data_pkg["data/"]
        DS["dataset.py<br/>Dataset, load_data, TripletDict"]
        DL["dataloader.py<br/>collate()"]
        DH["dict_hub.py<br/>singleton caches"]
        PP["preprocess.py<br/>TripleExample JSON writer"]
    end

    PATHS --> DS
    DS --> DL
    DH --> DS
    DH --> EVAL["Evaluator<br/>filtered ranking masks"]

    subgraph caches["dict_hub caches (lazy init)"]
        ED["entity_dict"]
        RD["relation_id_map"]
        TG["train_triplet_dict / all_triplet_dict"]
        LG["link_graph"]
        TK["tokenizer (text models)"]
    end

    DH --> caches
```

**Roles**

- **`preprocess.py`:** converts raw TSV triples + entity metadata into enriched JSON (`TripleExample`: ids, text, descriptions).
- **`Dataset`:** reads JSON lines for training/eval batches.
- **`dict_hub`:** global caches used for filtered link prediction (all known true triples) and entity indexing.
- **`collate`:** batches examples into tensors or token dicts depending on model type.

---

## 5. Training paradigms (strategies)

`build_pipeline()` infers a **paradigm** from the strategy path and wires loss/sampler accordingly.

```mermaid
flowchart TB
    SP["model_strategy_path"]
    SP --> PAR["_strategy_paradigm()"]

    PAR --> NE["negsamp<br/>RotatE, pRotatE, DaBR"]
    PAR --> KV["kvsall<br/>DistMult"]
    PAR --> OV["1vsall<br/>ComplEx"]
    PAR --> KG["kgau<br/>*-AU models"]
    PAR --> IB["inbatch<br/>SimKGC"]

    NE --> NE_FLOW["Sampler corrupts triples<br/>→ score_spo(pos/neg)<br/>→ BCE / adversarial BCE / logistic"]
    KV --> KV_FLOW["Full entity matrix per batch<br/>→ KL / softmax over all tails"]
    OV --> OV_FLOW["Broadcast (h,r) vs all entities<br/>→ cross-entropy"]
    KG --> KG_FLOW["Extract AU representations<br/>→ KGAULoss (alignment + uniformity)<br/>no sampler"]
    IB --> IB_FLOW["In-batch negatives + masking<br/>→ InfoNCE"]
```

| Model family | Strategy | Sampler | Loss | Negative signal |
|--------------|----------|---------|------|-----------------|
| DistMult | `kvsall_strategy` | none | `kl_loss` | all entities (K vs all) |
| ComplEx | `1vsall_strategy` | none | `ce_loss` | all entities (1 vs all) |
| RotatE / pRotatE | `negsamp_strategy` | `filtered_1_to_n_sampler` | `bce_loss` / `adversarial_bce_loss` | sampled negatives |
| DaBR | `negsamp_strategy` | `uniform_pointwise_sampler` | `pointwise_logistic_loss` | pointwise negatives |
| SimKGC | `inbatch_strategy` | `masking_sampler` | `infonce_loss` | in-batch + mask |
| *-AU (KGAU) | `kgau_strategy` | none | `au_loss` | positive batch geometry only |

---

## 6. Inner training loop (index KGE strategies)

Most lookup-table strategies share helpers in `models/builder.py`: `run_index_kge_train_loop()` or `run_step_based_kge_train_loop()`.

```mermaid
sequenceDiagram
    participant S as Strategy
    participant M as KGEModel
    participant SC as Scorer
    participant LF as Loss fn
    participant EV as Evaluator
    participant CK as checkpoint.py

    loop epochs / steps
        S->>S: iter_training_batches()
        S->>M: embed + score (paradigm-specific)
        M->>SC: score_spo / score_sp_ / au_representations
        SC-->>M: scores or reps
        M-->>S: tensors
        S->>LF: compute loss
        LF-->>S: loss (+ optional regularization)
        S->>S: optimizer.step(), LR schedule

        alt validation interval
            S->>EV: eval_index_kge_epoch()
            EV->>M: filtered link prediction on valid
            EV-->>S: MRR, Hits@k
            S->>CK: save best_model.mdl if improved
        end
    end
```

**Shared training utilities (`models/builder.py`)**

- `build_optimizer`, `build_lr_scheduler`, `apply_kge_regularization`
- `init_index_kge_trainer` — attaches `Evaluator`, entity dict, memory tracker
- Early stopping / step cadence via `utils/training_cadence.py`

**KGAU strategy (`kgau_strategy.py`)** uses its own loop but the same validation/checkpoint pattern: builds `KGAULoss`, optional learnable gammas/`tuni`, gamma linear schedule, and calls `model` AU representation hooks on the scorer.

---

## 7. Single training batch (conceptual)

How components interact for one batch depends on the paradigm:

```mermaid
flowchart LR
    subgraph batch["Batch from DataLoader"]
        B["(h, r, t) indices or token batch"]
    end

    subgraph forward["Forward"]
        E1["ent_embedder / rel_embedder"]
        E2["scorer"]
        B --> E1 --> E2
    end

    subgraph objective["Objective"]
        SAM{"Sampler<br/>needed?"}
        E2 --> SAM
        SAM -->|negsamp| NEG["Corrupted (h', r, t) etc."]
        SAM -->|1vsall/kvsall| ALL["Scores vs all |E| entities"]
        SAM -->|kgau| REP["Query + target + entity reps"]
        SAM -->|inbatch| IB["Contrastive logits in batch"]

        NEG & ALL & REP & IB --> LOSS["loss_fn / KGAULoss"]
    end

    LOSS --> REG["+ L3 regularization<br/>(optional)"]
    REG --> BW["backward + optimizer"]
```

---

## 8. Evaluation pipeline (post-training)

After `train_loop()` returns, `main.py` does **not** use the trainer for testing—it reloads weights into a fresh `Evaluator`.

```mermaid
flowchart TB
    MAIN["main.py"]
    BEST["best_checkpoint_path<br/>or best_model.mdl"]
    EV["Evaluator.load()"]
    REBUILD["Rebuild model via config paths<br/>(same five pillars)"]

    MAIN --> BEST --> EV
    EV --> REBUILD

    subgraph tasks["task flag: lp / tc / both"]
        LP["Link prediction<br/>evaluate_link_prediction_inplace<br/>forward + backward queries"]
        TC["Triple classification<br/>evaluate_test_triple_classification"]
    end

    REBUILD --> LP & TC

    LP --> MET["metrics/ranking.py<br/>MRR, MR, Hits@k"]
    TC --> CLS["metrics/classification.py<br/>accuracy, F1, AUC"]

    MET & CLS --> RES["utils/logger.py<br/>write_results_report → results.txt"]
```

**Filtered ranking:** `Evaluator` uses `all_triplet_dict` from `dict_hub` to mask known true tails/heads so ranks reflect *filtered* link prediction.

---

## 9. Configuration → component map (example)

`ComplEx-AU` on WN18RR (`configs/ComplEx-AU_WN18RR_learnable_gammas_tuni.json`):

```mermaid
flowchart LR
    JSON["JSON config"]

    JSON --> E["lookup_embedder.py<br/>entity + relation tables"]
    JSON --> SC["complex_scorer.py<br/>ComplEx + AU reps"]
    JSON --> L["au_loss.py<br/>KGAULoss"]
    JSON --> ST["kgau_strategy.py<br/>KGAU train loop"]

    E & SC --> M["KGEModel"]
    M & L & ST --> RUN["python main.py --config-path …"]
```

Equivalent CLI overrides (any registered key): `--batch-size`, `--lr`, `--epoch-per-eval`, etc. See `configs/config.py` → `build_parser()`.

---

## 10. Outputs and HPO

```mermaid
flowchart LR
    RUN["Training run"]
    RUN --> LOG["logs/&lt;prefix&gt;_&lt;timestamp&gt;/"]
    LOG --> RL["run.log"]
    LOG --> RM["results.txt"]
    LOG --> BM["best_model.mdl"]
    LOG --> LM["last_model.mdl"]

    HPO["scripts/hpo/run_search.py"]
    TRIAL["scripts/hpo/train_trial.py"]
    HPO --> TRIAL
    TRIAL --> BUILD["build_pipeline() + train_loop()"]
    BUILD --> LOG
```

HPO trials are thin wrappers: they load a trial JSON config, call the same `build_pipeline()` path as `main.py`, and report `best_mrr` as the objective.

---

## 11. Component dependency graph (summary)

```mermaid
graph TD
    main["main.py"]
    cfg["configs/config.py + *.json"]
    builder["models/builder.py"]
    embed["models/embedders/"]
    scorer["models/scorers/"]
    loss["models/losses/"]
    sampler["models/samplers/"]
    strategy["models/strategies/"]
    model["base/model.py"]
    data["data/"]
    metrics["metrics/"]
    utils["utils/"]
    eval["base/evaluator.py"]

    cfg --> main
    main --> builder
    main --> eval
    builder --> embed & scorer & loss & sampler & strategy
    embed & scorer --> model
    model --> strategy
    loss --> strategy
    sampler --> strategy
    data --> strategy & eval
    strategy --> utils
    eval --> metrics & utils
    strategy --> eval
```

---

## Quick reference: who owns what?

| Concern | Owner |
|---------|--------|
| Hyperparameters & paths | `configs/*.json` + `configs/config.py` |
| GPU / DDP / parameter counts | `utils/device.py` |
| Experiment wiring | `models/builder.py` → `build_pipeline()` |
| Embedding tables / text encoding | `models/embedders/` |
| Score functions & AU representations | `models/scorers/` |
| Training objective | `models/losses/` |
| Negative corruption | `models/samplers/` (when used) |
| Epoch/step loop, optim, validation | `models/strategies/` |
| Model binding & 1-vs-all API | `base/model.py` (`KGEModel`, `TextKGEModel`) |
| Test metrics | `base/evaluator.py` + `metrics/` |
| Checkpoint I/O | `utils/checkpoint.py` |
| Logging & result files | `utils/logger.py` |
