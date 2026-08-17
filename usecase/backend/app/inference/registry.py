from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from app.paths import MODEL_DIR, REGISTRY_JSON
from app.schemas import ModelCardOut


@dataclass
class RegisteredModel:
    id: str
    label: str
    available: bool
    checkpoint_path: Path
    config_path: Path
    config: dict


def _as_card(item: RegisteredModel) -> ModelCardOut:
    cfg = item.config
    return ModelCardOut(
        id=item.id,
        label=item.label,
        available=item.available,
        framework=str(cfg.get("framework") or "KGAU"),
        backbone=str(cfg.get("backbone") or ""),
        embedding_dim=cfg.get("embedding_dim"),
        training_strategy=str(cfg.get("training_strategy") or ""),
        negative_sampling=str(cfg.get("negative_sampling") or ""),
        best_epoch=cfg.get("best_epoch"),
        checkpoint=item.checkpoint_path.name,
        tuni=cfg.get("tuni"),
        gamma_q=cfg.get("gamma_q"),
        gamma_t=cfg.get("gamma_t"),
        gamma_ent=cfg.get("gamma_ent"),
        head_eval_mode=str(cfg.get("head_eval_mode") or "rt_forward"),
    )


def load_registry() -> tuple[str, list[RegisteredModel]]:
    payload = json.loads(REGISTRY_JSON.read_text(encoding="utf-8"))
    models: list[RegisteredModel] = []
    for item in payload["models"]:
        config_path = MODEL_DIR / item["config"]
        checkpoint_path = MODEL_DIR / item["checkpoint"]
        config = json.loads(config_path.read_text(encoding="utf-8")) if config_path.exists() else {}
        models.append(
            RegisteredModel(
                id=item["id"],
                label=item["label"],
                available=bool(item.get("available")) and checkpoint_path.exists(),
                checkpoint_path=checkpoint_path,
                config_path=config_path,
                config=config,
            )
        )
    return str(payload["default_model_id"]), models


def model_cards(models: list[RegisteredModel]) -> list[ModelCardOut]:
    return [_as_card(model) for model in models]


def find_model(models: list[RegisteredModel], model_id: str) -> RegisteredModel:
    for model in models:
        if model.id == model_id:
            return model
    raise KeyError(model_id)
