"""Verify triple-classification eval fix scope across all training configs."""

from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace

from metrics.classification import find_global_threshold
from models.losses.bce_loss import bce_logit_offset, uses_bce_logit_offset

import numpy as np


def audit_configs() -> list[dict]:
    rows = []
    for path in sorted(Path("configs").glob("*.json")):
        with open(path, encoding="utf-8") as f:
            cfg = json.load(f)
        args = SimpleNamespace(**cfg)
        rows.append(
            {
                "config": path.name,
                "model": cfg.get("model"),
                "loss": os.path.basename(cfg.get("model_loss_path", "")),
                "strategy": os.path.basename(cfg.get("model_strategy_path", "")),
                "uses_bce_offset": uses_bce_logit_offset(args),
                "bce_offset": bce_logit_offset(args),
                "loss_arg": cfg.get("loss_arg"),
                "margin": cfg.get("margin"),
                "task": cfg.get("task"),
            }
        )
    return rows


def test_threshold_tiebreak_preserves_accuracy_optimum() -> None:
    """F1 tie-break should not lower the best achievable accuracy."""
    rng = np.random.default_rng(0)
    y_true = np.array([1] * 500 + [0] * 500)
    y_prob = np.concatenate([rng.beta(2, 5, 500), rng.beta(5, 2, 500)])
    threshold = find_global_threshold(y_true, y_prob)
    y_pred = (y_prob > threshold).astype(int)
    best_acc = max(
        float((y_pred == y_true).mean())
        for y_pred in ((y_prob > t).astype(int) for t in np.linspace(y_prob.min(), y_prob.max(), 100))
    )
    actual_acc = float((y_pred == y_true).mean())
    assert actual_acc == best_acc, f"accuracy dropped: {actual_acc} vs {best_acc}"


def test_training_loss_paths_untouched() -> None:
    """Training strategies must not import triple-classification threshold helpers."""
    strategy_dir = Path("models/strategies")
    forbidden = ("find_global_threshold", "_scores_to_classification_probs", "evaluate_test_triple_classification")
    for path in strategy_dir.glob("*.py"):
        text = path.read_text(encoding="utf-8")
        for token in forbidden:
            assert token not in text, f"{path.name} unexpectedly references {token}"


def main() -> None:
    rows = audit_configs()
    print("Config impact matrix (triple-classification test eval only):")
    print(f"{'config':42} {'loss':28} offset?   offset     loss_arg")
    print("-" * 95)
    for r in rows:
        print(
            f"{r['config']:42} {r['loss']:28} {str(r['uses_bce_offset']):8} "
            f"{r['bce_offset']:10.4f} {r['loss_arg']}"
        )

    nonzero = [r["config"] for r in rows if r["uses_bce_offset"] and r["bce_offset"] != 0]
    flagged_zero = [r["config"] for r in rows if r["uses_bce_offset"] and r["bce_offset"] == 0]
    print()
    print("Configs with NON-ZERO BCE offset at eval:", nonzero or "none")
    print("Configs flagged uses_bce_offset but offset=0:", flagged_zero or "none")

    test_threshold_tiebreak_preserves_accuracy_optimum()
    print("OK: find_global_threshold still maximizes accuracy")

    test_training_loss_paths_untouched()
    print("OK: training strategies do not use triple-classification eval helpers")

    # Sanity: only standard LibKGE BCE should apply offset
    assert nonzero == ["RotatE_WN18RR.json"], f"unexpected nonzero offset configs: {nonzero}"
    assert not flagged_zero, f"configs should not be flagged without loss_arg: {flagged_zero}"

    print("OK: all verification checks passed")


if __name__ == "__main__":
    main()
