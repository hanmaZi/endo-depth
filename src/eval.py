"""Point d'entrée d'évaluation pour endo-depth (smoke test, zero-shot ou checkpoint fine-tuné).

Exemples :
    python -m src.eval --config configs/base.yaml --smoke-test
    python -m src.eval --config configs/c3vd_zeroshot.yaml
    python -m src.eval --config configs/c3vd_lora_grad_norm.yaml --checkpoint results/c3vd_lora_grad_norm/checkpoints/best.pt
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.datasets import build_dataset, make_synthetic_endoscopy_batch
from src.metrics import DepthMetricTracker
from src.models import SmokeDepthNet, build_model
from src.utils import (
    get_device,
    load_checkpoint,
    load_config,
    save_config_copy,
    save_json,
    set_seed,
)


def parse_args() -> argparse.Namespace:
    """Analyse les arguments passés en ligne de commande."""
    parser = argparse.ArgumentParser(description="Évaluation endo-depth")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/base.yaml",
        help="Chemin vers le fichier de configuration YAML.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Chemin optionnel vers un checkpoint .pt/.pth à évaluer.",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Remplace data.dataset_path (utile sur Google Drive).",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default=None,
        help="Remplace paths.results_dir (utile sur Google Drive).",
    )
    parser.add_argument(
        "--split",
        type=str,
        default=None,
        help="Split à évaluer (train, val, test, all).",
    )
    parser.add_argument(
        "--align-mode",
        type=str,
        default=None,
        choices=["median", "scale_shift", "none"],
        help="Mode d'alignement monoculaire.",
    )
    parser.add_argument(
        "--max-batches",
        type=int,
        default=None,
        help="Limite le nombre de batches évalués (pour test rapide).",
    )
    parser.add_argument(
        "--from-config-only",
        action="store_true",
        help="Instancie l'architecture sans télécharger les poids HuggingFace (test local).",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Lance une évaluation rapide sur données synthétiques.",
    )
    return parser.parse_args()


def apply_eval_overrides(config: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    """Applique les surcharges CLI à la configuration YAML."""
    if args.data_dir:
        config.setdefault("data", {})["dataset_path"] = args.data_dir
    if args.results_dir:
        config.setdefault("paths", {})["results_dir"] = args.results_dir
    if args.split:
        config.setdefault("evaluation", {})["split"] = args.split
    if args.align_mode:
        config.setdefault("evaluation", {})["align_mode"] = args.align_mode
    if getattr(args, "from_config_only", False):
        config.setdefault("model", {})["from_config_only"] = True
    return config


@torch.no_grad()
def evaluate_loader(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    pred_is_disparity: bool = False,
    align_mode: str = "median",
    min_depth: float = 1.0,
    max_depth: float = 100.0,
    max_batches: int | None = None,
) -> dict[str, float]:
    """Évalue un modèle sur un DataLoader et retourne les métriques standard moyennées."""
    model.eval()
    tracker = DepthMetricTracker()

    for batch_idx, batch in enumerate(tqdm(loader, desc="Évaluation", leave=False)):
        if max_batches is not None and batch_idx >= max_batches:
            break
        images = batch["image"].to(device)
        depths = batch["depth"].to(device)
        masks = batch["mask"].to(device)

        preds = model(images)
        tracker.update_batch(
            preds,
            depths,
            masks,
            pred_is_disparity=pred_is_disparity,
            align_mode=align_mode,  # type: ignore[arg-type]
            min_depth=min_depth,
            max_depth=max_depth,
        )

    metrics = tracker.compute()
    metrics["num_evaluated_frames"] = float(tracker.count)
    return metrics


def run_smoke_evaluation(
    config: dict[str, Any],
    checkpoint_path: str | None = None,
) -> dict[str, float]:
    """Évalue SmokeDepthNet sur un lot synthétique et enregistre les métriques."""
    set_seed(int(config["seed"]))
    device = get_device(config.get("device", "auto"))

    data_cfg = config["data"]
    height, width = map(int, data_cfg["image_size"])
    batch_size = int(data_cfg["batch_size"])

    model = SmokeDepthNet().to(device)
    if checkpoint_path:
        ckpt = load_checkpoint(checkpoint_path, model=model, device=device)
        print(f"[Info] Checkpoint chargé (epoch={ckpt['epoch']}) : {checkpoint_path}")

    model.eval()
    images, target_depth = make_synthetic_endoscopy_batch(
        batch_size=batch_size,
        height=height,
        width=width,
        device=device,
    )

    with torch.no_grad():
        pred_depth = model(images)
        if config.get("evaluation", {}).get("median_scaling", True):
            scale = torch.median(target_depth) / torch.median(pred_depth).clamp(min=1e-6)
            pred_depth = pred_depth * scale

        mae = F.l1_loss(pred_depth, target_depth).item()
        rmse = torch.sqrt(F.mse_loss(pred_depth, target_depth)).item()

    metrics = {"smoke_mae": round(mae, 6), "smoke_rmse": round(rmse, 6)}
    results_dir = Path(config["paths"]["results_dir"])
    metrics_path = save_json(metrics, results_dir / "smoke_metrics.json")
    print(f"[OK] Métriques smoke sauvegardées dans {metrics_path} : {metrics}")
    return metrics


def run_evaluation(
    config: dict[str, Any],
    checkpoint_path: str | None = None,
    max_batches: int | None = None,
) -> dict[str, Any]:
    """Évalue un modèle réel (zero-shot ou fine-tuné) sur un dataset endoscopique."""
    set_seed(int(config.get("seed", 42)))
    device = get_device(config.get("device", "auto"))
    eval_cfg = config.get("evaluation", {})
    data_cfg = config["data"]

    split = str(eval_cfg.get("split", "test"))
    dataset = build_dataset(config, split=split)
    num_workers = 0 if device.type == "cpu" else int(data_cfg.get("num_workers", 2))
    loader = DataLoader(
        dataset,
        batch_size=int(data_cfg.get("batch_size", 4)),
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )

    model = build_model(config).to(device)
    if checkpoint_path:
        ckpt = load_checkpoint(checkpoint_path, model=model, device=device)
        print(f"[Info] Checkpoint chargé (epoch={ckpt['epoch']}) : {checkpoint_path}")

    align_mode = str(eval_cfg.get("align_mode", "median"))
    pred_is_disparity = bool(eval_cfg.get("pred_is_disparity", False))
    metrics = evaluate_loader(
        model=model,
        loader=loader,
        device=device,
        pred_is_disparity=pred_is_disparity,
        align_mode=align_mode,
        min_depth=float(eval_cfg.get("min_depth", data_cfg.get("min_depth_mm", 1.0))),
        max_depth=float(eval_cfg.get("max_depth", data_cfg.get("max_depth_mm", 100.0))),
        max_batches=max_batches,
    )

    exp_name = str(config.get("experiment_name", "eval_run"))
    exp_dir = Path(config["paths"]["results_dir"]) / exp_name
    payload = {
        "experiment_name": exp_name,
        "dataset": data_cfg["dataset_name"],
        "split": split,
        "align_mode": align_mode,
        "pred_is_disparity": pred_is_disparity,
        "checkpoint": str(checkpoint_path) if checkpoint_path else None,
        "metrics": metrics,
    }
    out_json = save_json(payload, exp_dir / f"eval_{split}_{align_mode}.json")
    save_config_copy(config, exp_dir / "config_used.yaml")
    print(f"[OK] Évaluation terminée -> {out_json}")
    print(f"     Résultats : {metrics}")
    return payload


def main() -> None:
    args = parse_args()
    config = apply_eval_overrides(load_config(args.config), args)
    dataset_name = str(config.get("data", {}).get("dataset_name", ""))

    if args.smoke_test or dataset_name == "synthetic_smoke":
        run_smoke_evaluation(config, checkpoint_path=args.checkpoint)
        return

    run_evaluation(config, checkpoint_path=args.checkpoint, max_batches=args.max_batches)


if __name__ == "__main__":
    main()
