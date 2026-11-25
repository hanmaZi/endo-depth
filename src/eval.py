"""Point d'entrée d'évaluation pour endo-depth.

Usage rapide (test CPU) :
    python -m src.eval --config configs/base.yaml --smoke-test
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from src.datasets import make_synthetic_endoscopy_batch
from src.models import SmokeDepthNet
from src.utils import get_device, load_checkpoint, load_config, set_seed


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
        "--smoke-test",
        action="store_true",
        help="Lance une évaluation rapide sur données synthétiques.",
    )
    return parser.parse_args()


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

    # Mode évaluation : désactive Dropout/BatchNorm training et le suivi des gradients
    model.eval()
    images, target_depth = make_synthetic_endoscopy_batch(
        batch_size=batch_size,
        height=height,
        width=width,
        device=device,
    )

    with torch.no_grad():
        pred_depth = model(images)

        # Alignement d'échelle médian : corrige l'ambiguïté d'échelle monoculaire
        if config.get("evaluation", {}).get("median_scaling", True):
            scale = torch.median(target_depth) / torch.median(pred_depth).clamp(min=1e-6)
            pred_depth = pred_depth * scale

        mae = F.l1_loss(pred_depth, target_depth).item()
        rmse = torch.sqrt(F.mse_loss(pred_depth, target_depth)).item()

    metrics = {
        "smoke_mae": round(mae, 6),
        "smoke_rmse": round(rmse, 6),
    }

    results_dir = Path(config["paths"]["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = results_dir / "smoke_metrics.json"
    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)

    print(f"[OK] Métriques smoke sauvegardées dans {metrics_path} : {metrics}")
    return metrics


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    dataset_name = config.get("data", {}).get("dataset_name", "")

    if args.smoke_test or dataset_name == "synthetic_smoke":
        run_smoke_evaluation(config, checkpoint_path=args.checkpoint)
        return

    raise NotImplementedError(
        f"Dataset '{dataset_name}' non encore câblé : les métriques endoscopiques complètes "
        "(AbsRel, SqRel, RMSE, RMSE_log, delta) seront ajoutées à l'étape D."
    )


if __name__ == "__main__":
    main()
