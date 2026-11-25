"""Point d'entrée d'entraînement pour endo-depth.

Usage rapide (test CPU) :
    python -m src.train --config configs/base.yaml --smoke-test
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from src.datasets import make_synthetic_endoscopy_batch
from src.models import SmokeDepthNet
from src.utils import (
    count_parameters,
    get_device,
    load_checkpoint,
    load_config,
    save_checkpoint,
    set_seed,
)


def parse_args() -> argparse.Namespace:
    """Analyse les arguments passés en ligne de commande."""
    parser = argparse.ArgumentParser(description="Entraînement endo-depth")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/base.yaml",
        help="Chemin vers le fichier de configuration YAML.",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Lance un mini entraînement synthétique rapide pour vérifier le pipeline.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Remplace le nombre d'époques défini dans le fichier YAML.",
    )
    return parser.parse_args()


def run_smoke_training(config: dict[str, Any], epochs_override: int | None = None) -> Path:
    """Exécute une boucle d'entraînement complète sur données synthétiques (CPU ou GPU)."""
    set_seed(int(config["seed"]))
    device = get_device(config.get("device", "auto"))
    print(f"[Info] Périphérique utilisé : {device}")

    data_cfg = config["data"]
    train_cfg = config["training"]
    height, width = map(int, data_cfg["image_size"])
    batch_size = int(data_cfg["batch_size"])
    epochs = int(epochs_override or train_cfg["epochs"])
    save_every = max(1, int(train_cfg.get("save_every_n_epochs", 1)))

    # 1. Modèle léger + optimiseur AdamW
    model = SmokeDepthNet().to(device)
    total_params, trainable_params = count_parameters(model)
    print(f"[Info] SmokeDepthNet : {trainable_params:,} / {total_params:,} paramètres entraînables")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_cfg["learning_rate"]),
        weight_decay=float(train_cfg.get("weight_decay", 0.0)),
    )

    # 2. Reprise éventuelle depuis un checkpoint (utile après coupure Colab/Kaggle)
    start_epoch = 1
    resume_path = train_cfg.get("resume_from")
    if resume_path:
        ckpt = load_checkpoint(resume_path, model=model, device=device, optimizer=optimizer)
        start_epoch = int(ckpt["epoch"]) + 1
        print(f"[Info] Reprise depuis {resume_path} à l'époque {start_epoch}")

    # 3. Boucle d'entraînement
    ckpt_dir = Path(config["paths"]["checkpoint_dir"])
    last_ckpt_path = ckpt_dir / "smoke_last.pt"

    model.train()
    for epoch in range(start_epoch, epochs + 1):
        images, target_depth = make_synthetic_endoscopy_batch(
            batch_size=batch_size,
            height=height,
            width=width,
            device=device,
        )

        # Étape A : remettre les gradients précédents à zéro
        optimizer.zero_grad()

        # Étape B : passe avant (forward pass) -> prédiction de la profondeur
        pred_depth = model(images)

        # Étape C : fonction de perte (loss) -> mesure l'écart entre prédiction et vérité terrain
        loss = F.l1_loss(pred_depth, target_depth)

        # Étape D : rétropropagation (backpropagation) -> calcule le gradient pour chaque poids
        loss.backward()

        # Étape E : mise à jour des poids via l'optimiseur
        optimizer.step()

        print(f"[Epoch {epoch}/{epochs}] L1 loss = {loss.item():.4f}")

        if (epoch % save_every == 0) or (epoch == epochs):
            save_checkpoint(
                checkpoint_path=last_ckpt_path,
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                loss=loss.item(),
            )

    print(f"[OK] Checkpoint sauvegardé : {last_ckpt_path}")
    return last_ckpt_path


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    dataset_name = config.get("data", {}).get("dataset_name", "")

    if args.smoke_test or dataset_name == "synthetic_smoke":
        run_smoke_training(config, epochs_override=args.epochs)
        return

    raise NotImplementedError(
        f"Dataset '{dataset_name}' non encore câblé : les datasets endoscopiques réels "
        "et modèles baseline seront intégrés aux étapes C et D."
    )


if __name__ == "__main__":
    main()
