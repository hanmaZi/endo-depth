"""Point d'entrée d'entraînement pour endo-depth.

Exemples :
    python -m src.train --config configs/base.yaml --smoke-test
    python -m src.train --config configs/c3vd_lora_baseline.yaml --auto-resume
    python -m src.train --config configs/c3vd_lora_grad_norm.yaml --auto-resume
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.datasets import build_dataset, make_synthetic_endoscopy_batch
from src.eval import evaluate_loader
from src.losses import build_loss_from_config
from src.models import SmokeDepthNet, build_model
from src.utils import (
    count_parameters,
    get_device,
    load_checkpoint,
    load_config,
    save_checkpoint,
    save_config_copy,
    save_json,
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
        "--data-dir",
        type=str,
        default=None,
        help="Remplace data.dataset_path (utile sur Google Drive).",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default=None,
        help="Remplace paths.results_dir et paths.checkpoint_dir.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Remplace le nombre d'époques défini dans le fichier YAML.",
    )
    parser.add_argument(
        "--max-batches",
        type=int,
        default=None,
        help="Limite le nombre de batches par époque (pratique pour test CPU rapide).",
    )
    parser.add_argument(
        "--auto-resume",
        action="store_true",
        help="Reprend automatiquement depuis checkpoints/last.pt s'il existe.",
    )
    parser.add_argument(
        "--from-config-only",
        action="store_true",
        help="Instancie l'architecture sans télécharger les poids HuggingFace (test local).",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Lance un mini entraînement synthétique rapide pour vérifier le pipeline.",
    )
    return parser.parse_args()


def apply_train_overrides(config: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    """Applique les surcharges CLI à la configuration YAML."""
    if args.data_dir:
        config.setdefault("data", {})["dataset_path"] = args.data_dir
    if args.results_dir:
        exp_name = str(config.get("experiment_name", "train_run"))
        config.setdefault("paths", {})["results_dir"] = args.results_dir
        config["paths"]["checkpoint_dir"] = str(Path(args.results_dir) / exp_name / "checkpoints")
    if args.epochs is not None:
        config.setdefault("training", {})["epochs"] = int(args.epochs)
    if args.from_config_only:
        config.setdefault("model", {})["from_config_only"] = True
    return config


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

    model = SmokeDepthNet().to(device)
    total_params, trainable_params = count_parameters(model)
    print(f"[Info] SmokeDepthNet : {trainable_params:,} / {total_params:,} paramètres entraînables")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_cfg["learning_rate"]),
        weight_decay=float(train_cfg.get("weight_decay", 0.0)),
    )

    start_epoch = 1
    resume_path = train_cfg.get("resume_from")
    if resume_path:
        ckpt = load_checkpoint(resume_path, model=model, device=device, optimizer=optimizer)
        start_epoch = int(ckpt["epoch"]) + 1
        print(f"[Info] Reprise depuis {resume_path} à l'époque {start_epoch}")

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
        optimizer.zero_grad()
        pred_depth = model(images)
        loss = F.l1_loss(pred_depth, target_depth)
        loss.backward()
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


def run_training(
    config: dict[str, Any],
    auto_resume: bool = False,
    max_batches: int | None = None,
) -> Path:
    """Boucle d'entraînement complète (LoRA + Loss Gradients & Normales) avec reprise Colab."""
    set_seed(int(config.get("seed", 42)))
    device = get_device(config.get("device", "auto"))
    print(f"[Info] Périphérique utilisé : {device}")

    data_cfg = config["data"]
    train_cfg = config["training"]
    eval_cfg = config.get("evaluation", {})
    exp_name = str(config.get("experiment_name", "train_run"))

    exp_dir = Path(config["paths"]["results_dir"]) / exp_name
    ckpt_dir = Path(config["paths"].get("checkpoint_dir") or (exp_dir / "checkpoints"))
    exp_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    save_config_copy(config, exp_dir / "config_used.yaml")

    train_ds = build_dataset(config, split="train")
    val_ds = build_dataset(config, split="val")
    test_ds = build_dataset(config, split="test")
    print(f"[Info] Dataset {data_cfg['dataset_name']} : train={len(train_ds)}, val={len(val_ds)}, test={len(test_ds)}")

    num_workers = 0 if device.type == "cpu" else int(data_cfg.get("num_workers", 2))
    batch_size = int(data_cfg.get("batch_size", 4))
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )

    model = build_model(config).to(device)
    total_params, trainable_params = count_parameters(model)
    pct = 100.0 * trainable_params / max(1, total_params)
    print(
        f"[Info] Modèle {config['model']['name']} : "
        f"{trainable_params:,} / {total_params:,} paramètres entraînables ({pct:.2f}%)"
    )

    criterion = build_loss_from_config(config)
    trainable_list = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_list,
        lr=float(train_cfg.get("learning_rate", 1e-4)),
        weight_decay=float(train_cfg.get("weight_decay", 1e-4)),
    )

    use_amp = bool(train_cfg.get("use_amp", True)) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda") if use_amp else None
    grad_clip = float(train_cfg.get("grad_clip_norm", 1.0))

    last_ckpt_path = ckpt_dir / "last.pt"
    best_ckpt_path = ckpt_dir / "best.pt"
    history_path = exp_dir / "history.json"

    start_epoch = 1
    best_val_abs_rel = float("inf")
    history: list[dict[str, Any]] = []

    resume_path = train_cfg.get("resume_from")
    if not resume_path and auto_resume and last_ckpt_path.is_file():
        resume_path = str(last_ckpt_path)

    if resume_path:
        ckpt = load_checkpoint(resume_path, model=model, device=device, optimizer=optimizer)
        start_epoch = int(ckpt["epoch"]) + 1
        best_val_abs_rel = float(ckpt.get("best_metric", float("inf")))
        if history_path.is_file():
            import json
            with history_path.open("r", encoding="utf-8") as handle:
                loaded_hist = json.load(handle)
                if isinstance(loaded_hist, list):
                    history = [h for h in loaded_hist if int(h.get("epoch", 0)) < start_epoch]
        print(f"[Info] Reprise depuis {resume_path} à l'époque {start_epoch} (best_abs_rel={best_val_abs_rel:.4f})")

    epochs = int(train_cfg.get("epochs", 10))
    save_every = max(1, int(train_cfg.get("save_every_n_epochs", 1)))
    min_depth = float(eval_cfg.get("min_depth", data_cfg.get("min_depth_mm", 1.0)))
    max_depth = float(eval_cfg.get("max_depth", data_cfg.get("max_depth_mm", 100.0)))
    align_mode = str(eval_cfg.get("align_mode", "median"))

    for epoch in range(start_epoch, epochs + 1):
        model.train()
        running: dict[str, float] = {"total": 0.0, "l1": 0.0, "silog": 0.0, "grad": 0.0, "normal": 0.0}
        steps = 0

        for batch_idx, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}", leave=False)):
            if max_batches is not None and batch_idx >= max_batches:
                break

            images = batch["image"].to(device)
            depths = batch["depth"].to(device)
            masks = batch["mask"].to(device)
            normals = batch["normals"].to(device)
            has_normals = batch["has_normals"].to(device)

            optimizer.zero_grad()
            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                pred_depth = model(images)
                loss, breakdown = criterion(
                    pred_depth,
                    depths,
                    masks,
                    target_normals=normals,
                    has_normals=has_normals,
                )

            if scaler is not None:
                scaler.scale(loss).backward()
                if grad_clip > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(trainable_list, grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                if grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(trainable_list, grad_clip)
                optimizer.step()

            for k, v in breakdown.items():
                running[k] += v
            steps += 1

        train_losses = {k: round(v / max(1, steps), 6) for k, v in running.items()}
        val_metrics = evaluate_loader(
            model=model,
            loader=val_loader,
            device=device,
            pred_is_disparity=False,
            align_mode=align_mode,
            min_depth=min_depth,
            max_depth=max_depth,
            max_batches=max_batches,
        )

        record = {
            "epoch": epoch,
            "train_loss": train_losses,
            "val_metrics": val_metrics,
        }
        history.append(record)
        save_json(history, history_path)

        val_abs_rel = float(val_metrics.get("abs_rel", float("inf")))
        print(
            f"[Epoch {epoch}/{epochs}] "
            f"loss={train_losses['total']:.4f} (l1={train_losses['l1']:.3f}, "
            f"grad={train_losses['grad']:.3f}, norm={train_losses['normal']:.3f}) | "
            f"val_AbsRel={val_abs_rel:.4f}, val_RMSE={val_metrics.get('rmse', 0.0):.3f}, "
            f"val_d1={val_metrics.get('delta_1', 0.0):.4f}"
        )

        if val_abs_rel < best_val_abs_rel:
            best_val_abs_rel = val_abs_rel
            save_checkpoint(
                best_ckpt_path,
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                loss=train_losses["total"],
                best_metric=best_val_abs_rel,
            )

        if (epoch % save_every == 0) or (epoch == epochs):
            save_checkpoint(
                last_ckpt_path,
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                loss=train_losses["total"],
                best_metric=best_val_abs_rel,
            )

    # Évaluation finale sur le split test avec le meilleur checkpoint
    if best_ckpt_path.is_file():
        load_checkpoint(best_ckpt_path, model=model, device=device)
    test_metrics = evaluate_loader(
        model=model,
        loader=test_loader,
        device=device,
        pred_is_disparity=False,
        align_mode=align_mode,
        min_depth=min_depth,
        max_depth=max_depth,
        max_batches=max_batches,
    )
    test_payload = {
        "experiment_name": exp_name,
        "dataset": data_cfg["dataset_name"],
        "split": "test",
        "align_mode": align_mode,
        "best_val_abs_rel": round(best_val_abs_rel, 6),
        "trainable_params": trainable_params,
        "total_params": total_params,
        "metrics": test_metrics,
    }
    save_json(test_payload, exp_dir / f"eval_test_{align_mode}.json")
    print(f"[OK] Entraînement terminé. Test final : {test_metrics}")
    return best_ckpt_path if best_ckpt_path.is_file() else last_ckpt_path


def main() -> None:
    args = parse_args()
    config = apply_train_overrides(load_config(args.config), args)
    dataset_name = str(config.get("data", {}).get("dataset_name", ""))

    if args.smoke_test or dataset_name == "synthetic_smoke":
        run_smoke_training(config, epochs_override=args.epochs)
        return

    run_training(config, auto_resume=args.auto_resume, max_batches=args.max_batches)


if __name__ == "__main__":
    main()
