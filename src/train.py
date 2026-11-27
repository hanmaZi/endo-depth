"""Point d'entrée d'entraînement pour endo-depth.

Exemples :
    python -m src.train --config configs/base.yaml --smoke-test
    python -m src.train --config configs/c3vd_lora_baseline.yaml --auto-resume
    python -m src.train --config configs/c3vd_lora_grad_norm.yaml --auto-resume
"""

from __future__ import annotations

import argparse
import math
import re
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.datasets import build_dataset, make_synthetic_endoscopy_batch, summarize_records_by_sequence
from src.eval import evaluate_loader
from src.losses import build_loss_from_config
from src.metrics import compute_macro_metrics
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
        "--seed",
        type=int,
        default=None,
        help="Remplace la graine aléatoire (met à jour _seedN dans experiment_name).",
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
    if getattr(args, "seed", None) is not None:
        new_seed = int(args.seed)
        config["seed"] = new_seed
        exp_name = str(config.get("experiment_name", "train_run"))
        if re.search(r"_seed\d+$", exp_name):
            exp_name = re.sub(r"_seed\d+$", f"_seed{new_seed}", exp_name)
        else:
            exp_name = f"{exp_name}_seed{new_seed}"
        config["experiment_name"] = exp_name
        results_root = str(config.get("paths", {}).get("results_dir", "results"))
        config.setdefault("paths", {})["checkpoint_dir"] = str(Path(results_root) / exp_name / "checkpoints")
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
    split_sequences = {
        "train": summarize_records_by_sequence(train_ds.records) if hasattr(train_ds, "records") else {},
        "val": summarize_records_by_sequence(val_ds.records) if hasattr(val_ds, "records") else {},
        "test": summarize_records_by_sequence(test_ds.records) if hasattr(test_ds, "records") else {},
    }
    print(f"[Info] Dataset {data_cfg['dataset_name']} : train={len(train_ds)}, val={len(val_ds)}, test={len(test_ds)}")
    print(f"[Info] Séquences par split : {split_sequences}")

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
    base_lr = float(train_cfg.get("learning_rate", 1e-4))
    epochs = int(train_cfg.get("epochs", 10))
    optimizer = torch.optim.AdamW(
        trainable_list,
        lr=base_lr,
        weight_decay=float(train_cfg.get("weight_decay", 1e-4)),
    )
    scheduler_name = str(train_cfg.get("lr_scheduler") or "none").lower()
    t_max = max(epochs, 1)
    eta_min = base_lr * 0.1
    scheduler = None
    if scheduler_name == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=t_max,
            eta_min=eta_min,
        )
    elif scheduler_name not in {"none", ""}:
        raise ValueError(f"lr_scheduler non supporté : {scheduler_name}")

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
        if scheduler is not None and start_epoch > 1:
            completed = min(start_epoch - 1, t_max)
            resumed_lr = eta_min + 0.5 * (base_lr - eta_min) * (1.0 + math.cos(math.pi * completed / t_max))
            n_groups = len(optimizer.param_groups)
            for group in optimizer.param_groups:
                group["lr"] = resumed_lr
            scheduler.load_state_dict(
                {
                    "base_lrs": [base_lr] * n_groups,
                    "last_epoch": completed,
                    "_step_count": completed + 1,
                    "_last_lr": [resumed_lr] * n_groups,
                }
            )
        print(f"[Info] Reprise depuis {resume_path} à l'époque {start_epoch} (best_abs_rel={best_val_abs_rel:.4f})")
    save_every = max(1, int(train_cfg.get("save_every_n_epochs", 1)))
    min_depth = float(eval_cfg.get("min_depth", data_cfg.get("min_depth_mm", 1.0)))
    max_depth = float(eval_cfg.get("max_depth", data_cfg.get("max_depth_mm", 100.0)))
    align_mode = str(eval_cfg.get("align_mode", "median"))

    eval_json_path = exp_dir / f"eval_test_{align_mode}.json"
    if resume_path and start_epoch > epochs and eval_json_path.is_file():
        print(
            f"[Skip] {exp_name} déjà terminé ({start_epoch - 1}/{epochs} époques) -> {eval_json_path}"
        )
        return best_ckpt_path if best_ckpt_path.is_file() else last_ckpt_path

    for epoch in range(start_epoch, epochs + 1):
        epoch_t0 = time.monotonic()
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
        epoch_sec = round(time.monotonic() - epoch_t0, 2)

        record = {
            "epoch": epoch,
            "duration_sec": epoch_sec,
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
            f"val_GradMAE={val_metrics.get('grad_mae', 0.0):.3f}, "
            f"val_d1={val_metrics.get('delta_1', 0.0):.4f} | {epoch_sec:.1f}s"
        )

        if scheduler is not None:
            scheduler.step()

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
    eval_t0 = time.monotonic()
    if best_ckpt_path.is_file():
        load_checkpoint(best_ckpt_path, model=model, device=device)
    test_metrics, per_sequence_metrics = evaluate_loader(
        model=model,
        loader=test_loader,
        device=device,
        pred_is_disparity=False,
        align_mode=align_mode,
        min_depth=min_depth,
        max_depth=max_depth,
        max_batches=max_batches,
        return_per_sequence=True,
    )
    eval_duration_sec = round(time.monotonic() - eval_t0, 2)
    macro_metrics = compute_macro_metrics(per_sequence_metrics)
    total_train_sec = round(sum(float(h.get("duration_sec", 0.0)) for h in history), 2)
    device_name = torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu"

    test_payload = {
        "experiment_name": exp_name,
        "seed": int(config.get("seed", 42)),
        "dataset": data_cfg["dataset_name"],
        "split": "test",
        "split_sequences": split_sequences,
        "align_mode": align_mode,
        "device_name": device_name,
        "training_duration_sec": total_train_sec,
        "eval_duration_sec": eval_duration_sec,
        "best_val_abs_rel": round(best_val_abs_rel, 6),
        "trainable_params": trainable_params,
        "total_params": total_params,
        "metrics": test_metrics,
        "macro_metrics": macro_metrics,
        "per_sequence_metrics": per_sequence_metrics,
    }
    save_json(test_payload, exp_dir / f"eval_test_{align_mode}.json")
    print(f"[OK] Entraînement terminé ({total_train_sec:.1f}s = {total_train_sec / 60.0:.2f} min sur {device_name}).")
    print(f"     Moyenne par image (micro)   : {test_metrics}")
    if macro_metrics:
        print(f"     Moyenne par séquence (macro): {macro_metrics}")
    if per_sequence_metrics:
        print(f"     Par séquence (test)         : {per_sequence_metrics}")
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
