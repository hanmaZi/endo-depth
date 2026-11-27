"""Point d'entrée d'évaluation pour endo-depth (smoke test, zero-shot ou checkpoint fine-tuné).

Exemples :
    python -m src.eval --config configs/base.yaml --smoke-test
    python -m src.eval --config configs/c3vd_zeroshot.yaml
    python -m src.eval --config configs/c3vd_lora_grad_norm.yaml --checkpoint results/c3vd_lora_grad_norm/checkpoints/best.pt
"""

from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.datasets import (
    build_dataset,
    extract_colon_model,
    make_synthetic_endoscopy_batch,
    summarize_records_by_sequence,
)
from src.metrics import DepthMetricTracker, compute_macro_metrics
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
    return_per_sequence: bool = False,
) -> dict[str, float] | tuple[dict[str, float], dict[str, dict[str, float]]]:
    """Évalue un modèle sur un DataLoader et retourne les métriques standard moyennées."""
    model.eval()
    tracker = DepthMetricTracker()

    for batch_idx, batch in enumerate(tqdm(loader, desc="Évaluation", leave=False)):
        if max_batches is not None and batch_idx >= max_batches:
            break
        images = batch["image"].to(device)
        depths = batch["depth"].to(device)
        masks = batch["mask"].to(device)
        sequences = batch.get("sequence")

        preds = model(images)
        tracker.update_batch(
            preds,
            depths,
            masks,
            pred_is_disparity=pred_is_disparity,
            align_mode=align_mode,  # type: ignore[arg-type]
            min_depth=min_depth,
            max_depth=max_depth,
            sequences=sequences,
        )

    metrics = tracker.compute()
    metrics["num_evaluated_frames"] = float(tracker.count)
    if return_per_sequence:
        return metrics, tracker.compute_per_sequence()
    return metrics


def _summarize_metric_dicts(rows: list[dict[str, float]]) -> dict[str, dict[str, Any]]:
    """Agrège une liste de dictionnaires de métriques (moyenne ± écart-type ddof=1 si n>=2)."""
    if not rows:
        return {}
    keys = [k for k in DepthMetricTracker.METRIC_KEYS if k in rows[0]]
    summary: dict[str, dict[str, Any]] = {}
    for key in keys:
        vals = [float(r[key]) for r in rows if key in r]
        mean_val = sum(vals) / len(vals)
        if len(vals) > 1:
            var = sum((v - mean_val) ** 2 for v in vals) / (len(vals) - 1)
            std_val: float | None = math.sqrt(var)
            formatted = f"{mean_val:.4f} ± {std_val:.4f}"
        else:
            std_val = None
            formatted = f"{mean_val:.4f} (n=1)"
        summary[key] = {
            "mean": round(mean_val, 6),
            "std": round(std_val, 6) if std_val is not None else None,
            "formatted": formatted,
            "n_seeds": len(vals),
        }
    return summary


def aggregate_ablation_results(
    results_dir: str | Path,
    variant_runs: dict[str, list[str]],
    align_mode: str = "median",
) -> dict[str, Any]:
    """Agrège les JSON multi-graines (micro par image, macro par séquence, et détail par séquence)."""
    base = Path(results_dir)
    aggregated: dict[str, Any] = {}

    for label, exp_dirs in variant_runs.items():
        global_rows: list[dict[str, float]] = []
        macro_rows: list[dict[str, float]] = []
        per_seq_rows: dict[str, list[dict[str, float]]] = {}
        used_exps: list[str] = []

        for exp_dir_name in exp_dirs:
            eval_path = base / exp_dir_name / f"eval_test_{align_mode}.json"
            if not eval_path.is_file():
                continue
            with eval_path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
            used_exps.append(exp_dir_name)
            if "metrics" in payload:
                global_rows.append(payload["metrics"])
            per_seq_dict = payload.get("per_sequence_metrics", {})
            for seq_name, seq_metrics in per_seq_dict.items():
                per_seq_rows.setdefault(seq_name, []).append(seq_metrics)
            macro = payload.get("macro_metrics") or compute_macro_metrics(per_seq_dict)
            if macro:
                macro_rows.append(macro)

        if not global_rows:
            continue

        aggregated[label] = {
            "num_seeds": len(global_rows),
            "experiments": used_exps,
            "overall": _summarize_metric_dicts(global_rows),
            "macro": _summarize_metric_dicts(macro_rows),
            "per_sequence": {
                seq: _summarize_metric_dicts(rows)
                for seq, rows in sorted(per_seq_rows.items())
            },
        }

    return aggregated


def compare_variants_per_sequence(
    results_dir: str | Path,
    baseline_runs: list[str],
    contrib_runs: list[str],
    align_mode: str = "median",
) -> dict[str, Any]:
    """Compare appariée graine-à-graine baseline vs contribution sur chaque séquence de test.

    Calcule `delta = contrib - baseline` (négatif = meilleur pour AbsRel/SqRel/RMSE/RMSE_log/GradMAE ;
    positif = meilleur pour delta_1..3) et compte sur combien de séquences la contribution gagne.
    """
    base = Path(results_dir)
    higher_is_better = {"delta_1", "delta_2", "delta_3"}

    paired_payloads: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for b_exp, c_exp in zip(baseline_runs, contrib_runs):
        bp = base / b_exp / f"eval_test_{align_mode}.json"
        cp = base / c_exp / f"eval_test_{align_mode}.json"
        if not (bp.is_file() and cp.is_file()):
            continue
        with bp.open("r", encoding="utf-8") as fb, cp.open("r", encoding="utf-8") as fc:
            paired_payloads.append((json.load(fb), json.load(fc)))

    if not paired_payloads:
        return {}

    common_seqs = sorted(
        set.intersection(
            *(
                set(b.get("per_sequence_metrics", {}).keys())
                & set(c.get("per_sequence_metrics", {}).keys())
                for b, c in paired_payloads
            )
        )
    )

    def _paired_block(pairs_list: list[tuple[dict[str, float], dict[str, float]]]) -> dict[str, Any]:
        keys = [k for k in DepthMetricTracker.METRIC_KEYS if k in pairs_list[0][0] and k in pairs_list[0][1]]
        out: dict[str, Any] = {}
        for k in keys:
            b_vals = [float(b[k]) for b, _ in pairs_list]
            c_vals = [float(c[k]) for _, c in pairs_list]
            diffs = [c_v - b_v for b_v, c_v in zip(b_vals, c_vals)]
            n = len(diffs)
            b_mean = sum(b_vals) / n
            c_mean = sum(c_vals) / n
            d_mean = sum(diffs) / n
            if n > 1:
                d_std: float | None = math.sqrt(sum((d - d_mean) ** 2 for d in diffs) / (n - 1))
                d_fmt = f"{d_mean:+.4f} ± {d_std:.4f}"
            else:
                d_std = None
                d_fmt = f"{d_mean:+.4f} (n=1)"
            wins = (d_mean > 0.0) if k in higher_is_better else (d_mean < 0.0)
            out[k] = {
                "baseline_mean": round(b_mean, 6),
                "contrib_mean": round(c_mean, 6),
                "paired_diff_mean": round(d_mean, 6),
                "paired_diff_std": round(d_std, 6) if d_std is not None else None,
                "paired_diff_formatted": d_fmt,
                "contrib_wins": wins,
                "n_pairs": n,
            }
        return out

    per_sequence: dict[str, Any] = {}
    for seq in common_seqs:
        seq_pairs = [
            (b["per_sequence_metrics"][seq], c["per_sequence_metrics"][seq])
            for b, c in paired_payloads
        ]
        per_sequence[seq] = _paired_block(seq_pairs)

    win_summary: dict[str, Any] = {}
    if common_seqs:
        for k in DepthMetricTracker.METRIC_KEYS:
            seq_entries = [per_sequence[s][k] for s in common_seqs if k in per_sequence[s]]
            if not seq_entries:
                continue
            wins = sum(1 for e in seq_entries if e["contrib_wins"])
            total_seq = len(seq_entries)
            macro_diff = sum(e["paired_diff_mean"] for e in seq_entries) / total_seq
            win_summary[k] = {
                "contrib_wins_sequences": wins,
                "total_sequences": total_seq,
                "win_ratio_str": f"{wins}/{total_seq}",
                "mean_paired_diff_across_sequences": round(macro_diff, 6),
            }

    micro_pairs = [(b.get("metrics", {}), c.get("metrics", {})) for b, c in paired_payloads]
    macro_pairs = [
        (
            b.get("macro_metrics") or compute_macro_metrics(b.get("per_sequence_metrics", {})),
            c.get("macro_metrics") or compute_macro_metrics(c.get("per_sequence_metrics", {})),
        )
        for b, c in paired_payloads
    ]

    return {
        "num_paired_seeds": len(paired_payloads),
        "baseline_experiments": [b.get("experiment_name") for b, _ in paired_payloads],
        "contrib_experiments": [c.get("experiment_name") for _, c in paired_payloads],
        "micro_paired": _paired_block(micro_pairs),
        "macro_paired": _paired_block(macro_pairs) if all(m[0] and m[1] for m in macro_pairs) else {},
        "per_sequence": per_sequence,
        "sequence_win_summary": win_summary,
    }


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
    metrics, per_sequence_metrics = evaluate_loader(
        model=model,
        loader=loader,
        device=device,
        pred_is_disparity=pred_is_disparity,
        align_mode=align_mode,
        min_depth=float(eval_cfg.get("min_depth", data_cfg.get("min_depth_mm", 1.0))),
        max_depth=float(eval_cfg.get("max_depth", data_cfg.get("max_depth_mm", 100.0))),
        max_batches=max_batches,
        return_per_sequence=True,
    )

    macro_metrics = compute_macro_metrics(per_sequence_metrics)
    exp_name = str(config.get("experiment_name", "eval_run"))
    exp_dir = Path(config["paths"]["results_dir"]) / exp_name
    payload = {
        "experiment_name": exp_name,
        "dataset": data_cfg["dataset_name"],
        "split": split,
        "split_sequences": summarize_records_by_sequence(dataset.records)
        if hasattr(dataset, "records")
        else {},
        "align_mode": align_mode,
        "pred_is_disparity": pred_is_disparity,
        "checkpoint": str(checkpoint_path) if checkpoint_path else None,
        "metrics": metrics,
        "macro_metrics": macro_metrics,
        "per_sequence_metrics": per_sequence_metrics,
    }
    out_json = save_json(payload, exp_dir / f"eval_{split}_{align_mode}.json")
    save_config_copy(config, exp_dir / "config_used.yaml")
    print(f"[OK] Évaluation terminée -> {out_json}")
    print(f"     Moyenne par image (micro)    : {metrics}")
    if macro_metrics:
        print(f"     Moyenne par séquence (macro) : {macro_metrics}")
    if per_sequence_metrics:
        print(f"     Par séquence                 : {per_sequence_metrics}")
    return payload


def evaluate_checkpoint_on_sequences(
    config: dict[str, Any],
    dataset_dir: str | Path,
    test_sequences: list[str],
    checkpoint_path: str | Path | None = None,
    align_mode: str = "median",
    max_batches: int | None = None,
    train_sequences_used: list[str] | None = None,
) -> dict[str, Any]:
    """Réévalue n'importe quel modèle (Zero-Shot, v2, v3...) sur un même jeu de séquences test.

    Détecte automatiquement si un modèle de côlon présent en test figurait déjà dans le train
    d'origine (`has_model_leakage`) pour signaler tout biais éventuel.
    """
    cfg = copy.deepcopy(config)
    cfg.setdefault("data", {})["dataset_name"] = "c3vd"
    cfg["data"]["dataset_path"] = str(dataset_dir)
    cfg["data"]["selected_sequences"] = list(test_sequences)
    cfg["data"]["expected_sequences"] = list(test_sequences)
    cfg["data"]["sequence_splits"] = {"train": [], "val": [], "test": list(test_sequences)}

    device = get_device(cfg.get("device", "auto"))
    eval_cfg = cfg.get("evaluation", {})
    dataset = build_dataset(cfg, split="test")
    num_workers = 0 if device.type == "cpu" else int(cfg["data"].get("num_workers", 2))
    loader = DataLoader(
        dataset,
        batch_size=int(cfg["data"].get("batch_size", 4)),
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )

    model = build_model(cfg).to(device)
    ckpt_epoch = None
    if checkpoint_path is not None:
        ckpt = load_checkpoint(checkpoint_path, model=model, device=device)
        ckpt_epoch = int(ckpt.get("epoch", 0))

    pred_is_disparity = bool(eval_cfg.get("pred_is_disparity", False))
    metrics, per_seq = evaluate_loader(
        model=model,
        loader=loader,
        device=device,
        pred_is_disparity=pred_is_disparity,
        align_mode=align_mode,
        min_depth=float(eval_cfg.get("min_depth", cfg["data"].get("min_depth_mm", 1.0))),
        max_depth=float(eval_cfg.get("max_depth", cfg["data"].get("max_depth_mm", 100.0))),
        max_batches=max_batches,
        return_per_sequence=True,
    )

    train_seqs = list(train_sequences_used or [])
    train_models = sorted({extract_colon_model(s) for s in train_seqs})
    test_models = sorted({extract_colon_model(s) for s in test_sequences})
    overlapping_models = sorted(set(train_models) & set(test_models))

    return {
        "experiment_name": str(cfg.get("experiment_name", "eval_run")),
        "checkpoint": str(checkpoint_path) if checkpoint_path else None,
        "checkpoint_epoch": ckpt_epoch,
        "align_mode": align_mode,
        "pred_is_disparity": pred_is_disparity,
        "test_sequences": summarize_records_by_sequence(dataset.records),
        "train_models": train_models,
        "test_models": test_models,
        "overlapping_models": overlapping_models,
        "has_model_leakage": bool(overlapping_models),
        "metrics": metrics,
        "macro_metrics": compute_macro_metrics(per_seq),
        "per_sequence_metrics": per_seq,
    }


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
