"""Métriques standard d'estimation de profondeur monoculaire avec alignement d'échelle."""

from __future__ import annotations

from typing import Literal

import torch


AlignMode = Literal["median", "scale_shift", "none"]


def valid_depth_mask(
    target: torch.Tensor,
    mask: torch.Tensor,
    min_depth: float = 1e-3,
    max_depth: float | None = None,
) -> torch.Tensor:
    """Exclut les coins sans profondeur (< min_depth) et valeurs tronquées (>= max_depth)."""
    gt = target.squeeze().float()
    m = mask.squeeze().bool() & torch.isfinite(gt) & (gt >= min_depth)
    if max_depth is not None:
        m = m & (gt < (max_depth - 1e-3))
    return m


def align_prediction(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    pred_is_disparity: bool = False,
    align_mode: AlignMode = "median",
    min_depth: float = 1e-3,
    max_depth: float = 200.0,
) -> torch.Tensor:
    """Aligne une carte prédite (1, H, W) ou (H, W) sur la vérité terrain en millimètres.

    - Si `pred_is_disparity=True` et `align_mode='median'`, convertit d'abord en pseudo-profondeur
      via 1 / max(pred, eps) puis applique le facteur d'échelle médian.
    - Si `pred_is_disparity=True` et `align_mode='scale_shift'`, ajuste s*pred + t vers 1/target
      par moindres carrés (protocole standard Depth Anything V2 / MiDaS) puis repasse en profondeur.
    """
    p = pred.squeeze().float()
    gt = target.squeeze().float()
    m = valid_depth_mask(gt, mask, min_depth=min_depth, max_depth=max_depth)

    if not m.any():
        return p.clamp(min=min_depth, max=max_depth)

    eps = 1e-6
    if pred_is_disparity:
        if align_mode == "scale_shift":
            # Ajuste a * p + b ~= 1 / gt dans l'espace disparité
            gt_disp = 1.0 / gt[m].clamp(min=eps)
            p_m = p[m]
            A = torch.stack([p_m, torch.ones_like(p_m)], dim=-1)
            sol = torch.linalg.lstsq(A, gt_disp.unsqueeze(-1)).solution.squeeze(-1)
            scale, shift = sol[0], sol[1]
            # Garde-fou si la régression s'inverse numériquement
            if scale <= eps:
                pred_depth = 1.0 / p.clamp(min=eps)
                ratio = torch.median(gt[m]) / torch.median(pred_depth[m]).clamp(min=eps)
                return (pred_depth * ratio).clamp(min=min_depth, max=max_depth)
            aligned_disp = (scale * p + shift).clamp(min=1.0 / max_depth)
            return (1.0 / aligned_disp).clamp(min=min_depth, max=max_depth)

        p = 1.0 / p.clamp(min=eps)

    if align_mode == "median":
        ratio = torch.median(gt[m]) / torch.median(p[m]).clamp(min=eps)
        p = p * ratio
    elif align_mode == "scale_shift":
        p_m = p[m]
        gt_m = gt[m]
        A = torch.stack([p_m, torch.ones_like(p_m)], dim=-1)
        sol = torch.linalg.lstsq(A, gt_m.unsqueeze(-1)).solution.squeeze(-1)
        if sol[0] > eps:
            p = sol[0] * p + sol[1]
        else:
            ratio = torch.median(gt_m) / torch.median(p_m).clamp(min=eps)
            p = p * ratio
    elif align_mode != "none":
        raise ValueError(f"align_mode inconnu : {align_mode}")

    return p.clamp(min=min_depth, max=max_depth)


def compute_depth_metrics_single(
    pred_mm: torch.Tensor,
    target_mm: torch.Tensor,
    mask: torch.Tensor,
    min_depth: float = 1e-3,
    max_depth: float | None = None,
) -> dict[str, float]:
    """Calcule AbsRel, SqRel, RMSE, RMSE_log et delta < 1.25 / 1.25^2 / 1.25^3 sur 1 image."""
    p = pred_mm.squeeze().float()
    gt = target_mm.squeeze().float()
    m = valid_depth_mask(gt, mask, min_depth=min_depth, max_depth=max_depth)
    if not m.any():
        return {}

    p_val = p[m].clamp(min=min_depth)
    gt_val = gt[m].clamp(min=min_depth)

    thresh = torch.maximum(gt_val / p_val, p_val / gt_val)
    delta_1 = (thresh < 1.25).float().mean().item()
    delta_2 = (thresh < (1.25**2)).float().mean().item()
    delta_3 = (thresh < (1.25**3)).float().mean().item()

    diff = p_val - gt_val
    abs_rel = (torch.abs(diff) / gt_val).mean().item()
    sq_rel = ((diff**2) / gt_val).mean().item()
    rmse = torch.sqrt((diff**2).mean()).item()
    rmse_log = torch.sqrt(((torch.log(p_val) - torch.log(gt_val)) ** 2).mean()).item()

    # MAE des gradients spatiaux (mm/px) sur paires de pixels adjacents tous deux valides
    grad_errs: list[torch.Tensor] = []
    if p.ndim == 2 and p.shape[1] > 1:
        mx = m[:, 1:] & m[:, :-1]
        if mx.any():
            dx_p = p[:, 1:] - p[:, :-1]
            dx_gt = gt[:, 1:] - gt[:, :-1]
            grad_errs.append(torch.abs(dx_p[mx] - dx_gt[mx]))
    if p.ndim == 2 and p.shape[0] > 1:
        my = m[1:, :] & m[:-1, :]
        if my.any():
            dy_p = p[1:, :] - p[:-1, :]
            dy_gt = gt[1:, :] - gt[:-1, :]
            grad_errs.append(torch.abs(dy_p[my] - dy_gt[my]))
    grad_mae = torch.cat(grad_errs).mean().item() if grad_errs else 0.0

    return {
        "abs_rel": abs_rel,
        "sq_rel": sq_rel,
        "rmse": rmse,
        "rmse_log": rmse_log,
        "grad_mae": grad_mae,
        "delta_1": delta_1,
        "delta_2": delta_2,
        "delta_3": delta_3,
    }


class DepthMetricTracker:
    """Accumule les métriques image par image (globales et par séquence) sans saturer la RAM."""

    METRIC_KEYS = (
        "abs_rel",
        "sq_rel",
        "rmse",
        "rmse_log",
        "grad_mae",
        "delta_1",
        "delta_2",
        "delta_3",
    )

    def __init__(self) -> None:
        self.sums: dict[str, float] = {k: 0.0 for k in self.METRIC_KEYS}
        self.count: int = 0
        self.seq_sums: dict[str, dict[str, float]] = {}
        self.seq_counts: dict[str, int] = {}

    def update_batch(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor,
        pred_is_disparity: bool = False,
        align_mode: AlignMode = "median",
        min_depth: float = 1e-3,
        max_depth: float = 200.0,
        sequences: list[str] | tuple[str, ...] | None = None,
    ) -> None:
        """Aligne et évalue chaque image du batch."""
        batch_size = pred.shape[0]
        for idx in range(batch_size):
            aligned = align_prediction(
                pred[idx],
                target[idx],
                mask[idx],
                pred_is_disparity=pred_is_disparity,
                align_mode=align_mode,
                min_depth=min_depth,
                max_depth=max_depth,
            )
            metrics = compute_depth_metrics_single(
                aligned, target[idx], mask[idx], min_depth=min_depth, max_depth=max_depth
            )
            if not metrics:
                continue
            for key in self.METRIC_KEYS:
                self.sums[key] += metrics[key]
            self.count += 1

            if sequences is not None and idx < len(sequences):
                seq = str(sequences[idx])
                if seq not in self.seq_sums:
                    self.seq_sums[seq] = {k: 0.0 for k in self.METRIC_KEYS}
                    self.seq_counts[seq] = 0
                for key in self.METRIC_KEYS:
                    self.seq_sums[seq][key] += metrics[key]
                self.seq_counts[seq] += 1

    def compute(self) -> dict[str, float]:
        """Retourne la moyenne de chaque métrique sur toutes les images évaluées (micro)."""
        if self.count == 0:
            return {k: 0.0 for k in self.METRIC_KEYS}
        return {k: round(self.sums[k] / self.count, 6) for k in self.METRIC_KEYS}

    def compute_per_sequence(self) -> dict[str, dict[str, float]]:
        """Retourne les métriques moyennes détaillées par nom de séquence."""
        out: dict[str, dict[str, float]] = {}
        for seq in sorted(self.seq_counts):
            cnt = self.seq_counts[seq]
            if cnt == 0:
                continue
            seq_metrics = {k: round(self.seq_sums[seq][k] / cnt, 6) for k in self.METRIC_KEYS}
            seq_metrics["num_evaluated_frames"] = float(cnt)
            out[seq] = seq_metrics
        return out

    def compute_macro(self) -> dict[str, float]:
        """Retourne la moyenne des moyennes par séquence (macro, chaque séquence pèse autant)."""
        return compute_macro_metrics(self.compute_per_sequence())


def compute_macro_metrics(per_sequence_metrics: dict[str, dict[str, float]]) -> dict[str, float]:
    """Calcule la moyenne non pondérée (macro) sur les séquences évaluées."""
    seq_rows = [row for row in per_sequence_metrics.values() if row]
    if not seq_rows:
        return {}
    out: dict[str, float] = {}
    for key in DepthMetricTracker.METRIC_KEYS:
        vals = [float(row[key]) for row in seq_rows if key in row]
        if vals:
            out[key] = round(sum(vals) / len(vals), 6)
    out["num_evaluated_sequences"] = float(len(seq_rows))
    out["num_evaluated_frames"] = float(
        sum(float(row.get("num_evaluated_frames", 0.0)) for row in seq_rows)
    )
    return out
