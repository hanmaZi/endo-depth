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

    return {
        "abs_rel": abs_rel,
        "sq_rel": sq_rel,
        "rmse": rmse,
        "rmse_log": rmse_log,
        "delta_1": delta_1,
        "delta_2": delta_2,
        "delta_3": delta_3,
    }


class DepthMetricTracker:
    """Accumule les métriques image par image sans saturer la RAM."""

    METRIC_KEYS = ("abs_rel", "sq_rel", "rmse", "rmse_log", "delta_1", "delta_2", "delta_3")

    def __init__(self) -> None:
        self.sums: dict[str, float] = {k: 0.0 for k in self.METRIC_KEYS}
        self.count: int = 0

    def update_batch(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor,
        pred_is_disparity: bool = False,
        align_mode: AlignMode = "median",
        min_depth: float = 1e-3,
        max_depth: float = 200.0,
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

    def compute(self) -> dict[str, float]:
        """Retourne la moyenne de chaque métrique sur toutes les images évaluées."""
        if self.count == 0:
            return {k: 0.0 for k in self.METRIC_KEYS}
        return {k: round(self.sums[k] / self.count, 6) for k in self.METRIC_KEYS}
