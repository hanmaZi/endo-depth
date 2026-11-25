"""Fonctions de coût pour l'estimation de profondeur en coloscopie.

Inclut la contribution du PFE : perte combinée L1 + SiLog + gradients spatiaux
multi-échelles + cohérence des normales 3D de surface (pour préserver le relief
des polypes et des plis coliques).
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


def depth_to_normals(
    depth_mm: torch.Tensor,
    fx: float | None = None,
    fy: float | None = None,
) -> torch.Tensor:
    """Calcule les normales 3D unitaires (B, 3, H, W) à partir d'une carte de profondeur.

    Convention repère caméra : +X droite, +Y bas, +Z devant la caméra.
    Les normales retournées pointent vers la caméra (Nz <= 0).
    """
    b, _, h, w = depth_mm.shape
    device = depth_mm.device
    dtype = depth_mm.dtype

    # Si les focales exactes ne sont pas connues, approximation standard proportionnelle à la largeur
    fx_val = float(fx) if fx is not None else float(w)
    fy_val = float(fy) if fy is not None else float(w)
    cx = (w - 1) * 0.5
    cy = (h - 1) * 0.5

    u = torch.arange(w, device=device, dtype=dtype).view(1, 1, 1, w)
    v = torch.arange(h, device=device, dtype=dtype).view(1, 1, h, 1)

    z = depth_mm
    x = (u - cx) * z / fx_val
    y = (v - cy) * z / fy_val
    points_3d = torch.cat([x, y, z], dim=1)  # (B, 3, H, W)

    # Différences centrées avec padding répliqué pour conserver (H, W)
    padded = F.pad(points_3d, (1, 1, 1, 1), mode="replicate")
    dp_du = padded[:, :, 1:-1, 2:] - padded[:, :, 1:-1, :-2]  # vecteur droite
    dp_dv = padded[:, :, 2:, 1:-1] - padded[:, :, :-2, 1:-1]  # vecteur bas

    # Produit vectoriel (bas x droite) => pointe vers la caméra (-Z)
    normals = torch.cross(dp_dv, dp_du, dim=1)
    norm = torch.linalg.norm(normals, dim=1, keepdim=True).clamp(min=1e-6)
    return normals / norm


class SiLogLoss(nn.Module):
    """Scale-Invariant Logarithmic Loss (Eigen et al.) pour profondeur positive."""

    def __init__(self, variance_focus: float = 0.85, eps: float = 1e-4) -> None:
        super().__init__()
        self.variance_focus = float(variance_focus)
        self.eps = float(eps)

    def forward(
        self, pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        valid = mask.bool()
        if not valid.any():
            return pred.sum() * 0.0

        log_diff = torch.log(pred[valid].clamp(min=self.eps)) - torch.log(
            target[valid].clamp(min=self.eps)
        )
        mean_sq = (log_diff**2).mean()
        sq_mean = (log_diff.mean()) ** 2
        return torch.sqrt((mean_sq - self.variance_focus * sq_mean).clamp(min=1e-8))


class MultiScaleGradientLoss(nn.Module):
    """Perte L1 multi-échelles sur les gradients spatiaux (dx, dy) pour préserver les bords."""

    def __init__(self, num_scales: int = 3) -> None:
        super().__init__()
        self.num_scales = int(num_scales)

    def forward(
        self, pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        total = pred.sum() * 0.0
        p = pred
        gt = target
        m = mask.float()

        for scale in range(self.num_scales):
            if scale > 0:
                p = F.avg_pool2d(p, kernel_size=2, stride=2)
                gt = F.avg_pool2d(gt, kernel_size=2, stride=2)
                m = (F.avg_pool2d(m, kernel_size=2, stride=2) > 0.99).float()

            if p.shape[-1] < 2 or p.shape[-2] < 2:
                break

            m_x = (m[:, :, :, 1:] * m[:, :, :, :-1]) > 0.5
            m_y = (m[:, :, 1:, :] * m[:, :, :-1, :]) > 0.5

            grad_p_x = p[:, :, :, 1:] - p[:, :, :, :-1]
            grad_gt_x = gt[:, :, :, 1:] - gt[:, :, :, :-1]
            grad_p_y = p[:, :, 1:, :] - p[:, :, :-1, :]
            grad_gt_y = gt[:, :, 1:, :] - gt[:, :, :-1, :]

            loss_x = (
                torch.abs(grad_p_x[m_x] - grad_gt_x[m_x]).mean() if m_x.any() else p.sum() * 0.0
            )
            loss_y = (
                torch.abs(grad_p_y[m_y] - grad_gt_y[m_y]).mean() if m_y.any() else p.sum() * 0.0
            )
            total = total + 0.5 * (loss_x + loss_y)

        return total / max(1, self.num_scales)


class SurfaceNormalLoss(nn.Module):
    """Perte cosinus (1 - cos_sim) entre normales issues de la profondeur prédite et normales cibles."""

    def __init__(self, fx: float | None = None, fy: float | None = None) -> None:
        super().__init__()
        self.fx = fx
        self.fy = fy

    def forward(
        self,
        pred_depth: torch.Tensor,
        target_depth: torch.Tensor,
        mask: torch.Tensor,
        target_normals: torch.Tensor | None = None,
        has_normals: torch.Tensor | bool | None = None,
    ) -> torch.Tensor:
        pred_normals = depth_to_normals(pred_depth, fx=self.fx, fy=self.fy)
        pseudo_normals = depth_to_normals(target_depth, fx=self.fx, fy=self.fy)

        if target_normals is not None and has_normals is not None:
            if isinstance(has_normals, bool):
                gt_normals = target_normals if has_normals else pseudo_normals
            else:
                use_gt = has_normals.to(pred_depth.device).view(-1, 1, 1, 1).bool()
                gt_normals = torch.where(use_gt, target_normals, pseudo_normals)
        else:
            gt_normals = pseudo_normals

        # Harmonise l'orientation globale du repère si les normales GT pointent vers +Z
        m_2d = mask.squeeze(1).bool()
        if m_2d.any():
            nz_mean = gt_normals[:, 2, :, :][m_2d].mean()
            if nz_mean > 0.1:
                gt_normals = -gt_normals

        cos_sim = F.cosine_similarity(pred_normals, gt_normals, dim=1)
        if not m_2d.any():
            return pred_depth.sum() * 0.0
        return (1.0 - cos_sim[m_2d]).mean()


class EndoDepthLoss(nn.Module):
    """Loss composite configurable depuis YAML : L1 + SiLog + Gradients + Normales."""

    def __init__(
        self,
        w_l1: float = 1.0,
        w_silog: float = 1.0,
        w_grad: float = 0.5,
        w_normal: float = 0.5,
        grad_scales: int = 3,
        fx: float | None = None,
        fy: float | None = None,
    ) -> None:
        super().__init__()
        self.w_l1 = float(w_l1)
        self.w_silog = float(w_silog)
        self.w_grad = float(w_grad)
        self.w_normal = float(w_normal)

        self.silog = SiLogLoss()
        self.grad = MultiScaleGradientLoss(num_scales=grad_scales)
        self.normal = SurfaceNormalLoss(fx=fx, fy=fy)

    def forward(
        self,
        pred_depth: torch.Tensor,
        target_depth: torch.Tensor,
        mask: torch.Tensor,
        target_normals: torch.Tensor | None = None,
        has_normals: torch.Tensor | bool | None = None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        valid = mask.bool()
        zero = pred_depth.sum() * 0.0

        loss_l1 = torch.abs(pred_depth[valid] - target_depth[valid]).mean() if valid.any() else zero
        loss_silog = self.silog(pred_depth, target_depth, mask) if self.w_silog > 0 else zero
        loss_grad = self.grad(pred_depth, target_depth, mask) if self.w_grad > 0 else zero
        loss_normal = (
            self.normal(
                pred_depth,
                target_depth,
                mask,
                target_normals=target_normals,
                has_normals=has_normals,
            )
            if self.w_normal > 0
            else zero
        )

        total = (
            self.w_l1 * loss_l1
            + self.w_silog * loss_silog
            + self.w_grad * loss_grad
            + self.w_normal * loss_normal
        )

        breakdown = {
            "total": float(total.detach().item()),
            "l1": float(loss_l1.detach().item()),
            "silog": float(loss_silog.detach().item()),
            "grad": float(loss_grad.detach().item()),
            "normal": float(loss_normal.detach().item()),
        }
        return total, breakdown


def build_loss_from_config(config: dict[str, Any]) -> EndoDepthLoss:
    """Instancie EndoDepthLoss à partir de la section `loss` d'un fichier YAML."""
    loss_cfg = config.get("loss", {})
    return EndoDepthLoss(
        w_l1=float(loss_cfg.get("w_l1", 1.0)),
        w_silog=float(loss_cfg.get("w_silog", 1.0)),
        w_grad=float(loss_cfg.get("w_grad", 0.5)),
        w_normal=float(loss_cfg.get("w_normal", 0.5)),
        grad_scales=int(loss_cfg.get("grad_scales", 3)),
        fx=loss_cfg.get("fx"),
        fy=loss_cfg.get("fy"),
    )
