"""Générateur de données synthétiques pour tester le pipeline sans dataset réel."""

from __future__ import annotations

import torch


def make_synthetic_endoscopy_batch(
    batch_size: int,
    height: int,
    width: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Crée un lot d'images et de profondeurs synthétiques simulant une cavité tubulaire.

    Le centre de l'image représente la partie profonde (lumen sombre) et les bords
    représentent les parois proches (plus lumineuses).
    """
    y = torch.linspace(-1.0, 1.0, height, device=device)
    x = torch.linspace(-1.0, 1.0, width, device=device)
    grid_y, grid_x = torch.meshgrid(y, x, indexing="ij")
    radius = torch.sqrt(grid_x**2 + grid_y**2).clamp(max=1.5)

    # Profondeur élevée au centre, faible sur les parois périphériques
    base_depth = (2.5 - 1.5 * (radius / 1.5)).unsqueeze(0).unsqueeze(0)
    depths = base_depth.repeat(batch_size, 1, 1, 1)

    # Image RGB simple corrélée à la proximité des parois (éclairage inversement lié à la distance)
    intensity = (1.0 / (0.5 + depths)).clamp(0.0, 1.0)
    rgb = torch.cat([intensity * 0.9, intensity * 0.55, intensity * 0.5], dim=1)
    noise = 0.03 * torch.randn_like(rgb)
    images = (rgb + noise).clamp(0.0, 1.0)
    return images, depths
