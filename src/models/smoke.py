"""Modèle CNN minimaliste utilisé uniquement pour les tests rapides (smoke test) sur CPU."""

from __future__ import annotations

import torch
import torch.nn as nn


class SmokeDepthNet(nn.Module):
    """Petit réseau convolutif pour vérifier le pipeline de bout en bout.

    Entrée : tenseur image RGB de forme (B, 3, H, W)
    Sortie : carte de profondeur positive de forme (B, 1, H, W)
    """

    def __init__(self, hidden_channels: int = 16) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, 1, kernel_size=3, padding=1),
            nn.Softplus(),  # Garantit une profondeur strictement positive (> 0)
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Passe avant (forward pass) : transforme les images en cartes de profondeur."""
        return self.net(images)
