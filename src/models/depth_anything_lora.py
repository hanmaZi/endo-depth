"""Wrapper Depth-Anything-V2-Small avec adaptation LoRA pour l'endoscopie."""

from __future__ import annotations

import math
from typing import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    AutoModelForDepthEstimation,
    DepthAnythingConfig,
    DepthAnythingForDepthEstimation,
)


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class LoRALinear(nn.Module):
    """Couche linéaire figée augmentée de deux matrices de rang faible (Low-Rank Adaptation).

    y = W_0(x) + (alpha / r) * B(A(dropout(x)))
    À l'initialisation, B = 0 donc le modèle pré-entraîné est rigoureusement préservé.
    """

    def __init__(
        self,
        base_linear: nn.Linear,
        rank: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.05,
    ) -> None:
        super().__init__()
        if rank <= 0:
            raise ValueError(f"Le rang LoRA doit être > 0 (reçu {rank}).")

        self.base_linear = base_linear
        for param in self.base_linear.parameters():
            param.requires_grad = False

        self.rank = int(rank)
        self.scaling = float(alpha) / float(rank)
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

        in_features = base_linear.in_features
        out_features = base_linear.out_features
        self.lora_a = nn.Linear(in_features, self.rank, bias=False)
        self.lora_b = nn.Linear(self.rank, out_features, bias=False)

        nn.init.kaiming_uniform_(self.lora_a.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_b.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base_linear(x)
        lora_out = self.lora_b(self.lora_a(self.dropout(x))) * self.scaling
        return base_out + lora_out


def inject_lora_adapters(
    module: nn.Module,
    target_names: Iterable[str] = ("q_proj", "v_proj"),
    rank: int = 8,
    alpha: float = 16.0,
    dropout: float = 0.05,
) -> int:
    """Remplace les couches nn.Linear ciblées par des modules LoRALinear."""
    targets = set(target_names)
    replaced = 0
    for child_name, child in list(module.named_children()):
        if isinstance(child, nn.Linear) and child_name in targets:
            setattr(module, child_name, LoRALinear(child, rank=rank, alpha=alpha, dropout=dropout))
            replaced += 1
        else:
            replaced += inject_lora_adapters(
                child,
                target_names=targets,
                rank=rank,
                alpha=alpha,
                dropout=dropout,
            )
    return replaced


class DepthAnythingV2LoRA(nn.Module):
    """Depth-Anything-V2-Small utilisable en zero-shot (disparité) ou fine-tuné LoRA (mm).

    - mode='zeroshot' : retourne la disparité relative brute du modèle pré-entraîné.
    - mode='metric'   : remplace la ReLU finale par une Sigmoid bornée sur [0, max_depth_mm]
                        (méthode officielle Depth Anything V2 Metric) et applique LoRA sur
                        l'encodeur DINOv2.
    """

    def __init__(
        self,
        pretrained_id: str = "depth-anything/Depth-Anything-V2-Small-hf",
        mode: str = "metric",
        max_depth_mm: float = 100.0,
        min_depth_mm: float = 1.0,
        use_lora: bool = True,
        lora_rank: int = 8,
        lora_alpha: float = 16.0,
        lora_dropout: float = 0.05,
        lora_targets: tuple[str, ...] = ("q_proj", "v_proj"),
        train_neck: bool = True,
        train_head: bool = True,
        from_config_only: bool = False,
    ) -> None:
        super().__init__()
        if mode not in {"zeroshot", "metric"}:
            raise ValueError(f"Mode inconnu : {mode}. Attendu : 'zeroshot' ou 'metric'.")

        self.mode = mode
        self.min_depth_mm = float(min_depth_mm)
        self.max_depth_mm = float(max_depth_mm)

        if from_config_only:
            # Utile pour les tests unitaires rapides sur CPU sans téléchargement internet
            self.model = DepthAnythingForDepthEstimation(DepthAnythingConfig())
        else:
            self.model = AutoModelForDepthEstimation.from_pretrained(pretrained_id)

        self.patch_size = int(getattr(self.model.config.backbone_config, "patch_size", 14))

        mean = torch.tensor(IMAGENET_MEAN, dtype=torch.float32).view(1, 3, 1, 1)
        std = torch.tensor(IMAGENET_STD, dtype=torch.float32).view(1, 3, 1, 1)
        self.register_buffer("mean", mean, persistent=False)
        self.register_buffer("std", std, persistent=False)

        if self.mode == "metric":
            # Remplace ReLU finale pour éviter les gradients nuls et garder les poids conv1..3
            self.model.head.activation2 = nn.Identity()
            # Facteur négatif initial : une forte disparité pré-entraînée -> faible profondeur (proche)
            self.metric_scale = nn.Parameter(torch.tensor(-0.5))
            self.metric_shift = nn.Parameter(torch.tensor(0.0))

            # Gèle tout par défaut, puis débloque LoRA + head (+ neck si activé)
            for param in self.model.parameters():
                param.requires_grad = False

            if use_lora:
                num_replaced = inject_lora_adapters(
                    self.model.backbone,
                    target_names=lora_targets,
                    rank=lora_rank,
                    alpha=lora_alpha,
                    dropout=lora_dropout,
                )
                if num_replaced == 0:
                    raise RuntimeError(f"Aucune couche {lora_targets} trouvée dans le backbone.")

            if train_neck:
                for param in self.model.neck.parameters():
                    param.requires_grad = True
            if train_head:
                for param in self.model.head.parameters():
                    param.requires_grad = True
        else:
            for param in self.model.parameters():
                param.requires_grad = False

    def _prepare_inputs(self, images: torch.Tensor) -> tuple[torch.Tensor, tuple[int, int]]:
        """Normalise ImageNet et ajuste (H, W) au multiple supérieur de patch_size (14)."""
        _, _, h, w = images.shape
        normed = (images - self.mean.to(images.device, images.dtype)) / self.std.to(
            images.device, images.dtype
        )

        target_h = math.ceil(h / self.patch_size) * self.patch_size
        target_w = math.ceil(w / self.patch_size) * self.patch_size
        if target_h != h or target_w != w:
            normed = F.interpolate(
                normed, size=(target_h, target_w), mode="bilinear", align_corners=False
            )
        return normed, (h, w)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Retourne un tenseur (B, 1, H, W) : disparité en zeroshot ou profondeur en mm en metric."""
        pixel_values, orig_hw = self._prepare_inputs(images)
        raw_out = self.model(pixel_values=pixel_values).predicted_depth.unsqueeze(1)

        if raw_out.shape[-2:] != orig_hw:
            raw_out = F.interpolate(raw_out, size=orig_hw, mode="bilinear", align_corners=False)

        if self.mode == "zeroshot":
            return raw_out

        # Conversion bornée sur [min_depth_mm, max_depth_mm]
        span = self.max_depth_mm - self.min_depth_mm
        return self.min_depth_mm + span * torch.sigmoid(self.metric_scale * raw_out + self.metric_shift)
