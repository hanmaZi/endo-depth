"""Fabrique de modèles à partir de la configuration YAML."""

from __future__ import annotations

from typing import Any

import torch.nn as nn

from src.models.depth_anything_lora import DepthAnythingV2LoRA
from src.models.smoke import SmokeDepthNet


def build_model(config: dict[str, Any]) -> nn.Module:
    """Instancie le modèle décrit dans `config['model']`."""
    model_cfg = config["model"]
    name = str(model_cfg["name"]).lower()

    if name == "smoke":
        return SmokeDepthNet(hidden_channels=int(model_cfg.get("hidden_channels", 16)))

    if name in {"depth_anything_v2_small", "depth_anything_v2_lora"}:
        data_cfg = config.get("data", {})
        lora_cfg = model_cfg.get("lora", {})
        mode = str(model_cfg.get("mode", "metric"))
        return DepthAnythingV2LoRA(
            pretrained_id=str(
                model_cfg.get("pretrained_id", "depth-anything/Depth-Anything-V2-Small-hf")
            ),
            mode=mode,
            max_depth_mm=float(data_cfg.get("max_depth_mm", 100.0)),
            min_depth_mm=float(data_cfg.get("min_depth_mm", 1.0)),
            use_lora=bool(model_cfg.get("use_lora", mode == "metric")),
            lora_rank=int(lora_cfg.get("rank", 8)),
            lora_alpha=float(lora_cfg.get("alpha", 16.0)),
            lora_dropout=float(lora_cfg.get("dropout", 0.05)),
            lora_targets=tuple(lora_cfg.get("targets", ["q_proj", "v_proj"])),
            train_neck=bool(model_cfg.get("train_neck", True)),
            train_head=bool(model_cfg.get("train_head", True)),
            from_config_only=bool(model_cfg.get("from_config_only", False)),
        )

    raise ValueError(f"Modèle inconnu dans la configuration : '{name}'.")
