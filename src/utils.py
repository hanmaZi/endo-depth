"""Fonctions utilitaires partagées entre l'entraînement et l'évaluation."""

from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml


def load_config(config_path: str | Path) -> dict[str, Any]:
    """Charge un fichier de configuration YAML et retourne un dictionnaire."""
    path = Path(config_path)
    if not path.is_file():
        raise FileNotFoundError(f"Fichier de configuration introuvable : {path}")
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Le fichier YAML doit contenir un mapping clé-valeur : {path}")
    return config


def set_seed(seed: int) -> None:
    """Fixe les graines aléatoires pour garantir la reproductibilité des résultats."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(preference: str = "auto") -> torch.device:
    """Sélectionne le périphérique de calcul (GPU CUDA si disponible, sinon CPU)."""
    pref = preference.lower()
    if pref == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if pref == "cuda" and not torch.cuda.is_available():
        print("[Attention] CUDA demandé mais indisponible ici -> bascule sur CPU.")
        return torch.device("cpu")
    return torch.device(pref)


def count_parameters(model: torch.nn.Module) -> tuple[int, int]:
    """Retourne (nombre total de paramètres, nombre de paramètres entraînables)."""
    total = sum(param.numel() for param in model.parameters())
    trainable = sum(param.numel() for param in model.parameters() if param.requires_grad)
    return total, trainable


def save_checkpoint(
    checkpoint_path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    loss: float,
    best_metric: float | None = None,
) -> Path:
    """Sauvegarde un checkpoint léger pour reprendre après une coupure Colab/Kaggle."""
    path = Path(checkpoint_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "epoch": int(epoch),
        "loss": float(loss),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }
    if best_metric is not None:
        payload["best_metric"] = float(best_metric)
    torch.save(payload, path)
    return path


def save_json(data: Any, json_path: str | Path) -> Path:
    """Enregistre un objet JSON lisible avec création automatique des dossiers parents."""
    path = Path(json_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, ensure_ascii=False)
    return path


def save_config_copy(config: dict[str, Any], yaml_path: str | Path) -> Path:
    """Sauvegarde la configuration effective d'une expérience pour la reproductibilité."""
    path = Path(yaml_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False, allow_unicode=True)
    return path


def load_checkpoint(
    checkpoint_path: str | Path,
    model: torch.nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> dict[str, Any]:
    """Recharge un checkpoint sauvegardé sur le périphérique cible."""
    path = Path(checkpoint_path)
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint introuvable : {path}")

    checkpoint = torch.load(path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return checkpoint
