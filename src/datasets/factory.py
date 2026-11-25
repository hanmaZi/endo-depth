"""Fabrique de datasets et générateur d'échantillons C3VD synthétiques pour tests."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np

from src.datasets.c3vd import C3VDDataset
from src.datasets.simcol3d import SimCol3DDataset


def create_synthetic_c3vd_demo(
    output_dir: str | Path,
    num_sequences: int = 3,
    frames_per_seq: int = 4,
    height: int = 128,
    width: int = 128,
) -> Path:
    """Génère un mini dataset sur disque au format exact C3VD (16-bit TIFF) pour tests/démos."""
    root = Path(output_dir)
    yy, xx = np.meshgrid(np.linspace(-1.0, 1.0, height), np.linspace(-1.0, 1.0, width), indexing="ij")
    radius = np.sqrt(xx**2 + yy**2)

    for seq_idx in range(num_sequences):
        seq_dir = root / f"demo_seq_{seq_idx + 1}"
        seq_dir.mkdir(parents=True, exist_ok=True)

        for frame_idx in range(frames_per_seq):
            shift = 0.08 * frame_idx
            # Profondeur entre ~15 mm (parois) et ~80 mm (fond du lumen) + petite bosse (polype)
            polyp_bump = 12.0 * np.exp(-((xx - 0.3 + shift) ** 2 + (yy + 0.1) ** 2) / 0.03)
            depth_mm = np.clip(75.0 - 55.0 * np.clip(radius, 0.0, 1.2) / 1.2 - polyp_bump, 5.0, 95.0)

            # Normales analytiques approximatives (Nx, Ny, Nz) pointant vers la caméra (+Z ou -Z)
            dz_dy, dz_dx = np.gradient(depth_mm)
            nx = -dz_dx / 15.0
            ny = -dz_dy / 15.0
            nz = np.ones_like(depth_mm)
            norm = np.sqrt(nx**2 + ny**2 + nz**2)
            normals_rgb = np.stack([nx / norm, ny / norm, nz / norm], axis=-1)

            # Image RGB réaliste avec reflet spéculaire synthétique
            intensity = np.clip(18.0 / (depth_mm + 5.0), 0.1, 1.0)
            specular = 0.5 * np.exp(-((xx + 0.2) ** 2 + (yy - 0.25) ** 2) / 0.005)
            rgb = np.stack(
                [
                    np.clip(intensity * 0.92 + specular, 0.0, 1.0),
                    np.clip(intensity * 0.45 + specular, 0.0, 1.0),
                    np.clip(intensity * 0.40 + specular, 0.0, 1.0),
                ],
                axis=-1,
            )

            rgb_u8 = (rgb * 255.0).astype(np.uint8)
            depth_u16 = np.clip((depth_mm / 100.0) * 65535.0, 0, 65535).astype(np.uint16)
            normals_u16 = np.clip(((normals_rgb + 1.0) * 0.5) * 65535.0, 0, 65535).astype(np.uint16)

            stem = f"{frame_idx:04d}"
            cv2.imencode(".png", cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2BGR))[1].tofile(
                str(seq_dir / f"{stem}_color.png")
            )
            cv2.imencode(".tiff", depth_u16)[1].tofile(str(seq_dir / f"{stem}_depth.tiff"))
            # OpenCV écrit en BGR -> inverse RGB->BGR avant écriture pour respecter lecture RGB
            cv2.imencode(".tiff", normals_u16[:, :, ::-1])[1].tofile(
                str(seq_dir / f"{stem}_normals.tiff")
            )

    return root


def build_dataset(config: dict[str, Any], split: str) -> C3VDDataset | SimCol3DDataset:
    """Instancie le dataset configuré dans le YAML pour un split donné."""
    data_cfg = config["data"]
    name = str(data_cfg["dataset_name"]).lower()
    image_size = (int(data_cfg["image_size"][0]), int(data_cfg["image_size"][1]))
    root_dir = Path(data_cfg.get("dataset_path") or Path(config["paths"]["data_dir"]) / name)

    common_kwargs = {
        "root_dir": root_dir,
        "split": split,
        "image_size": image_size,
        "min_depth_mm": float(data_cfg.get("min_depth_mm", 1.0)),
        "max_depth_mm": float(data_cfg.get("max_depth_mm", 100.0 if name == "c3vd" else 200.0)),
        "val_ratio": float(data_cfg.get("val_ratio", 0.15)),
        "test_ratio": float(data_cfg.get("test_ratio", 0.15)),
        "seed": int(config.get("seed", 42)),
        "augment": (split == "train") and bool(data_cfg.get("augment", True)),
    }

    if name == "c3vd":
        return C3VDDataset(**common_kwargs)
    if name == "simcol3d":
        common_kwargs["depth_max_mm"] = float(data_cfg.get("depth_max_mm", 200.0))
        return SimCol3DDataset(**common_kwargs)
    raise ValueError(f"Dataset '{name}' non supporté par build_dataset (attendu: c3vd, simcol3d).")
