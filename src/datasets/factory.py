"""Fabrique de datasets et générateur d'échantillons C3VD synthétiques pour tests."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np

from src.datasets.c3vd import C3VDDataset, discover_c3vd_records, prepare_c3vd_from_source
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

            # Normales analytiques approximatives (Nx, Ny, Nz) pointant vers la caméra (-Z)
            dz_dy, dz_dx = np.gradient(depth_mm)
            nx = dz_dx / 15.0
            ny = dz_dy / 15.0
            nz = -np.ones_like(depth_mm)
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
        "sequence_splits": data_cfg.get("sequence_splits"),
    }

    if name == "c3vd":
        common_kwargs["expected_sequences"] = data_cfg.get("expected_sequences")
        return C3VDDataset(**common_kwargs)
    if name == "simcol3d":
        common_kwargs["depth_max_mm"] = float(data_cfg.get("depth_max_mm", 200.0))
        return SimCol3DDataset(**common_kwargs)
    raise ValueError(f"Dataset '{name}' non supporté par build_dataset (attendu: c3vd, simcol3d).")


def resolve_dataset_dir(
    dataset_name: str,
    drive_data_base: str | Path,
    local_data_base: str | Path,
    demo_fallback_dir: str | Path,
    expected_sequences: list[str] | tuple[str, ...] | None = None,
    num_demo_sequences: int = 4,
    frames_per_seq: int = 6,
) -> tuple[Path, str, bool]:
    """Résout le dossier du dataset depuis Drive/local (y compris c3vd_zips mixte zip+dossiers).

    Retourne `(dataset_dir, effective_dataset_name, is_synthetic_demo)`.
    """
    drive_base = Path(drive_data_base)
    local_base = Path(local_data_base)
    name = dataset_name.lower()

    if name == "c3vd":
        zips_dir = drive_base / "c3vd_zips"
        raw_dir = drive_base / "c3vd"
        target_dir = local_base / "c3vd"
        found_real = False

        if zips_dir.is_dir() and any(zips_dir.iterdir()):
            prepare_c3vd_from_source(zips_dir, target_dir)
            found_real = True

        if raw_dir.is_dir() and any(raw_dir.iterdir()):
            if raw_dir.resolve() != target_dir.resolve():
                prepare_c3vd_from_source(raw_dir, target_dir)
            else:
                target_dir = raw_dir
            found_real = True

        if found_real:
            discover_c3vd_records(target_dir, expected_sequences=expected_sequences)
            return target_dir, "c3vd", False

    else:
        candidate = drive_base / name
        if candidate.is_dir() and any(candidate.iterdir()):
            return candidate, name, False

    demo_dir = create_synthetic_c3vd_demo(
        demo_fallback_dir,
        num_sequences=num_demo_sequences,
        frames_per_seq=frames_per_seq,
    )
    return demo_dir, "c3vd", True
