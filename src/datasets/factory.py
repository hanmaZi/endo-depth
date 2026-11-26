"""Fabrique de datasets et générateur d'échantillons C3VD synthétiques pour tests."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

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
        "color_augment": (split == "train") and bool(data_cfg.get("color_augment", False)),
        "sequence_splits": data_cfg.get("sequence_splits"),
    }

    if name == "c3vd":
        common_kwargs["expected_sequences"] = data_cfg.get("expected_sequences")
        common_kwargs["include_sequences"] = data_cfg.get("selected_sequences")
        return C3VDDataset(**common_kwargs)
    if name == "simcol3d":
        common_kwargs["depth_max_mm"] = float(data_cfg.get("depth_max_mm", 200.0))
        return SimCol3DDataset(**common_kwargs)
    raise ValueError(f"Dataset '{name}' non supporté par build_dataset (attendu: c3vd, simcol3d).")


def _find_subdir_bounded(root: Path, target_names: tuple[str, ...], max_depth: int = 3) -> list[Path]:
    """Cherche des sous-dossiers non vides portant un nom donné jusqu'à une profondeur max."""
    matches: list[Path] = []
    if not root.is_dir():
        return matches

    def _walk(current: Path, depth: int) -> None:
        if depth > max_depth:
            return
        try:
            entries = sorted(current.iterdir())
        except OSError:
            return
        for entry in entries:
            if not entry.is_dir() or entry.name.startswith("."):
                continue
            if entry.name.lower() in target_names and any(entry.iterdir()):
                matches.append(entry)
            elif depth < max_depth:
                _walk(entry, depth + 1)

    _walk(root, 1)
    return matches


def resolve_dataset_dir(
    dataset_name: str,
    drive_data_base: str | Path,
    local_data_base: str | Path,
    demo_fallback_dir: str | Path,
    expected_sequences: list[str] | tuple[str, ...] | None = None,
    selected_sequences: Iterable[str] | None = None,
    allow_demo_fallback: bool | None = None,
    num_demo_sequences: int = 4,
    frames_per_seq: int = 6,
) -> tuple[Path, str, bool]:
    """Résout le dossier du dataset depuis Drive/local (y compris c3vd_zips mixte zip+dossiers).

    Si `selected_sequences` est fourni, ne copie/extrait et ne vérifie que ces séquences.
    Si `expected_sequences` ou `selected_sequences` est fourni et `allow_demo_fallback` n'est pas
    forcé à True, échoue explicitement si aucun dataset réel n'est trouvé.
    """
    drive_base = Path(drive_data_base)
    local_base = Path(local_data_base)
    name = dataset_name.lower()
    selected_list = [str(s) for s in selected_sequences] if selected_sequences is not None else None
    if allow_demo_fallback is None:
        allow_demo_fallback = not bool(expected_sequences or selected_list)

    if name == "c3vd":
        zips_dir = drive_base / "c3vd_zips"
        raw_dir = drive_base / "c3vd"
        target_dir = local_base / "c3vd"
        source_dirs: list[Path] = []

        if zips_dir.is_dir() and any(zips_dir.iterdir()):
            source_dirs.append(zips_dir)
        if raw_dir.is_dir() and any(raw_dir.iterdir()):
            source_dirs.append(raw_dir)

        # Recherche bornée dans MyDrive si placé dans un autre sous-dossier (ex. PFE/endo-depth/data)
        if not source_dirs:
            mydrive_root = Path("/content/drive/MyDrive")
            if not mydrive_root.is_dir():
                for parent in drive_base.parents:
                    if parent.name == "MyDrive":
                        mydrive_root = parent
                        break
            if mydrive_root.is_dir():
                found_alt = _find_subdir_bounded(mydrive_root, ("c3vd_zips", "c3vd"), max_depth=4)
                if found_alt:
                    print(f"[Auto-detect] Dossier(s) C3VD trouvé(s) dans MyDrive : {found_alt}")
                    source_dirs.extend(found_alt)

        if source_dirs:
            for src_dir in source_dirs:
                if src_dir.resolve() != target_dir.resolve():
                    prepare_c3vd_from_source(src_dir, target_dir, sequences=selected_list)
            discover_c3vd_records(
                target_dir,
                expected_sequences=expected_sequences,
                include_sequences=selected_list,
            )
            return target_dir, "c3vd", False

        if target_dir.is_dir() and any(target_dir.iterdir()):
            discover_c3vd_records(
                target_dir,
                expected_sequences=expected_sequences,
                include_sequences=selected_list,
            )
            return target_dir, "c3vd", False

    else:
        candidate = drive_base / name
        if candidate.is_dir() and any(candidate.iterdir()):
            return candidate, name, False

    if not allow_demo_fallback:
        mydrive_listing: list[str] = []
        mydrive = Path("/content/drive/MyDrive")
        if mydrive.is_dir():
            mydrive_listing = sorted(p.name for p in mydrive.iterdir() if not p.name.startswith("."))[:20]
        raise FileNotFoundError(
            f"Aucune donnée réelle '{name}' trouvée sous {drive_base} "
            f"(attendu : {expected_sequences}). "
            f"Dossiers visibles dans MyDrive : {mydrive_listing}. "
            "Vérifie le chemin Drive ou mets EXPECTED_SEQUENCES = [] pour tester en mode démo."
        )

    demo_dir = create_synthetic_c3vd_demo(
        demo_fallback_dir,
        num_sequences=num_demo_sequences,
        frames_per_seq=frames_per_seq,
    )
    return demo_dir, "c3vd", True
