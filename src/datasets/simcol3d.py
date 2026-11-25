"""Loader PyTorch pour le dataset SimCol3D (données synthétiques de coloscopie)."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from src.datasets.base import (
    BaseEndoscopyDataset,
    SampleRecord,
    read_image_unchanged,
    split_records_by_sequence,
)


def _match_depth_file(rgb_path: Path, seq_dir: Path) -> Path | None:
    """Associe une frame SimCol3D (FrameBuffer_XXXX.png ou FrameBuffer/XXXX.png) à sa profondeur."""
    name = rgb_path.name
    if name.startswith("FrameBuffer_"):
        candidate = rgb_path.with_name(name.replace("FrameBuffer_", "Depth_", 1))
        if candidate.is_file():
            return candidate
    if rgb_path.parent.name.lower() in {"framebuffer", "rgb"}:
        for depth_folder in ("Depth", "depth"):
            candidate = seq_dir / depth_folder / name
            if candidate.is_file():
                return candidate
    return None


def discover_simcol3d_records(root_dir: str | Path) -> list[SampleRecord]:
    """Parcourt un dossier SimCol3D et collecte les paires (RGB, Depth)."""
    root = Path(root_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"Dossier SimCol3D introuvable : {root}")

    records: list[SampleRecord] = []
    # Supporte root/seq/... ou root/synthetic_train/seq/...
    candidate_dirs = sorted(p for p in root.rglob("*") if p.is_dir())
    if root not in candidate_dirs:
        candidate_dirs.insert(0, root)

    seen_rgb: set[Path] = set()
    for seq_dir in candidate_dirs:
        rgb_files = sorted(seq_dir.glob("FrameBuffer_*.png"))
        for sub_name in ("FrameBuffer", "rgb"):
            sub_dir = seq_dir / sub_name
            if sub_dir.is_dir():
                rgb_files.extend(sorted(sub_dir.glob("*.png")))

        for rgb_path in rgb_files:
            if rgb_path in seen_rgb:
                continue
            depth_path = _match_depth_file(rgb_path, seq_dir)
            if depth_path is None:
                continue
            seen_rgb.add(rgb_path)
            frame_id = rgb_path.stem.replace("FrameBuffer_", "")
            seq_label = str(seq_dir.relative_to(root)) if seq_dir != root else root.name
            records.append(
                SampleRecord(
                    rgb_path=rgb_path,
                    depth_path=depth_path,
                    sequence_name=seq_label,
                    frame_id=frame_id,
                )
            )

    if not records:
        raise RuntimeError(
            f"Aucune paire (FrameBuffer, Depth) trouvée dans {root}. "
            "Vérifie l'arborescence SimCol3D."
        )
    return records


class SimCol3DDataset(BaseEndoscopyDataset):
    """Dataset PyTorch pour SimCol3D (profondeur 16-bit PNG sur [0, depth_max_mm])."""

    def __init__(
        self,
        root_dir: str | Path,
        split: str = "all",
        image_size: tuple[int, int] = (256, 256),
        min_depth_mm: float = 1.0,
        max_depth_mm: float = 200.0,
        depth_max_mm: float = 200.0,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        seed: int = 42,
        augment: bool = False,
        sequence_splits: dict[str, list[str]] | None = None,
    ) -> None:
        self.depth_max_mm = float(depth_max_mm)
        all_records = discover_simcol3d_records(root_dir)
        split_records = split_records_by_sequence(
            all_records,
            split=split,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            seed=seed,
            sequence_splits=sequence_splits,
        )
        super().__init__(
            records=split_records,
            image_size=image_size,
            min_depth_mm=min_depth_mm,
            max_depth_mm=max_depth_mm,
            augment=augment,
        )

    def _decode_depth_mm(self, record: SampleRecord) -> np.ndarray:
        raw = read_image_unchanged(record.depth_path)
        if raw.ndim == 3:
            raw = raw[:, :, 0]
        scale = 65535.0 if raw.dtype == np.uint16 else 255.0
        return (raw.astype(np.float32) / scale) * self.depth_max_mm
