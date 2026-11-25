"""Loader PyTorch pour le dataset C3VD (Colonoscopy 3D Video Dataset)."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from src.datasets.base import (
    BaseEndoscopyDataset,
    SampleRecord,
    read_image_unchanged,
    split_records_by_sequence,
)


def discover_c3vd_records(root_dir: str | Path) -> list[SampleRecord]:
    """Parcourt un dossier C3VD (format plat ou sous-dossiers color/depth/normals)."""
    root = Path(root_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"Dossier C3VD introuvable : {root}")

    records: list[SampleRecord] = []
    sequence_dirs = sorted(path for path in root.iterdir() if path.is_dir())
    if not sequence_dirs:
        sequence_dirs = [root]

    for seq_dir in sequence_dirs:
        color_candidates = sorted(seq_dir.glob("*_color.png"))
        if not color_candidates and (seq_dir / "color").is_dir():
            color_candidates = sorted((seq_dir / "color").glob("*_color.png"))

        for rgb_path in color_candidates:
            frame_id = rgb_path.stem.replace("_color", "")
            # Cherche d'abord à côté de l'image, sinon dans les sous-dossiers depth/ normals/
            depth_path = rgb_path.with_name(f"{frame_id}_depth.tiff")
            if not depth_path.is_file():
                alt_depth = seq_dir / "depth" / f"{frame_id}_depth.tiff"
                depth_path = alt_depth if alt_depth.is_file() else depth_path
            if not depth_path.is_file():
                continue

            normal_path = rgb_path.with_name(f"{frame_id}_normals.tiff")
            if not normal_path.is_file():
                alt_normal = seq_dir / "normals" / f"{frame_id}_normals.tiff"
                normal_path = alt_normal if alt_normal.is_file() else None

            occ_path = rgb_path.with_name(f"{frame_id}_occlusion.png")
            if not occ_path.is_file():
                alt_occ = seq_dir / "occlusion" / f"{frame_id}_occlusion.png"
                occ_path = alt_occ if alt_occ.is_file() else None

            records.append(
                SampleRecord(
                    rgb_path=rgb_path,
                    depth_path=depth_path,
                    normal_path=normal_path,
                    occlusion_path=occ_path,
                    sequence_name=seq_dir.name,
                    frame_id=frame_id,
                )
            )

    if not records:
        raise RuntimeError(
            f"Aucune paire (*_color.png, *_depth.tiff) trouvée dans {root}. "
            "Vérifie que les séquences C3VD sont bien décompressées."
        )
    return records


class C3VDDataset(BaseEndoscopyDataset):
    """Dataset PyTorch pour C3VD : profondeur 16-bit (0-100 mm) et normales 16-bit."""

    DEPTH_MAX_MM = 100.0

    def __init__(
        self,
        root_dir: str | Path,
        split: str = "all",
        image_size: tuple[int, int] = (256, 256),
        min_depth_mm: float = 1.0,
        max_depth_mm: float = 100.0,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        seed: int = 42,
        augment: bool = False,
    ) -> None:
        all_records = discover_c3vd_records(root_dir)
        split_records = split_records_by_sequence(
            all_records, split=split, val_ratio=val_ratio, test_ratio=test_ratio, seed=seed
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
        if raw.dtype == np.uint16:
            return (raw.astype(np.float32) / 65535.0) * self.DEPTH_MAX_MM
        return (raw.astype(np.float32) / 255.0) * self.DEPTH_MAX_MM

    def _decode_normals(self, record: SampleRecord) -> np.ndarray | None:
        if record.normal_path is None or not record.normal_path.is_file():
            return None
        raw_bgr = read_image_unchanged(record.normal_path)
        if raw_bgr.ndim != 3:
            return None
        # OpenCV charge les canaux en BGR : on repasse en RGB = (Nx, Ny, Nz)
        raw_rgb = cv2.cvtColor(raw_bgr[:, :, :3], cv2.COLOR_BGR2RGB)
        scale = 65535.0 if raw_rgb.dtype == np.uint16 else 255.0
        return (raw_rgb.astype(np.float32) / scale) * 2.0 - 1.0
