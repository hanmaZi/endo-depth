"""Loader PyTorch et utilitaire de préparation pour le dataset C3VD."""

from __future__ import annotations

import shutil
import zipfile
from pathlib import Path

import cv2
import numpy as np

from src.datasets.base import (
    BaseEndoscopyDataset,
    SampleRecord,
    read_image_unchanged,
    split_records_by_sequence,
)


def _unwrap_duplicate_seq_dir(seq_dir: Path) -> Path:
    """Évite les dossiers doublement imbriqués (ex. desc_t4_a/desc_t4_a/0000_color.png)."""
    nested = seq_dir / seq_dir.name
    entries = [p for p in seq_dir.iterdir() if not p.name.startswith(".") and p.name != "__MACOSX"]
    if len(entries) == 1 and entries[0] == nested and nested.is_dir():
        return nested
    return seq_dir


def _extract_c3vd_zip(zip_path: Path, target_dir: Path) -> Path:
    """Extrait une archive C3VD (.zip) vers target_dir/<sequence>/ sans dossier parasite."""
    with zipfile.ZipFile(zip_path, "r") as zf:
        members = [
            info
            for info in zf.infolist()
            if not info.is_dir()
            and not info.filename.startswith("__MACOSX")
            and "/__MACOSX/" not in info.filename
            and not Path(info.filename).name.startswith(".")
            and Path(info.filename).name != "Thumbs.db"
        ]
        if not members:
            raise RuntimeError(f"Archive zip vide : {zip_path}")

        parts_list = [Path(info.filename).parts for info in members if Path(info.filename).parts]
        has_root_files = any(len(parts) == 1 for parts in parts_list)
        top_dirs = {parts[0] for parts in parts_list}

        if not has_root_files and len(top_dirs) == 1:
            seq_name = next(iter(top_dirs))
            dest_seq = target_dir / seq_name
        else:
            seq_name = zip_path.stem
            dest_seq = target_dir / seq_name

        if dest_seq.is_dir() and any(dest_seq.rglob("*_color.png")):
            print(f"[Skip] Archive déjà extraite : {dest_seq}")
            return dest_seq

        tmp_extract = target_dir / f"._tmp_{zip_path.stem}"
        if tmp_extract.exists():
            shutil.rmtree(tmp_extract)
        tmp_extract.mkdir(parents=True, exist_ok=True)
        resolved_root = tmp_extract.resolve()
        for info in members:
            rel_path = Path(info.filename)
            out_path = (tmp_extract / rel_path).resolve()
            # Protection contre la traversée de chemin (Zip Slip)
            if not out_path.is_relative_to(resolved_root):
                raise RuntimeError(f"Chemin dangereux détecté dans {zip_path} : {info.filename}")
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(info, "r") as src, out_path.open("wb") as dst:
                shutil.copyfileobj(src, dst)

    src_content = tmp_extract / seq_name if (not has_root_files and len(top_dirs) == 1) else tmp_extract
    src_content = _unwrap_duplicate_seq_dir(src_content)
    if dest_seq.exists():
        shutil.rmtree(dest_seq)
    shutil.move(str(src_content), str(dest_seq))
    if tmp_extract.exists():
        shutil.rmtree(tmp_extract)

    print(f"[Extract] {zip_path.name} -> {dest_seq}")
    return dest_seq


def prepare_c3vd_from_source(source_dir: str | Path, target_dir: str | Path) -> Path:
    """Prépare C3VD depuis un dossier mixte (.zip et dossiers déjà décompressés).

    - Si un élément est un `.zip`, il est décompressé vers `target_dir/<sequence>/`.
    - Si un élément est un dossier de séquence, il est copié vers `target_dir/<sequence>/`.
    Les deux cas produisent exactement le même arbre `target_dir/<sequence>/...`.
    """
    src = Path(source_dir)
    dst = Path(target_dir)
    if not src.is_dir():
        raise FileNotFoundError(f"Dossier source C3VD introuvable : {src}")
    dst.mkdir(parents=True, exist_ok=True)

    prepared: list[Path] = []
    for item in sorted(src.iterdir()):
        if item.name.startswith(".") or item.name == "__MACOSX":
            continue
        if item.is_file() and item.suffix.lower() == ".zip":
            prepared.append(_extract_c3vd_zip(item, dst))
        elif item.is_dir():
            dest_seq = dst / item.name
            if dest_seq.is_dir() and any(dest_seq.rglob("*_color.png")):
                print(f"[Skip] Dossier déjà présent : {dest_seq}")
                prepared.append(dest_seq)
                continue
            src_seq = _unwrap_duplicate_seq_dir(item)
            tmp_copy = dst / f"._tmp_{item.name}"
            if tmp_copy.exists():
                shutil.rmtree(tmp_copy)
            shutil.copytree(src_seq, tmp_copy)
            if dest_seq.exists():
                shutil.rmtree(dest_seq)
            shutil.move(str(tmp_copy), str(dest_seq))
            print(f"[Copy] {item.name}/ -> {dest_seq}")
            prepared.append(dest_seq)

    if not prepared:
        raise RuntimeError(f"Aucune archive .zip ni dossier de séquence trouvé dans {src}")
    return dst


def discover_c3vd_records(root_dir: str | Path) -> list[SampleRecord]:
    """Parcourt un dossier C3VD (format plat, sous-dossiers color/depth/normals, ou imbriqué)."""
    root = Path(root_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"Dossier C3VD introuvable : {root}")

    records: list[SampleRecord] = []
    color_candidates = sorted(
        p
        for p in root.rglob("*_color.png")
        if "__MACOSX" not in p.parts and not any(part.startswith(".") for part in p.parts)
    )

    for rgb_path in color_candidates:
        seq_dir = rgb_path.parent.parent if rgb_path.parent.name.lower() == "color" else rgb_path.parent
        frame_id = rgb_path.stem.replace("_color", "")

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
