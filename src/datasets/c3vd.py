"""Loader PyTorch et utilitaire de préparation pour le dataset C3VD."""

from __future__ import annotations

import shutil
import zipfile
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np

from src.datasets.base import (
    BaseEndoscopyDataset,
    SampleRecord,
    read_image_unchanged,
    split_records_by_sequence,
    summarize_records_by_sequence,
)


def _unwrap_duplicate_seq_dir(seq_dir: Path) -> Path:
    """Évite les dossiers doublement imbriqués (ex. desc_t4_a/desc_t4_a/0000_color.png)."""
    nested = seq_dir / seq_dir.name
    entries = [p for p in seq_dir.iterdir() if not p.name.startswith(".") and p.name != "__MACOSX"]
    if len(entries) == 1 and entries[0] == nested and nested.is_dir():
        return nested
    return seq_dir


def _find_matching_map(
    rgb_path: Path, seq_dir: Path, frame_id: str, kind: str, extensions: tuple[str, ...]
) -> Path | None:
    """Recherche une carte associée (depth, normals, occlusion) sous plusieurs conventions C3VD."""
    candidates: list[Path] = []
    for ext in extensions:
        candidates.append(rgb_path.with_name(f"{frame_id}_{kind}{ext}"))
        candidates.append(seq_dir / kind / f"{frame_id}_{kind}{ext}")
        candidates.append(seq_dir / kind / f"{frame_id}{ext}")
        if kind == "occlusion":
            candidates.append(seq_dir / "occlusions" / f"{frame_id}_{kind}{ext}")
            candidates.append(seq_dir / "occlusions" / f"{frame_id}{ext}")
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def _is_valid_c3vd_dir(seq_dir: Path) -> bool:
    """Vérifie qu'un dossier de séquence contient au moins une paire RGB + profondeur valide."""
    if not seq_dir.is_dir():
        return False
    try:
        return len(discover_c3vd_records(seq_dir)) > 0
    except RuntimeError:
        return False


def _extract_c3vd_zip(zip_path: Path, target_dir: Path) -> Path:
    """Extrait une archive C3VD (.zip) vers target_dir/<zip_stem>/ et vérifie le résultat."""
    dest_seq = target_dir / zip_path.stem
    if _is_valid_c3vd_dir(dest_seq):
        print(f"[Skip] Archive déjà extraite : {dest_seq}")
        return dest_seq

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

        tmp_extract = target_dir / f"._tmp_{zip_path.stem}"
        if tmp_extract.exists():
            shutil.rmtree(tmp_extract)
        tmp_extract.mkdir(parents=True, exist_ok=True)
        resolved_root = tmp_extract.resolve()
        for info in members:
            rel_path = Path(info.filename)
            out_path = (tmp_extract / rel_path).resolve()
            if not out_path.is_relative_to(resolved_root):
                raise RuntimeError(f"Chemin dangereux détecté dans {zip_path} : {info.filename}")
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(info, "r") as src, out_path.open("wb") as dst:
                shutil.copyfileobj(src, dst)

    if not has_root_files and len(top_dirs) == 1:
        src_content = tmp_extract / next(iter(top_dirs))
    else:
        src_content = tmp_extract
    src_content = _unwrap_duplicate_seq_dir(src_content)

    if dest_seq.exists():
        shutil.rmtree(dest_seq)
    shutil.move(str(src_content), str(dest_seq))
    if tmp_extract.exists():
        shutil.rmtree(tmp_extract)

    if not _is_valid_c3vd_dir(dest_seq):
        sample_files = [str(p.relative_to(dest_seq)) for p in sorted(dest_seq.rglob("*"))[:10]]
        raise RuntimeError(
            f"Archive extraite mais aucune paire RGB/profondeur détectée dans {dest_seq} "
            f"(exemples de fichiers extraits : {sample_files})."
        )

    print(f"[Extract] {zip_path.name} -> {dest_seq}")
    return dest_seq


def prepare_c3vd_from_source(source_dir: str | Path, target_dir: str | Path) -> Path:
    """Prépare C3VD depuis un dossier mixte (.zip et dossiers déjà décompressés).

    - Si un élément est un `.zip`, il est décompressé vers `target_dir/<sequence>/`.
    - Si un élément est un dossier de séquence, il est copié vers `target_dir/<sequence>/`.
    Chaque séquence préparée est immédiatement vérifiée pour ne jamais passer une archive sous silence.
    """
    src = Path(source_dir)
    dst = Path(target_dir)
    if not src.is_dir():
        raise FileNotFoundError(f"Dossier source C3VD introuvable : {src}")
    dst.mkdir(parents=True, exist_ok=True)

    items = [
        item
        for item in sorted(src.iterdir())
        if not item.name.startswith(".") and item.name != "__MACOSX"
    ]
    print(f"[Source] {src} contient : {[p.name for p in items]}")

    prepared: list[Path] = []
    for item in items:
        if item.is_file() and item.suffix.lower() == ".zip":
            prepared.append(_extract_c3vd_zip(item, dst))
        elif item.is_dir():
            dest_seq = dst / item.name
            if _is_valid_c3vd_dir(dest_seq):
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
            if not _is_valid_c3vd_dir(dest_seq):
                raise RuntimeError(
                    f"Le dossier copié {dest_seq} ne contient aucune paire RGB/profondeur valide."
                )
            print(f"[Copy] {item.name}/ -> {dest_seq}")
            prepared.append(dest_seq)

    if not prepared:
        raise RuntimeError(f"Aucune archive .zip ni dossier de séquence trouvé dans {src}")
    return dst


def discover_c3vd_records(
    root_dir: str | Path,
    expected_sequences: Iterable[str] | None = None,
) -> list[SampleRecord]:
    """Parcourt un dossier C3VD (plat, color/depth, rgb/depth, .tiff/.tif) et vérifie les séquences."""
    root = Path(root_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"Dossier C3VD introuvable : {root}")

    candidates: list[tuple[Path, Path, str]] = []
    seen_rgb: set[Path] = set()

    for rgb_path in sorted(root.rglob("*")):
        if not rgb_path.is_file():
            continue
        if "__MACOSX" in rgb_path.parts or any(part.startswith(".") for part in rgb_path.parts):
            continue
        if rgb_path.suffix.lower() != ".png":
            continue

        stem_lower = rgb_path.stem.lower()
        parent_lower = rgb_path.parent.name.lower()

        if stem_lower.endswith("_color"):
            seq_dir = rgb_path.parent.parent if parent_lower == "color" else rgb_path.parent
            frame_id = rgb_path.stem[: -len("_color")]
        elif parent_lower in {"color", "rgb"} and not any(
            tag in stem_lower for tag in ("mask", "occlusion", "depth", "normal")
        ):
            seq_dir = rgb_path.parent.parent
            frame_id = rgb_path.stem
        else:
            continue

        if rgb_path in seen_rgb:
            continue
        seen_rgb.add(rgb_path)
        candidates.append((rgb_path, seq_dir, frame_id))

    records: list[SampleRecord] = []
    for rgb_path, seq_dir, frame_id in candidates:
        depth_path = _find_matching_map(
            rgb_path, seq_dir, frame_id, kind="depth", extensions=(".tiff", ".tif", ".TIFF", ".TIF")
        )
        if depth_path is None:
            continue

        normal_path = _find_matching_map(
            rgb_path, seq_dir, frame_id, kind="normals", extensions=(".tiff", ".tif", ".TIFF", ".TIF")
        )
        occ_path = _find_matching_map(
            rgb_path, seq_dir, frame_id, kind="occlusion", extensions=(".png", ".tiff", ".tif")
        )
        fov_mask_path = rgb_path.with_name(f"{frame_id}_mask.png")
        if not fov_mask_path.is_file():
            seq_mask = seq_dir / "mask.png"
            fov_mask_path = seq_mask if seq_mask.is_file() else None

        records.append(
            SampleRecord(
                rgb_path=rgb_path,
                depth_path=depth_path,
                normal_path=normal_path,
                occlusion_path=occ_path,
                fov_mask_path=fov_mask_path,
                sequence_name=seq_dir.name,
                frame_id=frame_id,
            )
        )

    if not records:
        raise RuntimeError(
            f"Aucune paire (RGB, profondeur) trouvée dans {root}. "
            "Vérifie que les séquences C3VD sont bien décompressées."
        )

    if expected_sequences is not None:
        counts = summarize_records_by_sequence(records)
        expected_list = [str(s) for s in expected_sequences]
        missing = [s for s in expected_list if counts.get(s, 0) == 0]
        if missing:
            raise RuntimeError(
                f"Séquences C3VD attendues manquantes ou vides : {missing}. "
                f"Séquences chargées : {counts}"
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
        expected_sequences: Iterable[str] | None = None,
        sequence_splits: dict[str, list[str]] | None = None,
    ) -> None:
        all_records = discover_c3vd_records(root_dir, expected_sequences=expected_sequences)
        if split in {"all", "train"}:
            counts = summarize_records_by_sequence(all_records)
            print(f"[C3VD] Séquences trouvées (total={len(all_records)} frames) : {counts}")

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
