"""Loader PyTorch et utilitaire de préparation pour le dataset C3VD."""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import time
import zipfile
from collections import Counter
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


_IGNORED_NAMES = {"__MACOSX", "desktop.ini", "Thumbs.db"}


def _is_ignored_entry(path: Path) -> bool:
    return path.name.startswith(".") or path.name in _IGNORED_NAMES


def _summarize_dir_contents(folder: Path) -> str:
    """Résume le contenu d'un dossier (sous-dossiers, extensions, premiers et derniers fichiers)."""
    all_files = sorted(p for p in folder.rglob("*") if p.is_file() and not _is_ignored_entry(p))
    subdirs = sorted({str(p.parent.relative_to(folder)) for p in all_files})[:10]
    exts = dict(Counter(p.suffix.lower() for p in all_files))
    first_10 = [str(p.relative_to(folder)) for p in all_files[:10]]
    last_10 = [str(p.relative_to(folder)) for p in all_files[-10:]]
    return (
        f"{len(all_files)} fichier(s) | dossiers={subdirs} | extensions={exts} | "
        f"premiers={first_10} | derniers={last_10}"
    )


def _extract_frame_id_candidates(stem: str) -> list[str]:
    """Génère les identifiants de frame possibles depuis un nom de fichier RGB (ex. 9_color -> 0009)."""
    base_candidates: list[str] = [stem]
    for suffix in ("_color", "_rgb", "_img", "_image", "_undistorted"):
        if stem.lower().endswith(suffix):
            base_candidates.append(stem[: -len(suffix)])
    for prefix in ("color_", "rgb_", "frame_", "img_", "image_"):
        if stem.lower().startswith(prefix):
            base_candidates.append(stem[len(prefix) :])
    digit_match = re.search(r"(\d+)$", stem)
    if digit_match:
        base_candidates.append(digit_match.group(1))
    any_digits = re.search(r"(\d+)", stem)
    if any_digits:
        base_candidates.append(any_digits.group(1))

    expanded: list[str] = []
    for cand in base_candidates:
        if not cand:
            continue
        expanded.append(cand)
        if cand.isdigit():
            num_str = str(int(cand))
            for width in (4, 5, 6):
                expanded.append(num_str.zfill(width))
            expanded.append(num_str)
    # Déduplique tout en conservant l'ordre de priorité
    return list(dict.fromkeys(expanded))


def _unwrap_duplicate_seq_dir(seq_dir: Path) -> Path:
    nested = seq_dir / seq_dir.name
    entries = [p for p in seq_dir.iterdir() if not _is_ignored_entry(p)]
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
        candidates.append(seq_dir / f"{frame_id}_{kind}{ext}")
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


def _copy_file_fast(src: Path, dst: Path) -> None:
    """Copie un fichier d'un seul bloc (via `cp` système sous Linux/Colab si disponible)."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    if os.name == "posix":
        cp_bin = shutil.which("cp")
        if cp_bin:
            subprocess.run([cp_bin, "-f", str(src), str(dst)], check=True)
            return
    shutil.copyfile(src, dst)


def _extract_c3vd_zip(zip_path: Path, target_dir: Path) -> Path:
    """Copie localement une archive C3VD (.zip) d'un bloc, la décompresse puis supprime le zip temporaire."""
    dest_seq = target_dir / zip_path.stem
    if _is_valid_c3vd_dir(dest_seq):
        print(f"[Skip] Séquence déjà présente : {dest_seq}")
        return dest_seq

    target_dir.mkdir(parents=True, exist_ok=True)
    need_stage = zip_path.parent.resolve() != target_dir.resolve()
    local_zip = (target_dir / f"._tmp_{zip_path.name}") if need_stage else zip_path
    tmp_extract = target_dir / f"._tmp_{zip_path.stem}"

    try:
        if need_stage:
            if local_zip.exists():
                local_zip.unlink()
            size_gb = zip_path.stat().st_size / (1024**3)
            t0 = time.monotonic()
            print(f"[CopyZip] {zip_path.name} ({size_gb:.2f} Go) -> {local_zip}")
            _copy_file_fast(zip_path, local_zip)
            print(f"[CopyZip] Terminé en {time.monotonic() - t0:.1f}s")

        t1 = time.monotonic()
        with zipfile.ZipFile(local_zip, "r") as zf:
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
                with zf.open(info, "r") as src_f, out_path.open("wb") as dst_f:
                    shutil.copyfileobj(src_f, dst_f)

        if not has_root_files and len(top_dirs) == 1:
            src_content = tmp_extract / next(iter(top_dirs))
        else:
            src_content = tmp_extract
        src_content = _unwrap_duplicate_seq_dir(src_content)

        if dest_seq.exists():
            shutil.rmtree(dest_seq)
        shutil.move(str(src_content), str(dest_seq))

        if not _is_valid_c3vd_dir(dest_seq):
            sample_files = [str(p.relative_to(dest_seq)) for p in sorted(dest_seq.rglob("*"))[:10]]
            raise RuntimeError(
                f"Archive extraite mais aucune paire RGB/profondeur détectée dans {dest_seq} "
                f"(exemples de fichiers extraits : {sample_files})."
            )

        print(f"[Extract] {zip_path.name} -> {dest_seq} ({time.monotonic() - t1:.1f}s)")
        return dest_seq
    finally:
        if need_stage and local_zip.exists():
            local_zip.unlink()
        if tmp_extract.exists():
            shutil.rmtree(tmp_extract, ignore_errors=True)


def prepare_c3vd_from_source(
    source_dir: str | Path,
    target_dir: str | Path,
    sequences: Iterable[str] | None = None,
) -> Path:
    """Prépare C3VD depuis un dossier mixte (.zip et dossiers déjà décompressés).

    - Si `sequences` est fourni, ne traite que les séquences sélectionnées.
    - Saute immédiatement toute séquence déjà extraite et valide dans `target_dir`.
    - Pour chaque `.zip` manquant, copie l'archive d'un bloc sur disque local avant de la décompresser.
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

    allowed = {str(s) for s in sequences} if sequences is not None else None
    prepared: list[Path] = []
    for item in items:
        is_zip = item.is_file() and item.suffix.lower() == ".zip"
        if not is_zip and not item.is_dir():
            continue
        seq_name = item.stem if is_zip else item.name
        if allowed is not None and seq_name not in allowed:
            print(f"[Filter] Séquence non sélectionnée ignorée : {seq_name}")
            continue

        if is_zip:
            prepared.append(_extract_c3vd_zip(item, dst))
        else:
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

            # Si le dossier contenait en réalité un .zip non décompressé, l'extrait aussi
            inner_zips = [p for p in dest_seq.glob("*.zip") if not _is_ignored_entry(p)]
            if not _is_valid_c3vd_dir(dest_seq) and len(inner_zips) == 1:
                inner_zip = inner_zips[0]
                extracted = _extract_c3vd_zip(inner_zip, dst)
                if extracted != dest_seq:
                    dest_seq = extracted

            if not _is_valid_c3vd_dir(dest_seq):
                summary = _summarize_dir_contents(dest_seq)
                raise RuntimeError(
                    f"Le dossier copié {dest_seq} ne contient aucune paire RGB/profondeur valide "
                    f"({summary})."
                )
            print(f"[Copy] {item.name}/ -> {dest_seq}")
            prepared.append(dest_seq)

    if not prepared:
        existing = [p for p in dst.iterdir() if p.is_dir() and _is_valid_c3vd_dir(p)] if dst.is_dir() else []
        if not existing:
            raise RuntimeError(
                f"Aucune archive .zip ni dossier de séquence correspondant trouvé dans {src} "
                f"(filtre={sorted(allowed) if allowed is not None else 'toutes'})."
            )
    return dst


def discover_c3vd_records(
    root_dir: str | Path,
    expected_sequences: Iterable[str] | None = None,
    include_sequences: Iterable[str] | None = None,
) -> list[SampleRecord]:
    """Parcourt un dossier C3VD et apparie souplement RGB + profondeur (.tiff/.tif/.png)."""
    root = Path(root_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"Dossier C3VD introuvable : {root}")

    allowed_seqs = {str(s) for s in include_sequences} if include_sequences is not None else None
    non_rgb_tags = ("mask", "occlusion", "depth", "normal", "flow", "edge")
    rgb_exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    depth_exts = (".tiff", ".tif", ".TIFF", ".TIF", ".png", ".PNG")
    normal_exts = (".tiff", ".tif", ".TIFF", ".TIF")
    occ_exts = (".png", ".tiff", ".tif")

    records: list[SampleRecord] = []
    seen_rgb: set[Path] = set()

    for rgb_path in sorted(root.rglob("*")):
        if not rgb_path.is_file():
            continue
        if "__MACOSX" in rgb_path.parts or any(part.startswith(".") for part in rgb_path.parts):
            continue
        if rgb_path.suffix.lower() not in rgb_exts:
            continue

        stem_lower = rgb_path.stem.lower()
        if any(tag in stem_lower for tag in non_rgb_tags):
            continue
        if any(tag in rgb_path.parent.name.lower() for tag in ("depth", "normal", "occlusion", "flow")):
            continue

        # Cherche le seq_dir (dossier courant ou parent) et le frame_id qui possèdent une profondeur
        seq_candidates = [rgb_path.parent]
        if rgb_path.parent != root and rgb_path.parent.parent != root.parent:
            seq_candidates.append(rgb_path.parent.parent)

        matched: tuple[Path, str, Path] | None = None
        for seq_dir in seq_candidates:
            if allowed_seqs is not None and seq_dir.name not in allowed_seqs:
                continue
            for frame_id in _extract_frame_id_candidates(rgb_path.stem):
                depth_path = _find_matching_map(
                    rgb_path, seq_dir, frame_id, kind="depth", extensions=depth_exts
                )
                if depth_path is not None:
                    matched = (seq_dir, frame_id, depth_path)
                    break
            if matched is not None:
                break

        if matched is None or rgb_path in seen_rgb:
            continue
        seen_rgb.add(rgb_path)
        seq_dir, frame_id, depth_path = matched

        normal_path = _find_matching_map(
            rgb_path, seq_dir, frame_id, kind="normals", extensions=normal_exts
        )
        occ_path = _find_matching_map(
            rgb_path, seq_dir, frame_id, kind="occlusion", extensions=occ_exts
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
            f"Aucune paire (RGB, profondeur) trouvée dans {root} ({_summarize_dir_contents(root)})."
        )

    records.sort(key=lambda r: (r.sequence_name, r.frame_id))

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
        color_augment: bool = False,
        expected_sequences: Iterable[str] | None = None,
        sequence_splits: dict[str, list[str]] | None = None,
        include_sequences: Iterable[str] | None = None,
    ) -> None:
        all_records = discover_c3vd_records(
            root_dir,
            expected_sequences=expected_sequences,
            include_sequences=include_sequences,
        )
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
            color_augment=color_augment,
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
