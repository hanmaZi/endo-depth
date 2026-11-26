"""Socle commun des datasets PyTorch pour l'endoscopie (C3VD, SimCol3D, etc.)."""

from __future__ import annotations

import random
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass(frozen=True)
class SampleRecord:
    """Métadonnées d'une image et de ses annotations associées."""

    rgb_path: Path
    depth_path: Path
    sequence_name: str
    frame_id: str
    normal_path: Path | None = None
    occlusion_path: Path | None = None
    fov_mask_path: Path | None = None


def read_image_unchanged(path: Path) -> np.ndarray:
    """Lit une image (8-bit ou 16-bit PNG/TIFF) en gérant les chemins Unicode."""
    raw_bytes = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(raw_bytes, cv2.IMREAD_UNCHANGED)
    if image is None:
        raise IOError(f"Impossible de décoder l'image : {path}")
    return image


def detect_specular_mask(
    rgb_uint8: np.ndarray,
    valid_mask: np.ndarray | None = None,
    v_thresh: int = 235,
    s_thresh: int = 45,
) -> np.ndarray:
    """Détecte les reflets spéculaires (forte luminosité, faible saturation) dans le champ valide.

    Si `valid_mask` est fourni, exclut les coins noirs, bords hors champ et marqueurs UI.
    """
    hsv = cv2.cvtColor(rgb_uint8, cv2.COLOR_RGB2HSV)
    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]
    spec = (value >= v_thresh) & (saturation <= s_thresh)
    if valid_mask is not None:
        spec = spec & valid_mask.astype(bool)
    return spec


def summarize_records_by_sequence(records: list[SampleRecord]) -> dict[str, int]:
    """Retourne le nombre d'images par séquence trié alphabétiquement."""
    counts = Counter(rec.sequence_name for rec in records)
    return {seq: counts[seq] for seq in sorted(counts)}


_C3VD_TEXTURE_RE = re.compile(r"^(.+?)_(t\d+)(?:_[a-z0-9]+)?$", re.IGNORECASE)


def extract_texture_group(sequence_name: str, by_texture_tag_only: bool = False) -> str:
    """Extrait l'identifiant de texture C3VD afin d'éviter toute fuite entre splits.

    Exemples :
    - `trans_t3_a`, `trans_t3_b` -> `"trans_t3"` (ou `"t3"` si `by_texture_tag_only=True`).
    - Hors convention C3VD `<segment>_t<id>_...`, retourne `sequence_name`.
    """
    match = _C3VD_TEXTURE_RE.match(sequence_name)
    if not match:
        return sequence_name
    segment, tex_tag = match.group(1), match.group(2).lower()
    return tex_tag if by_texture_tag_only else f"{segment}_{tex_tag}"


def validate_no_texture_leakage(
    assignment: dict[str, list[str]],
    by_texture_tag_only: bool = False,
) -> None:
    """Vérifie qu'aucun groupe de texture n'est partagé entre deux splits (train/val/test)."""
    group_to_splits: dict[str, set[str]] = {}
    for split_name, seqs in assignment.items():
        for seq in seqs:
            grp = extract_texture_group(seq, by_texture_tag_only=by_texture_tag_only)
            group_to_splits.setdefault(grp, set()).add(split_name)

    leaks = {grp: sorted(splits) for grp, splits in group_to_splits.items() if len(splits) > 1}
    if leaks:
        raise ValueError(
            f"Fuite de texture détectée entre splits (même texture dans train/val/test) : {leaks}"
        )


def assign_sequence_splits(
    records: list[SampleRecord],
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
    sequence_splits: dict[str, list[str]] | None = None,
) -> dict[str, list[str]]:
    """Assigne des séquences entières à train / val / test sans fuite temporelle ni de texture.

    - Si `sequence_splits` est fourni, vérifie présence, unicité et absence de fuite `(segment, tN)`.
    - Sinon, regroupe d'abord par identifiant de texture (ex. `trans_t3_a` et `trans_t3_b` ensemble).
      Si `>= 3` tags globaux `tN` distincts existent (ex. `t1..t4`), regroupe par `tN`.
      Si exactement 3 groupes existent, assigne par taille décroissante (train -> val -> test).
      Si `>= 4` groupes existent, découpe aléatoirement (selon `seed`) par groupe de texture entier.
    """
    counts = summarize_records_by_sequence(records)
    available = set(counts.keys())

    if sequence_splits is not None:
        assignment: dict[str, list[str]] = {}
        seen: set[str] = set()
        for key in ("train", "val", "test"):
            seqs = list(sequence_splits.get(key, []))
            missing = set(seqs) - available
            if missing:
                raise ValueError(
                    f"Séquences introuvables pour split='{key}' : {sorted(missing)} "
                    f"(disponibles : {sorted(available)})"
                )
            overlap = seen & set(seqs)
            if overlap:
                raise ValueError(f"Séquences en double entre splits : {sorted(overlap)}")
            seen.update(seqs)
            assignment[key] = seqs
        validate_no_texture_leakage(assignment, by_texture_tag_only=False)
        return assignment

    # Regroupement par texture globale `tN` si >= 3 tags distincts, sinon par `(segment, tN)`
    by_tag: dict[str, list[str]] = {}
    by_seg_tag: dict[str, list[str]] = {}
    for seq in sorted(counts):
        by_tag.setdefault(extract_texture_group(seq, by_texture_tag_only=True), []).append(seq)
        by_seg_tag.setdefault(extract_texture_group(seq, by_texture_tag_only=False), []).append(seq)

    groups_map = by_tag if len(by_tag) >= 3 else by_seg_tag
    if len(groups_map) < 3:
        raise ValueError(
            f"Découpage sans fuite de texture impossible avec {len(groups_map)} groupe(s) : "
            f"{dict(sorted(groups_map.items()))}. "
            "Il faut au moins 3 groupes de texture distincts pour séparer train, val et test."
        )

    if len(groups_map) == 3:
        group_sizes = {
            grp: sum(counts[s] for s in seqs) for grp, seqs in groups_map.items()
        }
        ordered_groups = sorted(groups_map.keys(), key=lambda g: (-group_sizes[g], g))
        return {
            "train": sorted(groups_map[ordered_groups[0]]),
            "val": sorted(groups_map[ordered_groups[1]]),
            "test": sorted(groups_map[ordered_groups[2]]),
        }

    group_names = sorted(groups_map.keys())
    rng = random.Random(seed)
    rng.shuffle(group_names)
    n_val = max(1, round(len(group_names) * val_ratio))
    n_test = max(1, round(len(group_names) * test_ratio))
    if n_val + n_test >= len(group_names):
        n_val = 1
        n_test = 1
    return {
        "val": sorted(s for g in group_names[:n_val] for s in groups_map[g]),
        "test": sorted(s for g in group_names[n_val : n_val + n_test] for s in groups_map[g]),
        "train": sorted(s for g in group_names[n_val + n_test :] for s in groups_map[g]),
    }


def split_records_by_sequence(
    records: list[SampleRecord],
    split: str,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
    sequence_splits: dict[str, list[str]] | None = None,
) -> list[SampleRecord]:
    """Sépare les échantillons en train/val/test par séquence entière (aucune fuite temporelle)."""
    if split not in {"train", "val", "test", "all"}:
        raise ValueError(f"Split inconnu : {split}. Attendu : train, val, test ou all.")
    if split == "all" or not records:
        return records

    assignment = assign_sequence_splits(
        records,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        seed=seed,
        sequence_splits=sequence_splits,
    )
    target_seqs = set(assignment[split])
    return [rec for rec in records if rec.sequence_name in target_seqs]


class BaseEndoscopyDataset(Dataset):
    """Classe de base gérant le décodage, redimensionnement et l'augmentation."""

    def __init__(
        self,
        records: list[SampleRecord],
        image_size: tuple[int, int] = (256, 256),
        min_depth_mm: float = 1.0,
        max_depth_mm: float = 200.0,
        augment: bool = False,
        color_augment: bool = False,
    ) -> None:
        self.records = records
        self.height, self.width = int(image_size[0]), int(image_size[1])
        self.min_depth_mm = float(min_depth_mm)
        self.max_depth_mm = float(max_depth_mm)
        self.augment = augment
        self.color_augment = color_augment

    def __len__(self) -> int:
        return len(self.records)

    def _decode_depth_mm(self, record: SampleRecord) -> np.ndarray:
        """À implémenter dans chaque dataset pour convertir les pixels bruts en millimètres."""
        raise NotImplementedError

    def _decode_normals(self, record: SampleRecord) -> np.ndarray | None:
        """Retourne les normales (H, W, 3) dans [-1, 1] ou None si absentes."""
        return None

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]

        bgr_or_rgb = read_image_unchanged(record.rgb_path)
        if bgr_or_rgb.ndim == 2:
            rgb = cv2.cvtColor(bgr_or_rgb, cv2.COLOR_GRAY2RGB)
        else:
            rgb = cv2.cvtColor(bgr_or_rgb[:, :, :3], cv2.COLOR_BGR2RGB)
        if rgb.dtype != np.uint8:
            rgb = np.clip(rgb / 256.0, 0, 255).astype(np.uint8)

        depth_mm = self._decode_depth_mm(record).astype(np.float32)
        normals = self._decode_normals(record)

        fov_mask: np.ndarray | None = None
        if record.fov_mask_path is not None and record.fov_mask_path.is_file():
            raw_fov = read_image_unchanged(record.fov_mask_path)
            if raw_fov.ndim == 3:
                raw_fov = raw_fov[:, :, 0]
            fov_mask = (raw_fov > 0).astype(np.uint8)

        # Redimensionnement : bilinéaire pour RGB, plus proche voisin pour profondeur/masques
        rgb_resized = cv2.resize(rgb, (self.width, self.height), interpolation=cv2.INTER_LINEAR)
        depth_resized = cv2.resize(
            depth_mm, (self.width, self.height), interpolation=cv2.INTER_NEAREST
        )
        if fov_mask is not None:
            fov_resized = (
                cv2.resize(fov_mask, (self.width, self.height), interpolation=cv2.INTER_NEAREST) > 0
            )
        else:
            fov_resized = np.ones((self.height, self.width), dtype=bool)

        if normals is not None:
            normals_resized = cv2.resize(
                normals.astype(np.float32),
                (self.width, self.height),
                interpolation=cv2.INTER_NEAREST,
            )
        else:
            normals_resized = np.zeros((self.height, self.width, 3), dtype=np.float32)

        # Augmentation légère cohérente géométriquement (flip horizontal)
        if self.augment and random.random() < 0.5:
            rgb_resized = np.ascontiguousarray(rgb_resized[:, ::-1])
            depth_resized = np.ascontiguousarray(depth_resized[:, ::-1])
            fov_resized = np.ascontiguousarray(fov_resized[:, ::-1])
            if normals is not None:
                normals_resized = np.ascontiguousarray(normals_resized[:, ::-1])
                normals_resized[:, :, 0] *= -1.0  # Miroir gauche-droite => inverse Nx

        # Exclut les coins sans profondeur (< min_depth_mm) ET les pixels tronqués (= max_depth_mm)
        valid_mask = (
            fov_resized
            & np.isfinite(depth_resized)
            & (depth_resized >= self.min_depth_mm)
            & (depth_resized < (self.max_depth_mm - 1e-3))
        )

        if normals is not None:
            norm_len = np.linalg.norm(normals_resized, axis=-1, keepdims=True)
            valid_normals = norm_len.squeeze(-1) > 1e-3
            valid_mask = valid_mask & valid_normals
            normals_resized = normals_resized / np.clip(norm_len, 1e-6, None)

        # Met explicitement à zéro les zones hors masque (coins, bords, 100 mm tronqués)
        depth_resized = np.where(valid_mask, depth_resized, 0.0).astype(np.float32)
        normals_resized = np.where(valid_mask[:, :, None], normals_resized, 0.0).astype(np.float32)

        specular_mask = detect_specular_mask(rgb_resized, valid_mask=valid_mask)

        # Augmentation photométrique (luminosité/gamma/teinte) pour limiter le surapprentissage couleur
        rgb_float = rgb_resized.astype(np.float32) / 255.0
        if self.augment and self.color_augment and random.random() < 0.8:
            gamma = random.uniform(0.75, 1.35)
            brightness = random.uniform(0.75, 1.25)
            channel_gains = np.array(
                [random.uniform(0.88, 1.12), random.uniform(0.88, 1.12), random.uniform(0.88, 1.12)],
                dtype=np.float32,
            ).reshape(1, 1, 3)
            rgb_float = np.clip((rgb_float ** gamma) * brightness * channel_gains, 0.0, 1.0)

        image_tensor = torch.from_numpy(rgb_float.transpose(2, 0, 1)).float()
        depth_tensor = torch.from_numpy(depth_resized).unsqueeze(0).float()
        mask_tensor = torch.from_numpy(valid_mask).unsqueeze(0).bool()
        normals_tensor = torch.from_numpy(normals_resized.transpose(2, 0, 1)).float()
        specular_tensor = torch.from_numpy(specular_mask).unsqueeze(0).bool()

        return {
            "image": image_tensor,
            "depth": depth_tensor,
            "mask": mask_tensor,
            "normals": normals_tensor,
            "specular_mask": specular_tensor,
            "has_normals": bool(normals is not None),
            "sequence": record.sequence_name,
            "frame_id": record.frame_id,
        }
