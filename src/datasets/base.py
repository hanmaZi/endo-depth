"""Socle commun des datasets PyTorch pour l'endoscopie (C3VD, SimCol3D, etc.)."""

from __future__ import annotations

import random
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


def assign_sequence_splits(
    records: list[SampleRecord],
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
    sequence_splits: dict[str, list[str]] | None = None,
) -> dict[str, list[str]]:
    """Assigne des séquences entières à train / val / test sans jamais couper une séquence.

    - Si `sequence_splits` est fourni, utilise ces listes explicitement.
    - Si exactement 3 séquences sont présentes, assigne automatiquement par taille décroissante :
      plus grande séquence -> train, 2e -> val, plus petite -> test
      (ex. train=cecum_t1_a, val=desc_t4_a, test=trans_t1_a).
    - Si >= 4 séquences sont présentes, découpe aléatoirement (seed) par séquence entière.
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
        return assignment

    if len(counts) < 3:
        raise ValueError(
            f"Découpage par séquence entière impossible avec {len(counts)} séquence(s) : {counts}. "
            "Il faut au moins 3 séquences complètes (ex. cecum_t1_a, desc_t4_a, trans_t1_a) "
            "pour séparer train, val et test sans couper une séquence."
        )

    if len(counts) == 3:
        by_size = sorted(counts.keys(), key=lambda s: (-counts[s], s))
        return {
            "train": [by_size[0]],
            "val": [by_size[1]],
            "test": [by_size[2]],
        }

    sequences = sorted(counts.keys())
    rng = random.Random(seed)
    rng.shuffle(sequences)
    n_val = max(1, round(len(sequences) * val_ratio))
    n_test = max(1, round(len(sequences) * test_ratio))
    if n_val + n_test >= len(sequences):
        n_val = 1
        n_test = 1
    return {
        "val": sorted(sequences[:n_val]),
        "test": sorted(sequences[n_val : n_val + n_test]),
        "train": sorted(sequences[n_val + n_test :]),
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
