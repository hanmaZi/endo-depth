# Journal de bord du PFE — `endo-depth`

Ce journal trace chronologiquement ce qui a été fait, les choix techniques pris, les résultats observés et les blocages éventuels.

---

## 25 novembre 2026 — Étape A : Mise en place du projet
- **Fait :** Création du dépôt Git isolé `endo-depth`, `.gitignore` (datasets, checkpoints, poids), `.venv` Python 3.10 (`torch 2.12.0+cpu`, `transformers 5.20.2`, `timm 1.0.31`), `configs/base.yaml`, `src/train.py`, `src/eval.py`, `notebooks/00_smoke_test.ipynb`.
- **Résultats observés :**
  - Smoke test CPU validé (`SmokeDepthNet` 2 913 params : loss L1 passée de `1.0470` à `1.0449` sur 3 époques, checkpoint sauvé et rechargé).
  - Smoke test GPU validé par Anis sur **Google Colab (GPU T4)** avec `depth-anything/Depth-Anything-V2-Small-hf`.
- **Problème résolu :** Découverte d'un `.git` parent dans `C:\Users\hichem` → isolement strict de `endo-depth/` dans son propre repo.

## 25 novembre 2026 — Étapes B & E : État de l'art et plan d'apprentissage
- **Fait :** Rédaction de [`docs/etat_de_l_art.md`](etat_de_l_art.md) (Niu et al. 2026, TinyDepth 2024, Tan et al. 2025, Monodepth2, Endo-SfMLearner, AF-SfMLearner, EndoDAC, DARES, MiDaS, Depth Anything V2, EndoOmni, Marigold, DepthFM, travaux coloscopie SimCol3D/C3VD/Rau/Du) et [`docs/plan_apprentissage.md`](plan_apprentissage.md) sur 5 semaines jusqu'au 31 décembre 2026.
- **Décisions validées avec Anis :**
  - **Baseline :** fine-tuning léger par **LoRA sur `Depth-Anything-V2-Small`**.
  - **Contribution principale :** ajout d'une **loss sur les gradients spatiaux et les normales de surface** (pour préserver le relief des polypes et plis coliques).
  - **Option ultérieure :** Piste 2 (masquage/pondération des reflets spéculaires).
- **Rigueur :** Détails inaccessibles derrière paywall (ex. Tan et al. 2025) explicitement marqués `à vérifier`.

## 25 novembre 2026 — Étape C : Datasets endoscopiques et visualisation
- **Fait :**
  - Rédaction de [`docs/datasets.md`](datasets.md) recensant licences, tailles, formats et procédures d'accès de **C3VD**, **SimCol3D**, **EndoSLAM**, **SCARED**, **Hamlyn** et **EndoMapper**.
  - Implémentation des loaders PyTorch : [`src/datasets/base.py`](../src/datasets/base.py), [`src/datasets/c3vd.py`](../src/datasets/c3vd.py), [`src/datasets/simcol3d.py`](../src/datasets/simcol3d.py), [`src/datasets/factory.py`](../src/datasets/factory.py).
  - Création du notebook Colab prêt à lancer [`notebooks/01_dataset_viz.ipynb`](../notebooks/01_dataset_viz.ipynb) (montage Drive, démo synthétique auto si données absentes, stats JSON, visualisation RGB / profondeur / normales / reflets).
- **Décisions techniques :**
  - Découpage train/val/test **par séquence** (`split_records_by_sequence`) plutôt que frame par frame pour éviter toute fuite temporelle (*data leakage*).
  - Redimensionnement par `INTER_NEAREST` pour profondeurs/normales afin d'éviter des profondeurs interpolées artificielles aux bords des plis ; inversion correcte de `Nx` lors du flip horizontal des normales.
  - Suppression du filtrage naïf via `*_occlusion.png` après vérification : dans C3VD, `occlusion.png` encode les faces occluses/occluantes pour le rendu 3D, tandis que `depth == 0` marque déjà les pixels hors plage/masqués.
- **Vérifications :** Tests unitaires locaux passés sur fixtures C3VD (16-bit TIFF profondeur `0–100 mm` + normales unitaires) et SimCol3D (16-bit PNG `0–200 mm`), et validation du notebook (5 cellules).
- **En attente côté utilisateur :** Téléchargement des archives C3VD (et éventuellement SimCol3D) dans Google Drive (ou inscription Synapse pour EndoMapper plus tard).

## 25 novembre 2026 — Étape D : Métriques, Baseline LoRA, Contribution Gradients/Normales et Notebooks Colab
- **Fait :**
  - Métriques monoculaires (`AbsRel`, `SqRel`, `RMSE`, `RMSE_log`, $\delta < 1.25 / 1.25^2 / 1.25^3$ + alignement `median` / `scale_shift`) dans [`src/metrics/depth_metrics.py`](../src/metrics/depth_metrics.py).
  - Modèle [`DepthAnythingV2LoRA`](../src/models/depth_anything_lora.py) : adaptateurs `LoRALinear` sur `q_proj` et `v_proj` de l'encodeur DINOv2, redimensionnement automatique au multiple de 14 (`patch_size`), et tête métrique bornée par Sigmoid sur `[min_depth_mm, max_depth_mm]`.
  - Contribution [`EndoDepthLoss`](../src/losses/depth_losses.py) : `L1 + SiLog + Gradients multi-échelles + Normales 3D` (avec dérivation analytique de normales depuis la profondeur si un dataset ne fournit pas de normales GT).
  - Trois configs YAML (`c3vd_zeroshot.yaml`, `c3vd_lora_baseline.yaml`, `c3vd_lora_grad_norm.yaml`), boucles [`src/train.py`](../src/train.py) (`--auto-resume`, FP16 AMP, sauvegarde `last.pt`/`best.pt`/`history.json`/`config_used.yaml`) et [`src/eval.py`](../src/eval.py).
  - Notebooks Colab « Tout exécuter » : [`notebooks/02_baseline_zeroshot.ipynb`](../notebooks/02_baseline_zeroshot.ipynb) et [`notebooks/03_train_lora_contrib.ipynb`](../notebooks/03_train_lora_contrib.ipynb), plus résumé [`docs/resume_etape_d.md`](resume_etape_d.md).
- **Vérifications locales (CPU, `--from-config-only` sur démo synthétique C3VD — pas de vrais résultats C3VD encore) :**
  - Tests unitaires métriques (`abs_rel < 1e-5` après alignement médian et `scale_shift`), vérification LoRA (24 couches `lora_b` reçoivent un gradient, 24 `base_linear` restent gelées ; `175 203 / 24 932 547` params avec neck gelé, `2 875 971 / 24 932 547` = `11,54 %` avec neck dégelé).
  - Tests CLI bout-en-bout (`eval.py` zero-shot + `train.py` époque 1 → `--auto-resume` époque 2 → `history.json` + `eval_test_median.json`) : tous passés.
  - Validation `nbformat` des 4 notebooks (`00` à `03`) : OK.
- **Blocage / Action requise côté utilisateur :**
  1. Télécharger les séquences C3VD (ou SimCol3D) dans Google Drive (aucun téléchargement automatique effectué).
  2. Exécuter [`01_dataset_viz.ipynb`](../notebooks/01_dataset_viz.ipynb), [`02_baseline_zeroshot.ipynb`](../notebooks/02_baseline_zeroshot.ipynb) puis [`03_train_lora_contrib.ipynb`](../notebooks/03_train_lora_contrib.ipynb) sur GPU Colab T4 (« Tout exécuter »).

## 25 novembre 2026 — Mise à jour : Préparation mixte `c3vd_zips/` (.zip + dossiers) vers `/content/data/c3vd`
- **Fait :** Ajout de [`prepare_c3vd_from_source`](../src/datasets/c3vd.py) et [`resolve_dataset_dir`](../src/datasets/factory.py), et mise à jour des notebooks `01`, `02`, `03` pour traiter uniformément archives `.zip` (extraction) et dossiers déjà décompressés (copie) depuis `MyDrive/endo-depth/data/c3vd_zips/` vers `/content/data/c3vd/<sequence>/`, avec suppression des doubles imbrications éventuelles et skip si déjà préparé.
- **Vérifications :** Test local avec mélange de `.zip` plats/imbriqués et dossiers plats/doublement imbriqués + validation `nbformat` des 3 notebooks : tout passe.

## 25 novembre 2026 — Correction suite au retour sur C3VD réel (Notebook 01 validé)
- **Constat :** Sur les vraies séquences C3VD dans `01_dataset_viz.ipynb`, les pixels saturés à `100 mm` sont des profondeurs tronquées, et la détection de reflets marquait le carré blanc en bas à droite ainsi que des zones saturées hors champ.
- **Corrections appliquées :**
  1. **Exclusion des pixels tronqués (`= max_depth_mm`, soit `100 mm` sur C3VD)** : borne stricte `depth < max_depth_mm - 1e-3` appliquée à la fois dans [`BaseEndoscopyDataset`](../src/datasets/base.py), [`EndoDepthLoss`](../src/losses/depth_losses.py) et [`depth_metrics.py`](../src/metrics/depth_metrics.py) (`align_prediction` + `compute_depth_metrics_single`).
  2. **`detect_specular_mask` limité au masque valide** : prise en compte de `valid_mask` dans [`detect_specular_mask`](../src/datasets/base.py) et filtrage strict dans [`01_dataset_viz.ipynb`](../notebooks/01_dataset_viz.ipynb) pour exclure coins noirs, bords hors champ et carré en bas à droite.
  3. **Masquage complet des coins sans profondeur (`depth == 0` / `mask.png`)** : mise à zéro explicite de `depth`/`normals` hors masque dans le loader, érosion 1px sur les bords invalides lors du calcul de pseudo-normales dans [`SurfaceNormalLoss`](../src/losses/depth_losses.py), et masquage des normales/reflets dans [`01_dataset_viz.ipynb`](../notebooks/01_dataset_viz.ipynb).

## 25 novembre 2026 — Correction : inclusion stricte de `cecum_t1_a` et découpage par séquence entière
- **Constat :** `dataset_stats.json` n'affichait que 209 images (`desc_t4_a` 148 + `trans_t1_a` 61), sans `cecum_t1_a` (276 images), et le fallback 2-séquences coupait `desc_t4_a` entre `val` et `test`.
- **Corrections appliquées :**
  1. **Chargement et extraction C3VD robustes + échec explicite (`fail-fast`)** : [`discover_c3vd_records`](../src/datasets/c3vd.py) et [`prepare_c3vd_from_source`](../src/datasets/c3vd.py) gèrent désormais aussi `rgb/`, `color/`, `.tif`/`.tiff`, fusionnent `c3vd_zips/` + `c3vd/` si les deux existent sur Drive, vérifient immédiatement qu'aucune archive/dossier ne donne 0 paire, affichent le compte d'images par séquence, et lèvent une erreur si une séquence de `expected_sequences` manque.
  2. **Découpage strict par séquence entière** : [`assign_sequence_splits`](../src/datasets/base.py) ne coupe plus jamais une séquence entre `val` et `test` (erreur si `< 3` séquences ; avec exactement 3 séquences, assignation déterministe par taille décroissante → `train=['cecum_t1_a']` (276), `val=['desc_t4_a']` (148), `test=['trans_t1_a']` (61) en attendant le lot 2).
  3. **Notebooks Colab** : ajout de `EXPECTED_SEQUENCES = ["cecum_t1_a", "desc_t4_a", "trans_t1_a"]`, `git pull --ff-only` automatique si clone GitHub présent, et affichage par séquence dans `01_dataset_viz.ipynb`.

## 25 novembre 2026 — Correctif Colab : priorité GitHub (`REPO_URL`), auto-détection `MyDrive` et blocage démo silencieuse
- **Constat :** Le dernier lancement Colab a affiché les graphiques sur `demo_seq_1..4` tout en laissant l'ancien `dataset_stats.json` (209 images), signe d'un ancien `src/` sur Drive et/ou d'un chemin `MyDrive/.../data/c3vd_zips` non trouvé à la racine attendue avec bascule silencieuse en mode démo.
- **Corrections appliquées :**
  1. **Priorité au code GitHub dans `01`, `02`, `03`** : `REPO_URL = "https://github.com/hanmaZi/endo-depth.git"` est pré-rempli et cloné/mis à jour dans `/content/endo-depth` avant d'utiliser un éventuel `src/` sur Drive.
  2. **Auto-détection bornée dans `MyDrive`** : [`resolve_dataset_dir`](../src/datasets/factory.py) cherche `c3vd_zips` / `c3vd` jusqu'à 4 niveaux dans `MyDrive` si absent du chemin par défaut.
  3. **Interdiction de bascule silencieuse en démo** : lorsque `expected_sequences` est renseigné, `resolve_dataset_dir` lève `FileNotFoundError` (avec liste des dossiers visibles dans `MyDrive`) au lieu de générer `demo_seq_1..4`.

## 25 novembre 2026 — Cause exacte de l'absence de `cecum_t1_a` résolue
- **Constat (via diagnostic dossier) :** Dans `cecum_t1_a`, les images couleur ne sont **pas** paddées sur 4 chiffres (`0_color.png`, `9_color.png`, ..., `99_color.png`, ..., `275_color.png`), alors que les cartes de profondeur/normales/occlusion le sont (`0000_depth.tiff`, `0009_depth.tiff`, ..., `0275_depth.tiff`).
- **Correction :** [`_extract_frame_id_candidates`](../src/datasets/c3vd.py) génère désormais les variantes paddées (`zfill(4..6)`) et non paddées, et [`discover_c3vd_records`](../src/datasets/c3vd.py) trie les paires par `(sequence_name, frame_id)` (testé localement sur `0_color.png..11_color.png` + `0000_depth.tiff..0011_depth.tiff`).
