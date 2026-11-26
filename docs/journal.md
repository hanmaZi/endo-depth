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

## 25–26 novembre 2026 — Premiers résultats Colab T4 (Lot 1 C3VD) et préparation v2
- **Protocole réel Colab T4 (Lot 1) :** `cecum_t1_a` (train, 276 images), `desc_t4_a` (val, 148 images), `trans_t1_a` (test, 61 images), alignement médian par frame (`align_mode: "median"`, `eval_test_median.json`).
- **Résultats mesurés sur `trans_t1_a` (61 images test) :**
  - `Zero-Shot DAV2-Small` : `AbsRel = 0.379189`, `SqRel = 3.155410`, `RMSE = 8.462212`, `RMSE_log = 0.407756`, `δ < 1.25 = 0.475590`.
  - `Baseline LoRA (L1+SiLog)` (`v1`) : `AbsRel = 0.163542`, `SqRel = 1.096292`, `RMSE = 7.446611`, `RMSE_log = 0.209548`, `δ < 1.25 = 0.739860`, `δ < 1.25² = 0.955987`, `δ < 1.25³ = 0.992991`.
  - `Contrib LoRA + Grad/Norm` (`v1`) : `AbsRel = 0.167094`, `SqRel = 1.308990`, `RMSE = 8.464066`, `RMSE_log = 0.221717`, `δ < 1.25 = 0.732430`, `δ < 1.25² = 0.941260`, `δ < 1.25³ = 0.987169`.
- **Observations et hypothèses explorées pour `v2` (non isolées à ce stade) :**
  1. **Observation (courbes) :** En `v1`, `val_AbsRel` sur `desc_t4_a` atteint son minimum dès les époques 1–2 (`~0.165–0.169`) puis remonte vers `~0.205–0.215` à l'époque 15. **Hypothèse :** le faible volume/diversité du train (`cecum_t1_a` uniquement), combiné à `train_neck: true` (2 875 971 paramètres, 11,54 % du modèle), `lr = 1e-4` et l'absence d'augmentation photométrique, favorise le surapprentissage.
  2. **Hypothèse (normales Scaramuzza vs pinhole) :** C3VD utilise une calibration omnidirectionnelle Scaramuzza ; comparer des normales estimées par une approximation pinhole (`fx = W`) aux `*_normals.tiff` GT introduit un résidu géométrique même à profondeur exacte, ce qui pourrait pénaliser le `RMSE` de `Contrib v1` (`8.464` vs `7.447`).
- **Modifications appliquées (`v2`, nouveaux `experiment_name`) :**
  1. [`BaseEndoscopyDataset`](../src/datasets/base.py) (`color_augment: true`) : jitter photométrique multiplicatif préservant à `0` les coins noirs du masque endoscopique.
  2. [`MultiScaleGradientLoss`](../src/losses/depth_losses.py) (`use_log: true`) et [`SurfaceNormalLoss`](../src/losses/depth_losses.py) (`use_gt_normals: false`, `w_normal: 0.25`) : gradients sur `log(depth)` et normales extraites symétriquement de `pred_depth` et `target_depth`.
  3. [`c3vd_lora_baseline.yaml`](../configs/c3vd_lora_baseline.yaml) et [`c3vd_lora_grad_norm.yaml`](../configs/c3vd_lora_grad_norm.yaml) (`*_v2`) + [`src/train.py`](../src/train.py) : `train_neck: false` (175 203 paramètres entraînables, 0,70 %), `epochs: 12`, `learning_rate: 5.0e-5`, `weight_decay: 1.0e-2`, `lr_scheduler: "cosine"`.

## 26 novembre 2026 — Résultats Colab T4 Lot 1 (`v2`) vs `v1`
- **Observations mesurées (courbes) :**
  - La loss d'entraînement `v2` décroît régulièrement (`~9.4` à l'époque 1 → `~1.3` à l'époque 12) au lieu de chuter sous `1.0` dès les époques 4–5 (`v1`).
  - Sur `desc_t4_a` (`val_AbsRel`), la remontée observée en `v1` n'apparaît plus : `v2` atteint son minimum à l'époque 8 (`~0.171` Baseline v2, `~0.175` Contrib v2) et reste autour de `~0.181` à l'époque 12 (contre `~0.205–0.215` en fin de `v1`).
- **Résultats mesurés sur le split test `trans_t1_a` (61 images, `eval_test_median.json`) :**
  - `Baseline LoRA v2 (L1+SiLog, Aug, NoNeck)` : `AbsRel = 0.159225`, `SqRel = 1.421037`, `RMSE = 9.053005`, `RMSE_log = 0.223624`, `δ < 1.25 = 0.753037`, `δ < 1.25² = 0.936134`, `δ < 1.25³ = 0.984224`.
  - `Contrib LoRA v2 + LogGrad/DepthNorm` : `AbsRel = 0.164859`, `SqRel = 1.473564`, `RMSE = 9.183605`, `RMSE_log = 0.229352`, `δ < 1.25 = 0.740674`, `δ < 1.25² = 0.932627`, `δ < 1.25³ = 0.982114`.
- **Pistes et hypothèses (non isolées car plusieurs changements simultanés entre `v1` et `v2`) :**
  1. **Hypothèse :** La combinaison `train_neck: false` + `lr = 5e-5` + `color_augment: true` régularise suffisamment pour stabiliser la validation et améliorer `AbsRel` / `δ < 1.25`, mais ralentit l'adaptation des couches restantes aux grandes profondeurs du lumen, d'où une hausse observée de `RMSE` (`7.447 → 9.053` même sans perte additionnelle).
  2. **Hypothèse :** Le passage à `use_gt_normals: false` (et `w_normal = 0.25`) pourrait expliquer la réduction de l'écart de `RMSE` entre `Contrib` et `Baseline` (`+1.017 mm` en `v1` vs `+0.131 mm` en `v2`), mais une ablation isolée est nécessaire pour le confirmer.
  3. **Hypothèse :** Avec une seule séquence d'entraînement (`cecum_t1_a`), l'apport géométrique de `Grad/Norm` est peu capté par `AbsRel`/`RMSE` seuls ; élargir les données (Lot 2) et mesurer les contours (`Grad_MAE`) est prioritaire avant tout changement d'hyperparamètres.

## 26 novembre 2026 — Préparation `v3` (Lot 2 C3VD, `Grad_MAE`, anti-fuite de texture et protocole d'ablation)
- **Données disponibles dans `c3vd_zips` (9 séquences au total) :** Lot 1 (`cecum_t1_a`, `desc_t4_a`, `trans_t1_a`) + Lot 2 vague 1 en `.zip` (`cecum_t2_a`, `sigmoid_t2_a`, `sigmoid_t3_a`, `trans_t2_a`, `trans_t3_a`, `trans_t3_b`).
- **Préparation rapide Colab ([`src/datasets/c3vd.py`](../src/datasets/c3vd.py)) :** sélection configurable via `SELECTED_SEQUENCES`, saut automatique des dossiers déjà valides dans `/content/data/c3vd`, copie séquentielle d'un bloc (`cp`) de chaque `.zip` vers le disque local suivie d'une décompression locale et suppression immédiate de l'archive temporaire.
- **Critère explicite de découpage sans fuite de texture ([`src/datasets/base.py`](../src/datasets/base.py)) :**
  - Dans C3VD, les suffixes `_t1.._t4` identifient les textures appliquées aux fantômes, et `_a`/`_b` correspondent à des trajectoires différentes sur le même segment et la même texture (ex. `trans_t3_a` et `trans_t3_b`).
  - Règle appliquée : découpage par **séquences entières** avec **disjonction stricte des groupes de texture** entre `train`, `val` et `test` (vérifiée automatiquement par [`validate_no_texture_leakage`](../src/datasets/base.py)) :
    - **`train` (`t2` + `t3`, 6 séquences / 3 segments) :** `cecum_t2_a`, `sigmoid_t2_a`, `sigmoid_t3_a`, `trans_t2_a`, `trans_t3_a`, `trans_t3_b`.
    - **`val` (`t4`, 1 séquence) :** `desc_t4_a`.
    - **`test` (`t1`, 2 séquences / 2 segments) :** `cecum_t1_a`, `trans_t1_a` *(permet de conserver `trans_t1_a` pour la comparaison par séquence avec le Lot 1 tout en évaluant un second segment inédit en Lot 2)*.
- **Évaluation et visualisation ajoutées en `v3` (hyperparamètres `v2` strictement conservés : `epochs=12`, `lr=5e-5`, `weight_decay=1e-2`, `cosine`, `train_neck=false`, `color_augment=true`) :**
  - Ajout de `grad_mae` (MAE des gradients spatiaux en mm/px) et sauvegarde systématique de `per_sequence_metrics` dans [`src/metrics/depth_metrics.py`](../src/metrics/depth_metrics.py), [`src/eval.py`](../src/eval.py) et [`src/train.py`](../src/train.py).
  - Grille qualitative dans [`notebooks/03_train_lora_contrib.ipynb`](../notebooks/03_train_lora_contrib.ipynb) : `RGB | Vérité terrain | Baseline v3 | Contrib v3 | Erreur |Baseline - GT| | Erreur |Contrib - GT|`.
  - Préparation de l'ablation isolée en 4 variantes ([`configs/c3vd_v3_*.yaml`](../configs)) : `baseline`, `grad_only`, `norm_only`, `grad_norm`, sur 3 graines (`42, 123, 2026`), agrégée en moyenne ± écart-type globalement et par séquence de test via [`aggregate_ablation_results`](../src/eval.py).
