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
