# Datasets endoscopiques pour l'estimation de profondeur monoculaire

Ce document recense les principaux datasets publics évalués pour ce PFE (focus **coloscopie**), leurs licences, leurs formats et la procédure d'accès.

> **Règle :** Les informations ci-dessous proviennent de la documentation officielle de chaque dataset. Les champs non directement vérifiés dans les archives sont marqués **`à vérifier`**.

---

## 1. Tableau comparatif des datasets

| Dataset | Domaine | Source officielle | Licence | Taille & Volume | Vérité terrain profondeur | Procédure d'accès | Recommandation PFE |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **C3VD** *(Colonoscopy 3D Video Dataset)* | Coloscopie (endoscope clinique sur fantômes côlon) | [DurrLab / C3VD](https://durrlab.github.io/C3VD/) • [GitHub](https://github.com/DurrLab/C3VD) • [`arXiv:2206.08903`](https://arxiv.org/abs/2206.08903) | **CC BY-NC-SA 4.0** *(non-commercial)* | Archives `.zip` par séquence (~0,59 Go à 11,06 Go selon la séquence ; ex. `trans_t1_a` 61 frames / 0,59 Go, `desc_t4_a` 148 frames / 1,24 Go, `cecum_t1_a` 276 frames / 2,86 Go) | **Oui (dense)** : TIFF 16-bit (`0–100 mm`) + **normales de surface** 16-bit + masques d'occlusion + poses 6-DoF | Téléchargement direct par `.zip` sur le site officiel (pas de formulaire d'inscription) | **Prioritaire n°1** (données cliniques sur fantôme + profondeur + normales pour notre contribution) |
| **SimCol3D** | Coloscopie synthétique (rendu CT/Unity) + extraits réels | [UCL WEISS](https://www.ucl.ac.uk/interventional-surgical-sciences/simcol3d-data) • [GitHub](https://github.com/anitarau/simcol) • [`arXiv:2307.11261`](https://arxiv.org/abs/2307.11261) | **CC BY 4.0** *(UCL Figshare)* | > 37 000 frames synthétiques (taille exacte des `.zip` `à vérifier` sur Figshare) | **Oui sur le synthétique** : PNG 16-bit (plage `0–20 cm` = `0–200 mm`) + intrinsèques + poses. *Non sur la partie réelle EndoMapper.* | Téléchargement via UCL Figshare / page officielle | **Prioritaire n°2** (complément supervisé massif et test *sim-to-real* vers C3VD) |
| **EndoSLAM** | Gastro-intestinal (côlon, estomac, intestin grêle — porc ex-vivo & capsule synthétique) | [GitHub](https://github.com/CapsuleEndoscope/EndoSLAM) • [Mendeley Data](https://data.mendeley.com/datasets/cd2rtzm23r/1) • [`arXiv:2006.12892`](https://arxiv.org/abs/2006.12892) | **CC BY 4.0** | 35 sous-datasets (18 côlon) : 42 700 frames *ex-vivo* + 21 887 frames synthétiques (taille exacte `à vérifier` sur Mendeley) | **Partielle** : nuages de points 3D `.ply` + poses sur *ex-vivo* (pas de carte 2D directe sans projection) ; cartes de profondeur sur synthétique ($320 \times 320$) | Téléchargement public par sous-ensembles sur Mendeley Data | **Secondaire** (utile pour tests croisés *ex-vivo* ou auto-supervisé) |
| **SCARED** | Chirurgie robotique abdominale porcine (da Vinci) | [Portail EndoVis / Challenge](https://endovissub2019-scared.grand-challenge.org/) *(variante `SCARED-C` sur Mendeley : CC BY 4.0)* | **Accord d'utilisation recherche restrictif** *(sauf `SCARED-C` CC BY 4.0 `à vérifier`)* | 9 séquences / ~17k+ frames stéréo (`taille exacte à vérifier`) | **Oui** (lumière structurée sur tissus porcins) | Inscription et demande d'accès + acceptation DUA EndoVis / organisateurs | **Optionnel** (domaine = laparoscopie, pas coloscopie ; référence historique EndoDAC/AF-SfMLearner) |
| **Hamlyn** | Laparoscopie & endoscopie générale | [Hamlyn Centre / miroirs](https://huggingface.co/datasets/vslamlab/Hamlyn_Rectified_Dataset) • `Endo-Depth-and-Motion` (GPL-3.0) | **Usage académique** (outils souvent GPL-3.0 ; `Hamlyn-C` CC BY 4.0) | Plusieurs dizaines de vidéos stéréo (`taille exacte à vérifier`) | **Non native** (pseudo-profondeur stéréo calculée par Libelas sur paires rectifiées) | Téléchargement selon miroir (conditions d'origine Imperial College `à vérifier`) | **Optionnel** (test qualitatif / *zero-shot* hors coloscopie) |
| **EndoMapper** | Vraies coloscopies cliniques humaines | [Synapse `syn26707219`](https://www.synapse.org/#!Synapse:syn26707219) • [`arXiv:2204.14240`](https://arxiv.org/abs/2204.14240) | **Conditions projet EndoMapper** *(recherche académique ; docs CC BY-NC-SA 4.0)* | Séquences complètes de coloscopies cliniques (`taille totale à vérifier`) | **Non** (vidéos cliniques calibrées pour SLAM / auto-supervisé, sans capteur de profondeur) | **Inscription Synapse obligatoire + signature du Data Use Agreement (DUA)** | **Utile en fin de PFE** pour test qualitatif sur vraies coloscopies humaines |

---

## 2. Format des datasets supportés dans `src/datasets/`

### 2.1. C3VD ([`src/datasets/c3vd.py`](../src/datasets/c3vd.py))
Deux organisations de dossiers sont gérées automatiquement :
1. **Format plat** : `data/c3vd/<sequence>/0000_color.png`, `0000_depth.tiff`, `0000_normals.tiff`, `0000_occlusion.png`
2. **Format sous-dossiers** : `data/c3vd/<sequence>/color/0000_color.png`, `depth/0000_depth.tiff`, `normals/0000_normals.tiff`

Encodage vérifié :
- **Profondeur** : TIFF 16-bit non signé (`0..65535`) → $\text{depth\_mm} = (\text{raw} / 65535.0) \times 100.0$ (plage `0` à `100 mm`). Les pixels à `0` (ou masqués par `*_occlusion.png`) sont exclus du masque valide.
- **Normales de surface** : TIFF RGB 16-bit (`0..65535`) → $n = (\text{raw} / 65535.0) \times 2.0 - 1.0 \in [-1, 1]^3$ dans le repère caméra (attention : OpenCV lit en BGR, le loader convertit en RGB `(X, Y, Z)` puis renormalise à norme unitaire).

### 2.2. SimCol3D ([`src/datasets/simcol3d.py`](../src/datasets/simcol3d.py))
Formats supportés : `FrameBuffer/*.png` ou `FrameBuffer_*.png` avec `Depth/*.png` ou `Depth_*.png`.
- **Profondeur** : PNG 16-bit (`0..65535`) → $\text{depth\_mm} = (\text{raw} / 65535.0) \times \text{depth\_max\_mm}$ (paramètre `depth_max_mm: 200.0` par défaut, soit `20 cm`, modifiable en YAML).

---

## 3. Ce que tu dois télécharger / valider de ton côté

Pour respecter les licences médicales et la limite de stockage Google Drive gratuit (15 Go), **ne télécharge pas tout d'un coup** :

1. **Choix recommandé pour démarrer rapidement (sans inscription, ~4,7 Go au total) :**
   Va sur [https://durrlab.github.io/C3VD/](https://durrlab.github.io/C3VD/) (licence `CC BY-NC-SA 4.0`, usage académique autorisé) et télécharge **3 à 4 séquences représentatives de zones différentes du côlon**, par exemple :
   - `trans_t1_a.zip` (61 frames — 0,59 Go) → bon jeu rapide pour tests & validation
   - `desc_t4_a.zip` (148 frames — 1,24 Go)
   - `cecum_t1_a.zip` (276 frames — 2,86 Go)
   - *(Optionnel si espace Drive suffisant)* `sigmoid_t1_a.zip` (700 frames — 5,20 Go)
   Décompresse-les dans un dossier `endo-depth/data/c3vd/` sur ton Google Drive (ex. `MyDrive/endo-depth/data/c3vd/trans_t1_a/...`).
2. **Optionnel (SimCol3D, CC BY 4.0) :**
   Télécharge un sous-ensemble synthétique depuis [SimCol3D](https://www.ucl.ac.uk/interventional-surgical-sciences/simcol3d-data) et place les dossiers de séquences dans `data/simcol3d/`.
3. **Pour plus tard (EndoMapper — vraies coloscopies cliniques) :**
   Si tu veux tester qualitativement notre modèle sur des vidéos humaines réelles en fin de PFE, il faudra créer un compte sur [Synapse](https://www.synapse.org/#!Synapse:syn26707219) et signer le Data Use Agreement (`syn26707219`).
