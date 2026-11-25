# État de l'art — Estimation de profondeur monoculaire en endoscopie (focus coloscopie)

> **Règle de rigueur :** Ce document ne cite que des informations bibliographiques et techniques vérifiées via les sources publiques (arXiv, dépôts officiels, notices d'éditeurs). Tout détail non consulté directement dans un PDF (ex. articles sous paywall ou chiffres exacts non lus) est explicitement marqué **`à vérifier`**.

---

## 1. Mini-glossaire des concepts clés (pour bien lire la suite)

- **Supervision supervisée vs auto-supervisée (self-supervised)** :
  - *Supervisée* : le réseau apprend avec une vraie carte de profondeur de référence (*ground truth*) pour chaque image (rare en chirurgie réelle, disponible surtout en simulation ou sur fantômes).
  - *Auto-supervisée* : aucune carte de profondeur n'est nécessaire à l'entraînement. À partir d'une séquence vidéo, un réseau prédit la profondeur et un autre prédit le mouvement de la caméra (*PoseNet*) ; on reprojette une image $t$ vers $t+1$ et on minimise l'erreur visuelle (*perte photométrique*).
- **Ambiguïté d'échelle (monoculaire)** : avec une seule caméra, un objet petit et proche ressemble à un objet grand et loin. La profondeur monoculaire est souvent relative ; pour évaluer en millimètres, on applique un **alignement d'échelle médian** (*median scaling*).
- **CNN vs ViT (Vision Transformer)** :
  - *CNN* (ex. ResNet) : capte bien les détails locaux (bords, textures) avec peu de paramètres, mais a un champ de vision limité.
  - *ViT* (ex. DINOv2) : découpe l'image en patchs et modélise les relations globales sur toute l'image ; très robuste mais plus coûteux en mémoire.
- **LoRA (Low-Rank Adaptation)** : technique de *fine-tuning* léger qui gèle le gros modèle pré-entraîné et n'entraîne que de toutes petites matrices additives (~1–2 % des paramètres), idéale pour un GPU gratuit type Colab T4.
- **Modèles de diffusion / Flow Matching** : génèrent la profondeur par débruitage itératif (diffusion) ou transport direct (Flow Matching). Bords très nets, mais plus lourds/lents qu'un modèle direct (*discriminatif*).

---

## 2. Analyse des 3 références de départ

### 2.1. Niu et al., 2026 — *Endoscopic depth estimation based on deep learning: A survey*
- **Référence :** *Neurocomputing*, Vol. 696, Article 133958 (2026) | DOI : [`10.1016/j.neucom.2026.133958`](https://doi.org/10.1016/j.neucom.2026.133958) | Preprint : [`arXiv:2507.20881`](https://arxiv.org/abs/2507.20881)
- **Auteurs :** Ke Niu, Zeyun Liu, Xue Feng, Heng Li, Naian Xiao, Binghua Su, Qika Lin, Kaize Shi.
- **Synthèse :**
  1. **Niveau données :** recense les datasets publics (Hamlyn, SCARED, SERV-CT, données synthétiques/simulation) et les métriques standard (`AbsRel`, `SqRel`, `RMSE`, `RMSE log`, seuils $\delta < 1.25 / 1.25^2 / 1.25^3$).
  2. **Taxonomie des méthodes :** sépare les approches **monoculaires** (supervisées, semi-supervisées, auto-supervisées, adaptation de domaine non supervisée / UDA) des approches **stéréo** (appariement par deep learning, réseaux stéréo bout-en-bout).
  3. **Verrous cliniques identifiés :** reflets spéculaires sur tissus humides, fumée/fluides chirurgicaux, déformations non rigides des organes, zones pauvres en texture et éclairage mobile non uniforme.
  4. **Directions ouvertes :** adaptation de domaine inter-patients/inter-caméras, contraintes temps réel et fusion multimodale.
- **Éléments `à vérifier` :** tableaux complets des scores numériques par papier dans la version finale publiée (Elsevier).

### 2.2. Cheng et al., 2024 — *TinyDepth: Lightweight self-supervised monocular depth estimation based on transformer*
- **Référence :** *Engineering Applications of Artificial Intelligence*, Vol. 138, 109313 (2024) | DOI : `10.1016/j.engappai.2024.109313` | Code : [`ZYCheng777/TinyDepth`](https://github.com/ZYCheng777/TinyDepth)
- **Auteurs :** Zeyu Cheng, Yi Zhang, Yang Yu, Zhe Song, Chengkai Tang.
- **Synthèse :**
  - **Idée clé :** combiner la capacité de modélisation globale des Transformers avec un budget de calcul très faible (**~6,2 M de paramètres**), afin d'obtenir une inférence rapide en auto-supervisé.
  - **Architecture :** encodeur léger intégrant des mécanismes d'attention multi-échelle + décodeur à **fusion éparse (*sparse fusion upsampling*)** qui réduit la redondance entre niveaux de résolution.
  - **Datasets d'origine :** **KITTI** (conduite routière) et **NYU Depth V2** (scènes d'intérieur).
  - **Limites pour notre PFE :** conçu et testé sur scènes naturelles (non endoscopiques) ; comme tout modèle auto-supervisé par perte photométrique standard, il reste sensible aux surfaces peu texturées ou réfléchissantes (très fréquentes dans le côlon) et n'intègre pas de correction d'éclairage endoscopique.
- **Éléments `à vérifier` :** scores exacts `AbsRel` / `RMSE` et latence mesurée par les auteurs sur KITTI/NYU (article Elsevier sous paywall), ainsi que son comportement réel une fois réentraîné sur C3VD/SimCol3D.

### 2.3. Tan et al., 2025 — *Accurate and fast monocular endoscopic depth estimation of structure-content integrated diffusion*
- **Référence :** *Computerized Medical Imaging and Graphics*, Vol. 125, p. 102640 (2025) | DOI : [`10.1016/j.compmedimag.2025.102640`](https://doi.org/10.1016/j.compmedimag.2025.102640)
- **Auteurs :** Min Tan, Yu-Shun Tao, Boyun Zheng, Gaosheng Xie, Ze-Yang Xia, Jing Xiong.
- **Synthèse :**
  - **Idée clé :** les modèles de diffusion préservent très bien les frontières anatomiques dans les milieux peu texturés ou déformables, mais sont habituellement trop lents pour la chirurgie. Ce papier intègre des contraintes structure-contenu dans un cadre de diffusion pour améliorer la précision géométrique tout en accélérant l'inférence en endoscopie.
- **Éléments `à vérifier` (paywall ScienceDirect) :** architecture exacte du débruiteur, nombre d'étapes d'inférence, liste exacte des datasets (ex. SCARED, Hamlyn) et résultats quantitatifs (`AbsRel`, `RMSE`, FPS) non accessibles sans le PDF complet.

---

## 3. Panorama des méthodes clés (classiques, fondation, diffusion, coloscopie)

### 3.1. Approches auto-supervisées classiques & endoscopiques
1. **Monodepth2** (Godard et al., ICCV 2019 — [`arXiv:1806.01260`](https://arxiv.org/abs/1806.01260)) : socle historique de l'auto-supervision vidéo/stéréo (ResNet `DepthNet` + `PoseNet`, *minimum reprojection loss*, *auto-masking*, régularisation de lissage). Hypothèse de luminosité constante et scène rigide → échoue sur l'endoscopie où la lampe bouge avec la caméra et crée des reflets spéculaires.
2. **Endo-SfMLearner** (Özyörük et al., *Medical Image Analysis* 2021 — [`arXiv:2006.12892`](https://arxiv.org/abs/2006.12892)) : introduit le dataset **EndoSLAM** et adapte `SfMLearner` avec attention spatiale dans le `PoseNet` et une perte photométrique tenant compte des variations globales de luminosité.
3. **AF-SfMLearner** (Shao et al., 2022 — [`arXiv:2112.01711`](https://arxiv.org/abs/2112.01711)) : ajoute un module **Appearance Flow (AF)** sur une base ResNet-18 pour modéliser explicitement les variations locales d'illumination et les reflets sur tissus humides (SCARED, EndoVis, Hamlyn, SERV-CT). Référence classique en endoscopie, mais backbone CNN à champ réceptif limité.

### 3.2. Modèles de fondation & adaptation légère (ViT / LoRA)
4. **MiDaS** (Ranftl et al., TPAMI — [`arXiv:1907.01341`](https://arxiv.org/abs/1907.01341)) & **DPT** ([`arXiv:2103.13413`](https://arxiv.org/abs/2103.13413)) : entraînement supervisé multi-datasets avec perte invariante à l'échelle et au décalage (*scale- and shift-invariant loss*) et décodeur dense sur Vision Transformer (DPT). Prédit une profondeur relative robuste sur scènes naturelles, mais subit un fort *domain gap* en endoscopie (couleurs roses homogènes, géométrie tubulaire, reflets).
5. **Depth Anything V2** (Yang et al., NeurIPS 2024 — [`arXiv:2406.09414`](https://arxiv.org/abs/2406.09414)) : encodeur **DINOv2** + décodeur **DPT**. Un gros modèle enseignant (Giant) est entraîné uniquement sur ~595K images synthétiques précises (5 datasets : Hypersim, Virtual KITTI 2, BlendedMVS, TartanAir, IRS), puis génère des pseudo-labels sur **62M+ images réelles** pour entraîner des modèles étudiants (dont **Small ~25M params**). Plus de 10× plus rapide que la diffusion type Marigold, mais non entraîné sur imagerie médicale.
6. **EndoDAC** (Cui et al., MICCAI 2024 — [`arXiv:2405.08672`](https://arxiv.org/abs/2405.08672)) : adapte *Depth Anything* à l'endoscopie en auto-supervisé via **DV-LoRA** (*Dynamic Vector-Based LoRA*) + *Convolutional Neck* + estimation automatique des intrinsèques caméra. N'entraîne que **~1,6M paramètres (~1,6 %)** tout en dépassant `AF-SfMLearner` sur SCARED (15 351 train / 1 705 val / 551 test) et en généralisant en zero-shot sur Hamlyn (~17,7 ms/image annoncés).
7. **DARES** (Zeinoddin et al., 2024 — [`arXiv:2408.17433`](https://arxiv.org/abs/2408.17433)) : adapte *Depth Anything* via **Vector-LoRA** auto-supervisé sur SCARED et EndoVis.
8. **EndoOmni** (Tian et al., 2024 — [`arXiv:2409.05442`](https://arxiv.org/abs/2409.05442), code [`TianCuteQY/EndoOmni`](https://github.com/TianCuteQY/EndoOmni)) : modèle fondation dédié à l'endoscopie (profondeur relative et métrique en zero-shot) par auto-apprentissage enseignant-étudiant sur un méta-dataset massif (>200k images labellisées + >500k non labellisées, incluant SCARED, C3VD, SimCol3D, Hamlyn, EndoSLAM, EndoMapper). Très fort en généralisation, mais l'entraînement complet demande beaucoup de données et de calcul.

### 3.3. Modèles de diffusion génériques pour la profondeur
9. **Marigold** (Ke et al., CVPR 2024 — [`arXiv:2312.02145`](https://arxiv.org/abs/2312.02145)) : réutilise un U-Net de Stable Diffusion pour prédire la profondeur par débruitage DDIM (10–50 étapes à l'origine, réduit à 1–4 avec LCM). Bords très nets, mais empreinte VRAM élevée, besoin fréquent d'*ensembling* (plusieurs passes moyennées) et risque d'hallucinations hors domaine.
10. **DepthFM** (Gui et al., 2024 / AAAI 2025 — [`arXiv:2403.13788`](https://arxiv.org/abs/2403.13788)) : remplace la diffusion classique par du **Flow Matching** pour obtenir une profondeur nette en **1 à 2 évaluations (NFE)** avec perte auxiliaire sur les normales, mais reste basé sur une lourde architecture générative de type Stable Diffusion (peu adaptée à la coloscopie sans réentraînement lourd).

### 3.4. Travaux spécifiques à la coloscopie et aux polypes
11. **Rau et al., 2019** (*IJCARS 2019* — *Implicit domain adaptation with conditional GANs*) : prédiction de profondeur en coloscopie par **cGAN (pix2pix)** entraîné en simulation Unity + adaptation implicite sur images cliniques. Pionnier en coloscopie, mais instabilité des GAN et traitement image par image.
12. **C3VD** (Bobrow et al., *Medical Image Analysis* 2023 — [`arXiv:2206.08903`](https://arxiv.org/abs/2206.08903)) : référence matérielle coloscopique (endoscope clinique réel sur fantômes anatomiques de côlon) fournissant **profondeur dense, normales de surface, flux optique et poses 6-DoF** via recalage 2D-3D multimodal.
13. **SimCol3D** (Rau et al., challenge MICCAI EndoVis — [`arXiv:2307.11261`](https://arxiv.org/abs/2307.11261)) : benchmark de reconstruction 3D en coloscopie (simulations CT + séquences réelles). Le bilan du challenge montre que les réseaux hybrides CNN-Transformer prédisent très bien la profondeur sur synthétique (erreur sub-millimétrique), mais que le transfert *sim-to-real* et la pose caméra restent difficiles.
14. **Du et al., 2024** (*Uncertainty-based Global and Local Fusion Network* — [`arXiv:2409.15006`](https://arxiv.org/abs/2409.15006)) : dédié à la généralisation en coloscopie face aux **reflets spéculaires non-lambertiens** ; fusionne des features CNN locales et Transformer globales guidées par l'incertitude pour généraliser de la simulation vers la coloscopie réelle.
15. **EndoMapper** ([`arXiv:2204.14240`](https://arxiv.org/abs/2204.14240), Azagra et al., 2022) & **Polyp-sizing** (ex. *PolypSense3D*, NeurIPS ; `BseNet` / reconstruction 3D pour mesure de polypes $\le 5\text{ mm}$ / $\le 10\text{ mm}$) : illustrent l'usage clinique direct d'une bonne profondeur locale pour mesurer la taille réelle d'un polype — *détails architecturaux et métriques exactes des papiers polyp-sizing `à vérifier`*.

---

## 4. Tableau comparatif synthétique

| Méthode | Supervision | Architecture | Datasets | Métriques | Points faibles | Lien |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Niu et al. (2026)** | Survey (toutes) | Taxonomie (Mono / Stéréo) | Hamlyn, SCARED, SERV-CT, Sim | AbsRel, SqRel, RMSE, RMSElog, $\delta$ | Survey (pas de nouveau modèle) ; tableaux complets `à vérifier` | [`arXiv:2507.20881`](https://arxiv.org/abs/2507.20881) |
| **TinyDepth** (Cheng 2024) | Auto-supervisée | Transformer léger + décodeur *sparse fusion* (~6,2M params) | KITTI, NYU Depth V2 | AbsRel, RMSE, $\delta$ (`valeurs exactes à vérifier`) | Non testé en endoscopie ; perte photométrique fragile sur reflets/zones lisses | [`GitHub`](https://github.com/ZYCheng777/TinyDepth) |
| **Tan et al. (2025)** | `à vérifier` | Diffusion intégrée structure-contenu | Endoscopie (SCARED/Hamlyn `à vérifier`) | `à vérifier` (paywall) | Coût mémoire/latence de diffusion (`détails à vérifier`) | [`DOI`](https://doi.org/10.1016/j.compmedimag.2025.102640) |
| **Monodepth2** (2019) | Auto-supervisée (vidéo/stéréo) | ResNet DepthNet + PoseNet | KITTI, Cityscapes, Make3D | AbsRel, SqRel, RMSE, RMSElog, $\delta$ | Hypothèse scène rigide & lumière fixe → échoue en endoscopie | [`arXiv:1806.01260`](https://arxiv.org/abs/1806.01260) |
| **Endo-SfMLearner** (2021) | Auto-supervisée (luminosité-aware) | DepthNet + PoseNet à attention spatiale | EndoSLAM (porc ex-vivo, fantôme, synthétique) | RMSE, AbsRel, ATE/RPE (`à vérifier`) | Sensible aux forts reflets spéculaires et déformations | [`arXiv:2006.12892`](https://arxiv.org/abs/2006.12892) |
| **AF-SfMLearner** (2022) | Auto-supervisée + *Appearance Flow* | ResNet-18 DepthNet + PoseNet + module AF | SCARED, EndoVis, Hamlyn, SERV-CT | AbsRel, SqRel, RMSE, RMSElog, $\delta$ | Champ réceptif CNN limité ; sensible aux grands plis/déformations | [`arXiv:2112.01711`](https://arxiv.org/abs/2112.01711) |
| **EndoDAC** (2024) | Auto-supervisée | Depth Anything (ViT) + DV-LoRA (~1,6M params entraînables) | SCARED, Hamlyn (zero-shot) | AbsRel, SqRel, RMSE, RMSElog, $\delta$ (`scores exacts à vérifier`) | Plus lourd qu'un petit CNN ; domaine laparoscopie ≠ coloscopie | [`arXiv:2405.08672`](https://arxiv.org/abs/2405.08672) |
| **DARES** (2024) | Auto-supervisée | Depth Anything + Vector-LoRA | SCARED, EndoVis | AbsRel, RMSE, $\delta$ (`à vérifier`) | Même sensibilité aux occlusions/fumée et transfert côlon | [`arXiv:2408.17433`](https://arxiv.org/abs/2408.17433) |
| **MiDaS / DPT** (2020/21) | Supervisée multi-datasets | CNN / ViT + décodeur DPT | Mélanges scènes naturelles | Erreur relative / $\delta$ | Fort *domain gap* médical ; profondeur relative uniquement | [`arXiv:1907.01341`](https://arxiv.org/abs/1907.01341) |
| **Depth Anything V2** (2024) | Semi-supervisée (Teacher synthétique + 62M pseudo-labels) | DINOv2 (Small 25M → Giant 1.3B) + DPT | 5 datasets synthétiques + 62M réels + DA-2K | AbsRel, $\delta_1$, RMSE | Non entraîné sur muqueuses humides/reflets ; relative par défaut | [`arXiv:2406.09414`](https://arxiv.org/abs/2406.09414) |
| **EndoOmni** (2024) | Semi-supervisée (Teacher-Student) | ViT fondation endoscopique | Méta-dataset >700k frames (SCARED, C3VD, SimCol3D, Hamlyn...) | AbsRel, SqRel, RMSE, RMSElog, $\delta_1$ | Entraînement complet trop lourd pour GPU gratuit (`scores exacts à vérifier`) | [`arXiv:2409.05442`](https://arxiv.org/abs/2409.05442) |
| **Marigold / DepthFM** (2024) | Supervisée (priors Stable Diffusion) | U-Net Diffusion (DDIM) / Flow Matching (1-2 NFE) | Hypersim, Virtual KITTI 2 (transfert zero-shot) | AbsRel, $\delta_1$ | VRAM élevée, risque d'hallucination médicale, lent (Marigold) | [`arXiv:2312.02145`](https://arxiv.org/abs/2312.02145) / [`2403.13788`](https://arxiv.org/abs/2403.13788) |
| **Rau et al. / SimCol3D** (2019–24) | Supervisée / adaptation cGAN | cGAN (pix2pix) ; CNN-Transformer hybrides | SimCol3D, Unity colon, séquences réelles | RMSE, L1, AbsRel (`à vérifier`) | Transfert *sim-to-real* et dérive de pose sur côlon réel | [`arXiv:2307.11261`](https://arxiv.org/abs/2307.11261) |
| **Du et al.** (2024) | Supervisée (sim-to-real) | Fusion CNN local + Transformer global par incertitude | Coloscopie synthétique/fantôme → réel (`à vérifier`) | AbsRel, RMSE, $\delta$ (`à vérifier`) | Code/reproductibilité et coût exact `à vérifier` | [`arXiv:2409.15006`](https://arxiv.org/abs/2409.15006) |

---

## 5. Pistes d'amélioration réalistes faisables sans GPU puissant (Colab T4 16 Go / Kaggle)

Étant donné ta contrainte matérielle (**pas de GPU local**, entraînement sur **T4 16 Go gratuit** avec coupures de session) et ton focus **coloscopie / polypes**, voici **4 pistes concrètes et réalistes** :

### Piste 1 — Adaptation par LoRA / DV-LoRA d'un modèle fondation (`Depth-Anything-V2-Small`) sur la coloscopie
- **Principe :** Garder `Depth-Anything-V2-Small` (~25M paramètres) comme socle figé et n'entraîner que des couches **LoRA** (~1 à 2M paramètres) + ajustement léger sur des données coloscopiques (ex. C3VD ou SimCol3D).
- **Faisabilité T4 :** Consomme peu de VRAM (< 8 Go en précision mixte FP16), converge vite, et produit des checkpoints légers (quelques Mo) faciles à sauver sur Google Drive.
- **Intérêt PFE :** Mesurer le gain de généralisation entre un zero-shot pur, un fine-tuning LoRA supervisé (sur C3VD/SimCol3D) et/ou auto-supervisé (EndoMapper/EndoSLAM).

### Piste 2 — Masquage et pondération des reflets spéculaires et variations d'éclairage
- **Principe :** En coloscopie, la muqueuse mouillée renvoie des taches blanches saturées (reflets spéculaires) et le fond du côlon reste sombre. On peut :
  1. Détecter automatiquement les reflets spéculaires (seuillage HSV : forte luminance + faible saturation) pour les masquer ou réduire leur poids dans la loss d'entraînement ;
  2. Ajouter une *data augmentation* photométrique réaliste (vignettage, variation d'éclairage directionnel, reflets synthétiques) pendant le fine-tuning.
- **Faisabilité T4 :** Quasi **zéro surcoût GPU** (simple pré-traitement / masquage dans la loss).
- **Intérêt PFE :** Attaque directement le verrou principal de la fiche PFE sur les images réelles et les polypes brillants.

### Piste 3 — Préservation des reliefs et bords de polypes via une loss géométrique (gradients / normales de surface)
- **Principe :** Les pertes classiques (`L1` / `RMSE`) ont tendance à lisser les petits reliefs comme les polypes plats ou sessiles. On ajoute à la fonction de perte un terme de **cohérence des gradients spatiaux** ($\partial_x d, \partial_y d$) et/ou de **normales de surface** (directement disponibles dans C3VD !).
- **Faisabilité T4 :** Calcul analytique très léger sur tenseurs PyTorch, sans réseau additionnel.
- **Intérêt PFE :** Justification directe par l'objectif médical (mieux restituer la géométrie locale des polypes et des plis coliques).

### Piste 4 — Modèle léger hybride / Distillation depuis un modèle fondation (inspiration *TinyDepth*)
- **Principe :** Entraîner un petit modèle rapide (~5–6M paramètres, ex. *TinyDepth* ou encodeur léger type MobileNet/ConvNeXt-Tiny) en **distillant** les prédictions d'un modèle fondation plus gros (`Depth-Anything-V2`) sur des séquences de coloscopie.
- **Faisabilité T4 :** Un petit modèle étudiant s'entraîne très vite sur T4 et tourne quasiment en temps réel à l'inférence.
- **Intérêt PFE :** Permet de présenter une étude comparative rigoureuse **Précision vs Vitesse / Taille mémoire** (critère clé souligné dans Niu et al., 2026).

> **Conseil méthodologique :** La combinaison la plus solide et réaliste avant la soutenance est souvent **Piste 1 (base Depth Anything V2 Small + LoRA)** combinée avec **Piste 2 (gestion des reflets)** ou **Piste 3 (loss gradients/normales pour les polypes)**, puis une comparaison face à un modèle léger (**Piste 4** / TinyDepth) comme baseline alternative.
