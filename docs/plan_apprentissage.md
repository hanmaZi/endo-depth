# Plan d'apprentissage ciblé — Deep Learning pour l'estimation de profondeur (Novembre – Décembre 2026)

> **Objectif :** Passer de **zéro base en deep learning** à un socle solide et pratique **avant janvier 2027** (bases PyTorch + compréhension des architectures + baseline coloscopie reproductible sur Colab/Kaggle), sans te perdre dans des cours théoriques trop larges.
>
> **Rythme réaliste :** ~**6 à 8 heures par semaine** sur **5 semaines** (du 26 novembre au 31 décembre 2026), en apprenant **en pratiquant** directement dans le dépôt `endo-depth`.

---

## Comment utiliser ce plan (méthode « apprendre en faisant »)

1. **Règle des 30 / 70 :** 30 % de lecture/vidéo pour comprendre l'intuition, 70 % de pratique dans le code (`endo-depth`) et les notebooks Colab.
2. **Ne cherche pas l'exhaustivité mathématique au début :** retiens d'abord *à quoi sert chaque brique*, *quelles dimensions entrent et sortent* `(B, C, H, W)`, et *comment vérifier que ça apprend* (courbe de loss qui baisse).
3. **À chaque concept :** lis la fiche rapide, modifie un fichier de `endo-depth`, et observe l'effet.

---

## Parcours sur 5 semaines (26 novembre → 31 décembre 2026)

### Semaine 1 (26 nov – 2 déc 2026) : Fondamentaux Deep Learning & PyTorch
- **Notions à maîtriser :**
  - Qu'est-ce qu'un **tenseur** PyTorch (tableau multidimensionnel `(Batch, Canaux, Hauteur, Largeur)` qui tourne sur CPU ou GPU).
  - **Forward pass** (calcul des prédictions), **fonction de perte / Loss** (mesure l'erreur, ex. L1 / MSE), **Autograd & Backpropagation** (calcul automatique des gradients), **Optimizer (`AdamW`)** (mise à jour des poids).
  - Rôle d'une **époque** (*epoch*), d'un **batch**, du **learning rate**, et des **checkpoints** (sauvegarde sur Drive).
- **Ressources gratuites (dans l'ordre) :**
  1. Intuition visuelle (1 h 30) : [3Blue1Brown — Neural Networks](https://www.3blue1brown.com/topics/neural-networks) (épisodes 1 à 4 : neurones, descente de gradient, backpropagation).
  2. Pratique guidée (3–4 h) : [PyTorch — Learn the Basics](https://pytorch.org/tutorials/beginner/basics/index.html) (sections *Tensors*, *Datasets & DataLoaders*, *Build the Neural Network*, *Autograd*, *Optimization Loop*).
- **Mini-TP dans `endo-depth` :**
  - Lire et exécuter [`src/models/smoke.py`](../src/models/smoke.py) et [`src/train.py`](../src/train.py).
  - Changer le `learning_rate` dans [`configs/base.yaml`](../configs/base.yaml) (`1e-2`, `1e-4`, `1e-6`) et observer comment la loss évolue sur 10 époques.
- **Critère de validation :** Tu sais expliquer à l'oral la différence entre `optimizer.zero_grad()`, `loss.backward()` et `optimizer.step()`.

---

### Semaine 2 (3 – 9 déc 2026) : CNN, Encodeur-Décodeur & Chargement d'Images
- **Notions à maîtriser :**
  - **Convolution 2D (`nn.Conv2d`)** : comment un filtre détecte des bords, textures, reflets.
  - **Stride, Padding, Pooling / Upsampling** : comment réduire puis reconstruire la résolution spatiale.
  - Architecture **Encodeur-Décodeur / U-Net (skip connections)** : indispensable pour produire une carte pixel par pixel (même taille que l'image d'entrée).
  - **Dataset & DataLoader PyTorch**, normalisation d'images et *data augmentation* (flip, luminosité/contraste).
- **Ressources gratuites :**
  1. Intuition & schémas (2 h) : [Stanford CS231n — Convolutional Neural Networks](https://cs231n.github.io/convolutional-networks/).
  2. Vision pratique (2–3 h) : [Hugging Face Computer Vision Course](https://huggingface.co/learn/computer-vision-course) (modules d'introduction CNN & segmentation/tâches denses).
- **Mini-TP dans `endo-depth` (couplé à l'Étape C) :**
  - Implémenter/lire les loaders PyTorch des datasets endoscopiques dans `src/datasets/`.
  - Visualiser les tenseurs d'images et de cartes de profondeur (dimensions, min/max, valeurs invalides/masques).
- **Critère de validation :** Tu sais écrire un `Dataset` PyTorch qui charge une paire `(image RGB, depth map)` et vérifier la forme du batch `(B, 3, H, W)` et `(B, 1, H, W)`.

---

### Semaine 3 (10 – 16 déc 2026) : Métriques de Profondeur & Géométrie Monoculaire
- **Notions à maîtriser :**
  - Pourquoi une seule caméra a une **ambiguïté d'échelle** (*scale ambiguity*) : différence entre **profondeur métrique** (mm/m), **profondeur relative**, et **disparité** ($1/z$).
  - **Alignement d'échelle médian (*median scaling*)** avant évaluation.
  - Les 5 métriques standard de l'état de l'art :
    - `AbsRel` : erreur relative absolue moyenne ($\frac{|d - d^*|}{d^*}$).
    - `SqRel` : pénalise davantage les grosses erreurs ($\frac{|d - d^*|^2}{d^*}$).
    - `RMSE` et `RMSE log` : erreur quadratique (linéaire et logarithmique).
    - `δ < 1.25`, `1.25²`, `1.25³` : pourcentage de pixels « assez proches » de la vérité terrain.
  - Pertes spécifiques à la profondeur : **L1**, **Scale-Invariant Log Loss (SiLog)**, perte de **gradient / lissage**.
- **Ressources gratuites :**
  1. Lecture ciblée (1 h 30) : section « Métriques » et taxonomie de Niu et al. 2026 ([`arXiv:2507.20881`](https://arxiv.org/abs/2507.20881)).
  2. Relire [`docs/etat_de_l_art.md`](etat_de_l_art.md) (sections 1 et 4).
- **Mini-TP dans `endo-depth` (couplé à l'Étape D.1–D.2) :**
  - Implémenter les métriques dans `src/metrics/`.
  - Évaluer `Depth-Anything-V2-Small` en *zero-shot* sur un premier sous-ensemble coloscopique et analyser ses erreurs sur les reflets spéculaires et les plis.
- **Critère de validation :** Tu sais calculer manuellement sur un petit exemple NumPy `AbsRel` et $\delta_1$, et expliquer pourquoi la disparité de Depth Anything V2 doit être inversée/alignée avant comparaison.

---

### Semaine 4 (17 – 23 déc 2026) : Vision Transformers (ViT), Modèles Fondation & LoRA
- **Notions à maîtriser :**
  - Comment fonctionne l'**Attention (*Self-Attention*)** : Q, K, V intuitivement.
  - Comment un **Vision Transformer (ViT)** découpe une image en patchs 14×14 ou 16×16 (ex. DINOv2) et pourquoi il capte mieux la géométrie globale du côlon qu'un petit CNN.
  - Décodeur **DPT** (*Dense Prediction Transformer*) utilisé par MiDaS et Depth Anything V2.
  - **Fine-tuning léger par LoRA** : geler le modèle pré-entraîné et n'entraîner que de petites matrices de rang faible pour tenir dans 16 Go de VRAM (Colab T4).
  - **Précision mixte (AMP / FP16)** et reprise sur checkpoint.
- **Ressources gratuites :**
  1. Intuition attention (1 h 30) : [The Illustrated Transformer (Jay Alammar)](https://jalammar.github.io/illustrated-transformer/).
  2. ViT & Fine-tuning (2–3 h) : [Hugging Face Computer Vision Course](https://huggingface.co/learn/computer-vision-course) (chapitres Vision Transformers & Fine-tuning).
  3. Lecture ciblée (1 h) : résumé de **Depth Anything V2** ([`arXiv:2406.09414`](https://arxiv.org/abs/2406.09414)) et **EndoDAC** ([`arXiv:2405.08672`](https://arxiv.org/abs/2405.08672)).
- **Mini-TP dans `endo-depth` (couplé à l'Étape D.3) :**
  - Lancer le premier fine-tuning léger (ou LoRA) sur Colab T4, avec sauvegarde automatique des checkpoints sur Google Drive à chaque époque.
  - Logger et sauvegarder les courbes train/val et résultats dans `results/`.
- **Critère de validation :** Tu as une **baseline fine-tunée reproductible** avec métriques enregistrées dans `results/` et un notebook Colab qui reprend proprement après interruption.

---

### Semaine 5 (24 – 31 déc 2026) : Apprentissage Auto-Supervisé & Bilan Pré-2027
- **Notions à maîtriser :**
  - Comment entraîner *sans carte de profondeur* à partir d'une vidéo : **DepthNet + PoseNet**.
  - **Reprojection photométrique** : projeter les pixels de l'image $t+1$ vers l'image $t$ grâce à la profondeur prédite et aux intrinsèques de la caméra $K$.
  - Pourquoi l'endoscopie casse l'auto-supervision naïve : **reflets spéculaires**, lumière mobile, déformations (solutions de *Monodepth2* → *Endo-SfMLearner* → *AF-SfMLearner*).
- **Ressources gratuites :**
  1. Lecture structurée (2 h) : introduction & méthode de **Monodepth2** ([`arXiv:1806.01260`](https://arxiv.org/abs/1806.01260)) + **AF-SfMLearner** ([`arXiv:2112.01711`](https://arxiv.org/abs/2112.01711)).
  2. Vérification bibliographique (1–2 h) : préparer une fiche de 1 page résumant ta baseline et la piste d'amélioration choisie pour en discuter avec ton encadrante début janvier 2027.
- **Mini-TP dans `endo-depth` :**
  - Mettre au propre le `README.md`, comparer tes résultats zero-shot vs fine-tuned dans un tableau clair.
- **Critère de validation (Objectif Janvier 2027 atteint) :**
  - Tu comprends toute la chaîne `données → modèle → loss → backprop → métriques`.
  - Tu as un dépôt GitHub propre, des résultats chiffrés de baseline, et une piste claire à développer de janvier à juin 2027.

---

## Récapitulatif des ressources gratuites (favoris à garder)

| Thème | Ressource | Coût |
| :--- | :--- | :--- |
| Bases réseaux de neurones | [3Blue1Brown — Neural Networks](https://www.3blue1brown.com/topics/neural-networks) | Gratuit |
| PyTorch de zéro | [PyTorch — Learn the Basics](https://pytorch.org/tutorials/beginner/basics/index.html) | Gratuit |
| CNN & vision | [Stanford CS231n — CNN Notes](https://cs231n.github.io/convolutional-networks/) | Gratuit |
| Attention & Transformers | [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) | Gratuit |
| ViT & Hugging Face pratique | [Hugging Face Computer Vision Course](https://huggingface.co/learn/computer-vision-course) | Gratuit |
| GPU cloud gratuit | Google Colab (T4) & Kaggle Notebooks (30 h/semaine GPU) | Gratuit |
