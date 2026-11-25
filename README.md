# endo-depth

**Endoscopic monocular depth estimation in minimally invasive surgeries using deep learning**

Projet de Fin d'Études (Master M-IV, USTHB, 2026/2027) — estimation de profondeur monoculaire sur images endoscopiques (focus coloscopie).

## Structure du dépôt

```text
endo-depth/
├── configs/                # Fichiers de configuration YAML
│   ├── base.yaml
│   ├── c3vd_zeroshot.yaml
│   ├── c3vd_lora_baseline.yaml
│   └── c3vd_lora_grad_norm.yaml
├── data/                   # Datasets locaux (ignorés par Git)
├── docs/                   # État de l'art, datasets, journal, résumés et plan d'apprentissage
├── notebooks/              # Notebooks Google Colab / Kaggle ("Tout exécuter")
│   ├── 00_smoke_test.ipynb
│   ├── 01_dataset_viz.ipynb
│   ├── 02_baseline_zeroshot.ipynb
│   └── 03_train_lora_contrib.ipynb
├── results/                # Résultats, métriques JSON et checkpoints (poids ignorés par Git)
├── src/
│   ├── datasets/           # Loaders PyTorch (C3VD, SimCol3D, démo synthétique)
│   ├── models/             # DepthAnythingV2LoRA + SmokeDepthNet
│   ├── losses/             # L1 + SiLog + Gradients multi-échelles + Normales 3D
│   ├── metrics/            # AbsRel, SqRel, RMSE, RMSE_log, delta (< 1.25/1.25^2/1.25^3)
│   ├── utils.py            # Configuration YAML, seed, GPU/CPU, checkpoints, JSON
│   ├── train.py            # Boucle d'entraînement (+ --auto-resume)
│   └── eval.py             # Boucle d'évaluation (zero-shot et fine-tuned)
├── environment.yml         # Environnement Conda
├── requirements.txt        # Dépendances pip
└── README.md
```

## Installation locale (tests CPU)

```bash
git clone <URL_DU_REPO>
cd endo-depth

python -m venv .venv
# Windows (PowerShell) :
.\.venv\Scripts\Activate.ps1
# Linux / macOS :
# source .venv/bin/activate

pip install -r requirements.txt
```

Ou avec Conda :

```bash
conda env create -f environment.yml
conda activate endo-depth
```

## Vérification rapide (Smoke Test)

### 1. Pipeline Python en local (CPU)

```bash
python -m src.train --config configs/base.yaml --smoke-test --epochs 3
python -m src.eval --config configs/base.yaml --checkpoint results/checkpoints/smoke_last.pt --smoke-test
```

### 2. Environnement GPU Cloud (Google Colab / Kaggle)

Ouvre [`notebooks/00_smoke_test.ipynb`](notebooks/00_smoke_test.ipynb) dans Google Colab ou Kaggle avec un accélérateur GPU activé pour vérifier CUDA, PyTorch et le chargement de `depth-anything/Depth-Anything-V2-Small-hf`.

## Feuille de route

- [x] **Étape A** : Mise en place du dépôt, environnement, config YAML et smoke tests CPU/GPU
- [x] **Étape B** : État de l'art ([`docs/etat_de_l_art.md`](docs/etat_de_l_art.md))
- [x] **Étape C** : Préparation et visualisation des datasets endoscopiques ([`docs/datasets.md`](docs/datasets.md), [`notebooks/01_dataset_viz.ipynb`](notebooks/01_dataset_viz.ipynb))
- [x] **Étape D** : Baseline reproductible ([`notebooks/02_baseline_zeroshot.ipynb`](notebooks/02_baseline_zeroshot.ipynb), [`notebooks/03_train_lora_contrib.ipynb`](notebooks/03_train_lora_contrib.ipynb))
- [x] **Étape E** : Plan d'apprentissage ciblé ([`docs/plan_apprentissage.md`](docs/plan_apprentissage.md))
