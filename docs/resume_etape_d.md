# Résumé — Étape D : Baseline LoRA et contribution (gradients & normales)

1. Nous avons implémenté les 7 métriques standard de profondeur (`AbsRel`, `SqRel`, `RMSE`, `RMSE_log`, `delta < 1.25 / 1.25² / 1.25³`) dans [`depth_metrics.py`](../src/metrics/depth_metrics.py) avec alignement d'échelle (médian et affine).
2. Un modèle complet [`DepthAnythingV2LoRA`](../src/models/depth_anything_lora.py) a été construit autour de `Depth-Anything-V2-Small-hf`, en gérant automatiquement le format des patchs DINOv2 (multiples de 14).
3. En mode *zero-shot*, le modèle pré-entraîné prédit une disparité relative qui est convertie et alignée par médiane pour obtenir une première référence sans réentraînement.
4. Pour le *fine-tuning* sur GPU gratuit (Colab T4), l'encodeur ViT reste figé et seules des matrices légères **LoRA** (`q_proj`, `v_proj`) ainsi que la tête de prédiction sont entraînées.
5. La sortie finale utilise une activation Sigmoid bornée sur `[min_depth_mm, max_depth_mm]`, conforme à la méthode métrique officielle de Depth Anything V2.
6. Pour notre contribution ciblant les polypes et plis du côlon, [`depth_losses.py`](../src/losses/depth_losses.py) ajoute aux pertes `L1 + SiLog` une perte sur les gradients multi-échelles et une perte cosinus sur les normales 3D.
7. Lorsque les normales de vérité terrain ne sont pas fournies par un dataset (ex. SimCol3D), elles sont dérivées automatiquement de la profondeur cible.
8. Trois configurations YAML reproductibles (`c3vd_zeroshot.yaml`, `c3vd_lora_baseline.yaml`, `c3vd_lora_grad_norm.yaml`) permettent une ablation rigoureuse avec et sans notre contribution.
9. Les boucles d'entraînement et d'évaluation supportent la précision mixte (FP16), le suivi JSON par époque et la reprise automatique (`--auto-resume`) après coupure Colab.
10. Deux notebooks Colab prêts pour « Tout exécuter » ([`02_baseline_zeroshot.ipynb`](../notebooks/02_baseline_zeroshot.ipynb) et [`03_train_lora_contrib.ipynb`](../notebooks/03_train_lora_contrib.ipynb)) restent prêts à tourner sur GPU dès le téléchargement de C3VD.
