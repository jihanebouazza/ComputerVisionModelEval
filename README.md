# Évaluation comparative des modèles de Computer Vision

Projet expérimental réalisé dans le cadre du module **Computer Vision**. Ce dépôt présente un notebook Python comparant plusieurs approches de segmentation d'images, de détection d'objets, de segmentation sémantique et de segmentation d'instances.

## Objectifs

- Comprendre les différences entre regroupement de pixels, détection d'objets et segmentation pixel par pixel.
- Exécuter des algorithmes classiques et des réseaux de neurones pré-entraînés sur des images annotées.
- Comparer les performances prédictives, la qualité visuelle des résultats et les temps d'exécution.
- Étudier les compromis entre précision et coût de calcul.

## Contenu du dépôt

```text
.
├── Computer_Vision_Model_Eval.ipynb   # Code, expériences, figures et évaluations
└── README.md                         # Documentation du projet
```

Les expériences sont regroupées dans **un notebook Jupyter** ; aucun script Python séparé n'est nécessaire. Les jeux de données et les poids des modèles sont téléchargés lors de l'exécution et ne sont pas inclus dans le dépôt.

## Technologies

- **Python**, **Jupyter Notebook / Google Colab**
- **NumPy**, **Pandas**, **Matplotlib** : manipulation, tableaux et visualisation
- **scikit-image**, **scikit-learn** : image de démonstration, K-Means, Mean Shift et Silhouette Score
- **PyTorch**, **torchvision** : FCN, DeepLabV3 et Mask R-CNN
- **Ultralytics** : YOLOv8n, YOLO11n et YOLO11n-seg
- **Hugging Face Datasets** : accès à Pascal VOC 2012 en streaming
- **OpenCV**, **Pillow** : lecture, transformation et affichage d'images
- **pycocotools** : lecture des annotations COCO et évaluation des masques d'instances

## Installation et exécution

### Option recommandée : Google Colab

1. Ouvrir [Google Colab](https://colab.research.google.com/).
2. Importer `Computer_Vision_Model_Eval.ipynb` depuis le dépôt GitHub ou depuis votre ordinateur.
3. Choisir **Exécution > Tout exécuter**, en conservant l'ordre des cellules.
4. Autoriser l'accès Internet pour les téléchargements de données et de poids. Un GPU est facultatif ; l'exécution sur CPU est possible, mais plus lente.

Le notebook contient déjà les commandes d'installation suivantes :

```python
!pip -q install ultralytics
!pip -q install datasets
!pip -q install pycocotools
```

Dans un environnement Jupyter local, installer au préalable les dépendances principales :

```bash
python -m pip install numpy pandas matplotlib scikit-image scikit-learn torch torchvision ultralytics datasets opencv-python pillow pycocotools jupyter
jupyter notebook Computer_Vision_Model_Eval.ipynb
```

**Attention :** plusieurs chemins du notebook sont prévus pour Colab (`/content/datasets`, `/content/evaluation`, `/content/coco_annotations`). Pour une exécution locale, adapter ces chemins ainsi que les emplacements de téléchargement. Les versions exactes des bibliothèques ne sont pas verrouillées.

## Expérience 1 — Segmentation d'images

**Modèles :** K-Means et Mean Shift.

**Données :** image `astronaut` de `scikit-image`, redimensionnée à **64 × 64 pixels**.

**Principe :** chaque pixel RGB est considéré comme un vecteur de trois caractéristiques. K-Means répartit les pixels en **5 clusters prédéfinis**, tandis que Mean Shift estime automatiquement les groupes selon une bande passante calculée sur les couleurs.

**Étapes du code :** chargement de l'image, conversion en tableau de pixels, ajustement des deux algorithmes, reconstruction des images segmentées, calcul du Silhouette Score sur un échantillon commun de pixels, puis visualisation comparative.

**Métriques :** Silhouette Score, temps d'exécution et nombre de clusters.

| Modèle | Silhouette Score | Temps (s) | Clusters |
|---|---:|---:|---:|
| K-Means | 0,5250 | 0,3112 | 5 |
| Mean Shift | **0,5593** | 0,5037 | 3 |

**Interprétation :** Mean Shift présente une meilleure séparation des groupes dans l'espace RGB, tandis que K-Means est plus rapide. Il s'agit d'une segmentation par couleur, sans reconnaissance sémantique des objets.

## Expérience 2 — Détection d'objets

**Modèles :** `yolov8n.pt` et `yolo11n.pt` (poids pré-entraînés).

**Données :** **COCO128**, téléchargé depuis les ressources Ultralytics ; images et annotations de boîtes englobantes.

**Étapes du code :** chargement des deux modèles, prédiction sur une même image (`conf=0.25`, `imgsz=640`), affichage des boîtes et classes, validation sur COCO128 avec `model.val()`, comparaison des métriques et mesure répétée des temps d'inférence après échauffement.

**Métriques :** précision, rappel, mAP@50, mAP@50:95, temps moyen de prédiction.

| Métrique | YOLOv8n | YOLO11n |
|---|---:|---:|
| Précision | 0,6287 | **0,6403** |
| Rappel | 0,5291 | **0,6082** |
| mAP@50 | 0,6024 | **0,6707** |
| mAP@50:95 | 0,4439 | **0,5021** |
| Temps moyen (ms/image) | **185,54** | 210,15 |

**Interprétation :** YOLO11n obtient les meilleures métriques de détection sur cet échantillon, alors que YOLOv8n est légèrement plus rapide dans l'environnement testé.

## Expérience 3A — Segmentation sémantique

**Modèles :** `fcn_resnet50` et `deeplabv3_resnet50` de torchvision, avec poids pré-entraînés.

**Données :** **Pascal VOC 2012**, split `val`, chargé en streaming depuis `nateraw/pascal-voc-2012` sur Hugging Face ; **30 images** utilisées pour l'évaluation.

**Étapes du code :** chargement des images et masques, prétraitement des images à 320 × 320 avec normalisation ImageNet, inférence, redimensionnement des prédictions aux dimensions originales, conversion des masques RGB en identifiants de classes, accumulation d'une matrice de confusion et calcul des scores.

**Métriques :** Pixel Accuracy, mIoU, Mean Dice et temps moyen par image.

| Métrique | FCN-ResNet50 | DeepLabV3-ResNet50 |
|---|---:|---:|
| Pixel Accuracy | 0,9532 | **0,9603** |
| mIoU | 0,6814 | **0,7186** |
| Mean Dice | 0,7625 | **0,7897** |
| Temps (s/image) | **2,9407** | 3,0379 |

**Interprétation :** DeepLabV3 obtient de meilleurs scores de segmentation, avec un temps moyen légèrement supérieur. **Point de vigilance :** les masques de cette version du jeu de données sont RGB ; leur correspondance avec les classes Pascal VOC doit être vérifiée sur l'ensemble des images. Les scores ci-dessus dépendent de la validité de cette conversion.

## Expérience 3B — Segmentation d'instances

**Modèles :** `maskrcnn_resnet50_fpn` (torchvision) et `yolo11n-seg.pt` (Ultralytics), avec poids pré-entraînés.

**Données :** images de **COCO128** et annotations officielles `instances_train2017.json` ; évaluation sur **30 images** avec **184 annotations**.

**Étapes du code :** chargement des modèles, affichage de masques individuels sur une image commune, conversion des prédictions en format COCO (identifiants de catégories, masques RLE et scores), évaluation via `COCOeval(..., "segm")`, puis comparaison des métriques et des temps moyens.

**Métriques :** Mask mAP@50:95, Mask mAP@50, Mask AP@75 et temps moyen par image.

| Métrique | Mask R-CNN | YOLO11n-seg |
|---|---:|---:|
| Mask mAP@50:95 | **0,5282** | 0,4473 |
| Mask mAP@50 | **0,7930** | 0,6747 |
| Mask AP@75 | **0,5888** | 0,4686 |
| Temps (s/image) | 9,8652 | **0,7504** |

**Interprétation :** Mask R-CNN présente une meilleure qualité des masques selon les métriques utilisées. YOLO11n-seg est environ **13,1 fois plus rapide** dans cette expérience.

## Différences entre les tâches

| Tâche | Sortie du modèle | Exemple |
|---|---|---|
| Segmentation par clustering | Groupes de pixels de couleurs similaires | K-Means, Mean Shift |
| Détection d'objets | Boîtes englobantes, classes et scores | YOLOv8n, YOLO11n |
| Segmentation sémantique | Classe attribuée à chaque pixel | FCN, DeepLabV3 |
| Segmentation d'instances | Masque distinct pour chaque objet | Mask R-CNN, YOLO11n-seg |

## Organisation du notebook

Les numéros ci-dessous correspondent aux **cellules du notebook fourni**, comptées à partir de 1 :

| Cellules | Contenu |
|---|---|
| 1–15 | Segmentation d'images : K-Means et Mean Shift |
| 16–36 | Détection d'objets : YOLOv8n et YOLO11n |
| 37–55 | Segmentation sémantique : FCN et DeepLabV3 |
| 56–73 | Segmentation d'instances : Mask R-CNN et YOLO11n-seg |
| 74 | Conclusion générale |

## Reproductibilité et limites

- Les modèles de deep learning sont **pré-entraînés** ; le notebook réalise de l'inférence et de l'évaluation, sans réentraînement.
- Les expériences ont été exécutées principalement sur **CPU**. Les temps varient selon la machine, les versions logicielles et la charge système.
- Les évaluations utilisent des **sous-ensembles réduits**. Les conclusions ne doivent pas être généralisées à tous les jeux de données.
- COCO128 contient des images issues de **COCO train2017** : les résultats obtenus ne constituent pas un benchmark indépendant de généralisation des modèles pré-entraînés.
- Les données, annotations et poids sont récupérés en ligne : leur téléchargement peut prendre du temps ou échouer selon la connexion.
- Le calcul des métriques sémantiques suppose une **conversion correcte des masques RGB vers les identifiants de classes**.
- Les résultats numériques présentés correspondent à l'exécution documentée dans le notebook et peuvent varier lors d'une nouvelle exécution.

## Références

- [Documentation Ultralytics](https://docs.ultralytics.com/)
- [Modèles torchvision](https://pytorch.org/vision/stable/models.html)
- [Documentation scikit-learn](https://scikit-learn.org/stable/)
- [Hugging Face Datasets](https://huggingface.co/docs/datasets/)
- [COCO Dataset](https://cocodataset.org/)

## Conclusion

Ce projet illustre l'intérêt d'une évaluation comparative tenant compte à la fois de la qualité prédictive et du temps de calcul. Selon la tâche et les contraintes opérationnelles, le modèle le plus précis n'est pas nécessairement le plus rapide.
