# Dataset: TensorFlow Flowers (`tf_flowers`)

**Official Source:** [TensorFlow Datasets - tf_flowers](https://www.tensorflow.org/datasets/catalog/tf_flowers)  
**Direct Download URL:** `https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz`

---

## Description

The TensorFlow Flowers dataset contains natural RGB flower photographs belonging to five botanical categories, photographed under diverse real-world lighting conditions, perspectives, and environmental backgrounds.

| Property | Value |
|---|---|
| **Total Instances** | 3,670 images |
| **Number of Classes** | 5 flower classes |
| **Class Names** | Daisy, Dandelion, Roses, Sunflowers, Tulips |
| **Format** | RGB JPEG images of variable native resolution |
| **Input Resolution** | Resized to $224 \times 224 \times 3$ |
| **Normalization** | ImageNet statistics ($\mu = [0.485, 0.456, 0.406]$, $\sigma = [0.229, 0.224, 0.225]$) |
| **Partitioning Split** | 70% Train (2,569) : 15% Validation (550) : 15% Test (551) |
| **Task** | Multi-class visual category classification and attention interpretability |

---

## Class Distribution

| Class Index | Class Name | Total Sample Count | Visual Characteristics |
|:---:|:---:|:---:|:---|
| 0 | **Daisy** | 633 | White radiating petals with yellow disc florets; fine texture |
| 1 | **Dandelion** | 898 | Dense yellow serrated ray florets; high intra-class background foliage |
| 2 | **Roses** | 641 | Multilayered overlapping petals; subtle color variations and thorns |
| 3 | **Sunflowers** | 699 | Large dark center disk surrounded by broad yellow petals |
| 4 | **Tulips** | 799 | Smooth cup-shaped petals; distinct vertical stem morphology |
| **Total** | | **3,670** | |

---

## Preprocessing & Data Augmentation Pipeline

To guarantee unbiased architectural comparisons across all attention mechanisms and insertion stages, an identical preprocessing and augmentation pipeline is maintained:

```python
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]),
    'val_test': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
}
```

---

## Download & Automatic Extraction

The dataset archive (~218 MB) is automatically downloaded and unpacked by `run_experiments.py` or `setup.sh`:

```bash
mkdir -p dataset
curl -o dataset/flower_photos.tgz https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz
tar -xzf dataset/flower_photos.tgz -C dataset/
```
