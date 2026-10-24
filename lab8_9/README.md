# Experiments 8, 9: Effect of Attention Type and Attention Position in a Pretrained ResNet50

**Course:** CS3807 -- Deep Learning Laboratory  
**Institution:** Shiv Nadar University Chennai  
**Department:** Department of Artificial Intelligence & Data Science  
**Degree & Branch:** B.Tech Artificial Intelligence & Data Science, Semester V  
**Academic Year:** 2026--27  
**Student Name:** CY Pranav Menon  
**Roll Number:** 24011101014  
**GitHub Repository:** [https://github.com/PranavMenon07/Deep-Learning](https://github.com/PranavMenon07/Deep-Learning)

---

## Table of Contents
- [Objective](#objective)
- [Folder Structure](#folder-structure)
- [Architectural Overview](#architectural-overview)
  - [ResNet50 Stage Hierarchy](#resnet50-stage-hierarchy)
  - [Attention Mechanisms Evaluated](#attention-mechanisms-evaluated)
- [Quantitative Performance Results](#quantitative-performance-results)
  - [Main Results Summary](#main-results-summary)
  - [Ablation Studies](#ablation-studies)
- [Key Visualizations & Interpretability](#key-visualizations--interpretability)
- [Environment Setup & Execution](#environment-setup--execution)
- [LaTeX Report Compilation](#latex-report-compilation)
- [References](#references)

---

## Objective

This study systematically investigates how different attention mechanisms modify convolutional representations in an ImageNet-pretrained ResNet50 and how the topological insertion depth influences visual classification performance on the **TensorFlow Flowers** dataset.

Specific aspects investigated include:
1. **Attention Types:** Channel Attention (SE style), Spatial Attention ($7\times7$ conv), Sequential Channel-Spatial Attention (CBAM), Lightweight Self-Attention, ECA-Net, and Coordinate Attention.
2. **Insertion Positions:** Early stage $P_1$ ($\text{conv2\_x}$), intermediate stages $P_2$ ($\text{conv3\_x}$) & $P_3$ ($\text{conv4\_x}$), and late stage $P_4$ ($\text{conv5\_x}$).
3. **Multi-Stage & Ordering:** Multi-stage attention ($P_3+P_4$, $P_2+P_3+P_4$) and operator sequencing ($C \to S$ vs.\ $S \to C$).
4. **Spatial Kernel Size:** Receptive field ablation ($k \in \{3\times3, 5\times5, 7\times7\}$).
5. **Interpretability:** Internal attention weights ($M_c$), spatial heatmaps ($M_s$), self-attention matrices, and post-hoc Grad-CAM visualizations.

See [`DATASET.md`](DATASET.md) for complete dataset details and class distribution.

---

## Folder Structure

```
lab8_9_attention_resnet50/
├── README.md                                             -- Project overview, results, and documentation
├── DATASET.md                                            -- Dataset description, statistics, and pipeline
├── requirements.txt                                      -- Python library dependencies
├── setup.sh                                              -- Shell script for automated environment setup & run
├── .gitignore                                            -- Excluded binaries, dataset cache, and build files
├── CS3807_Experiments_8_9_CY_Pranav_Menon_24011101014.pdf -- Final compiled 18-page submission report
├── report.tex                                            -- Full LaTeX report source code
├── Experiments_8_9_Attention_ResNet50.ipynb             -- Complete self-contained Jupyter Notebook
├── models.py                                             -- PyTorch implementations of all attention layers & ResNet50
├── run_experiments.py                                    -- Main training, evaluation & metrics logging script
├── generate_remaining_plots.py                           -- Grad-CAM and multi-stage attention plotting script
├── make_notebook.py                                      -- Automated notebook generator script
├── experiment_results.json                               -- Saved quantitative evaluation metrics
└── plots/                                                -- 16 high-resolution figures (300 DPI)
    ├── plot01_sample_dataset_images.png                  -- Class samples (daisy, dandelion, roses, sunflowers, tulips)
    ├── plot02_train_val_accuracy_curves.png              -- Training and validation accuracy curves
    ├── plot03_train_val_loss_curves.png                  -- Cross-entropy loss curves across epochs
    ├── plot04_confusion_matrices.png                     -- Baseline vs. CBAM confusion matrices
    ├── plot05_feature_maps_before_after.png              -- Intermediate feature activations before vs. after attention
    ├── plot06_channel_attention_weights.png              -- Top-20 learned channel-attention weights (Mc)
    ├── plot07_spatial_attention_heatmap_overlay.png      -- Spatial attention map (Ms) and overlay
    ├── plot08_self_attention_response.png                -- Self-attention affinity matrix & spatial overlay
    ├── plot09_gradcam_baseline_vs_attention.png          -- Baseline vs. CBAM Grad-CAM comparison
    ├── plot10_attention_maps_by_position.png             -- Spatial attention progression (P2 vs. P3 vs. P4)
    ├── plot11_accuracy_vs_parameters.png                 -- Accuracy vs. Total Parameters trade-off
    ├── plot12_f1_vs_inference_time.png                   -- Macro F1 vs. Inference latency per image
    ├── plot13_attention_type_comparison.png              -- Performance across attention types
    ├── plot14_attention_position_comparison.png          -- Performance across insertion stages
    ├── plot15_spatial_kernel_ablation.png                -- Kernel size ablation (3x3, 5x5, 7x7)
    └── plot16_qualitative_multi_image_comparison.png     -- Multi-model Grad-CAM across 5 test images
```

---

## Architectural Overview

### ResNet50 Stage Hierarchy

| Stage | Insertion Point | Output Resolution ($H \times W$) | Channels ($C$) | Representation Level |
|---|:---:|:---:|:---:|---|
| Stem + MaxPool | -- | $56 \times 56$ | 64 | Low-level edges, pixel gradients |
| $\text{conv2\_x}$ | $P_1$ | $56 \times 56$ | 256 | Local textures, contours |
| $\text{conv3\_x}$ | $P_2$ | $28 \times 28$ | 512 | Intermediate motifs, petal shapes |
| $\text{conv4\_x}$ | $P_3$ | $14 \times 14$ | 1024 | Floral sub-structures, stamen, sepals |
| $\text{conv5\_x}$ | $P_4$ | $7 \times 7$ | 2048 | High-level class-discriminative semantics |

### Attention Mechanisms Evaluated

1. **Channel Attention (Squeeze-and-Excitation):**
   $$z = \text{GAP}(F) \in \mathbb{R}^C, \quad M_c(F) = \sigma(W_2 \, \text{ReLU}(W_1 z)), \quad F_c = F \odot M_c(F)$$
2. **Spatial Attention:**
   $$M_s(F) = \sigma(\text{Conv}_{7\times7}([\text{Mean}_c(F) \,\|\, \text{Max}_c(F)])), \quad F_s' = F \odot M_s(F)$$
3. **CBAM (Sequential Channel-Spatial):**
   $$F_c = M_c(F) \odot F, \quad F_{cs} = M_s(F_c) \odot F_c$$
4. **Lightweight Self-Attention:**
   $$A = \text{softmax}(Q K^T / \sqrt{d}), \quad Y = A V, \quad F_{SA} = F + \gamma \, \text{reshape}(Y)$$
5. **ECA-Net (Efficient Channel Attention):**
   Adaptive 1D convolution of kernel size $k$ across GAP descriptors without dimensionality reduction.
6. **Coordinate Attention:**
   Decomposes spatial pooling into horizontal ($1 \times W$) and vertical ($H \times 1$) coordinate-aware directions.

---

## Quantitative Performance Results

### Main Results Summary

| Model Configuration | Position | Accuracy | Precision | Recall | Macro F1 | Total Parameters | Inf. Latency |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Baseline ResNet50** | -- | 87.00% | 88.12% | 86.40% | 85.39% | 23,518,277 | 45.80 ms |
| **Channel Attention (SE)** | $P_3$ | **91.00%** | **91.80%** | **90.15%** | **89.54%** | 23,649,349 | 39.99 ms |
| **Spatial Attention** | $P_3$ | 85.50% | 86.20% | 84.80% | 83.92% | 23,518,375 | 40.21 ms |
| **CBAM (Channel+Spatial)** | $P_2$ | 87.00% | 88.05% | 86.30% | 85.50% | 23,551,143 | 42.38 ms |
| **CBAM (Channel+Spatial)** | $P_3$ | 88.00% | 89.15% | 87.40% | 86.45% | 23,649,447 | 40.36 ms |
| **CBAM (Channel+Spatial)** | $P_4$ | 88.50% | 89.40% | 87.80% | 87.14% | 24,042,663 | 40.83 ms |
| **Self-Attention** | $P_3$ | 87.00% | 87.90% | 86.20% | 84.58% | 24,828,998 | 45.06 ms |
| **Combined Attention** | $P_3$ | 90.00% | 90.75% | 89.30% | 88.86% | 24,960,168 | 45.05 ms |
| **ECA-Net** | $P_3$ | 87.00% | 88.10% | 86.35% | 85.57% | **23,518,282** | 49.78 ms |
| **Coordinate Attention** | $P_3$ | 88.50% | 89.50% | 87.95% | 86.96% | 23,717,125 | 49.89 ms |

### Ablation Studies

- **Multi-Stage Insertion:** $P_3+P_4$ ($88.50\%$, $40.32\text{ ms}$), $P_2+P_3+P_4$ ($89.50\%$, $51.32\text{ ms}$). Adding multiple stages increases inference latency by $27.1\%$ with modest accuracy gain.
- **Attention Ordering:** Channel $\to$ Spatial ($88.00\%$, Macro F1 $86.45\%$) vs.\ Spatial $\to$ Channel ($89.00\%$, Macro F1 $87.72\%$).
- **Spatial Kernel Size:** $3\times3$ ($85.00\%$), $5\times5$ (**87.00%**), $7\times7$ ($85.50\%$). A $5\times5$ receptive field achieves optimal spatial context on $14\times14$ feature tensors.

---

## Key Visualizations & Interpretability

| Visualization | Description | File Path |
|---|---|---|
| **Class Samples** | Representative images from all 5 flower classes | [`plots/plot01_sample_dataset_images.png`](plots/plot01_sample_dataset_images.png) |
| **Training Curves** | Accuracy & Loss convergence across epochs | [`plots/plot02_train_val_accuracy_curves.png`](plots/plot02_train_val_accuracy_curves.png) |
| **Confusion Matrix** | Error distribution before and after CBAM attention | [`plots/plot04_confusion_matrices.png`](plots/plot04_confusion_matrices.png) |
| **Internal Weights** | Top-20 learned channel-attention coefficients ($M_c$) | [`plots/plot06_channel_attention_weights.png`](plots/plot06_channel_attention_weights.png) |
| **Spatial Overlay** | 2D Spatial attention heatmap & input overlay | [`plots/plot07_spatial_attention_heatmap_overlay.png`](plots/plot07_spatial_attention_heatmap_overlay.png) |
| **Grad-CAM Comparison** | Baseline vs.\ CBAM gradient evidence localization | [`plots/plot09_gradcam_baseline_vs_attention.png`](plots/plot09_gradcam_baseline_vs_attention.png) |
| **Depth Progression** | Stage-wise spatial attention ($P_2 \to P_3 \to P_4$) | [`plots/plot10_attention_maps_by_position.png`](plots/plot10_attention_maps_by_position.png) |
| **Pareto Trade-off** | Accuracy vs.\ Model Parameter complexity | [`plots/plot11_accuracy_vs_parameters.png`](plots/plot11_accuracy_vs_parameters.png) |
| **Multi-Image CAM** | Comparison on 3 correct & 2 misclassified samples | [`plots/plot16_qualitative_multi_image_comparison.png`](plots/plot16_qualitative_multi_image_comparison.png) |

---

## Environment Setup & Execution

### 1. Clone & Install Dependencies
```bash
git clone https://github.com/PranavMenon07/Deep-Learning.git
cd Deep-Learning/lab8_9_attention_resnet50
pip install -r requirements.txt
```

### 2. Run Experiments & Generate Figures
```bash
python run_experiments.py
python generate_remaining_plots.py
```

### 3. Launch Jupyter Notebook
```bash
jupyter notebook Experiments_8_9_Attention_ResNet50.ipynb
```

---

## LaTeX Report Compilation

The submission PDF is typeset with LaTeX and compiled using [Tectonic](https://tectonic-typesetting.github.io/en-US/):
```bash
tectonic report.tex
# Generates report.pdf (18 pages)
```

---

## References
1. K. He, X. Zhang, S. Ren, and J. Sun, "Deep Residual Learning for Image Recognition," *IEEE CVPR*, 2016.
2. J. Hu, L. Shen, S. Albanie, G. Sun, and E. Wu, "Squeeze-and-Excitation Networks," *IEEE TPAMI*, 2020.
3. S. Woo, J. Park, J.-Y. Lee, and I. S. Kweon, "CBAM: Convolutional Block Attention Module," *ECCV*, 2018.
4. Q. Wang, B. Wu, P. Zhu, P. Li, W. Zuo, and Q. Hu, "ECA-Net: Efficient Channel Attention for Deep Convolutional Neural Networks," *IEEE CVPR*, 2020.
5. Q. Hou, D. Zhou, and J. Feng, "Coordinate Attention for Efficient Mobile Network Design," *IEEE CVPR*, 2021.
6. A. Vaswani et al., "Attention Is All You Need," *NeurIPS*, 2017.
7. R. R. Selvaraju et al., "Grad-CAM: Visual Explanations from Deep Networks via Gradient-Based Localization," *IEEE ICCV*, 2017.
